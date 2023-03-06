use std::{
    ops::{Deref, DerefMut},
    rc::Rc,
};

#[cfg(not(short_floats))]
pub type NodeValue = f64;

#[cfg(short_floats)]
type NodeValue = f32;

#[cfg(test)]
#[derive(Default)]
pub struct JsRng(std::cell::RefCell<rand::rngs::ThreadRng>);

#[cfg(not(test))]
#[derive(Default)]
pub struct JsRng;

impl RNG for JsRng {
    #[cfg(test)]
    fn rand(&self) -> NodeValue {
        use rand::Rng;
        self.0.borrow_mut().gen()
    }
    #[cfg(not(test))]
    fn rand(&self) -> NodeValue {
        js_sys::Math::random() as NodeValue
    }
}

pub trait RNG {
    fn rand(&self) -> NodeValue;
    fn rand_range(&self, min: usize, exclusive_max: usize) -> usize {
        (self.rand() * (exclusive_max - min) as NodeValue) as usize + min
    }
}

pub trait ShuffleRng {
    fn shuffle_vec<T>(&self, vec: &mut Vec<T>);
}

impl<T: RNG> ShuffleRng for T {
    fn shuffle_vec<E>(&self, vec: &mut Vec<E>) {
        let len = vec.len();

        for i in 0..len {
            let j = self.rand_range(i, len);
            vec.swap(i, j);
        }
    }
}
impl ShuffleRng for Rc<dyn RNG> {
    fn shuffle_vec<E>(&self, vec: &mut Vec<E>) {
        let len = vec.len();

        for i in 0..len {
            let j = self.rand_range(i, len);
            vec.swap(i, j);
        }
    }
}

#[derive(Debug)]
pub struct NetworkShape {
    inputs_count: (usize, LayerInitStrategy),
    outputs_count: (usize, LayerInitStrategy),
    hidden_layer_shape: Vec<(usize, LayerInitStrategy)>,
}

impl NetworkShape {
    pub fn new(
        inputs_count: usize,
        outputs_count: usize,
        hidden_layer_shape: Vec<usize>,
        init_stratergy: LayerInitStrategy,
    ) -> Self {
        Self {
            inputs_count: (inputs_count, LayerInitStrategy::Zero),
            outputs_count: (outputs_count, init_stratergy.clone()),
            hidden_layer_shape: hidden_layer_shape
                .into_iter()
                .map(|layer_size| (layer_size, init_stratergy.clone()))
                .collect(),
        }
    }

    pub fn new_embedding(
        token_count: usize,
        embedding_size: usize,
        hidden_layer_shape: Vec<usize>,
        hidden_layer_init_stratergy: LayerInitStrategy,
        rng: Rc<dyn RNG>,
    ) -> Self {
        let embedding_layer_shape = [(embedding_size, LayerInitStrategy::NoBias(rng))];
        let embedding_layer_shape = embedding_layer_shape.into_iter();
        let hidden_layer_shape = hidden_layer_shape.into_iter();
        let stratergy = hidden_layer_init_stratergy;
        Self {
            inputs_count: (token_count, LayerInitStrategy::Zero),
            outputs_count: (token_count, stratergy.clone()),
            hidden_layer_shape: embedding_layer_shape
                .chain(hidden_layer_shape.map(|n| (n, stratergy.clone())))
                .collect(),
        }
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (usize, LayerInitStrategy)> + 'a {
        [&self.inputs_count]
            .into_iter()
            .chain(self.hidden_layer_shape.iter())
            .chain([&self.outputs_count])
            .cloned()
    }
}

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
    activation_mode: NetworkActivationMode,
    softmax_output_enabled: bool,
    first_layer_activation_enabled: bool,
}

impl Network {
    pub fn new(shape: &NetworkShape) -> Self {
        let mut layers = vec![];

        for pair in shape.iter().collect::<Vec<_>>().windows(2) {
            if let [(lhs, _), (rhs, init_stratergy)] = pair {
                layers.push(Layer::new(*rhs, *lhs, &init_stratergy))
            } else {
                panic!("failed to window layer shape");
            }
        }

        Self {
            layers,
            activation_mode: Default::default(),
            softmax_output_enabled: false,
            first_layer_activation_enabled: true,
        }
    }

    pub fn set_activation_mode(&mut self, activation_mode: NetworkActivationMode) {
        self.activation_mode = activation_mode;
    }

    pub fn set_softmax_output_enabled(&mut self, enabled: bool) {
        self.softmax_output_enabled = enabled;
    }

    pub fn set_first_layer_activation_enabled(&mut self, enabled: bool) {
        self.first_layer_activation_enabled = enabled;
    }

    pub fn compute(&self, inputs: LayerValues) -> Result<LayerValues, ()> {
        inputs.assert_length_equals(self.inputs_count()?)?;

        let (last_layer, layers) = self.layers.split_last().expect("should have final layer");
        let mut output = inputs;

        for (layer_idx, layer) in layers.iter().enumerate() {
            output = layer.compute(&output)?;
            if self.first_layer_activation_enabled || layer_idx != 0 {
                output = self.activation_mode.apply(&output);
            }
        }

        output = last_layer.compute(&output)?;
        output = self.final_layer_activation().apply(&output);

        Ok(output)
    }

    pub fn compute_error(
        &self,
        inputs: LayerValues,
        target_outputs: &LayerValues,
    ) -> Result<LayerValues, ()> {
        let last_layer = self.layers.last().ok_or(())?;
        let outputs = self.compute(inputs)?;
        let errors = last_layer.calculate_error(&outputs, &target_outputs)?;
        Ok(errors)
    }

    pub fn learn(&mut self, strategy: NetworkLearnStrategy) -> Result<Option<NodeValue>, ()> {
        let cost = match strategy {
            NetworkLearnStrategy::MutateRng { rand_amount, rng } => {
                let strategy = LayerLearnStrategy::MutateRng {
                    weight_factor: rand_amount,
                    bias_factor: rand_amount,
                    rng,
                };
                for layer in self.layers.iter_mut() {
                    layer.learn(&strategy)?;
                }

                None
            }
            NetworkLearnStrategy::GradientDecent {
                inputs,
                target_outputs,
                learn_rate,
            } => {
                let layers_activations = self.compute_with_layers_activations(inputs)?;

                let output_layer_error = self.perform_gradient_decent_step(
                    layers_activations,
                    target_outputs,
                    learn_rate,
                    false,
                )?;

                Some(output_layer_error)
            }
            NetworkLearnStrategy::BatchGradientDecent {
                training_pairs,
                batch_size,
                learn_rate,
                batch_sampling,
            } => {
                use itertools::Itertools;
                if training_pairs.is_empty() {
                    return Ok(None);
                }

                let mut sum_error = 0.0;
                let training_pairs_len = training_pairs.len();

                let batches = match batch_sampling {
                    BatchSamplingStrategy::Sequential => {
                        training_pairs.into_iter().chunks(batch_size)
                    }
                    BatchSamplingStrategy::Shuffle(rng) => {
                        let mut training_pairs = training_pairs;
                        rng.shuffle_vec(&mut training_pairs);
                        training_pairs.into_iter().chunks(batch_size)
                    }
                };

                for training_pairs in batches.into_iter() {
                    for (inputs, target_outputs) in training_pairs {
                        let layers_activations = self.compute_with_layers_activations(inputs)?;

                        let output_layer_error = self.perform_gradient_decent_step(
                            layers_activations,
                            target_outputs,
                            learn_rate,
                            true,
                        )?;

                        sum_error += output_layer_error;
                    }
                }
                if training_pairs_len > 0 {
                    for layer in self.layers.iter_mut() {
                        layer.apply_batch_gradients(learn_rate)?;
                    }
                }

                Some(sum_error / training_pairs_len as NodeValue)
            }
            NetworkLearnStrategy::Multi(strategies) => {
                for strategy in strategies.into_iter() {
                    self.learn(strategy)?;
                }

                None
            }
        };
        Ok(cost)
    }

    fn compute_with_layers_activations(
        &mut self,
        inputs: LayerValues,
    ) -> Result<Vec<LayerValues>, ()> {
        let mut layers_activations = vec![inputs];
        let mut output = layers_activations.first().expect("missing input values");
        let (last_layer, layers) = self.layers.split_last().expect("should have final layer");

        for (layer_idx, layer) in layers.iter().enumerate() {
            let x = layer.compute(&output)?;
            let x = if self.first_layer_activation_enabled || layer_idx != 0 {
                self.activation_mode.apply(&x)
            } else {
                x
            };

            layers_activations.push(x);
            output = layers_activations
                .last()
                .expect("missing latest pushed layer");
        }

        let x = last_layer.compute(&output)?;
        let x = self.final_layer_activation().apply(&x);

        layers_activations.push(x);
        Ok(layers_activations)
    }

    fn perform_gradient_decent_step(
        &mut self,
        layers_activations: Vec<LayerValues>,
        target_outputs: LayerValues,
        learn_rate: f64,
        enqueue_batch: bool,
    ) -> Result<f64, ()> {
        let mut layer_d = None;
        let mut output_layer_error = 0.0;
        let final_activation_mode = self.final_layer_activation();

        for ((layer_idx, layer), layer_input) in self
            .layers
            .iter_mut()
            .enumerate()
            .rev()
            .zip(layers_activations.iter().rev().skip(1))
        {
            // key
            //  c <-> cost
            //  a <-> layer activation
            //  z <-> weighted inputs
            //  w <-> weigths
            //
            //  dc/dw = dz/dw * da/dz * dc/da
            //    da/dz = activation_d
            //    dc/da = error_d
            //    dz/dw = layer.learn (network layer inputs)

            // use stored weighted input sums here instead?
            let x = layer.compute(&layer_input)?;

            let (layer_activation_mode, error_d) = match layer_d.take() {
                Some(prev_layer_error_d) => {
                    let error_d = LayerValues(prev_layer_error_d);
                    let activation_mode = if self.first_layer_activation_enabled || layer_idx != 0 {
                        self.activation_mode
                    } else {
                        NetworkActivationMode::Linear
                    };

                    (activation_mode, error_d)
                }
                None => {
                    let last = layers_activations.last();
                    let layer_output = last.expect("should have a final layer");
                    let error = layer.calculate_error(&layer_output, &target_outputs)?.ave();
                    let error_d = layer.calculate_error_d(&layer_output, &target_outputs)?;
                    output_layer_error = error;

                    (final_activation_mode, error_d)
                }
            };

            let activation_d = layer_activation_mode.derivative(&x);
            let activated_layer_d = activation_d.multiply_iter(&error_d);
            let activated_layer_d = activated_layer_d.collect();
            let input_gradients = layer.compute_d(&activated_layer_d)?;
            let input_gradients = input_gradients.collect::<Vec<NodeValue>>();

            let strategy = {
                let gradients = activated_layer_d.clone();
                let layer_inputs = layer_input.clone(); // TODO: can remove clone?
                match enqueue_batch {
                    true => LayerLearnStrategy::EnqueueBatchGradientDecent {
                        gradients,
                        layer_inputs,
                    },
                    false => LayerLearnStrategy::GradientDecent {
                        gradients,
                        layer_inputs,
                        learn_rate,
                    },
                }
            };

            layer.learn(&strategy)?;
            layer_d = Some(input_gradients);
        }
        Ok(output_layer_error)
    }

    pub fn node_weights(
        &self,
        layer_idx: usize,
        node_index: usize,
    ) -> Result<impl Iterator<Item = &f64>, ()> {
        Ok(self
            .layers
            .get(layer_idx)
            .ok_or(())?
            .node_weights(node_index))
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    fn inputs_count(&self) -> Result<usize, ()> {
        let first_layer = &self.layers.iter().next().ok_or(())?;
        Ok(first_layer.inputs_count)
    }

    fn final_layer_activation(&self) -> NetworkActivationMode {
        if self.softmax_output_enabled {
            NetworkActivationMode::SoftMax
        } else {
            self.activation_mode
        }
    }
}

#[derive(Debug)]
pub struct Layer {
    weights: Vec<NodeValue>,
    bias: Option<Vec<NodeValue>>,
    pending_gradients: Option<PendingLayerGradients>,
    inputs_count: usize,
    size: usize,
}

pub struct PendingLayerGradients {
    weights: Vec<NodeValue>,
    bias: Vec<NodeValue>,
}

impl std::fmt::Debug for PendingLayerGradients {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingLayerGradients")
            .field("weights_count", &self.weights.len())
            .field("bias_count", &self.bias.len())
            .finish()
    }
}

impl PendingLayerGradients {
    pub fn new(parent: &Layer) -> Self {
        Self {
            weights: vec![0.0; parent.inputs_count * parent.size()],
            bias: vec![0.0; parent.size()],
        }
    }
}

impl Layer {
    pub fn new(size: usize, inputs_count: usize, init_stratergy: &LayerInitStrategy) -> Self {
        let weights_size = inputs_count * size;
        let mut layer = Self {
            weights: vec![0.0; weights_size],
            bias: match &init_stratergy {
                LayerInitStrategy::NoBias(_) => None,
                _ => Some(vec![0.0; size]),
            },
            pending_gradients: None,
            inputs_count,
            size,
        };

        init_stratergy.apply(
            layer
                .weights
                .iter_mut()
                .chain(layer.bias.iter_mut().flatten()),
            inputs_count,
        );

        layer
    }

    pub fn compute(&self, inputs: &LayerValues) -> Result<LayerValues, ()> {
        inputs
            .assert_length_equals(self.inputs_count)
            .map_err(|_| format!("{self:?}"))
            .unwrap();

        let mut outputs = match &self.bias {
            Some(bias) => bias.clone(),
            None => vec![0.0; self.size],
        };

        for (input_index, input) in inputs.iter().enumerate() {
            let weights = self.node_weights(input_index);
            for (output_idx, weight) in weights.enumerate() {
                outputs[output_idx] += input * weight;
            }
        }

        Ok(LayerValues(outputs))
    }

    pub fn compute_d<'a>(
        &'a self,
        forward_layer_gradients: &'a LayerValues,
    ) -> Result<impl Iterator<Item = NodeValue> + 'a, ()> {
        let iter = self.weights.chunks(self.size()).map(|weights| {
            weights
                .iter()
                .zip(forward_layer_gradients.iter())
                .map(|(weight, next_layer_grad)| next_layer_grad * weight)
                .sum::<NodeValue>()
        });

        Ok(iter)
    }

    pub fn learn(&mut self, strategy: &LayerLearnStrategy) -> Result<(), ()> {
        match strategy {
            LayerLearnStrategy::MutateRng {
                weight_factor,
                bias_factor,
                rng,
            } => {
                for x in self.weights.iter_mut() {
                    let old = *x;
                    *x = old + (old * weight_factor * rng.rand())
                }
                for x in self.bias.iter_mut().flatten() {
                    let old = *x;
                    *x = old + (old * bias_factor * rng.rand())
                }
            }
            LayerLearnStrategy::GradientDecent {
                gradients,
                layer_inputs,
                learn_rate,
            } => {
                let (weights_gradients, bias_gradients) =
                    self.compute_gradients_iter(gradients, layer_inputs)?;

                for (x, grad) in self.weights.iter_mut().zip(weights_gradients) {
                    *x = *x - (grad * learn_rate)
                }
                for (x, grad) in self.bias.iter_mut().flatten().zip(bias_gradients) {
                    *x = *x - (grad * learn_rate)
                }
            }
            LayerLearnStrategy::EnqueueBatchGradientDecent {
                gradients,
                layer_inputs,
            } => {
                let (weights_gradients, bias_gradients) =
                    self.compute_gradients_iter(gradients, layer_inputs)?;

                let pending_gradients = self
                    .pending_gradients
                    .get_or_insert(PendingLayerGradients::new(&self));

                for (grad, pending_grad) in
                    weights_gradients.zip(pending_gradients.weights.iter_mut())
                {
                    *pending_grad += grad;
                }
                for (grad, pending_grad) in bias_gradients.zip(pending_gradients.bias.iter_mut()) {
                    *pending_grad += grad;
                }
            }
        }
        Ok(())
    }

    pub fn apply_batch_gradients(&mut self, learn_rate: f64) -> Result<(), ()> {
        let pending_gradients = self.pending_gradients.take().ok_or(())?;

        for (x, grad) in self
            .weights
            .iter_mut()
            .zip(pending_gradients.weights.iter())
        {
            *x = *x - (grad * learn_rate)
        }

        for (x, grad) in self
            .bias
            .iter_mut()
            .flatten()
            .zip(pending_gradients.bias.iter())
        {
            *x = *x - (grad * learn_rate)
        }

        Ok(())
    }

    fn compute_gradients_iter<'a>(
        &self,
        gradients: &'a LayerValues,
        layer_inputs: &'a LayerValues,
    ) -> Result<
        (
            impl Iterator<Item = NodeValue> + 'a,
            impl Iterator<Item = NodeValue> + 'a,
        ),
        (),
    > {
        if self.weights.len() != gradients.len() * layer_inputs.len() {
            dbg!(self.weights.len(), gradients.len() * layer_inputs.len());
            return Err(());
        }

        let weights_gradients = layer_inputs
            .iter()
            .flat_map(|layer_input| gradients.iter().map(move |grad| (layer_input, grad)))
            .map(move |(layer_input, grad)| grad * *layer_input);

        let bias_gradients = gradients.iter().cloned();

        Ok((weights_gradients, bias_gradients))
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn calculate_error(
        &self,
        outputs: &LayerValues,
        expected_outputs: &LayerValues,
    ) -> Result<LayerValues, ()> {
        let mut error = vec![];
        for (actual, expected) in outputs.iter().zip(expected_outputs.iter()) {
            error.push((actual - expected).powi(2));
        }

        Ok(LayerValues(error))
    }

    pub fn calculate_error_d(
        &self,
        outputs: &LayerValues,
        expected_outputs: &LayerValues,
    ) -> Result<LayerValues, ()> {
        let mut error = vec![];
        for (actual, expected) in outputs.iter().zip(expected_outputs.iter()) {
            error.push((actual - expected) * 2.0);
        }

        Ok(LayerValues(error))
    }

    fn node_weights<'a>(&'a self, node_index: usize) -> impl Iterator<Item = &'a NodeValue> {
        // TODO: handle invalid idx case
        let n = self.size();
        let start = n * node_index;
        self.weights[start..start + n].into_iter()
    }
}

#[derive(Debug, Clone)]
pub struct LayerValues(Vec<NodeValue>);

impl<'a, T> From<T> for LayerValues
where
    T: Iterator<Item = &'a NodeValue>,
{
    fn from(value: T) -> Self {
        Self(value.cloned().collect())
    }
}

impl FromIterator<NodeValue> for LayerValues {
    fn from_iter<T: IntoIterator<Item = NodeValue>>(iter: T) -> Self {
        LayerValues(iter.into_iter().collect())
    }
}

impl Deref for LayerValues {
    type Target = Vec<NodeValue>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for LayerValues {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl LayerValues {
    pub fn new(inner: Vec<NodeValue>) -> Self {
        Self(inner)
    }
    fn assert_length_equals(&self, expected: usize) -> Result<(), ()> {
        if self.len() == expected {
            Ok(())
        } else {
            Err(())
        }
    }
    fn multiply_iter<'a>(&'a self, rhs: &'a Self) -> impl Iterator<Item = NodeValue> + 'a {
        assert_eq!(self.len(), rhs.len());
        self.multiply_by_iter(rhs.iter().copied())
    }
    fn multiply_by_iter<'a, T: Iterator<Item = NodeValue> + 'a>(
        &'a self,
        rhs: T,
    ) -> impl Iterator<Item = NodeValue> + 'a {
        self.iter().zip(rhs).map(|(x, y)| x * y)
    }
    pub fn ave(&self) -> NodeValue {
        if !self.is_empty() {
            self.iter().sum::<NodeValue>() / self.len() as NodeValue
        } else {
            0.0
        }
    }
    pub fn normalized_dot_product(&self, rhs: &LayerValues) -> Option<NodeValue> {
        let (v1, v2) = (self, rhs);
        if v1.len() != v2.len() {
            return None;
        }

        let dot_product = v1
            .iter()
            .zip(v2.iter())
            .map(|(&x, &y)| x * y)
            .sum::<NodeValue>();

        let norm1 = v1.iter().map(|&x| x * x).sum::<NodeValue>().sqrt();
        let norm2 = v2.iter().map(|&x| x * x).sum::<NodeValue>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return None;
        }

        Some(dot_product / (norm1 * norm2))
    }
}

#[derive(Clone)]
pub enum LayerInitStrategy {
    Zero,
    PositiveRandom(Rc<dyn RNG>),
    ScaledFullRandom(Rc<dyn RNG>),
    FullRandom(Rc<dyn RNG>),
    NoBias(Rc<dyn RNG>),
}

impl std::fmt::Debug for LayerInitStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zero => write!(f, "Zero"),
            Self::PositiveRandom(arg0) => f.debug_tuple("PositiveRandom").finish(),
            Self::ScaledFullRandom(arg0) => f.debug_tuple("ScaledFullRandom").finish(),
            Self::FullRandom(arg0) => f.debug_tuple("FullRandom").finish(),
            Self::NoBias(arg0) => f.debug_tuple("NoBias").finish(),
        }
    }
}

impl LayerInitStrategy {
    pub fn apply<'a>(&self, values: impl Iterator<Item = &'a mut NodeValue>, inputs_count: usize) {
        use LayerInitStrategy::*;

        match self {
            Zero => {}
            PositiveRandom(rng) => {
                let scale_factor = (inputs_count as NodeValue).powf(-0.5);
                for value in values {
                    *value = rng.rand() * scale_factor;
                }
            }
            ScaledFullRandom(rng) | NoBias(rng) => {
                let scale_factor = (inputs_count as NodeValue).powf(-0.5) * 2.0;
                for value in values {
                    *value = Self::full_rand(rng) * scale_factor;
                }
            }
            FullRandom(rng) => {
                for value in values {
                    *value = Self::full_rand(rng);
                }
            }
        }
    }
    fn full_rand(rng: &Rc<dyn RNG>) -> NodeValue {
        rng.rand() * 2.0 - 1.0
    }
}

pub enum LayerLearnStrategy {
    MutateRng {
        weight_factor: NodeValue,
        bias_factor: NodeValue,
        rng: Rc<dyn RNG>,
    },
    GradientDecent {
        gradients: LayerValues,
        layer_inputs: LayerValues,
        learn_rate: NodeValue,
    },
    EnqueueBatchGradientDecent {
        gradients: LayerValues,
        layer_inputs: LayerValues,
    },
}

pub enum NetworkLearnStrategy {
    MutateRng {
        rand_amount: NodeValue,
        rng: Rc<dyn RNG>,
    },
    GradientDecent {
        inputs: LayerValues,
        target_outputs: LayerValues,
        learn_rate: NodeValue,
    },
    BatchGradientDecent {
        training_pairs: Vec<(LayerValues, LayerValues)>,
        batch_size: usize,
        learn_rate: NodeValue,
        batch_sampling: BatchSamplingStrategy,
    },
    Multi(Vec<NetworkLearnStrategy>),
}

pub enum BatchSamplingStrategy {
    Sequential,
    Shuffle(Rc<dyn RNG>),
}

#[derive(Debug, Clone, Copy)]
pub enum NetworkActivationMode {
    Linear,
    SoftMax,
    Sigmoid,
    Tanh,
    RelU,
}

impl NetworkActivationMode {
    pub fn apply(&self, output: &LayerValues) -> LayerValues {
        match self {
            NetworkActivationMode::Linear => output.clone(),
            NetworkActivationMode::SoftMax => {
                let exp_iter = output.iter().map(|x| x.exp());
                let sum: NodeValue = exp_iter.clone().sum();
                LayerValues(exp_iter.map(|x| x / sum).collect())
            }
            NetworkActivationMode::Sigmoid => {
                LayerValues(output.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect())
            }
            NetworkActivationMode::Tanh => LayerValues(output.iter().map(|x| x.tanh()).collect()),
            NetworkActivationMode::RelU => LayerValues(output.iter().map(|x| x.max(0.0)).collect()),
        }
    }
    fn derivative(&self, output: &LayerValues) -> LayerValues {
        match self {
            NetworkActivationMode::Linear => output.iter().map(|_| 1.0).collect(),
            NetworkActivationMode::SoftMax => {
                let softmax = self.apply(output);
                softmax
                    .iter()
                    .enumerate()
                    .map(|(j, sj)| {
                        softmax
                            .iter()
                            .enumerate()
                            .map(|(i, si)| if i == j { si * (1.0 - si) } else { si * -sj })
                            .sum::<NodeValue>()
                    })
                    .collect()
            }
            NetworkActivationMode::Sigmoid => {
                let mut sigmoid = self.apply(output);
                sigmoid.iter_mut().for_each(|x| *x = *x * (1.0 - *x));
                sigmoid
            }
            NetworkActivationMode::Tanh => {
                let mut sigmoid = self.apply(output);
                sigmoid.iter_mut().for_each(|x| *x = 1.0 - x.powi(2));
                sigmoid
            }
            NetworkActivationMode::RelU => output.iter().map(|x| x.signum().max(0.0)).collect(),
        }
    }
}

impl Default for NetworkActivationMode {
    fn default() -> Self {
        Self::SoftMax
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;

    #[test]
    fn can_create_network() {
        let shape = NetworkShape::new(2, 2, vec![], LayerInitStrategy::Zero);
        let network = Network::new(&shape);

        assert_eq!(2, network.inputs_count().unwrap());
    }

    #[test]
    fn can_create_hidden_layer_network() {
        let shape = NetworkShape::new(2, 2, vec![8], LayerInitStrategy::Zero);
        let network = Network::new(&shape);

        assert_eq!(2, network.inputs_count().unwrap());
    }

    #[test]
    fn can_compute_hidden_layer_network_zeros_linear_activation() {
        let shape = NetworkShape::new(2, 2, vec![8], LayerInitStrategy::Zero);
        let mut network = Network::new(&shape);
        network.set_activation_mode(NetworkActivationMode::Linear);
        let output = network.compute([2.0, 2.0].iter().into()).unwrap();

        assert_eq!(vec![0.0, 0.0], *output);
    }

    #[test]
    fn can_compute_hidden_layer_network_zeros() {
        let shape = NetworkShape::new(2, 2, vec![8], LayerInitStrategy::Zero);
        let mut network = Network::new(&shape);
        network.set_activation_mode(NetworkActivationMode::SoftMax);
        let output = network.compute([2.0, 2.0].iter().into()).unwrap();

        assert_eq!(vec![0.5, 0.5], *output);
        assert_eq!(1.0, output.iter().sum::<NodeValue>());
    }

    #[test]
    fn can_compute_hidden_layer_network_rands() {
        let shape = NetworkShape::new(
            2,
            2,
            vec![8],
            LayerInitStrategy::PositiveRandom(Rc::new(JsRng::default())),
        );
        let mut network = Network::new(&shape);
        network.set_activation_mode(NetworkActivationMode::SoftMax);
        let output = network.compute([2.0, 2.0].iter().into()).unwrap();

        assert_ne!(vec![0.5, 0.5], *output);
        assert_eq!(output.iter().sum::<NodeValue>().min(0.999), 0.999);
    }

    #[test]
    fn can_network_learn_randomly() {
        let rng = Rc::new(JsRng::default());
        let shape = NetworkShape::new(
            2,
            2,
            vec![8],
            LayerInitStrategy::PositiveRandom(rng.clone()),
        );
        let mut network = Network::new(&shape);
        network.set_activation_mode(NetworkActivationMode::SoftMax);

        let output_1 = network.compute([2.0, 2.0].iter().into()).unwrap();
        network
            .learn(NetworkLearnStrategy::MutateRng {
                rand_amount: 0.1,
                rng: rng.clone(),
            })
            .unwrap();

        let output_2 = network.compute([2.0, 2.0].iter().into()).unwrap();

        assert_ne!(vec![0.5, 0.5], *output_1);
        assert_ne!(vec![0.5, 0.5], *output_2);
        assert_ne!(*output_1, *output_2);
    }

    #[test]
    fn can_network_learn_grad_descent_linear_simple_orthoganal_inputs() {
        let training_pairs = [([0.0, 1.0], [0.0, 1.0]), ([1.0, 0.0], [1.0, 0.0])];
        let mut network = multi_sample::create_network(
            NetworkShape::new(
                2,
                2,
                vec![],
                LayerInitStrategy::ScaledFullRandom(Rc::new(JsRng::default())),
            ),
            NetworkActivationMode::Linear,
        );
        let learn_rate = 1e-1;
        let batch_size = None;

        for round in 1..100 {
            let error = multi_sample::run_training_iteration(
                &mut network,
                &training_pairs,
                learn_rate,
                round,
                batch_size,
            );

            if error < 1e-6 {
                println!("DONE round = {round}, network_cost = {error}");
                break;
            }
        }

        multi_sample::assert_training_outputs(&training_pairs, network);
    }

    #[test]
    fn can_network_learn_grad_descent_sigmoid_simple_orthoganal_inputs() {
        let training_pairs = [([0.0, 1.0], [0.1, 0.9]), ([1.0, 0.0], [0.9, 0.1])];
        let mut network = multi_sample::create_network(
            NetworkShape::new(
                2,
                2,
                vec![],
                LayerInitStrategy::ScaledFullRandom(Rc::new(JsRng::default())),
            ),
            NetworkActivationMode::Sigmoid,
        );
        let learn_rate = 1e-0;
        let batch_size = None;

        for round in 1..10_000 {
            let error = multi_sample::run_training_iteration(
                &mut network,
                &training_pairs,
                learn_rate,
                round,
                batch_size,
            );

            if error < 1e-6 {
                println!("DONE round = {round}, network_cost = {error}");
                break;
            }
        }

        multi_sample::assert_training_outputs(&training_pairs, network);
    }

    #[test]
    fn can_network_learn_grad_descent_tanh_simple_orthoganal_inputs() {
        let training_pairs = [([0.0, 1.0], [0.1, 0.9]), ([1.0, 0.0], [0.9, 0.1])];
        let mut network = multi_sample::create_network(
            NetworkShape::new(
                2,
                2,
                vec![],
                LayerInitStrategy::ScaledFullRandom(Rc::new(JsRng::default())),
            ),
            NetworkActivationMode::Tanh,
        );
        let learn_rate = 1e-1;
        let batch_size = None;

        for round in 1..100_000 {
            let error = multi_sample::run_training_iteration(
                &mut network,
                &training_pairs,
                learn_rate,
                round,
                batch_size,
            );

            if error < 1e-6 {
                println!("DONE round = {round}, network_cost = {error}");
                break;
            }
        }

        multi_sample::assert_training_outputs(&training_pairs, network);
    }

    #[test]
    fn can_network_learn_grad_descent_sigmoid_simple_orthoganal_triple_inputs() {
        let training_pairs = [
            ([0.0, 1.0, 0.0], [0.1, 0.9, 0.1]),
            ([1.0, 0.0, 0.0], [0.9, 0.1, 0.9]),
            ([0.0, 0.0, 0.1], [0.3, 0.5, 0.2]),
        ];
        let mut network = multi_sample::create_network(
            NetworkShape::new(
                3,
                3,
                vec![],
                LayerInitStrategy::ScaledFullRandom(Rc::new(JsRng::default())),
            ),
            NetworkActivationMode::Sigmoid,
        );
        let learn_rate = 1e-0;
        let batch_size = None;

        for round in 1..10_000 {
            let error = multi_sample::run_training_iteration(
                &mut network,
                &training_pairs,
                learn_rate,
                round,
                batch_size,
            );

            if error < 1e-6 {
                println!("DONE round = {round}, network_cost = {error}");
                break;
            }
        }

        multi_sample::assert_training_outputs(&training_pairs, network);
    }

    #[test]
    fn can_network_learn_grad_descent_sigmoid_simple_nonorthoganal_triple_inputs() {
        let training_pairs = [
            ([0.1, 1.0, 0.0], [0.1, 0.9, 0.1]),
            ([1.0, 1.0, 0.2], [0.9, 0.1, 0.9]),
            ([0.3, 0.1, 1.0], [0.3, 0.5, 0.2]),
        ];
        let mut network = multi_sample::create_network(
            NetworkShape::new(
                3,
                3,
                vec![],
                LayerInitStrategy::ScaledFullRandom(Rc::new(JsRng::default())),
            ),
            NetworkActivationMode::Sigmoid,
        );
        let learn_rate = 1e-0;
        let batch_size = None;

        for round in 1..10_000 {
            let error = multi_sample::run_training_iteration(
                &mut network,
                &training_pairs,
                learn_rate,
                round,
                batch_size,
            );

            if error < 1e-6 {
                println!("DONE round = {round}, network_cost = {error}");
                break;
            }
        }

        multi_sample::assert_training_outputs(&training_pairs, network);
    }

    #[test]
    fn can_network_learn_grad_descent_linear() {
        let training_pairs = [([0.1, 0.9], [0.25, 0.75])];
        let mut network = multi_sample::create_network(
            NetworkShape::new(
                2,
                2,
                vec![6, 6],
                LayerInitStrategy::ScaledFullRandom(Rc::new(JsRng::default())),
            ),
            NetworkActivationMode::Linear,
        );
        let learn_rate = 1e-2;
        let batch_size = None;

        for round in 1..200 {
            let error = multi_sample::run_training_iteration(
                &mut network,
                &training_pairs,
                learn_rate,
                round,
                batch_size,
            );

            if error < 1e-6 {
                println!("DONE round = {round}, network_cost = {error}");
                break;
            }
        }

        multi_sample::assert_training_outputs(&training_pairs, network);
    }

    #[test]
    fn can_network_learn_grad_descent_sigmoid() {
        let training_pairs = [([0.1, 0.9], [0.25, 0.75])];
        let mut network = multi_sample::create_network(
            NetworkShape::new(
                2,
                2,
                vec![6, 6],
                LayerInitStrategy::ScaledFullRandom(Rc::new(JsRng::default())),
            ),
            NetworkActivationMode::Sigmoid,
        );
        let learn_rate = 1e-1;
        let batch_size = None;

        for round in 1..1_000 {
            let error = multi_sample::run_training_iteration(
                &mut network,
                &training_pairs,
                learn_rate,
                round,
                batch_size,
            );

            if error < 1e-6 {
                println!("DONE round = {round}, network_cost = {error}");
                break;
            }
        }

        multi_sample::assert_training_outputs(&training_pairs, network);
    }

    #[test]
    fn can_network_learn_batch_grad_descent_sigmoid_binary_classifier() {
        let training_pairs = [
            ([2.0, 3.0, -1.0], [1.0]),
            ([3.0, -1.0, 0.5], [-1.0]),
            ([0.5, 1.0, 1.0], [-1.0]),
            ([1.0, 1.0, -1.0], [1.0]),
        ];
        let mut network = multi_sample::create_network(
            NetworkShape::new(
                3,
                1,
                vec![4, 4],
                LayerInitStrategy::ScaledFullRandom(Rc::new(JsRng::default())),
            ),
            NetworkActivationMode::Tanh,
        );

        let learn_rate = 0.05;
        let batch_size = Some(training_pairs.len());

        for round in 1..10_000 {
            let error = multi_sample::run_training_iteration(
                &mut network,
                &training_pairs,
                learn_rate,
                round,
                batch_size,
            );

            if error < 1e-4 {
                println!("DONE round = {round}, network_cost = {error}");
                break;
            }
        }

        multi_sample::assert_training_outputs(&training_pairs, network);
    }

    #[test]
    #[ignore = "review later as failure cause unknown"]
    fn can_network_learn_batch_grad_descent_sigmoid_multi_sample() {
        let training_pairs = [
            ([0.0, 1.0], [0.3, 0.4]),
            ([0.5, 1.0], [0.4, 0.3]),
            ([0.2, 1.0], [0.5, 0.0]),
        ];
        let mut network = multi_sample::create_network(
            NetworkShape::new(
                2,
                2,
                vec![4, 4, 4],
                LayerInitStrategy::ScaledFullRandom(Rc::new(JsRng::default())),
            ),
            NetworkActivationMode::Sigmoid,
        );
        let learn_rate = 5e-1;
        let batch_size = Some(training_pairs.len());

        for round in 1..500_000 {
            let error = multi_sample::run_training_iteration(
                &mut network,
                &training_pairs,
                learn_rate,
                round,
                batch_size,
            );

            if error < 1e-6 {
                println!("DONE round = {round}, network_cost = {error}");
                break;
            }
        }

        multi_sample::assert_training_outputs(&training_pairs, network);
    }

    #[test]
    fn can_network_learn_every_layers_batch_grad_descent_sigmoid() {
        let training_pairs = [([0.1, 0.9], [0.25, 0.75])];
        let mut network = multi_sample::create_network(
            NetworkShape::new(
                2,
                2,
                vec![6, 6],
                LayerInitStrategy::ScaledFullRandom(Rc::new(JsRng::default())),
            ),
            NetworkActivationMode::Sigmoid,
        );
        let learn_rate = 1e-1;
        let batch_size = Some(5);

        // TODO: impl on Network?
        fn layer_weights<'a>(network: &'a Network) -> impl Iterator<Item = Vec<f64>> + 'a {
            (0..network.layer_count()).map(|layer_idx| {
                network
                    .node_weights(layer_idx, 0)
                    .unwrap()
                    .cloned()
                    .collect::<Vec<_>>()
            })
        }

        let each_layer_weights: Vec<Vec<_>> = layer_weights(&network).collect();

        multi_sample::gradient_decent::compute_batch_training_iteration(
            &mut network,
            &training_pairs,
            learn_rate,
            batch_size.unwrap(),
        );

        for (idx, (current, initial)) in layer_weights(&network)
            .zip(each_layer_weights.iter())
            .enumerate()
        {
            let diffs = current
                .iter()
                .zip(initial.iter())
                .map(|(c, i)| c - i)
                .collect::<Vec<_>>();
            dbg!((idx, diffs));

            assert_ne!(&current, initial);
        }
    }

    mod multi_sample {
        use super::*;
        use std::rc::Rc;

        pub(crate) fn create_network(
            shape: NetworkShape,
            activation_mode: NetworkActivationMode,
        ) -> Network {
            let rng = Rc::new(JsRng::default());
            let mut network = Network::new(&shape);
            network.set_activation_mode(activation_mode);
            network
        }

        pub(crate) fn run_training_iteration<const N: usize, const O: usize>(
            network: &mut Network,
            training_pairs: &[([f64; N], [f64; O])],
            learn_rate: f64,
            round: i32,
            batch_size: Option<usize>,
        ) -> f64 {
            let error = if let Some(batch_size) = batch_size {
                gradient_decent::compute_batch_training_iteration(
                    network,
                    training_pairs,
                    learn_rate,
                    batch_size,
                )
            } else {
                gradient_decent::compute_single_training_iteration(
                    network,
                    training_pairs,
                    learn_rate,
                )
            };

            if round < 50
                || (round < 1000 && round % 100 == 0)
                || (round < 10000 && round % 1000 == 0)
                || round % 10000 == 0
            {
                println!("round = {round}, network_cost = {error}");
            }

            error
        }

        pub(crate) fn assert_training_outputs<const N: usize, const O: usize>(
            training_pairs: &[([f64; N], [f64; O])],
            network: Network,
        ) {
            let mut outputs = vec![];
            for (network_input, target_output) in training_pairs {
                let output_2 = network.compute(network_input.iter().into()).unwrap();
                println!("target = {target_output:?}, final_output = {output_2:?}");
                outputs.push(output_2);
            }

            for ((_, target_output), output_2) in training_pairs.iter().zip(outputs) {
                for (x, y) in target_output.iter().zip(output_2.iter()) {
                    assert!(
                        (x - y).abs() < 0.05,
                        "{target_output:?} and {output_2:?} dont match"
                    );
                }
            }
        }

        pub mod gradient_decent {
            use super::*;

            pub(crate) fn compute_single_training_iteration<const N: usize, const O: usize>(
                network: &mut Network,
                training_pairs: &[([f64; N], [f64; O])],
                learn_rate: f64,
            ) -> f64 {
                let mut error = 0.0;
                for (input, output) in training_pairs {
                    error += network
                        .learn(NetworkLearnStrategy::GradientDecent {
                            inputs: input.iter().cloned().collect(),
                            target_outputs: output.iter().cloned().collect(),
                            learn_rate,
                        })
                        .unwrap()
                        .unwrap();
                    error = error / training_pairs.len() as f64;
                }
                error
            }

            pub(crate) fn compute_batch_training_iteration<const N: usize, const O: usize>(
                network: &mut Network,
                training_pairs: &[([f64; N], [f64; O])],
                learn_rate: f64,
                batch_size: usize,
            ) -> f64 {
                network
                    .learn(NetworkLearnStrategy::BatchGradientDecent {
                        training_pairs: training_pairs
                            .iter()
                            .map(|(i, o)| (i.iter().into(), o.iter().into()))
                            .collect(),
                        batch_size,
                        learn_rate,
                        batch_sampling: BatchSamplingStrategy::Sequential,
                    })
                    .unwrap()
                    .unwrap()
            }
        }
    }
}
