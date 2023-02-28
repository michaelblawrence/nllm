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

#[derive(Debug)]
pub struct NetworkShape {
    inputs_count: usize,
    outputs_count: usize,
    hidden_layer_shape: Vec<usize>,
}

impl NetworkShape {
    pub fn new(inputs_count: usize, outputs_count: usize, hidden_layer_shape: Vec<usize>) -> Self {
        Self {
            inputs_count,
            outputs_count,
            hidden_layer_shape,
        }
    }
}

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
    activation_mode: NetworkActivationMode,
    softmax_output_enabled: bool,
}

impl Network {
    pub fn new(shape: &NetworkShape, init_stratergy: LayerInitStrategy) -> Self {
        let shape: Vec<usize> = [&shape.inputs_count]
            .into_iter()
            .chain(shape.hidden_layer_shape.iter())
            .chain([&shape.outputs_count])
            .cloned()
            .collect();

        let mut layers = vec![];

        for pair in shape.windows(2) {
            if let [lhs, rhs] = pair {
                layers.push(Layer::new(*rhs, *lhs, &init_stratergy))
            } else {
                panic!("failed to window layer shape");
            }
        }

        Self {
            layers,
            activation_mode: Default::default(),
            softmax_output_enabled: Default::default(),
        }
    }

    pub fn set_activation_mode(&mut self, activation_mode: NetworkActivationMode) {
        self.activation_mode = activation_mode;
    }

    pub fn set_softmax_output_enabled(&mut self, enabled: bool) {
        self.softmax_output_enabled = enabled;
    }

    pub fn compute(&self, inputs: LayerValues) -> Result<LayerValues, ()> {
        inputs.assert_length_equals(self.inputs_count()?)?;

        let (last_layer, layers) = self.layers.split_last().expect("should have final layer");
        let mut output = inputs;

        for layer in layers {
            output = layer.compute(&output)?;
            output = self.activation_mode.apply(&output);
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
            } => {
                use itertools::Itertools;
                if training_pairs.is_empty() {
                    return Ok(None);
                }

                let mut sum_error = 0.0;
                let training_pairs_len = training_pairs.len();

                for training_pairs in training_pairs.into_iter().chunks(batch_size).into_iter() {
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

        for layer in layers {
            let x = layer.compute(&output)?;
            let x = self.activation_mode.apply(&x);

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
        let mut output_layer_d = None;
        let mut output_layer_error = 0.0;
        let final_activation_mode = self.final_layer_activation();

        for (layer, layer_input) in self
            .layers
            .iter_mut()
            .rev()
            .zip(layers_activations.iter().rev().skip(1))
        {
            // use stored weighted input sums here instead?
            let x = layer.compute(&layer_input)?;

            // key
            //  dc/dw = dz/dw * da/dz * dc/da
            //    da/dz = activation_d
            //    dc/da = error_d
            //    dz/dw = layer.learn (network weights)

            if let Some(prev_layer_d) = output_layer_d.take() {
                let activation_d = self.activation_mode.derivative(&x);
                let prev_layer_d = LayerValues(prev_layer_d);
                let layer_d = layer.compute_d(&prev_layer_d)?;
                let layer_d = activation_d.multiply_by_iter(layer_d);

                output_layer_d = Some(layer_d.collect::<Vec<NodeValue>>());
            } else {
                let activation_d = final_activation_mode.derivative(&x);
                let layer_output = final_activation_mode.apply(&x);
                output_layer_error = layer.calculate_error(&layer_output, &target_outputs)?.ave();

                let error_d = layer.calculate_error_d(&layer_output, &target_outputs)?;
                output_layer_d = Some(
                    activation_d
                        .multiply_iter(&error_d)
                        .collect::<Vec<NodeValue>>(),
                );
            }

            let gradients = output_layer_d
                .as_ref()
                .expect("should know output deviratives by now");

            if enqueue_batch {
                layer.learn(&LayerLearnStrategy::EnqueueBatchGradientDecent {
                    gradients: LayerValues(gradients.clone()),
                    layer_inputs: layer_input.clone(), // TODO: can remove clone?
                })?;
            } else {
                layer.learn(&LayerLearnStrategy::GradientDecent {
                    gradients: LayerValues(gradients.clone()),
                    layer_inputs: layer_input.clone(), // TODO: can remove clone?
                    learn_rate,
                })?;
            }
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
    bias: Vec<NodeValue>,
    pending_gradients: Option<PendingLayerGradients>,
    inputs_count: usize,
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
        let empty = Self {
            weights: vec![0.0; weights_size],
            bias: vec![0.0; size],
            pending_gradients: None,
            inputs_count,
        };

        let scale_factor = (inputs_count as NodeValue).powf(-0.5);

        match init_stratergy {
            LayerInitStrategy::Zero => Self { ..empty },
            LayerInitStrategy::Random(rng) => Self {
                weights: empty
                    .weights
                    .iter()
                    .map(|_| rng.rand() * scale_factor)
                    .collect(),
                bias: empty
                    .bias
                    .iter()
                    .map(|_| rng.rand() * scale_factor)
                    .collect(),
                ..empty
            },
        }
    }

    pub fn compute(&self, inputs: &LayerValues) -> Result<LayerValues, ()> {
        inputs
            .assert_length_equals(self.inputs_count)
            .map_err(|_| format!("{self:?}"))
            .unwrap();

        let mut outputs = self.bias.clone();

        for input_index in 0..self.inputs_count {
            let weights = self.node_weights(input_index);
            for (idx, (input, weight)) in inputs.iter().zip(weights).enumerate() {
                outputs[idx] += input * weight;
            }
        }

        Ok(LayerValues(outputs))
    }

    pub fn compute_d<'a>(
        &'a self,
        next_layer_gradients: &'a LayerValues,
    ) -> Result<impl Iterator<Item = NodeValue> + 'a, ()> {
        let layer_node_iter = 0..self.size();
        let weights_iter = self.weights.iter();

        let iter = layer_node_iter
            .flat_map(move |_| next_layer_gradients.iter())
            .zip(weights_iter)
            .map(|(next_layer_grad, weight)| next_layer_grad * weight);

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
                for x in self.bias.iter_mut() {
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
                for (x, grad) in self.bias.iter_mut().zip(bias_gradients) {
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

        for (x, grad) in self.weights.iter_mut().zip(pending_gradients.bias.iter()) {
            *x = *x - (grad * learn_rate)
        }
        for (x, grad) in self.bias.iter_mut().zip(pending_gradients.weights.iter()) {
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
            // print error?
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
        self.bias.len()
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

        let dot_product = v1.iter().zip(v2.iter()).map(|(&x, &y)| x * y).sum::<NodeValue>();
        let norm1 = v1.iter().map(|&x| x * x).sum::<NodeValue>().sqrt();
        let norm2 = v2.iter().map(|&x| x * x).sum::<NodeValue>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return None;
        }

        Some(dot_product / (norm1 * norm2))
    }
}

pub enum LayerInitStrategy {
    Zero,
    Random(Rc<dyn RNG>),
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
    },
    Multi(Vec<NetworkLearnStrategy>),
}

#[derive(Debug, Clone, Copy)]
pub enum NetworkActivationMode {
    Linear,
    SoftMax,
    Sigmoid,
}

impl NetworkActivationMode {
    fn apply(&self, output: &LayerValues) -> LayerValues {
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
        let shape = NetworkShape::new(2, 2, vec![]);
        let network = Network::new(&shape, LayerInitStrategy::Zero);

        assert_eq!(2, network.inputs_count().unwrap());
    }

    #[test]
    fn can_create_hidden_layer_network() {
        let shape = NetworkShape::new(2, 2, vec![8]);
        let network = Network::new(&shape, LayerInitStrategy::Zero);

        assert_eq!(2, network.inputs_count().unwrap());
    }

    #[test]
    fn can_compute_hidden_layer_network_zeros_linear_activation() {
        let shape = NetworkShape::new(2, 2, vec![8]);
        let mut network = Network::new(&shape, LayerInitStrategy::Zero);
        network.set_activation_mode(NetworkActivationMode::Linear);
        let output = network.compute([2.0, 2.0].iter().into()).unwrap();

        assert_eq!(vec![0.0, 0.0], *output);
    }

    #[test]
    fn can_compute_hidden_layer_network_zeros() {
        let shape = NetworkShape::new(2, 2, vec![8]);
        let mut network = Network::new(&shape, LayerInitStrategy::Zero);
        network.set_activation_mode(NetworkActivationMode::SoftMax);
        let output = network.compute([2.0, 2.0].iter().into()).unwrap();

        assert_eq!(vec![0.5, 0.5], *output);
        assert_eq!(1.0, output.iter().sum::<NodeValue>());
    }

    #[test]
    fn can_compute_hidden_layer_network_rands() {
        let shape = NetworkShape::new(2, 2, vec![8]);
        let mut network =
            Network::new(&shape, LayerInitStrategy::Random(Rc::new(JsRng::default())));
        network.set_activation_mode(NetworkActivationMode::SoftMax);
        let output = network.compute([2.0, 2.0].iter().into()).unwrap();

        assert_ne!(vec![0.5, 0.5], *output);
        assert_eq!(output.iter().sum::<NodeValue>().min(0.999), 0.999);
    }

    #[test]
    fn can_network_learn_randomly() {
        let shape = NetworkShape::new(2, 2, vec![8]);
        let rng = Rc::new(JsRng::default());
        let mut network = Network::new(&shape, LayerInitStrategy::Random(rng.clone()));
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
    fn can_network_learn_grad_descent_linear() {
        let shape = NetworkShape::new(2, 2, vec![6, 6]);
        let rng = Rc::new(JsRng::default());
        let network_input = [0.1, 0.9];
        let target_output = [0.25, 0.75];
        let mut network = Network::new(&shape, LayerInitStrategy::Random(rng.clone()));
        network.set_activation_mode(NetworkActivationMode::Linear);

        let output_1 = network.compute(network_input.iter().into()).unwrap();
        for round in 1..200 {
            network
                .learn(NetworkLearnStrategy::GradientDecent {
                    learn_rate: 0.1,
                    inputs: network_input.iter().into(),
                    target_outputs: target_output.iter().into(),
                })
                .unwrap();

            let error = network
                .compute_error(network_input.iter().into(), &target_output.iter().into())
                .unwrap()
                .iter()
                .sum::<NodeValue>();

            if round < 50
                || (round < 1000 && round % 100 == 0)
                || (round < 10000 && round % 1000 == 0)
            {
                println!("round = {round}, network_cost = {error}");
            }

            if error < 1e-6 {
                break;
            }
        }

        let output_2 = network.compute(network_input.iter().into()).unwrap();
        println!(
            "init_output = {output_1:?}, target = {target_output:?}, final_output = {output_2:?}"
        );
        println!("network = {network:#?}");

        assert_ne!(vec![0.5, 0.5], *output_1);
        assert!((target_output[0] - output_2[0]).abs() < 0.05);
        assert!((target_output[1] - output_2[1]).abs() < 0.05);
    }

    #[test]
    fn can_network_learn_grad_descent_sigmoid() {
        let shape = NetworkShape::new(2, 2, vec![6, 6]);
        let rng = Rc::new(JsRng::default());
        let network_input = [0.1, 0.9];
        let target_output = [0.25, 0.75];
        let mut network = Network::new(&shape, LayerInitStrategy::Random(rng.clone()));
        network.set_activation_mode(NetworkActivationMode::Sigmoid);

        let output_1 = network.compute(network_input.iter().into()).unwrap();
        for round in 1..1_000 {
            let error = network
                .learn(NetworkLearnStrategy::GradientDecent {
                    learn_rate: 1e-2,
                    inputs: network_input.iter().into(),
                    target_outputs: target_output.iter().into(),
                })
                .unwrap()
                .unwrap();

            if round < 50
                || (round < 1000 && round % 100 == 0)
                || (round < 10000 && round % 1000 == 0)
                || round % 10000 == 0
            {
                println!("round = {round}, network_cost = {error}");
            }

            if error < 1e-6 {
                println!("DONE round = {round}, network_cost = {error}");
                break;
            }
        }
        let output_2 = network.compute(network_input.iter().into()).unwrap();
        println!(
            "init_output = {output_1:?}, target = {target_output:?}, final_output = {output_2:?}"
        );
        println!("network = {network:#?}");

        assert_ne!(vec![0.5, 0.5], *output_1);
        assert!(
            (target_output[0] - output_2[0]).abs() < 0.05,
            "{target_output:?} and {output_2:?} dont match"
        );
        assert!(
            (target_output[1] - output_2[1]).abs() < 0.05,
            "{target_output:?} and {output_2:?} dont match"
        );
    }
}
