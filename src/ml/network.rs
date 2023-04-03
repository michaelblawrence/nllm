use std::{
    ops::{Deref, DerefMut},
    rc::Rc,
    str::FromStr,
};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::ml::{ShuffleRng, RNG};

use super::JsRng;

#[cfg(not(short_floats))]
pub type NodeValue = f64;

#[cfg(short_floats)]
type NodeValue = f32;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkShape {
    inputs_count: usize,
    layers_shape: Vec<LayerShape>,
    activation_mode: NetworkActivationMode,
}

impl NetworkShape {
    pub fn new(
        inputs_count: usize,
        outputs_count: usize,
        hidden_layer_shape: Vec<usize>,
        init_stratergy: LayerInitStrategy,
    ) -> Self {
        Self {
            inputs_count: inputs_count,
            layers_shape: hidden_layer_shape
                .into_iter()
                .map(|layer_size| (layer_size, init_stratergy.clone()))
                .chain([(outputs_count, init_stratergy.clone())])
                .map(|x| x.into())
                .collect(),
            activation_mode: Default::default(),
        }
    }

    pub fn new_custom(
        inputs_count: usize,
        layers_shape: Vec<LayerShape>,
        activation_mode: NetworkActivationMode,
    ) -> Self {
        Self {
            inputs_count,
            layers_shape,
            activation_mode,
        }
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = LayerShape> + 'a {
        [&(self.inputs_count, LayerInitStrategy::Zero).into()]
            .into_iter()
            .chain(self.layers_shape.iter())
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
    }

    pub fn layer_activation(&self, layer_idx: usize) -> Result<NetworkActivationMode> {
        let layer_shape = &self
            .layers_shape
            .get(layer_idx)
            .context("provided layer index should be valid")?;

        Ok(layer_shape.activation_mode.unwrap_or(self.activation_mode))
    }

    pub fn set_activation_mode(&mut self, activation_mode: NetworkActivationMode) {
        self.activation_mode = activation_mode;
    }

    pub fn len(&self) -> usize {
        self.layers_shape.len()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerShape {
    node_count: usize,
    strategy: LayerInitStrategy,
    stride_count: Option<usize>,
    activation_mode: Option<NetworkActivationMode>,
}

impl LayerShape {
    pub fn new(
        node_count: usize,
        strategy: LayerInitStrategy,
        stride_count: Option<usize>,
        activation_mode: Option<NetworkActivationMode>,
    ) -> Self {
        Self {
            node_count,
            strategy,
            stride_count,
            activation_mode,
        }
    }

    pub fn with_activation_mode(self, mode: NetworkActivationMode) -> Self {
        Self {
            activation_mode: Some(mode),
            ..self
        }
    }

    pub fn mode_override(&self) -> Option<NetworkActivationMode> {
        self.activation_mode
    }

    pub fn stride_count(&self) -> usize {
        self.stride_count.unwrap_or(1)
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }
}

impl From<(usize, LayerInitStrategy)> for LayerShape {
    fn from((node_count, strategy): (usize, LayerInitStrategy)) -> Self {
        LayerShape {
            node_count,
            strategy,
            stride_count: None,
            activation_mode: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    layers: Vec<Layer>,
    shape: NetworkShape,
}

impl Network {
    pub fn new(shape: NetworkShape) -> Self {
        Self::new_with_rng(shape, Rc::new(JsRng::default()))
    }

    pub fn new_with_rng(shape: NetworkShape, rng: Rc<dyn RNG>) -> Self {
        let mut layers = vec![];

        for (pair_idx, pair) in shape.iter().collect::<Vec<_>>().windows(2).enumerate() {
            if let [lhs, rhs] = pair {
                let layer_input_nodes = lhs.node_count * lhs.stride_count.unwrap_or(1);
                let mut layer = Layer::new(
                    rhs.node_count,
                    layer_input_nodes,
                    &rhs.strategy,
                    rng.clone(),
                );

                match rhs.stride_count {
                    Some(stride_count) => {
                        assert_eq!(
                            0, pair_idx,
                            "layer stride not yet supported for non-starting layers"
                        );
                        layer.set_stride_count(stride_count);
                    }
                    _ => (),
                }

                layers.push(layer)
            } else {
                panic!("failed to window layer shape");
            }
        }

        Self { layers, shape }
    }

    pub fn compute(&self, inputs: LayerValues) -> Result<LayerValues> {
        inputs.assert_length_equals(self.input_vector_size()?)?;
        let (network_output, _) = self.compute_inner(inputs, true)?;
        Ok(network_output)
    }

    fn compute_all_layers_activations(
        &self,
        inputs: LayerValues,
    ) -> Result<Vec<(LayerValues, LayerValues)>> {
        let (_, layers_activations) = self.compute_inner(inputs, false)?;
        Ok(layers_activations)
    }

    pub fn compute_error(
        &self,
        inputs: LayerValues,
        target_outputs: &LayerValues,
    ) -> Result<LayerValues> {
        let last_layer = self.layers.last().context("should have a final layer")?;
        let outputs = self.compute(inputs)?;
        let errors = last_layer.calculate_error(
            &outputs,
            &target_outputs,
            &self.final_layer_activation()?,
        )?;
        Ok(errors)
    }

    fn compute_error_precomputed(
        &self,
        layers_activations: &Vec<(LayerValues, LayerValues)>,
        target_outputs: &LayerValues,
    ) -> Result<f64> {
        let last = layers_activations.last();
        let (_, layer_output) = last.expect("should have a final layer");

        let output_layer_error = self
            .layers
            .last()
            .context("should have final layer")?
            .calculate_error(
                &layer_output,
                target_outputs,
                &self.final_layer_activation()?,
            )?
            .ave();

        Ok(output_layer_error)
    }

    pub fn learn(&mut self, strategy: NetworkLearnStrategy) -> Result<Option<NodeValue>> {
        let cost = match strategy {
            NetworkLearnStrategy::MutateRng { rand_amount, rng } => {
                let strategy = LayerLearnStrategy::MutateRng {
                    weight_factor: rand_amount,
                    bias_factor: rand_amount,
                    rng,
                };
                for layer in self.layers.iter_mut() {
                    layer.learn(&strategy, 1.0)?;
                }

                None
            }
            NetworkLearnStrategy::GradientDecent {
                inputs,
                target_outputs,
                learn_rate,
            } => {
                let layers_activations = self.compute_all_layers_activations(inputs)?;

                let output_layer_error =
                    self.compute_error_precomputed(&layers_activations, &target_outputs)?;

                let learn_actions =
                    self.perform_gradient_decent_step(&layers_activations, &target_outputs)?;

                self.apply_pending_layer_actions(learn_actions.iter(), learn_rate)?;

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

                let mut network_errors = vec![];

                for batch in batches.into_iter() {
                    let all_layers_activations = batch
                        .map(|(inputs, target_outputs)| {
                            let layers_activations = self.compute_all_layers_activations(inputs)?;
                            Ok((layers_activations, target_outputs))
                        })
                        .collect::<Result<Vec<_>>>()?;

                    let batch_errors = all_layers_activations
                        .iter()
                        .map(|(layers_activations, target_outputs)| {
                            let output_layer_error = self
                                .compute_error_precomputed(&layers_activations, &target_outputs);
                            output_layer_error
                        })
                        .collect::<Result<Vec<_>>>()?;

                    let layer_learn_actions = all_layers_activations
                        .iter()
                        .map(|(layers_activations, target_outputs)| {
                            let learn_actions = self
                                .perform_gradient_decent_step(&layers_activations, &target_outputs);
                            learn_actions
                        })
                        .collect::<Result<Vec<_>>>()?;

                    let learn_actions = layer_learn_actions.iter().flatten();
                    self.apply_pending_layer_actions(learn_actions, learn_rate)?;

                    let batch_errors = LayerValues::new(batch_errors);
                    network_errors.push(batch_errors.ave())
                }

                let network_errors = LayerValues::new(network_errors);
                Some(network_errors.ave())
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

    #[inline]
    fn compute_inner(
        &self,
        inputs: LayerValues,
        disable_alloc: bool,
    ) -> Result<(LayerValues, Vec<(LayerValues, LayerValues)>)> {
        let (last_layer, layers) = self.layers.split_last().expect("should have final layer");

        let (mut layers_activations, output) = if disable_alloc {
            (Vec::new(), Some(&inputs))
        } else {
            let mut vec = Vec::with_capacity(self.layers.len());
            vec.push((LayerValues::new(Vec::new()), inputs));
            (vec, None)
        };

        // TODO: refactor this.. currently needed to satisfy lifetimes
        let mut _last_cache = None;
        let mut output = layers_activations
            .first()
            .map(|(_x, z)| z)
            .or(output)
            .expect("missing input values");

        for (layer_idx, layer) in layers.iter().enumerate() {
            let x = layer.compute(&output)?;
            let z = self.layer_activation(layer_idx)?.apply(&x);

            output = if !disable_alloc {
                layers_activations.push((x, z));
                layers_activations
                    .last()
                    .map(|(_x, z)| z)
                    .expect("missing latest pushed layer")
            } else {
                _last_cache = Some(z);
                _last_cache.as_ref().unwrap()
            };
        }

        let x = last_layer.compute(&output)?;
        let z = self.final_layer_activation()?.apply(&x);

        if !disable_alloc {
            layers_activations.push((x, z.clone()));
        }
        Ok((z, layers_activations))
    }

    fn perform_gradient_decent_step(
        &self,
        layers_activations: &Vec<(LayerValues, LayerValues)>,
        target_outputs: &LayerValues,
    ) -> Result<Vec<(usize, LayerLearnStrategy)>> {
        struct LayerIntermediate<'a> {
            layer: &'a Layer,
            layer_id: usize,
            layer_input: &'a LayerValues,
            layer_weighted_outputs: &'a LayerValues,
            activation_mode: NetworkActivationMode,
        }
        let layer_io_iter = layers_activations
            .windows(2)
            .map(|pair| (&pair[0].1, &pair[1].0));

        let activations_iter = self
            .layers
            .iter()
            .enumerate()
            .map(|(i, layer)| (layer, i, self.layer_activation(i)));

        let mut layer_learn_actions = vec![];
        let mut layer_d = None;

        let layer_states =
            activations_iter
                .zip(layer_io_iter)
                .map(|((layer, idx, mode), (l_input, l_output))| {
                    Ok(LayerIntermediate {
                        layer,
                        layer_id: idx,
                        layer_input: l_input,
                        layer_weighted_outputs: l_output,
                        activation_mode: mode?,
                    })
                });

        let layer_states = layer_states.collect::<Result<Vec<_>>>()?;

        for state in layer_states.iter().rev() {
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

            let layer_input = state.layer_input;
            let layer_weighted_outputs = state.layer_weighted_outputs;

            let error_d = match layer_d.take() {
                Some(prev_layer_error_d) => LayerValues(prev_layer_error_d),
                None => {
                    let (_, layer_output) = layers_activations
                        .last()
                        .context("should always have final layer")?;
                    state.layer.calculate_error_d(
                        &layer_output,
                        &target_outputs,
                        &state.activation_mode,
                    )?
                }
            };

            let activation_d = state.activation_mode.derivative(&layer_weighted_outputs);
            let activated_layer_d = activation_d.multiply_iter(&error_d);
            let activated_layer_d = activated_layer_d.collect();
            let input_gradients = state.layer.compute_d(&activated_layer_d)?;
            let input_gradients = input_gradients.collect::<Vec<NodeValue>>();
            layer_d = Some(input_gradients);

            let strategy = LayerLearnStrategy::GradientDecent {
                gradients: activated_layer_d,
                layer_inputs: layer_input.clone(),
            };

            layer_learn_actions.push((state.layer_id, strategy));
        }

        Ok(layer_learn_actions)
    }

    pub fn node_weights(&self, layer_idx: usize, node_index: usize) -> Result<&[f64]> {
        Ok(self
            .layers
            .get(layer_idx)
            .with_context(|| format!("invalid layer index = {layer_idx}"))?
            .node_weights(node_index)?)
    }

    pub fn layer_weights(&self, layer_idx: usize) -> Result<&[f64]> {
        Ok(self
            .layers
            .get(layer_idx)
            .with_context(|| format!("invalid layer index = {layer_idx}"))?
            .weights())
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    fn apply_pending_layer_actions<'a, I: Iterator<Item = &'a (usize, LayerLearnStrategy)>>(
        &mut self,
        learn_actions: I,
        learn_rate: f64,
    ) -> Result<()> {
        for (layer_idx, learn_strategy) in learn_actions.into_iter() {
            let layer = self
                .layers
                .get_mut(*layer_idx)
                .context("this layer index should be present")?;

            layer.learn(&learn_strategy, learn_rate)?;
        }

        Ok(())
    }

    fn input_vector_size(&self) -> Result<usize> {
        self.layers
            .first()
            .map(|layer| layer.input_vector_size())
            .context("unable to get first layer")
    }

    #[inline]
    fn layer_activation(&self, layer_idx: usize) -> Result<NetworkActivationMode> {
        self.shape.layer_activation(layer_idx)
    }

    #[inline]
    fn final_layer_activation(&self) -> Result<NetworkActivationMode> {
        let last_idx = self.shape.layers_shape.len() - 1;
        self.layer_activation(last_idx)
    }

    pub fn shape(&self) -> &NetworkShape {
        &self.shape
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    weights: Vec<NodeValue>,
    bias: Option<Vec<NodeValue>>,
    inputs_count: usize,
    size: usize,
    stride_count: Option<usize>,
}

impl Layer {
    pub fn new(
        size: usize,
        inputs_count: usize,
        init_stratergy: &LayerInitStrategy,
        rng: Rc<dyn RNG>,
    ) -> Self {
        let weights_size = inputs_count * size;
        let mut layer = Self {
            weights: vec![0.0; weights_size],
            bias: match &init_stratergy {
                LayerInitStrategy::NoBias => None,
                _ => Some(vec![0.0; size]),
            },
            inputs_count,
            size,
            stride_count: None,
        };

        init_stratergy.apply(
            layer.weights.iter_mut(),
            layer.bias.iter_mut().flatten(),
            inputs_count,
            rng.as_ref(),
        );

        layer
    }

    pub fn compute(&self, inputs: &LayerValues) -> Result<LayerValues> {
        inputs.assert_length_equals(self.input_vector_size())?;

        let mut outputs = match (&self.bias, self.stride_count) {
            (None, None) => vec![0.0; self.size],
            (None, Some(stride_count)) => vec![0.0; self.size * stride_count],
            (Some(bias), None) => bias.clone(),
            (Some(bias), Some(stride_count)) => bias
                .iter()
                .cycle()
                .take(self.size * stride_count)
                .cloned()
                .collect(),
        };

        let size = self.size();

        for (input_index, input) in inputs.iter().enumerate() {
            let weights = self.node_weights(input_index % self.inputs_count)?;
            let stride_idx = input_index / self.inputs_count;
            for (weight_idx, weight) in weights.iter().enumerate() {
                let output_idx = weight_idx + (stride_idx * size);
                outputs[output_idx] += input * weight;
            }
        }

        Ok(LayerValues(outputs))
    }

    pub fn compute_d<'a>(
        &'a self,
        forward_layer_gradients: &'a LayerValues,
    ) -> Result<impl Iterator<Item = NodeValue> + 'a> {
        let iter = self.weights.chunks(self.size()).map(|weights| {
            weights
                .iter()
                .zip(forward_layer_gradients.iter())
                .map(|(weight, next_layer_grad)| next_layer_grad * weight)
                .sum::<NodeValue>()
        });

        Ok(iter)
    }

    pub fn learn(&mut self, strategy: &LayerLearnStrategy, learn_rate: NodeValue) -> Result<()> {
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
            } => {
                let weights_gradients = self.weights_gradients_iter(&gradients, &layer_inputs)?;

                for (x, grad) in self.weights.iter_mut().zip(weights_gradients.iter()) {
                    let next_val = *x - (grad * learn_rate);
                    if next_val.is_nan() {
                        return Err(anyhow!(
                            "failed to apply gradient (value={grad}) to optimise weights"
                        ));
                    }
                    *x = next_val
                }
                for (x, grad) in self.bias.iter_mut().flatten().zip(gradients.iter()) {
                    let next_val = *x - (grad * learn_rate);
                    if next_val.is_nan() {
                        return Err(anyhow!(
                            "failed to apply gradient (value={grad}) to optimise biases"
                        ));
                    }
                    *x = next_val
                }
            }
        }
        Ok(())
    }

    fn weights_gradients_iter<'a>(
        &self,
        gradients: &'a LayerValues,
        layer_inputs: &'a LayerValues,
    ) -> Result<Vec<NodeValue>> {
        let stride_count = self.stride_count.unwrap_or(1);
        let weights_len = self.weights.len();
        let scaled_weights_len = weights_len * stride_count * stride_count;

        if scaled_weights_len != gradients.len() * layer_inputs.len() {
            dbg!(
                weights_len,
                gradients.len() * layer_inputs.len(),
                stride_count
            );
            return Err(anyhow!("mismatched inputs/gradients dimensions"));
        }

        let weights_gradients = layer_inputs
            .chunks(self.inputs_count)
            .zip(gradients.chunks(self.size))
            .fold(
                vec![0.0; weights_len],
                |mut state, (layer_inputs, gradients)| {
                    layer_inputs
                        .into_iter()
                        .flat_map(|layer_input| gradients.iter().map(move |g| g * layer_input))
                        .zip(state.iter_mut())
                        .for_each(|(gradient, x)| *x += gradient);
                    state
                },
            );

        Ok(weights_gradients)
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn calculate_error(
        &self,
        outputs: &LayerValues,
        expected_outputs: &LayerValues,
        mode: &NetworkActivationMode,
    ) -> Result<LayerValues> {
        match mode {
            NetworkActivationMode::SoftMax => expected_outputs
                .iter()
                .zip(outputs.iter())
                .map(|(expected, actual)| {
                    if *expected == 1.0 {
                        Ok(-actual.ln())
                    } else if *expected == 0.0 {
                        Ok(0.0)
                    } else {
                        Err(anyhow!(
                            "target outputs should be one-hot encoded: {expected_outputs:?}"
                        ))
                    }
                })
                .collect(),
            _ => self.calculate_msd_error(outputs, expected_outputs),
        }
    }

    pub fn calculate_msd_error(
        &self,
        outputs: &LayerValues,
        expected_outputs: &LayerValues,
    ) -> Result<LayerValues> {
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
        mode: &NetworkActivationMode,
    ) -> Result<LayerValues> {
        match mode {
            // NetworkActivationMode::SoftMax => {
            //     expected_outputs
            //         .iter()
            //         .zip(outputs.iter())
            //         .map(|(expected, actual)| match *expected {
            //             1.0 => Ok(-1.0 / actual),
            //             0.0 => Ok(0.0),
            //             _ => Err(anyhow!("target outputs should be one-hot encoded"))
            //         })
            //         .collect()
            // },
            _ => self.calculate_msd_error_d(outputs, expected_outputs),
        }
    }

    pub fn calculate_msd_error_d(
        &self,
        outputs: &LayerValues,
        expected_outputs: &LayerValues,
    ) -> Result<LayerValues> {
        let mut error = vec![];
        for (actual, expected) in outputs.iter().zip(expected_outputs.iter()) {
            error.push((actual - expected) * 2.0);
        }

        Ok(LayerValues(error))
    }

    fn node_weights<'a>(&'a self, node_index: usize) -> Result<&[f64]> {
        let n = self.size();
        let start = n * node_index;
        let end = start + n;

        if end > self.weights.len() {
            return Err(anyhow!("provided index out of range"));
        }

        Ok(&self.weights[start..end])
    }

    fn set_stride_count(&mut self, stride_count: usize) {
        self.stride_count = Some(stride_count);
    }

    pub fn input_vector_size(&self) -> usize {
        let stride_count = self.stride_count.unwrap_or(1);
        self.inputs_count * stride_count
    }

    pub fn weights(&self) -> &[f64] {
        self.weights.as_ref()
    }
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    fn assert_length_equals(&self, expected: usize) -> Result<()> {
        if self.len() == expected {
            Ok(())
        } else {
            return Err(anyhow!(
                "lengths are not equal: expected={expected}, actual={actual}",
                actual = self.len()
            ));
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LayerInitStrategy {
    Zero,
    NoBiasCopied(LayerValues),
    PositiveRandom,
    ScaledFullRandom,
    FullRandom,
    NoBias,
}

impl LayerInitStrategy {
    pub fn apply<'a>(
        &self,
        weights: impl Iterator<Item = &'a mut NodeValue>,
        bias: impl Iterator<Item = &'a mut NodeValue>,
        inputs_count: usize,
        rng: &dyn RNG,
    ) {
        use LayerInitStrategy::*;

        match self {
            Zero => {}
            NoBiasCopied(values) => {
                for (value, copied) in weights.zip(values.iter()) {
                    *value = *copied;
                }
            }
            PositiveRandom => {
                let scale_factor = (inputs_count as NodeValue).powf(-0.5);
                for value in weights.chain(bias) {
                    *value = rng.rand() * scale_factor;
                }
            }
            ScaledFullRandom | NoBias => {
                let scale_factor = (inputs_count as NodeValue).powf(-0.5) * 2.0;
                for value in weights.chain(bias) {
                    *value = Self::full_rand(rng) * scale_factor;
                }
            }
            FullRandom => {
                for value in weights.chain(bias) {
                    *value = Self::full_rand(rng);
                }
            }
        }
    }
    fn full_rand(rng: &dyn RNG) -> NodeValue {
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NetworkActivationMode {
    Linear,
    SoftMax,
    Sigmoid,
    Tanh,
    RelU,
}

impl std::fmt::Display for NetworkActivationMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string(&self).map_err(|_| std::fmt::Error::default())?;
        let json = json.trim_matches('"');
        write!(f, "{}", json)
    }
}

impl FromStr for NetworkActivationMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        use NetworkActivationMode::*;
        let mut s = s.to_uppercase();
        s.retain(char::is_alphabetic);
        match s.as_str() {
            "LINEAR" => Ok(Linear),
            "SOFTMAX" => Ok(SoftMax),
            "SIGMOID" => Ok(Sigmoid),
            "TANH" => Ok(Tanh),
            "RELU" => Ok(RelU),
            _ => Err(anyhow!("Could not match activation mode type")),
        }
    }
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
            // TODO: fix this approx deriv.
            NetworkActivationMode::SoftMax => self.apply(output),
            // NetworkActivationMode::SoftMax => {
            //     let dx = 0.0001;
            //     let apply_1 = self.apply(&output);
            // // adding to all inputs here makes no meaningful to computed probs, so this is a very very poor dydx approx
            //     let apply_2 = self.apply(&output.iter().map(|x| x + dx).collect());
            //     let mut v = vec![];

            //     for (x1, x2) in apply_1.iter().zip(apply_2.iter()) {
            //         let dy = x2 - x1;
            //         let dydx = dy / dx;
            //         v.push(dydx);
            //     }
            //     LayerValues::new(v)
            // }
            // NetworkActivationMode::SoftMax => {
            //     let softmax = self.apply(output);
            //     softmax
            //         .iter()
            //         .enumerate()
            //         .map(|(j, sj)| {
            //             softmax
            //                 .iter()
            //                 .enumerate()
            //                 .map(|(i, si)| if i == j { si * (1.0 - si) } else { si * -sj })
            //                 .sum::<NodeValue>()
            //         })
            //         .collect()
            // }
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

    use test_log::test;
    use tracing::info;

    use crate::ml::JsRng;

    use super::*;

    #[test]
    fn can_create_network() {
        let shape = NetworkShape::new(2, 2, vec![], LayerInitStrategy::Zero);
        let network = Network::new(shape);

        assert_eq!(2, network.input_vector_size().unwrap());
    }

    #[test]
    fn can_create_hidden_layer_network() {
        let shape = NetworkShape::new(2, 2, vec![8], LayerInitStrategy::Zero);
        let network = Network::new(shape);

        assert_eq!(2, network.input_vector_size().unwrap());
    }

    #[test]
    fn can_compute_hidden_layer_network_zeros_linear_activation() {
        let mut shape = NetworkShape::new(2, 2, vec![8], LayerInitStrategy::Zero);
        shape.set_activation_mode(NetworkActivationMode::Linear);
        let network = Network::new(shape);
        let output = network.compute([2.0, 2.0].iter().into()).unwrap();

        assert_eq!(vec![0.0, 0.0], *output);
    }

    #[test]
    fn can_compute_hidden_layer_network_zeros() {
        let mut shape = NetworkShape::new(2, 2, vec![8], LayerInitStrategy::Zero);
        shape.set_activation_mode(NetworkActivationMode::SoftMax);
        let network = Network::new(shape);
        let output = network.compute([2.0, 2.0].iter().into()).unwrap();

        assert_eq!(vec![0.5, 0.5], *output);
        assert_eq!(1.0, output.iter().sum::<NodeValue>());
    }

    #[test]
    fn can_compute_hidden_layer_network_rands() {
        let mut shape = NetworkShape::new(2, 2, vec![8], LayerInitStrategy::PositiveRandom);
        shape.set_activation_mode(NetworkActivationMode::SoftMax);
        let network = Network::new(shape);
        let output = network.compute([2.0, 2.0].iter().into()).unwrap();

        assert_ne!(vec![0.5, 0.5], *output);
        assert_eq!(output.iter().sum::<NodeValue>().min(0.999), 0.999);
    }

    #[test]
    fn can_network_learn_randomly() {
        let rng = Rc::new(JsRng::default());
        let mut shape = NetworkShape::new(2, 2, vec![8], LayerInitStrategy::PositiveRandom);
        shape.set_activation_mode(NetworkActivationMode::SoftMax);
        let mut network = Network::new(shape);

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
            NetworkShape::new(2, 2, vec![], LayerInitStrategy::ScaledFullRandom),
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
                info!("DONE round = {round}, network_cost = {error}");
                break;
            }
        }

        multi_sample::assert_training_outputs(&training_pairs, network);
    }

    #[test]
    fn can_network_learn_grad_descent_sigmoid_simple_orthoganal_inputs() {
        let training_pairs = [([0.0, 1.0], [0.1, 0.9]), ([1.0, 0.0], [0.9, 0.1])];
        let mut network = multi_sample::create_network(
            NetworkShape::new(2, 2, vec![], LayerInitStrategy::ScaledFullRandom),
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
                info!("DONE round = {round}, network_cost = {error}");
                break;
            }
        }

        multi_sample::assert_training_outputs(&training_pairs, network);
    }

    #[test]
    fn can_network_learn_grad_descent_tanh_simple_orthoganal_inputs() {
        let training_pairs = [([0.0, 1.0], [0.1, 0.9]), ([1.0, 0.0], [0.9, 0.1])];
        let mut network = multi_sample::create_network(
            NetworkShape::new(2, 2, vec![], LayerInitStrategy::ScaledFullRandom),
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
                info!("DONE round = {round}, network_cost = {error}");
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
            NetworkShape::new(3, 3, vec![], LayerInitStrategy::ScaledFullRandom),
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
                info!("DONE round = {round}, network_cost = {error}");
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
            NetworkShape::new(3, 3, vec![], LayerInitStrategy::ScaledFullRandom),
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
                info!("DONE round = {round}, network_cost = {error}");
                break;
            }
        }

        multi_sample::assert_training_outputs(&training_pairs, network);
    }

    #[test]
    fn can_network_learn_grad_descent_linear() {
        let training_pairs = [([0.1, 0.9], [0.25, 0.75])];
        let mut network = multi_sample::create_network(
            NetworkShape::new(2, 2, vec![6, 6], LayerInitStrategy::ScaledFullRandom),
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
                info!("DONE round = {round}, network_cost = {error}");
                break;
            }
        }

        multi_sample::assert_training_outputs(&training_pairs, network);
    }

    #[test]
    fn can_network_learn_grad_descent_sigmoid() {
        let training_pairs = [([0.1, 0.9], [0.25, 0.75])];
        let mut network = multi_sample::create_network(
            NetworkShape::new(2, 2, vec![6, 6], LayerInitStrategy::ScaledFullRandom),
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
                info!("DONE round = {round}, network_cost = {error}");
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
            NetworkShape::new(3, 1, vec![4, 4], LayerInitStrategy::ScaledFullRandom),
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
                info!("DONE round = {round}, network_cost = {error}");
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
            NetworkShape::new(2, 2, vec![4, 4, 4], LayerInitStrategy::ScaledFullRandom),
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
                info!("DONE round = {round}, network_cost = {error}");
                break;
            }
        }

        multi_sample::assert_training_outputs(&training_pairs, network);
    }

    #[test]
    fn can_network_learn_every_layers_batch_grad_descent_sigmoid() {
        let training_pairs = [([0.1, 0.9], [0.25, 0.75])];
        let mut network = multi_sample::create_network(
            NetworkShape::new(2, 2, vec![6, 6], LayerInitStrategy::ScaledFullRandom),
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
                    .into_iter()
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

        pub(crate) fn create_network(
            mut shape: NetworkShape,
            activation_mode: NetworkActivationMode,
        ) -> Network {
            shape.set_activation_mode(activation_mode);
            let network = Network::new(shape);
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
                info!("round = {round}, network_cost = {error}");
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
                info!("target = {target_output:?}, final_output = {output_2:?}");
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
