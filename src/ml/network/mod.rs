use std::str::FromStr;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::ml::ShuffleRng;

use layer::{Layer, LayerInitStrategy, LayerLearnAction};
pub use layer::{LayerValues, NodeValue};

use super::RngStrategy;

pub mod layer;

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
        init_strategy: LayerInitStrategy,
    ) -> Self {
        Self {
            inputs_count,
            layers_shape: hidden_layer_shape
                .into_iter()
                .map(|layer_size| (layer_size, init_strategy.clone()))
                .chain([(outputs_count, init_strategy.clone())])
                .map(|x| x.into())
                .collect(),
            activation_mode: Default::default(),
        }
    }

    pub fn new_custom(
        inputs_count: usize,
        layers_shape: Vec<LayerShape>,
        activation_mode: NetworkActivationMode,
    ) -> Result<Self> {
        let residual_connections_mismatch = layers_shape
            .windows(2)
            .map(|l| (&l[0], &l[1]))
            .filter(|(lhs, _)| lhs.residual_connections)
            .any(|(lhs, rhs)| lhs.output_vector_len() != rhs.output_vector_len());

        if residual_connections_mismatch {
            return Err(anyhow!(
                "layers with residual connections must share vector dimensions"
            ));
        }

        Ok(Self {
            inputs_count,
            layers_shape,
            activation_mode,
        })
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

    pub fn desc_pretty(&self) -> String {
        let stride_count = |x: &LayerShape| x.stride().map(|x| format!("x{}", x));

        let mut node_counts = self
            .iter()
            .map(|x| format!("{}{}", x.node_count(), stride_count(&x).unwrap_or_default()));

        itertools::Itertools::join(&mut node_counts, " -> ")
    }

    pub fn layers_shape(&self) -> &[LayerShape] {
        self.layers_shape.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerShape {
    node_count: usize,
    strategy: LayerInitStrategy,
    stride_count: Option<usize>,
    activation_mode: Option<NetworkActivationMode>,
    #[serde(default)]
    residual_connections: bool,
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
            stride_count: stride_count.filter(|x| *x > 1),
            activation_mode,
            residual_connections: false,
        }
    }

    pub fn with_activation_mode(self, mode: NetworkActivationMode) -> Self {
        Self {
            activation_mode: Some(mode),
            ..self
        }
    }

    pub fn with_residual_connections(self, enable: bool) -> Self {
        Self {
            residual_connections: enable,
            ..self
        }
    }

    pub fn mode_override(&self) -> Option<NetworkActivationMode> {
        self.activation_mode
    }

    pub fn stride(&self) -> Option<usize> {
        self.stride_count
    }

    pub fn stride_count(&self) -> usize {
        self.stride_count.unwrap_or(1)
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn output_vector_len(&self) -> usize {
        self.node_count * self.stride_count.unwrap_or(1)
    }

    pub fn strategy(&self) -> &LayerInitStrategy {
        &self.strategy
    }

    pub fn residual_connections(&self) -> bool {
        self.residual_connections
    }
}

impl From<(usize, LayerInitStrategy)> for LayerShape {
    fn from((node_count, strategy): (usize, LayerInitStrategy)) -> Self {
        LayerShape {
            node_count,
            strategy,
            stride_count: None,
            activation_mode: None,
            residual_connections: false,
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
        Self::new_with_rng(shape, RngStrategy::default())
    }

    pub fn new_with_rng(shape: NetworkShape, rng: RngStrategy) -> Self {
        let mut layers = vec![];
        let rng = rng.upgrade();

        for (pair_idx, (lhs, rhs)) in itertools::Itertools::tuple_windows(shape.iter()).enumerate()
        {
            if pair_idx != 0 && rhs.stride().is_some() {
                todo!("layer stride not yet supported for non-starting layers");
            }
            let layer_input_nodes = lhs.output_vector_len();
            let layer = Layer::new(layer_input_nodes, &rhs, &rng);
            layers.push(layer)
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
            self.final_layer_activation()?.is_softmax(),
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
                self.final_layer_activation()?.is_softmax(),
            )?;

        Ok(output_layer_error.ave())
    }

    #[instrument(level = "info", name = "net_learn", skip_all)]
    pub fn learn(&mut self, strategy: NetworkLearnStrategy) -> Result<Option<NodeValue>> {
        let cost = match strategy {
            NetworkLearnStrategy::BatchGradientDecent {
                training_pairs,
                learn_rate,
                batch_sampling,
            } => {
                use itertools::Itertools;
                if training_pairs.is_empty() {
                    return Ok(None);
                }

                let batches = match batch_sampling {
                    BatchSamplingStrategy::None => {
                        let size = training_pairs.len();
                        training_pairs.into_iter().chunks(size)
                    }
                    BatchSamplingStrategy::Sequential(batch_size) => {
                        training_pairs.into_iter().chunks(batch_size)
                    }
                    BatchSamplingStrategy::Shuffle(batch_size, rng) => {
                        let mut training_pairs = training_pairs;
                        rng.shuffle_vec(&mut training_pairs);
                        training_pairs.into_iter().chunks(batch_size)
                    }
                };

                let mut network_errors = vec![];

                for batch in batches.into_iter() {
                    let (batch_errors, layer_learn_actions) =
                        self.process_gradient_descent_batch(batch.collect())?;

                    let learn_actions = layer_learn_actions.iter().flatten();
                    self.apply_pending_layer_actions(learn_actions, learn_rate)?;

                    let batch_errors = LayerValues::new(batch_errors);
                    network_errors.push(batch_errors.ave());
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

    #[cfg(feature = "threadpool")]
    #[instrument(level = "info", name = "grad_batch", skip_all)]
    fn process_gradient_descent_batch(
        &mut self,
        batch: Vec<(LayerValues, LayerValues)>,
    ) -> Result<(Vec<f64>, Vec<Vec<(usize, LayerLearnAction)>>)> {
        use rayon::prelude::*;

        let (batch_errors, layer_learn_actions) = batch
            .into_par_iter()
            .map(|(inputs, target_outputs)| {
                let layers_activations = self.compute_all_layers_activations(inputs)?;
                let output_layer_error =
                    self.compute_error_precomputed(&layers_activations, &target_outputs)?;
                let learn_actions =
                    self.perform_gradient_decent_step(&layers_activations, &target_outputs)?;
                Ok((output_layer_error, learn_actions))
            })
            .collect::<Result<_>>()?;

        Ok((batch_errors, layer_learn_actions))
    }

    #[cfg(not(feature = "threadpool"))]
    fn process_gradient_descent_batch(
        &mut self,
        batch: Vec<(LayerValues, LayerValues)>,
    ) -> Result<(Vec<f64>, Vec<Vec<(usize, LayerLearnAction)>>)> {
        let all_layers_activations = batch
            .into_iter()
            .map(|(inputs, target_outputs)| {
                let layers_activations = self.compute_all_layers_activations(inputs)?;
                Ok((layers_activations, target_outputs))
            })
            .collect::<Result<Vec<_>>>()?;

        let batch_errors = all_layers_activations
            .iter()
            .map(|(layers_activations, target_outputs)| {
                let output_layer_error =
                    self.compute_error_precomputed(&layers_activations, &target_outputs);
                output_layer_error
            })
            .collect::<Result<Vec<_>>>()?;

        let layer_learn_actions = all_layers_activations
            .iter()
            .map(|(layers_activations, target_outputs)| {
                let learn_actions =
                    self.perform_gradient_decent_step(&layers_activations, &target_outputs);
                learn_actions
            })
            .collect::<Result<Vec<_>>>()?;

        Ok((batch_errors, layer_learn_actions))
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
    ) -> Result<Vec<(usize, LayerLearnAction)>> {
        let layer_states = self.to_intermediate_states(&layers_activations)?;

        // BEWARE: a little bit of maths to compute MLP parameters via back-propagation
        //
        // --- forward pass ---
        // l = loss( a(z0(x)) ) = loss( a( w * a(z'(x)) + b ) )
        //      [ a(..) is activation function (tanh, sigmoid, etc.)]
        //      [ z(x)  is the weighted inputs of a given MLP layer]
        //      [ z(x)  is equal to either (w * a(z'(x)) + b) or the network inputs, 'x' ]
        //
        // --- backward pass ---
        // dl/da  = loss_d(..)
        // dl/dz0 = dl/da * da/dz  =  loss_d(..) * activation_d(z, ..)
        //              [this allows us to calc gradients for final MLP layer]
        // dl/dz' = dl/dz * dz/dz' =  dl/dz * (w * da/dz')  =  dl/dz * w * activation_d(z', ..)
        //              [this allows us to calc gradients for any MLP layer]
        // dl/dw  = dl/dz * dz/dw  =  dl/dz * a(z'(x))
        //              [as seen above we can calculate dl/dz for any layer]
        // dl/db  = dl/dz * dz/db  =  dl/dz

        let mut layer_learn_actions = vec![];
        let mut backprop_activation_gradients = None;

        for state in layer_states.iter().rev() {
            let dl_da = match backprop_activation_gradients.take() {
                Some(prev_layer_dl_dx) => LayerValues::new(prev_layer_dl_dx),
                None => {
                    let layer_activations = state.layer_activations;
                    let layer = &state.layer;

                    match state.activation_mode {
                        NetworkActivationMode::SoftMax => {
                            layer.cross_entropy_error_d(&layer_activations, &target_outputs)?
                        }
                        _ => layer.msd_error_d(&layer_activations, &target_outputs)?,
                    }
                }
            };

            let da_dz = state
                .activation_mode
                .derivative(&state.layer_weighted_outputs);

            let dl_dz = dl_da.multiply_iter(&da_dz).collect();
            let dl_dz_next = state.layer.multiply_weight_matrix(&dl_dz)?;

            let strategy = LayerLearnAction::GradientDecent {
                gradients: dl_dz,
                layer_inputs: state.layer_input.clone(),
            };

            layer_learn_actions.push((state.layer_id, strategy));
            backprop_activation_gradients = Some(dl_dz_next);
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

    fn to_intermediate_states<'a>(
        &'a self,
        layers_activations: &'a Vec<(LayerValues, LayerValues)>,
    ) -> Result<Vec<LayerIntermediate>> {
        let layer_io_iter = layers_activations
            .windows(2)
            .map(|pair| (&pair[0].1, &pair[1].0, &pair[1].1));

        let activations_iter = self
            .layers
            .iter()
            .enumerate()
            .map(|(i, layer)| (layer, i, self.layer_activation(i)));

        let layer_states = activations_iter.zip(layer_io_iter).map(
            |((layer, idx, mode), (l_input, l_output, l_activation))| {
                Ok(LayerIntermediate {
                    layer,
                    layer_id: idx,
                    layer_input: l_input,
                    layer_weighted_outputs: l_output,
                    layer_activations: l_activation,
                    activation_mode: mode?,
                })
            },
        );

        Ok(layer_states.collect::<Result<Vec<_>>>()?)
    }

    #[instrument(level = "info", skip_all)]
    #[cfg(feature = "threadpool")]
    fn apply_pending_layer_actions<'a, I: Iterator<Item = &'a (usize, LayerLearnAction)>>(
        &mut self,
        learn_actions: I,
        learn_rate: f64,
    ) -> Result<()> {
        use std::collections::HashMap;
        use itertools::Itertools;
        use rayon::prelude::*;

        let layer_actions: HashMap<usize, Vec<&LayerLearnAction>> = learn_actions
            .group_by(|x| x.0)
            .into_iter()
            .map(|(id, group)| (id, group.map(|(_, b)| b).collect_vec()))
            .collect();

        let each_layers_actions = self
            .layers
            .iter_mut()
            .enumerate()
            .map(|(idx, layer)| (layer, layer_actions.get(&idx)))
            .collect_vec();

        each_layers_actions
            .into_par_iter()
            .try_for_each(|(layer, layer_actions)| {
                for layer_action in layer_actions.unwrap_or(&vec![]) {
                    layer.learn(&layer_action)?;
                }
                layer.apply_gradients(learn_rate)?;
                Ok::<(), anyhow::Error>(())
            })?;

        Ok(())
    }

    #[instrument(level = "info", skip_all)]
    #[cfg(not(feature = "threadpool"))]
    fn apply_pending_layer_actions<'a, I: Iterator<Item = &'a (usize, LayerLearnAction)>>(
        &mut self,
        learn_actions: I,
        learn_rate: f64,
    ) -> Result<()> {
        for (layer_idx, learn_strategy) in learn_actions.into_iter() {
            let layer = self
                .layers
                .get_mut(*layer_idx)
                .context("this layer index should be present")?;

            layer.learn(&learn_strategy)?;
            layer.apply_gradients(learn_rate)?;
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
        let last_idx = self.shape.len() - 1;
        self.layer_activation(last_idx)
    }

    pub fn shape(&self) -> &NetworkShape {
        &self.shape
    }
}

struct LayerIntermediate<'a> {
    layer: &'a Layer,
    layer_id: usize,
    layer_input: &'a LayerValues,
    layer_weighted_outputs: &'a LayerValues,
    layer_activations: &'a LayerValues,
    activation_mode: NetworkActivationMode,
}

pub enum NetworkLearnStrategy {
    BatchGradientDecent {
        training_pairs: Vec<(LayerValues, LayerValues)>,
        learn_rate: NodeValue,
        batch_sampling: BatchSamplingStrategy,
    },
    Multi(Vec<NetworkLearnStrategy>),
}

pub enum BatchSamplingStrategy {
    None,
    Sequential(usize),
    Shuffle(usize, RngStrategy),
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
                let max = output
                    .iter()
                    .copied()
                    .max_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or_default();

                let exp_iter = output.iter().map(|x| (x - max).exp());
                let sum: NodeValue = exp_iter.clone().sum();
                LayerValues::new(exp_iter.map(|x| x / sum).collect())
            }
            NetworkActivationMode::Sigmoid => {
                LayerValues::new(output.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect())
            }
            NetworkActivationMode::Tanh => {
                LayerValues::new(output.iter().map(|x| x.tanh()).collect())
            }
            NetworkActivationMode::RelU => {
                LayerValues::new(output.iter().map(|x| x.max(0.0)).collect())
            }
        }
    }
    pub fn derivative(&self, output: &LayerValues) -> LayerValues {
        match self {
            NetworkActivationMode::Linear => output.iter().map(|_| 1.0).collect(),
            // TODO: cross-entroy loss derivative is calculate elsewhere for FINAL layer only.
            // can improve api?
            NetworkActivationMode::SoftMax => output.iter().map(|_| 1.0).collect(),
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

    pub fn is_softmax(&self) -> bool {
        match self {
            NetworkActivationMode::SoftMax => true,
            _ => false,
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
    use test_log::test;
    use tracing::info;

    use comparer::ApproxFloatingPointComparable;

    use super::*;

    #[test]
    fn can_create_network() {
        let shape = NetworkShape::new(2, 2, vec![], LayerInitStrategy::Zero);
        let rng = RngStrategy::testable(12345);
        let network = Network::new_with_rng(shape, rng);

        assert_eq!(2, network.input_vector_size().unwrap());
    }

    #[test]
    fn can_create_hidden_layer_network() {
        let shape = NetworkShape::new(2, 2, vec![8], LayerInitStrategy::Zero);
        let rng = RngStrategy::testable(12345);
        let network = Network::new_with_rng(shape, rng);

        assert_eq!(2, network.input_vector_size().unwrap());
    }

    #[test]
    fn can_compute_hidden_layer_network_zeros_linear_activation() {
        let mut shape = NetworkShape::new(2, 2, vec![8], LayerInitStrategy::Zero);
        shape.set_activation_mode(NetworkActivationMode::Linear);
        let rng = RngStrategy::testable(12345);
        let network = Network::new_with_rng(shape, rng);
        let output = network.compute([2.0, 2.0].iter().into()).unwrap();

        assert_eq!(vec![0.0, 0.0].approx(), output.approx());
    }

    #[test]
    fn can_compute_hidden_layer_network_zeros() {
        let mut shape = NetworkShape::new(2, 2, vec![8], LayerInitStrategy::Zero);
        shape.set_activation_mode(NetworkActivationMode::SoftMax);
        let rng = RngStrategy::testable(12345);
        let network = Network::new_with_rng(shape, rng);
        let output = network.compute([2.0, 2.0].iter().into()).unwrap();

        assert_eq!(vec![0.5, 0.5].approx(), output.approx());
        assert_eq!(1.0, output.iter().sum::<NodeValue>());
    }

    #[test]
    fn can_compute_hidden_layer_network_rands() {
        let mut shape = NetworkShape::new(2, 2, vec![8], LayerInitStrategy::PositiveRandom);
        shape.set_activation_mode(NetworkActivationMode::SoftMax);
        let rng = RngStrategy::testable(12345);
        let network = Network::new_with_rng(shape, rng);
        let output = network.compute([2.0, 2.0].iter().into()).unwrap();

        assert_ne!(vec![0.5, 0.5].approx(), output.approx());
        assert_eq!(output.iter().sum::<NodeValue>().min(0.999), 0.999);
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

        multi_sample::gradient_decent::compute_training_iteration(
            &mut network,
            &training_pairs,
            learn_rate,
            BatchSamplingStrategy::Sequential(batch_size.unwrap()),
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
            let rng = RngStrategy::testable(12345);
            let network = Network::new_with_rng(shape, rng);
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
                gradient_decent::compute_training_iteration(
                    network,
                    training_pairs,
                    learn_rate,
                    BatchSamplingStrategy::Sequential(batch_size),
                )
            } else {
                gradient_decent::compute_training_iteration(
                    network,
                    training_pairs,
                    learn_rate,
                    BatchSamplingStrategy::None,
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

            pub fn compute_training_iteration<const N: usize, const O: usize>(
                network: &mut Network,
                training_pairs: &[([f64; N], [f64; O])],
                learn_rate: f64,
                batch_sampling: BatchSamplingStrategy,
            ) -> f64 {
                network
                    .learn(NetworkLearnStrategy::BatchGradientDecent {
                        training_pairs: training_pairs
                            .iter()
                            .map(|(i, o)| (i.iter().into(), o.iter().into()))
                            .collect(),
                        learn_rate,
                        batch_sampling,
                    })
                    .unwrap()
                    .unwrap()
            }
        }
    }

    mod comparer {
        #[derive(Debug)]
        pub struct ApproxFloatingPointComparer<'a> {
            inner: &'a Vec<f64>,
        }

        impl<'a> PartialEq for ApproxFloatingPointComparer<'a> {
            fn eq(&self, other: &Self) -> bool {
                if self.inner == other.inner {
                    true
                } else if self.inner.len() == other.inner.len() {
                    for (a, b) in self.inner.iter().zip(other.inner.iter()) {
                        if (a - b).abs() > 0.0001 {
                            return false;
                        }
                    }
                    true
                } else {
                    false
                }
            }
        }

        pub trait ApproxFloatingPointComparable<'a> {
            fn approx(&'a self) -> ApproxFloatingPointComparer<'a>;
        }

        impl<'a> ApproxFloatingPointComparable<'a> for Vec<f64> {
            fn approx(&'a self) -> ApproxFloatingPointComparer<'a> {
                ApproxFloatingPointComparer { inner: &self }
            }
        }
    }
}
