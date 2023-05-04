use std::ops::{Deref, DerefMut};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::ml::{RngStrategy, RNG};

use super::LayerShape;

#[cfg(not(short_floats))]
pub type NodeValue = f64;

#[cfg(short_floats)]
pub type NodeValue = f32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    weights: Vec<NodeValue>,
    bias: Option<Vec<NodeValue>>,
    inputs_count: usize,
    size: usize,
    stride_count: Option<usize>,
    #[serde(default)]
    gradients: LayerGradients,
}

impl Layer {
    pub fn new(inputs_count: usize, shape: &LayerShape, rng: &RngStrategy) -> Self {
        let size = shape.node_count();
        let init_strategy = &shape.strategy();

        let mut weights = vec![0.0; inputs_count * size];
        let mut bias = match &init_strategy {
            x if x.requires_bias() => Some(vec![0.0; size]),
            _ => None,
        };

        init_strategy.apply(
            weights.iter_mut(),
            bias.iter_mut().flatten(),
            inputs_count,
            &rng,
        );

        Self {
            weights,
            bias,
            inputs_count,
            size,
            stride_count: shape.stride(),
            gradients: LayerGradients::default(),
        }
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

    pub fn multiply_weight_matrix(
        &self,
        outputs_dimension_vector: &LayerValues,
    ) -> Result<Vec<NodeValue>> {
        let layer_outputs_count = self.size() * self.stride_count.unwrap_or(1);
        if outputs_dimension_vector.len() != layer_outputs_count {
            return Err(anyhow!(
                "mismatched dimensions for matrix mul. expected={}, actual={}, (strideN={}, inputs={})",
                self.size(),
                outputs_dimension_vector.len(),
                self.stride_count.unwrap_or(1),
                self.inputs_count,
            ));
        }

        let output_weights_per_input = self.weights.chunks(self.size());

        Ok(output_weights_per_input
            .map(|output_weights| {
                outputs_dimension_vector
                    .iter()
                    .zip(output_weights.iter().cycle())
                    .map(|(x, weight)| x * weight)
                    .sum::<NodeValue>()
            })
            .collect())
    }

    pub fn learn(&mut self, strategy: &LayerLearnAction) -> Result<()> {
        match strategy {
            LayerLearnAction::GradientDecent {
                gradients,
                layer_inputs,
            } => {
                let weights_gradients = self.weights_gradients_iter(&gradients, &layer_inputs)?;
                let layer_gradients = self.gradients.get_or_insert(&self.weights, &self.bias);

                for (x, grad) in layer_gradients
                    .weights
                    .iter_mut()
                    .zip(weights_gradients.iter())
                {
                    let next_val = *x + grad;
                    *x = if next_val.is_finite() {
                        next_val
                    } else {
                        return Err(anyhow!(
                            "failed to increment gradient (value={grad}) to optimise weights"
                        ));
                    }
                }
                for (x, grad) in layer_gradients
                    .bias
                    .iter_mut()
                    .flatten()
                    .zip(gradients.iter())
                {
                    let next_val = *x + grad;
                    *x = if next_val.is_finite() {
                        next_val
                    } else {
                        return Err(anyhow!(
                            "failed to increment gradient (value={grad}) to optimise biases"
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn apply_gradients(&mut self, learn_rate: NodeValue) -> Result<()> {
        let weights_gradients = self.gradients.weights.iter_mut();
        let bias_gradients = self.gradients.bias.iter_mut().flat_map(|x| x.iter_mut());

        for (x, grad) in self.weights.iter_mut().zip(weights_gradients) {
            let next_val = *x - (*grad * learn_rate);
            *grad = 0.0;
            *x = if next_val.is_finite() {
                next_val
            } else {
                // continue;
                return Err(anyhow!(
                    "failed to apply gradient (value={grad}) to optimise weights"
                ));
            }
        }
        for (x, grad) in self.bias.iter_mut().flatten().zip(bias_gradients) {
            let next_val = *x - (*grad * learn_rate);
            *grad = 0.0;
            *x = if next_val.is_finite() {
                next_val
            } else {
                // continue;
                return Err(anyhow!(
                    "failed to apply gradient (value={grad}) to optimise biases"
                ));
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
            return Err(anyhow!(
                "mismatched inputs/gradients dimensions len(w)={}, len(x)={}, len(grad)={}, strideN={}",
                weights_len, layer_inputs.len(), gradients.len(),stride_count
            ));
        }

        let layer_input_strides = layer_inputs.chunks(self.inputs_count);
        let layer_output_gradient_strides = gradients.chunks(self.size);

        let weights_gradients = layer_input_strides.zip(layer_output_gradient_strides).fold(
            vec![0.0; weights_len],
            |mut weights_gradients, (layer_inputs, output_gradients)| {
                layer_inputs
                    .into_iter()
                    .flat_map(|layer_input| {
                        output_gradients.iter().map(move |grad| grad * layer_input)
                    })
                    .zip(weights_gradients.iter_mut())
                    .for_each(|(gradient, weights_gradient)| *weights_gradient += gradient);
                weights_gradients
            },
        );

        Ok(weights_gradients)
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn node_weights<'a>(&'a self, node_index: usize) -> Result<&[f64]> {
        let n = self.size();
        let start = n * node_index;
        let end = start + n;

        if end > self.weights.len() {
            return Err(anyhow!("provided index out of range"));
        }

        Ok(&self.weights[start..end])
    }

    pub fn input_vector_size(&self) -> usize {
        let stride_count = self.stride_count.unwrap_or(1);
        self.inputs_count * stride_count
    }

    pub fn weights(&self) -> &[f64] {
        self.weights.as_ref()
    }

    pub fn gradients(&self) -> &LayerGradients {
        &self.gradients
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayerGradients {
    weights: Vec<NodeValue>,
    bias: Option<Vec<NodeValue>>,
}

impl LayerGradients {
    fn get_or_insert(
        &mut self,
        weights: &Vec<NodeValue>,
        bias: &Option<Vec<NodeValue>>,
    ) -> &mut Self {
        if self.weights.len() == 0 {
            self.weights = vec![0.0; weights.len()];
        }
        if self.bias.is_none() && bias.is_some() {
            self.bias = Some(vec![0.0; bias.as_ref().unwrap().len()]);
        }
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerValues(Vec<NodeValue>);

impl<T> From<T> for LayerValues
where
    T: AsRef<[NodeValue]>,
{
    fn from(value: T) -> Self {
        Self(value.as_ref().to_vec())
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
    pub fn assert_length_equals(&self, expected: usize) -> Result<()> {
        if self.len() == expected {
            Ok(())
        } else {
            return Err(anyhow!(
                "lengths are not equal: expected={expected}, actual={actual}",
                actual = self.len()
            ));
        }
    }
    pub fn multiply_iter<'a>(&'a self, rhs: &'a Self) -> impl Iterator<Item = NodeValue> + 'a {
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
    pub fn position_max(&self) -> Option<usize> {
        itertools::Itertools::position_max_by(self.iter(), |x, y| {
            x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
        })
    }
    pub fn rank_iter<'a>(&'a self) -> impl Iterator<Item = usize> + 'a {
        use itertools::Itertools;
        self.iter()
            .enumerate()
            .sorted_by(|x, y| y.1.partial_cmp(x.1).unwrap_or(std::cmp::Ordering::Equal))
            .enumerate()
            .sorted_by_key(|(_rank, (idx, _value))| *idx)
            .map(|(rank, (_idx, _value))| rank + 1)
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

    pub fn cross_entropy_error(&self, expected_outputs: &LayerValues) -> Result<LayerValues> {
        expected_outputs
            .iter()
            .zip(self.iter())
            .map(|(expected, actual)| {
                if *expected == 1.0 {
                    Ok(-actual.ln())
                } else if *expected == 0.0 {
                    Ok(-(1.0 - actual).ln())
                } else {
                    Err(anyhow!(
                        "target outputs should be one-hot encoded: {expected_outputs:?}"
                    ))
                }
            })
            .collect()
    }

    pub fn msd_error(&self, expected_outputs: &LayerValues) -> Result<LayerValues> {
        let mut error = vec![];
        for (actual, expected) in self.iter().zip(expected_outputs.iter()) {
            error.push((actual - expected).powi(2));
        }

        Ok(LayerValues(error))
    }

    pub fn cross_entropy_error_d(&self, expected_outputs: &LayerValues) -> Result<LayerValues> {
        expected_outputs
            .iter()
            .zip(self.iter())
            .map(|(&expected, &actual)| {
                if expected == 0.0 {
                    Ok(actual)
                } else if expected == 1.0 {
                    Ok(actual - 1.0)
                } else {
                    Err(anyhow!(
                        "expected cross entropy outputs should be one-hot encoded"
                    ))
                }
            })
            .collect()
    }

    pub fn msd_error_d(&self, expected_outputs: &LayerValues) -> Result<LayerValues> {
        let mut error = vec![];
        for (actual, expected) in self.iter().zip(expected_outputs.iter()) {
            error.push((actual - expected) * 2.0);
        }

        Ok(LayerValues(error))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LayerInitStrategy {
    Zero,
    NoBiasCopied(LayerValues),
    PositiveRandom,
    ScaledFullRandom,
    ScaledFullRandomZeroBias,
    Kaiming,
    KaimingZeroBias,
    FullRandom,
    NoBias,
}

impl LayerInitStrategy {
    pub fn apply<'a>(
        &self,
        weights: impl Iterator<Item = &'a mut NodeValue>,
        bias: impl Iterator<Item = &'a mut NodeValue>,
        inputs_count: usize,
        rng: &RngStrategy,
    ) {
        use LayerInitStrategy::*;

        match self {
            Zero => {
                for value in weights.chain(bias) {
                    *value = 0.0;
                }
            }
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
            ScaledFullRandom => {
                let scale_factor = (inputs_count as NodeValue).powf(-0.5) * 2.0;
                for value in weights.chain(bias) {
                    *value = Self::full_rand(rng) * scale_factor;
                }
            }
            ScaledFullRandomZeroBias => {
                let scale_factor = (inputs_count as NodeValue).powf(-0.5) * 2.0;
                for value in weights {
                    *value = Self::full_rand(rng) * scale_factor;
                }
                for value in bias {
                    *value = 0.0;
                }
            }
            Kaiming | NoBias => {
                let scale_factor = (inputs_count as NodeValue).powf(-0.5) * 5.0 / 3.0;
                for value in weights.chain(bias) {
                    *value = Self::rand_normal(0.0, 1.0, rng) * scale_factor;
                }
            }
            KaimingZeroBias => {
                let scale_factor = (inputs_count as NodeValue).powf(-0.5) * 5.0 / 3.0;
                for value in weights {
                    *value = Self::rand_normal(0.0, 1.0, rng) * scale_factor;
                }
                for value in bias {
                    *value = 0.0;
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
        (rng.rand() * 2.0) - 1.0
    }

    fn rand_normal(mu: f64, sigma: f64, rng: &dyn RNG) -> f64 {
        use std::f64::consts::PI;
        let u1 = rng.rand();
        let u2 = rng.rand();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        return mu + sigma * z0;
    }

    pub fn requires_bias(&self) -> bool {
        match self {
            LayerInitStrategy::Zero
            | LayerInitStrategy::PositiveRandom
            | LayerInitStrategy::ScaledFullRandom
            | LayerInitStrategy::ScaledFullRandomZeroBias
            | LayerInitStrategy::Kaiming
            | LayerInitStrategy::KaimingZeroBias
            | LayerInitStrategy::FullRandom => true,

            LayerInitStrategy::NoBias | LayerInitStrategy::NoBiasCopied(_) => false,
        }
    }
}

pub enum LayerLearnAction {
    GradientDecent {
        gradients: LayerValues,
        layer_inputs: LayerValues,
    },
}
