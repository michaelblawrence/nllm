use serde::{Deserialize, Serialize};


pub use layer::{LayerValues, NodeValue};

use super::RngStrategy;

pub mod layer;
pub mod transformer;

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
    #[serde(alias = "SoftMax")]
    SoftMaxCrossEntropy,
    SoftMaxCounts,
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

impl NetworkActivationMode {
    pub fn apply(&self, output: &LayerValues) -> LayerValues {
        match self {
            NetworkActivationMode::Linear => output.clone(),
            NetworkActivationMode::SoftMaxCounts | NetworkActivationMode::SoftMaxCrossEntropy => {
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
    pub fn derivative(&self, activation_d: &LayerValues, activation: &LayerValues) -> LayerValues {
        match self {
            NetworkActivationMode::Linear | NetworkActivationMode::SoftMaxCrossEntropy => {
                activation.iter().map(|_| 1.0).collect()
            }
            NetworkActivationMode::SoftMaxCounts => {
                Self::softmax_d(activation_d, activation).collect()
            }
            NetworkActivationMode::Sigmoid => {
                let mut sigmoid = activation.clone();
                sigmoid.iter_mut().for_each(|x| *x = *x * (1.0 - *x));
                sigmoid
            }
            NetworkActivationMode::Tanh => {
                let mut tanh = activation.clone();
                tanh.iter_mut().for_each(|x| *x = 1.0 - x.powi(2));
                tanh
            }
            NetworkActivationMode::RelU => activation
                .iter()
                .map(|&x| if x == 0.0 { 0.0 } else { 1.0 })
                .collect(),
        }
    }

    pub fn is_softmax(&self) -> bool {
        match self {
            NetworkActivationMode::SoftMaxCrossEntropy => true,
            _ => false,
        }
    }

    pub fn softmax_d<'a>(
        softmax_d: &'a [NodeValue],
        softmax: &'a [NodeValue],
    ) -> impl Iterator<Item = NodeValue> + 'a {
        // dL/dx_i = sum(dL/dy_j * softmax(x_j) * (delta_ij - softmax(x_i)))   for i = 1, ..., n and j = 1, ..., n
        let dot_product = softmax
            .iter()
            .zip(softmax_d)
            .map(|(prob_j, grad_j)| prob_j * grad_j)
            .sum::<NodeValue>();

        softmax
            .iter()
            .zip(softmax_d)
            .map(move |(prob_i, grad_i)| prob_i * (grad_i - dot_product))
    }
}

impl Default for NetworkActivationMode {
    fn default() -> Self {
        Self::SoftMaxCrossEntropy
    }
}
