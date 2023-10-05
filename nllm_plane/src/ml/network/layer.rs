use std::ops::{Deref, DerefMut};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::ml::{RngStrategy, RNG};

#[cfg(not(feature = "short_floats"))]
pub type NodeValue = f64;

#[cfg(feature = "short_floats")]
pub type NodeValue = f32;

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
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

    fn rand_normal(mu: NodeValue, sigma: NodeValue, rng: &dyn RNG) -> NodeValue {
        use std::f64::consts::PI;
        let u1 = rng.rand();
        let u2 = rng.rand();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI as NodeValue * u2).cos();
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
