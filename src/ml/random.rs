use std::ops::Deref;

use anyhow::{Context, Result};

use crate::ml::NodeValue;

#[cfg(any(test, feature = "thread"))]
#[derive(Default)]
pub struct JsRng(std::cell::RefCell<rand::rngs::ThreadRng>);

#[cfg(not(any(test, feature = "thread")))]
#[derive(Default)]
pub struct JsRng;

impl RNG for JsRng {
    #[cfg(any(test, feature = "thread"))]
    fn rand(&self) -> NodeValue {
        use rand::Rng;
        self.0.borrow_mut().gen()
    }
    #[cfg(not(any(test, feature = "thread")))]
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

pub trait SamplingRng {
    fn sample_uniform(&self, probabilities: &Vec<NodeValue>) -> Result<usize>;
}

impl<T: Deref<Target = dyn RNG>> ShuffleRng for T {
    fn shuffle_vec<E>(&self, vec: &mut Vec<E>) {
        let len = vec.len();

        for i in 0..len {
            let j = self.rand_range(i, len);
            vec.swap(i, j);
        }
    }
}

impl<T: Deref<Target = dyn RNG>> SamplingRng for T {
    fn sample_uniform(&self, probabilities: &Vec<NodeValue>) -> Result<usize> {
        let (sampled_idx, _) = probabilities.iter().enumerate().fold(
            (None, self.rand()),
            |(sampled_idx, state), (p_idx, p)| {
                let next_state = state - p;
                match sampled_idx {
                    Some(sampled_idx) => (Some(sampled_idx), next_state),
                    None if next_state <= 0.0 => (Some(p_idx), next_state),
                    None => (None, next_state),
                }
            },
        );

        let sampled_idx = sampled_idx.with_context(|| {
            format!("failed to sample from provided probabilities: {probabilities:?}")
        })?;

        Ok(sampled_idx)
    }
}
