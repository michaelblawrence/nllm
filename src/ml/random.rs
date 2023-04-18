use std::{cell::UnsafeCell, ops::Deref, rc::Rc};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::ml::NodeValue;

#[derive(Clone, Serialize, Deserialize)]
pub enum RngStrategy {
    Default,
    Debug {
        seed: u32,
    },
    #[serde(serialize_with = "serialize_cached")]
    #[serde(skip_deserializing)]
    Cached(Rc<dyn RNG>, Box<RngStrategy>),
}

impl Default for RngStrategy {
    fn default() -> Self {
        Self::Default.upgrade()
    }
}

impl Deref for RngStrategy {
    type Target = dyn RNG;

    fn deref(&self) -> &Self::Target {
        self
    }
}

impl RNG for RngStrategy {
    fn rand(&self) -> NodeValue {
        self.with_rng(|x| x.rand())
    }
}

impl RngStrategy {
    pub fn testable(seed: u32) -> Self {
        RngStrategy::Debug { seed }.upgrade()
    }
    
    pub fn to_rc(&self) -> Rc<dyn RNG> {
        match self {
            RngStrategy::Cached(instance, _) => instance.clone(),
            rng => rng.factory().unwrap().into(),
        }
    }

    pub fn with_rng<F: Fn(&dyn RNG) -> O, O>(&self, func: F) -> O {
        match self {
            RngStrategy::Cached(instance, _) => func(instance.as_ref()),
            rng => func(&*rng.factory().unwrap()),
        }
    }

    pub fn upgrade(self) -> Self {
        match self {
            RngStrategy::Cached(instance, strategy) => RngStrategy::Cached(instance, strategy),
            rng => RngStrategy::Cached(rng.to_rc(), Box::new(rng)),
        }
    }

    fn factory(&self) -> Option<Box<dyn RNG>> {
        match self {
            RngStrategy::Default => Some(Box::new(JsRng::default())),
            RngStrategy::Debug { seed } => Some(Box::new(SeedableTestRng::new(*seed))),
            RngStrategy::Cached(_, _) => None,
        }
    }
}

fn serialize_cached<S>(
    _: &Rc<dyn RNG>,
    inner: &Box<RngStrategy>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    inner.serialize(serializer)
}

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

pub struct SeedableTestRng(UnsafeCell<algo::mersenne_twister::MersenneTwister>);

impl SeedableTestRng {
    pub fn new(seed: u32) -> Self {
        Self(UnsafeCell::new(
            algo::mersenne_twister::MersenneTwister::new(seed),
        ))
    }
}

impl RNG for SeedableTestRng {
    fn rand(&self) -> NodeValue {
        let rand = {
            let inner = unsafe { &mut *self.0.get() };
            let rand = inner.rand() - 1;
            rand as NodeValue
        };
        rand * algo::mersenne_twister::F64_MULTIPLIER
    }
}

pub struct FastSeedableTestRng(UnsafeCell<algo::park_miller::ParkMiller>);

impl FastSeedableTestRng {
    pub fn new(seed: u64) -> Self {
        Self(UnsafeCell::new(algo::park_miller::ParkMiller::new(seed)))
    }
}

impl RNG for FastSeedableTestRng {
    fn rand(&self) -> NodeValue {
        let rand = {
            let inner = unsafe { &mut *self.0.get() };
            let rand = inner.rand() - 1;
            rand as NodeValue
        };
        rand * algo::park_miller::F64_MULTIPLIER
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seedable_test_rng_samples_uniformly() {
        let rng = SeedableTestRng::new(6);
        assert_rng(&rng);
    }

    #[test]
    fn fast_seedable_test_rng_samples_uniformly() {
        let rng = FastSeedableTestRng::new(6);
        assert_rng(&rng);
    }

    fn assert_rng(rng: &dyn RNG) {
        let mut buckets = vec![0; 13];
        let span = 1.0 / buckets.len() as f64;

        let iters = 10_000;
        for _ in 0..iters {
            let rand = rng.rand();
            let bucket_idx = (rand / span) as usize;
            buckets[bucket_idx] += 1;
        }

        let min_expected = iters / (buckets.len() + 1).max((buckets.len() as f64 * 0.1) as usize);
        for (i, bucket) in buckets.iter().enumerate() {
            assert!(
                *bucket > min_expected,
                "bucket[{i}] distribution is not even {:?}",
                buckets
                    .iter()
                    .enumerate()
                    .map(|(i, c)| format!("bucket[{i}]: {:.1}%", 100.0 * *c as f64 / iters as f64))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
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
    fn take_rand<'a, E>(&self, vec: &'a [E], take: usize) -> Vec<&'a E>;
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

    fn take_rand<'a, E>(&self, vec: &'a [E], take: usize) -> Vec<&'a E> {
        let len = vec.len();
        let mut v = vec![];
        let mut idx = 0;

        for i in 0..take {
            let j = self.rand_range(0, len - i);
            idx += j;
            idx %= len;
            v.push(&vec[idx]);
        }

        v
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

mod algo {
    pub mod mersenne_twister {
        pub const F64_MULTIPLIER: f64 = 1.0 / u32::MAX as f64;

        pub struct MersenneTwister {
            state: [u32; 624],
            index: usize,
        }

        impl MersenneTwister {
            pub fn new(seed: u32) -> Self {
                let mut mt = Self {
                    state: [0; 624],
                    index: 624,
                };
                mt.state[0] = seed;
                for i in 1..624 {
                    let prev = mt.state[i - 1];
                    mt.state[i] = 0x6c078965_u32.wrapping_mul((prev ^ (prev >> 30)) + i as u32);
                }
                mt
            }

            pub fn rand(&mut self) -> u32 {
                if self.index >= 624 {
                    self.twist();
                }
                let mut y = self.state[self.index];
                y ^= y >> 11;
                y ^= (y << 7) & 0x9d2c_5680;
                y ^= (y << 15) & 0xefc6_0000;
                y ^= y >> 18;
                self.index += 1;
                y
            }

            fn twist(&mut self) {
                const MATRIX_A: u32 = 0x9908_b0df;
                const UPPER_MASK: u32 = 0x8000_0000;
                const LOWER_MASK: u32 = 0x7fff_ffff;
                for i in 0..624 {
                    let x = (self.state[i] & UPPER_MASK) + (self.state[(i + 1) % 624] & LOWER_MASK);
                    let mut x_a = x >> 1;
                    if x % 2 != 0 {
                        x_a ^= MATRIX_A;
                    }
                    self.state[i] = self.state[(i + 397) % 624] ^ x_a;
                }
                self.index = 0;
            }
        }
    }
    pub mod park_miller {
        const MODULUS: u64 = 2_147_483_647;
        const MULTIPLIER: u64 = 16_807;
        // const MULTIPLIER: u64 = 1_672_535_203;

        pub(crate) const F64_MULTIPLIER: f64 = 1.0 / 2_147_483_646 as f64;

        pub struct ParkMiller {
            state: u64,
        }

        impl ParkMiller {
            pub fn new(seed: u64) -> Self {
                Self {
                    state: seed % MODULUS,
                }
            }

            pub fn rand(&mut self) -> u64 {
                self.state = self.state.wrapping_mul(MULTIPLIER) % MODULUS;
                self.state
            }
        }
    }
}
