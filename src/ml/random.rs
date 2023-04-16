use std::{cell::UnsafeCell, ops::Deref, rc::Rc};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::ml::NodeValue;

#[derive(Default, Clone, Serialize, Deserialize)]
pub enum RngStrategy {
    #[default]
    Default,
    Debug {
        seed: u64,
    },
    #[serde(skip)]
    Cached(Rc<dyn RNG>),
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
    pub fn to_rc(&self) -> Rc<dyn RNG> {
        match self {
            RngStrategy::Default => Rc::new(Self::default_rng_factory()),
            RngStrategy::Debug { seed } => Rc::new(Self::seedable_rng_factory(*seed)),
            RngStrategy::Cached(instance) => instance.clone(),
        }
    }

    pub fn with_rng<F: Fn(&dyn RNG) -> O, O>(&self, func: F) -> O {
        match self {
            RngStrategy::Default => func(&Self::default_rng_factory()),
            RngStrategy::Debug { seed } => func(&Self::seedable_rng_factory(*seed)),
            RngStrategy::Cached(instance) => func(instance.as_ref()),
        }
    }

    pub fn upgrade(self) -> Self {
        match self {
            RngStrategy::Default => {
                let rng = self.to_rc();
                Self::Cached(rng)
            }
            x => x,
        }
    }

    fn default_rng_factory() -> impl RNG {
        JsRng::default()
    }

    fn seedable_rng_factory(seed: u64) -> impl RNG {
        SeedableTestRng::new(seed)
    }
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

pub struct SeedableTestRng(UnsafeCell<algo::park_miller::ParkMiller>);

impl SeedableTestRng {
    pub fn new(seed: u64) -> Self {
        Self(UnsafeCell::new(algo::park_miller::ParkMiller::new(seed)))
    }
}

impl RNG for SeedableTestRng {
    fn rand(&self) -> NodeValue {
        let rand = {
            let inner = unsafe { &mut *self.0.get() };
            let rand = inner.rand() - 1;
            rand as NodeValue
        };
        rand * algo::park_miller::F64_MULTIPLIER
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
