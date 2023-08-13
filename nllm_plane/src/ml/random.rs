use std::{collections::HashMap, hash::Hash, ops::Deref, rc::Rc, sync::Arc};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::ml::NodeValue;

use self::{rc::ArcRNG, store::GlobalRngStore};

use super::cell::MutexCell;

#[derive(Clone, Serialize, Deserialize)]
pub enum RngStrategy {
    Default,

    Debug {
        seed: u32,
    },

    #[serde(serialize_with = "serialize_cached")]
    #[serde(skip_deserializing)]
    Cached(ArcRNG, Arc<RngStrategy>),

    #[serde(alias = "Cached")]
    #[serde(serialize_with = "serialize_unknown")]
    #[serde(deserialize_with = "deserialize_unknown")]
    Unknown(Option<Arc<RngStrategy>>),
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

    pub fn to_arc(&self) -> Arc<dyn RNG> {
        match self {
            RngStrategy::Cached(instance, _) => instance.rng.clone(),
            RngStrategy::Unknown(Some(restored)) => GlobalRngStore::get(restored).to_arc(),
            rng => rng.factory().unwrap().into(),
        }
    }

    pub fn with_rng<F: Fn(&dyn RNG) -> O, O>(&self, func: F) -> O {
        match self {
            RngStrategy::Cached(instance, _) => func(instance.as_ref()),
            RngStrategy::Unknown(Some(restored)) => GlobalRngStore::get(restored).with_rng(func),
            rng => func(&*rng.factory().unwrap()),
        }
    }

    pub fn upgrade(self) -> Self {
        match self {
            RngStrategy::Cached(instance, strategy) => RngStrategy::Cached(instance, strategy),
            RngStrategy::Unknown(None) => RngStrategy::Default.upgrade(),
            RngStrategy::Unknown(Some(restored)) => {
                (*GlobalRngStore::get(&restored)).clone().upgrade()
            }
            rng => RngStrategy::Cached(rng.to_arc().into(), Arc::new(rng)),
        }
    }

    fn factory(&self) -> Option<Box<dyn RNG>> {
        match self {
            RngStrategy::Default | RngStrategy::Unknown(_) => Some(Box::new(JsRng::default())),
            RngStrategy::Debug { seed } => Some(Box::new(SeedableTestRng::new(*seed))),
            RngStrategy::Cached(_, _) => None,
        }
    }

    /// Returns `true` if the rng strategy is [`Cached`].
    ///
    /// [`Cached`]: RngStrategy::Cached
    #[must_use]
    pub fn is_cached(&self) -> bool {
        matches!(self, Self::Cached(..))
    }

    /// Returns `true` if the rng strategy is [`Debug`].
    ///
    /// [`Debug`]: RngStrategy::Debug
    #[must_use]
    pub fn is_debug(&self) -> bool {
        match self {
            Self::Debug { .. } => true,
            Self::Cached(_, inner) => inner.is_debug(),
            Self::Unknown(Some(restored)) => restored.is_debug(),
            _ => false,
        }
    }
}

unsafe impl Send for RngStrategy {}

fn serialize_cached<S>(
    _: &Arc<dyn RNG>,
    inner: &Arc<RngStrategy>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let mut inner = inner;
    while let RngStrategy::Cached(_, child) = &**inner {
        inner = child;
    }
    inner.serialize(serializer)
}

fn serialize_unknown<S>(
    inner: &Option<Arc<RngStrategy>>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match &inner {
        Some(child) => child.serialize(serializer),
        _ => Option::<RngStrategy>::None.serialize(serializer),
    }
}

fn deserialize_unknown<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<Arc<RngStrategy>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let rng = RngStrategy::deserialize(deserializer)?;
    Ok(Some(Arc::new(rng)))
}

impl std::fmt::Debug for RngStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Default => write!(f, "Default"),
            Self::Debug { seed } => f.debug_struct("Debug").field("seed", seed).finish(),
            Self::Cached(_, inner) => f.debug_struct("Cached").field("inner", inner).finish(),
            Self::Unknown(inner) => f.debug_struct("Unknown").field("inner", inner).finish(),
        }
    }
}

#[cfg(any(test, feature = "threadrng"))]
#[derive(Default)]
pub struct JsRng(std::cell::RefCell<rand::rngs::ThreadRng>);

#[cfg(not(any(test, feature = "threadrng")))]
#[derive(Default)]
pub struct JsRng;

impl RNG for JsRng {
    #[cfg(any(test, feature = "threadrng"))]
    fn rand(&self) -> NodeValue {
        use rand::Rng;
        self.0.borrow_mut().gen()
    }
    #[cfg(not(any(test, feature = "threadrng")))]
    fn rand(&self) -> NodeValue {
        js_sys::Math::random() as NodeValue
    }
}

mod rc {
    use std::{ops::Deref, sync::Arc};

    use super::{RngStrategy, RNG};

    #[derive(Clone)]
    pub struct ArcRNG {
        pub rng: Arc<dyn RNG>,
    }

    unsafe impl Sync for ArcRNG {}

    impl Default for ArcRNG {
        fn default() -> Self {
            Self {
                rng: Arc::new(RngStrategy::default()),
            }
        }
    }

    impl Deref for ArcRNG {
        type Target = Arc<dyn RNG>;

        fn deref(&self) -> &Self::Target {
            &self.rng
        }
    }

    impl From<Arc<dyn RNG>> for ArcRNG {
        fn from(value: Arc<dyn RNG>) -> Self {
            Self { rng: value }
        }
    }
}

mod store {
    use super::*;
    use std::cell::RefCell;
    thread_local! {
        static FOO: RefCell<HashMap<u64, Rc<RngStrategy>>> = RefCell::new(HashMap::default());
    }

    #[derive(Debug)]
    struct GlobalRngStoreEntry<'a>(&'a RngStrategy);

    impl<'a> GlobalRngStoreEntry<'a> {
        fn to_hash(&self) -> u64 {
            use std::hash::Hasher;

            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            self.hash(&mut hasher);
            hasher.finish()
        }
    }

    impl<'a> Eq for GlobalRngStoreEntry<'a> {}

    impl<'a> PartialEq for GlobalRngStoreEntry<'a> {
        fn eq(&self, other: &Self) -> bool {
            let hash1 = self.to_hash();
            let hash2 = other.to_hash();
            hash1 == hash2
        }
    }

    impl<'a> Hash for GlobalRngStoreEntry<'a> {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            match self.0 {
                RngStrategy::Default => "default".hash(state),
                RngStrategy::Debug { seed } => ("no_upgrade_debug", seed).hash(state),
                RngStrategy::Cached(_, inner) => Self(&inner).hash(state),
                RngStrategy::Unknown(Some(inner)) => {
                    ("no_upgrade", Self(&inner).hash(state)).hash(state)
                }
                RngStrategy::Unknown(None) => "unknown".hash(state),
            }
        }
    }

    pub struct GlobalRngStore;

    impl GlobalRngStore {
        pub fn get(rng: &RngStrategy) -> Rc<RngStrategy> {
            let entry = GlobalRngStoreEntry(rng);
            FOO.with(|foo| {
                foo.borrow_mut()
                    .entry(entry.to_hash())
                    .or_insert_with(|| Rc::new(rng.clone().upgrade()))
                    .clone()
            })
        }
    }
}

pub struct SeedableTestRng(MutexCell<algo::mersenne_twister::MersenneTwister>);

impl SeedableTestRng {
    pub fn new(seed: u32) -> Self {
        Self(MutexCell::new(
            algo::mersenne_twister::MersenneTwister::new(seed),
        ))
    }
}

impl RNG for SeedableTestRng {
    fn rand(&self) -> NodeValue {
        let rand = {
            let rand = self.0.with_inner(|inner| inner.rand() - 1);
            rand as NodeValue
        };
        rand * algo::mersenne_twister::F64_MULTIPLIER
    }
}

pub struct FastSeedableTestRng(MutexCell<algo::park_miller::ParkMiller>);

impl FastSeedableTestRng {
    pub fn new(seed: u64) -> Self {
        Self(MutexCell::new(algo::park_miller::ParkMiller::new(seed)))
    }
}

impl RNG for FastSeedableTestRng {
    fn rand(&self) -> NodeValue {
        let rand = {
            let rand = self.0.with_inner(|inner| inner.rand() - 1);
            rand as NodeValue
        };
        rand * algo::park_miller::F64_MULTIPLIER
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_strategy_can_be_serialized() {
        // default rng
        let rng_from_variant = RngStrategy::Default;
        let json = serde_json::to_string(&rng_from_variant).unwrap();
        assert!(!json.is_empty());

        let rng_upgraded = rng_from_variant.upgrade();
        let json = serde_json::to_string(&rng_upgraded).unwrap();
        assert!(!json.is_empty());

        let rng_from_factory = RngStrategy::default();
        let json = serde_json::to_string(&rng_from_factory).unwrap();
        assert!(!json.is_empty());

        // seeded rng
        let rng_from_variant = RngStrategy::Debug { seed: 1234 };
        let json = serde_json::to_string(&rng_from_variant).unwrap();
        assert!(!json.is_empty());

        let rng_upgraded = rng_from_variant.upgrade();
        let json = serde_json::to_string(&rng_upgraded).unwrap();
        assert!(!json.is_empty());

        let rng_from_factory = RngStrategy::testable(1234);
        let json = serde_json::to_string(&rng_from_factory).unwrap();
        assert!(!json.is_empty());
    }

    #[test]
    fn rng_strategy_can_be_deserialized() {
        // default rng
        let src_rng = RngStrategy::default();
        let json = serde_json::to_string(&src_rng).unwrap();
        let rng_from_json: RngStrategy = serde_json::from_str(dbg!(&json)).unwrap();
        let rng = rng_from_json.upgrade();
        assert!(rng.is_cached());
        assert!(!rng.is_debug());

        // seeded rng
        let src_rng = RngStrategy::testable(1234);
        let json = serde_json::to_string(&src_rng).unwrap();
        let rng_from_json: RngStrategy = serde_json::from_str(dbg!(&json)).unwrap();
        let rng = rng_from_json.upgrade();
        assert!(rng.is_cached());
        assert!(rng.is_debug());
    }

    #[test]
    fn rng_strategy_can_be_deserialized_and_joined() {
        // seeded rng
        let src_rng = RngStrategy::testable(1234);
        let json = serde_json::to_string(&src_rng).unwrap();
        let rng_from_json_1: RngStrategy = serde_json::from_str(dbg!(&json)).unwrap();
        let rng_from_json_2: RngStrategy = serde_json::from_str(dbg!(&json)).unwrap();

        let sample1 = rng_from_json_1.rand_range(0, 1000);
        let sample2 = rng_from_json_2.rand_range(0, 1000);

        assert_ne!(sample1, sample2);
        assert!(rng_from_json_1.is_debug());
        assert!(rng_from_json_2.is_debug());
    }

    #[test]
    fn rng_strategy_can_be_deserialized_repeatedly() {
        // seeded rng
        let src_rng = RngStrategy::testable(1234);

        let json_1 = serde_json::to_string(&src_rng).unwrap();
        let rng_from_json_1: RngStrategy = serde_json::from_str(dbg!(&json_1)).unwrap();

        let json_2 = serde_json::to_string(&rng_from_json_1).unwrap();
        let rng_from_json_2: RngStrategy = serde_json::from_str(dbg!(&json_2)).unwrap();

        assert!(rng_from_json_1.is_debug());
        assert!(rng_from_json_2.is_debug());
    }

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
        let sampled_idx = probabilities
            .iter()
            .enumerate()
            .scan(self.rand(), |state, (p_idx, p)| {
                if *state > 0.0 {
                    *state -= p;
                    Some(p_idx)
                } else {
                    None
                }
            })
            .last();

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
