use std::ops::Deref;

#[derive(Debug, Clone, Copy)]
pub enum TrainBatchConfig {
    Batches(usize),
    SingleBatch(usize),
}

impl From<usize> for TrainBatchConfig {
    fn from(v: usize) -> Self {
        Self::Batches(v)
    }
}

impl Deref for TrainBatchConfig {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        match self {
            TrainBatchConfig::Batches(x) => x,
            TrainBatchConfig::SingleBatch(x) => x,
        }
    }
}
