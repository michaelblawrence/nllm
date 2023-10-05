use anyhow::Result;
use tracing::info;

use plane::ml::{
    embeddings::TrainBatchConfig,
    gdt::{GenerativeDecoderTransformer, TrainContext},
    transformer::solver::{
        source::{DefaultOptimizerCache, DynamicOptimizerFactory},
        AdamOptimizer,
    },
    NodeValue,
};

use crate::training;

pub trait MLModel {
    fn snapshot(&self) -> Result<String>;
    fn as_gdt(&self) -> Option<&GenerativeDecoderTransformer>;
    fn train<B: Into<TrainBatchConfig>>(
        &mut self,
        phrases: &Vec<Vec<String>>,
        batch_size: B,
    ) -> Result<NodeValue>;
    fn set_learn_rate(&mut self, learn_rate: NodeValue);
    fn generate_sequence_string(&self) -> Result<String>;
    fn from_gdt(
        v: GenerativeDecoderTransformer,
        learn_rate: NodeValue,
        sample_from_pattern: Option<String>,
    ) -> Self;
}

pub enum EmbedModel {
    GDT(GenerativeDecoderTransformer, GDTTrainContext),
}

pub type OptType = DefaultOptimizerCache<DynamicOptimizerFactory<AdamOptimizer>, AdamOptimizer>;

pub struct GDTTrainContext(OptType, NodeValue, Option<String>, Option<String>);

impl EmbedModel {
    fn as_gdt(&self) -> Option<&GenerativeDecoderTransformer> {
        match self {
            Self::GDT(v, _) => Some(v),
        }
    }
}

impl MLModel for EmbedModel {
    fn train<B: Into<TrainBatchConfig>>(
        &mut self,
        phrases: &Vec<Vec<String>>,
        batch_size: B,
    ) -> Result<NodeValue> {
        match self {
            EmbedModel::GDT(model, GDTTrainContext(opt, _, train_corpus, sample_from_pattern)) => {
                let train_corpus = &*train_corpus.get_or_insert_with(move || {
                    training::TrainerState::<()>::join_phrases(phrases, Some(false))
                });
                let train_context = TrainContext::new(batch_size, sample_from_pattern.as_deref());
                model.train(train_corpus, opt, &train_context)
            }
        }
    }

    fn set_learn_rate(&mut self, learn_rate: NodeValue) {
        let next_learn_rate = learn_rate;
        match self {
            EmbedModel::GDT(_, GDTTrainContext(opt, learn_rate, ..)) => {
                if *learn_rate != next_learn_rate {
                    opt.for_each_mut(|o| AdamOptimizer::set_eta(o, next_learn_rate));
                    info!(
                        "AdamOptimizer has set learn_rate (eta) from {} to {}",
                        learn_rate, next_learn_rate
                    );
                    *learn_rate = next_learn_rate;
                }
            }
        }
    }

    fn snapshot(&self) -> Result<String> {
        match self {
            EmbedModel::GDT(model, _) => model.snapshot(),
        }
    }

    fn generate_sequence_string(&self) -> Result<String> {
        match self {
            EmbedModel::GDT(model, _) => model.generate_sequence_string(None),
        }
    }

    fn from_gdt(
        v: GenerativeDecoderTransformer,
        learn_rate: NodeValue,
        sample_from_pattern: Option<String>,
    ) -> Self {
        let new_cache = AdamOptimizer::new_cache(learn_rate);
        Self::GDT(
            v,
            GDTTrainContext(new_cache, learn_rate, None, sample_from_pattern),
        )
    }

    fn as_gdt(&self) -> Option<&GenerativeDecoderTransformer> {
        Self::as_gdt(&self)
    }
}
