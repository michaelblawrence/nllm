use anyhow::Result;
use tracing::info;

use plane::ml::{
    embeddings::{Embedding, TrainBatchConfig},
    gdt::{GenerativeDecoderTransformer, TrainContext},
    seq2seq::transformer::CharacterTransformer,
    transformer::solver::{
        source::{DefaultOptimizerCache, DynamicOptimizerFactory},
        AdamOptimizer,
    },
    NodeValue,
};

use crate::training;

pub trait MLModel {
    fn snapshot(&self) -> Result<String>;
    fn as_embedding(&self) -> Option<&Embedding>;
    fn as_s2s(&self) -> Option<&CharacterTransformer>;
    fn as_gdt(&self) -> Option<&GenerativeDecoderTransformer>;
    fn train<B: Into<TrainBatchConfig>>(
        &mut self,
        phrases: &Vec<Vec<String>>,
        batch_size: B,
    ) -> Result<NodeValue>;
    fn set_learn_rate(&mut self, learn_rate: NodeValue);
    fn generate_sequence_string(&self, token_separator: &str) -> Result<String>;
    fn from_embedding(v: Embedding, learn_rate: NodeValue) -> Self;
    fn from_s2s(v: CharacterTransformer, learn_rate: NodeValue) -> Self;
    fn from_gdt(
        v: GenerativeDecoderTransformer,
        learn_rate: NodeValue,
        sample_from_pattern: Option<String>,
    ) -> Self;
}

pub enum EmbedModel {
    Embedding(Embedding, EmbeddingTrainContext),
    S2S(CharacterTransformer, S2STrainContext),
    GDT(GenerativeDecoderTransformer, GDTTrainContext),
}

pub type OptType = DefaultOptimizerCache<DynamicOptimizerFactory<AdamOptimizer>, AdamOptimizer>;

pub struct EmbeddingTrainContext(NodeValue);
pub struct S2STrainContext(OptType, NodeValue, Option<String>);
pub struct GDTTrainContext(OptType, NodeValue, Option<String>, Option<String>);

impl EmbedModel {
    fn as_embedding(&self) -> Option<&Embedding> {
        if let Self::Embedding(v, _) = self {
            Some(v)
        } else {
            None
        }
    }

    fn as_s2s(&self) -> Option<&CharacterTransformer> {
        if let Self::S2S(v, _) = self {
            Some(v)
        } else {
            None
        }
    }

    fn as_gdt(&self) -> Option<&GenerativeDecoderTransformer> {
        if let Self::GDT(v, _) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn invalidate_corpus_cache(&mut self) {
        if let Self::S2S(_, S2STrainContext(_, _, train_corpus)) = self {
            train_corpus.take();
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
            EmbedModel::Embedding(model, EmbeddingTrainContext(learn_rate)) => {
                model.train(phrases, *learn_rate, batch_size)
            }
            EmbedModel::GDT(model, GDTTrainContext(opt, _, train_corpus, sample_from_pattern)) => {
                let char_mode = model.vocab().token_type().is_char();
                let train_corpus = &*train_corpus.get_or_insert_with(move || {
                    training::TrainerState::<()>::join_phrases(phrases, Some(char_mode))
                });
                let train_context = TrainContext::new(batch_size, sample_from_pattern.as_deref());
                model.train(train_corpus, opt, &train_context)
            }
            EmbedModel::S2S(model, S2STrainContext(opt, _, train_corpus)) => {
                let train_corpus = &*train_corpus.get_or_insert_with(move || {
                    training::TrainerState::<()>::join_phrases(phrases, Some(true))
                });
                model.train(train_corpus, opt, batch_size)
            }
        }
    }

    fn set_learn_rate(&mut self, learn_rate: NodeValue) {
        let next_learn_rate = learn_rate;
        match self {
            EmbedModel::Embedding(_, EmbeddingTrainContext(learn_rate)) => {
                *learn_rate = next_learn_rate
            }
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
            EmbedModel::S2S(_, S2STrainContext(opt, learn_rate, _)) => {
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
            EmbedModel::Embedding(model, _) => model.snapshot(),
            EmbedModel::GDT(model, _) => model.snapshot(),
            EmbedModel::S2S(model, _) => model.snapshot(),
        }
    }

    fn generate_sequence_string(&self, token_separator: &str) -> Result<String> {
        match self {
            EmbedModel::Embedding(model, _) => Ok(model.generate_sequence_string(token_separator)),
            // EmbedModel::S2S(model, _) => model.generate_sequence_string(token_separator, None),
            EmbedModel::GDT(model, _) => {
                model.generate_sequence_string(None)
                // model.generate_sequence_string(Some('\n'))
            }
            EmbedModel::S2S(model, _) => {
                Ok(model.generate_sequence_string(token_separator, Some('\n')))
            }
        }
    }

    fn from_embedding(v: Embedding, learn_rate: NodeValue) -> Self {
        Self::Embedding(v, EmbeddingTrainContext(learn_rate))
    }

    fn from_s2s(v: CharacterTransformer, learn_rate: NodeValue) -> Self {
        let new_cache = AdamOptimizer::new_cache(learn_rate);
        Self::S2S(v, S2STrainContext(new_cache, learn_rate, None))
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

    fn as_embedding(&self) -> Option<&Embedding> {
        Self::as_embedding(&self)
    }

    fn as_s2s(&self) -> Option<&CharacterTransformer> {
        Self::as_s2s(&self)
    }

    fn as_gdt(&self) -> Option<&GenerativeDecoderTransformer> {
        Self::as_gdt(&self)
    }
}
