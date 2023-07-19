use anyhow::Result;
use tracing::info;

use plane::ml::{
    embeddings::{Embedding, TrainBatchConfig},
    gdt::GenerativeDecoderTransformer,
    seq2seq::transformer::CharacterTransformer,
    transformer::solver::{
        source::{DefaultOptimizerCache, DynamicOptimizerFactory},
        AdamOptimizer,
    },
    NodeValue,
};

use crate::training;

pub trait MLModel:
    From<Embedding> + From<CharacterTransformer> + From<GenerativeDecoderTransformer>
{
    fn snapshot(&self) -> Result<String>;
    fn as_embedding(&self) -> Option<&Embedding>;
    fn as_s2s(&self) -> Option<&CharacterTransformer>;
    fn as_gdt(&self) -> Option<&GenerativeDecoderTransformer>;
    fn train<B: Into<TrainBatchConfig>>(
        &mut self,
        phrases: &Vec<Vec<String>>,
        learn_rate: NodeValue,
        batch_size: B,
    ) -> Result<NodeValue>;
    fn generate_sequence_string(&self, token_separator: &str) -> Result<String>;
    fn from_s2s(v: CharacterTransformer, learn_rate: NodeValue) -> Self;
    fn from_gdt(v: GenerativeDecoderTransformer, learn_rate: NodeValue) -> Self;
}

pub enum EmbedModel {
    Embedding(Embedding),
    S2S(CharacterTransformer, S2STrainContext),
    GDT(GenerativeDecoderTransformer, GDTTrainContext),
}

pub type OptType = DefaultOptimizerCache<DynamicOptimizerFactory<AdamOptimizer>, AdamOptimizer>;

pub struct S2STrainContext(OptType, NodeValue, Option<String>);
pub struct GDTTrainContext(OptType, NodeValue, Option<String>);

impl EmbedModel {
    fn as_embedding(&self) -> Option<&Embedding> {
        if let Self::Embedding(v) = self {
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

impl From<CharacterTransformer> for EmbedModel {
    fn from(v: CharacterTransformer) -> Self {
        Self::from_s2s(v, 0.001)
    }
}

impl From<GenerativeDecoderTransformer> for EmbedModel {
    fn from(v: GenerativeDecoderTransformer) -> Self {
        Self::from_gdt(v, 0.001)
    }
}

impl From<Embedding> for EmbedModel {
    fn from(v: Embedding) -> Self {
        Self::Embedding(v)
    }
}

impl MLModel for EmbedModel {
    fn train<B: Into<TrainBatchConfig>>(
        &mut self,
        phrases: &Vec<Vec<String>>,
        learn_rate: NodeValue,
        batch_size: B,
    ) -> Result<NodeValue> {
        match self {
            EmbedModel::Embedding(model) => model.train(phrases, learn_rate, batch_size),
            EmbedModel::GDT(model, GDTTrainContext(opt, lr, train_corpus)) => {
                if *lr != learn_rate {
                    opt.for_each_mut(|o| AdamOptimizer::set_eta(o, learn_rate));
                    info!(
                        "AdamOptimizer has set learn_rate (eta) from {} to {}",
                        lr, learn_rate
                    );
                    *lr = learn_rate;
                }
                let char_mode = model.vocab().token_type().is_char();
                let train_corpus = &*train_corpus.get_or_insert_with(move || {
                    training::TrainerState::<()>::join_phrases(phrases, Some(char_mode))
                });
                model.train(train_corpus, opt, batch_size)
            }
            EmbedModel::S2S(model, S2STrainContext(opt, lr, train_corpus)) => {
                if *lr != learn_rate {
                    opt.for_each_mut(|o| AdamOptimizer::set_eta(o, learn_rate));
                    info!(
                        "AdamOptimizer has set learn_rate (eta) from {} to {}",
                        lr, learn_rate
                    );
                    *lr = learn_rate;
                }
                let train_corpus = &*train_corpus.get_or_insert_with(move || {
                    training::TrainerState::<()>::join_phrases(phrases, Some(true))
                });
                model.train(train_corpus, opt, batch_size)
            }
        }
    }

    fn snapshot(&self) -> Result<String> {
        match self {
            EmbedModel::Embedding(model) => model.snapshot(),
            EmbedModel::GDT(model, _) => model.snapshot(),
            EmbedModel::S2S(model, _) => model.snapshot(),
        }
    }

    fn generate_sequence_string(&self, token_separator: &str) -> Result<String> {
        match self {
            EmbedModel::Embedding(model) => Ok(model.generate_sequence_string(token_separator)),
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

    fn from_s2s(v: CharacterTransformer, learn_rate: NodeValue) -> Self {
        let new_cache = AdamOptimizer::new_cache(learn_rate);
        Self::S2S(v, S2STrainContext(new_cache, learn_rate, None))
    }

    fn from_gdt(v: GenerativeDecoderTransformer, learn_rate: NodeValue) -> Self {
        let new_cache = AdamOptimizer::new_cache(learn_rate);
        Self::GDT(v, GDTTrainContext(new_cache, learn_rate, None))
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
