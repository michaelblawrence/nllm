use std::{
    collections::{HashMap, VecDeque},
    ops::Deref,
};

use anyhow::{Context, Result};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::ml::{
    embeddings::TrainBatchConfig,
    transformer::{
        decoder::{builder::DecoderBuilder, Decoder},
        linear::Linear,
        solver::source::OptimizerSource,
    },
    LayerValues, NetworkActivationMode, NodeValue, RngStrategy, SamplingRng,
};

pub mod token {
    pub use token::*;
    pub use vocab::*;

    mod token {
        use std::fmt::Display;

        use anyhow::{Context, Result};
        use serde::{Deserialize, Serialize};

        #[derive(Debug, Clone, Eq, Serialize, Deserialize)]
        pub struct Token {
            inner: GDTToken,
            id: Option<usize>,
            user_control_token: bool,
        }

        impl std::hash::Hash for Token {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.inner.hash(state);
            }
        }

        impl PartialOrd for Token {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                self.inner.partial_cmp(&other.inner)
            }
        }

        impl PartialEq for Token {
            fn eq(&self, other: &Self) -> bool {
                self.inner == other.inner
            }
        }

        impl Display for Token {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match &self.inner {
                    GDTToken::Word(x) => x.fmt(f),
                    GDTToken::Subword(x) => x.fmt(f),
                    GDTToken::Char(x) => x.fmt(f),
                }
            }
        }

        impl From<GDTToken> for Token {
            fn from(value: GDTToken) -> Self {
                Self {
                    inner: value,
                    id: None,
                    user_control_token: false,
                }
            }
        }

        impl Token {
            pub fn word(inner: &str) -> Self {
                GDTToken::Word(inner.to_string()).into()
            }
            pub fn subword(inner: &str) -> Self {
                GDTToken::Subword(inner.to_string()).into()
            }
            pub fn char(inner: char) -> Self {
                GDTToken::Char(inner).into()
            }
            pub fn try_char(inner: &str) -> Result<Self> {
                let mut chars = inner.chars();
                let inner = chars.next().context("missing char")?;
                chars
                    .next()
                    .ok_or(())
                    .err()
                    .context("expected single char")?;

                Ok(GDTToken::Char(inner).into())
            }
            pub fn with_id(self, id: usize) -> Self {
                Self {
                    id: Some(id),
                    ..self
                }
            }
            pub fn as_control_token(self) -> Self {
                Self {
                    user_control_token: true,
                    ..self
                }
            }
            pub fn id(&self) -> Option<usize> {
                self.id
            }
            pub fn len(&self) -> usize {
                match &self.inner {
                    GDTToken::Word(x) => x.chars().count(),
                    GDTToken::Subword(x) => x.chars().count(),
                    GDTToken::Char(_) => 1,
                }
            }
            pub fn appendable(&self) -> String {
                match &self.inner {
                    GDTToken::Word(x) => " ".to_string() + &x,
                    GDTToken::Subword(x) => x.to_string(),
                    GDTToken::Char(x) => x.to_string(),
                }
            }

            pub fn is_whitespace_or_control(&self) -> bool {
                if let Some(0) = self.id {
                    return true;
                }
                if self.user_control_token {
                    return true;
                }
                match &self.inner {
                    GDTToken::Word(x) => x.chars().all(|x| x.is_whitespace()),
                    GDTToken::Subword(x) => x.chars().all(|x| x.is_whitespace()),
                    GDTToken::Char(x) => x.is_whitespace(),
                }
            }

            pub fn inner(&self) -> &GDTToken {
                &self.inner
            }
        }

        #[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Serialize, Deserialize)]
        pub enum GDTToken {
            Word(String),
            Subword(String),
            Char(char),
        }

        impl Display for GDTToken {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    GDTToken::Word(x) => write!(f, "{}", x),
                    GDTToken::Subword(x) => write!(f, "{}", x),
                    GDTToken::Char(x) => write!(f, "{}", x),
                }
            }
        }

        #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
        pub enum VocabTokenType {
            Word,
            Char,
            BPE,
        }

        impl VocabTokenType {
            /// Returns `true` if the vocab token type is [`Char`].
            ///
            /// [`Char`]: VocabTokenType::Char
            #[must_use]
            pub fn is_char(&self) -> bool {
                matches!(self, Self::Char)
            }
        }
    }

    mod vocab {
        use std::{collections::HashMap, sync::Arc};

        use anyhow::{anyhow, Context, Result};
        use itertools::{Either, Itertools};
        use serde::{Deserialize, Serialize};

        use crate::ml::RngStrategy;

        pub use builder::VocabBuilder;

        use super::{GDTToken, Token, VocabTokenType};

        #[derive(Debug, Clone, Serialize, Deserialize)]
        #[serde(from = "Vocabulary")]
        pub struct Vocab {
            #[serde(flatten)]
            value: Vocabulary,

            #[serde(skip)]
            encoder: Option<Arc<HashMap<Token, usize>>>,
            #[serde(skip)]
            bpe_encoder: Option<Arc<[(usize, HashMap<Token, usize>)]>>,
        }

        impl From<Vocabulary> for Vocab {
            fn from(value: Vocabulary) -> Self {
                let encoder = Self::build_encoder_dictionary(
                    &value.inner,
                    value.fallback.as_deref(),
                    value.index_offset,
                );
                let bpe_encoder = Self::build_bpe_encoder_dictionary(&encoder);

                Self {
                    value,
                    encoder: Some(Arc::new(encoder)),
                    bpe_encoder: Some(bpe_encoder.into()),
                }
            }
        }

        impl std::ops::Deref for Vocab {
            type Target = Vocabulary;

            fn deref(&self) -> &Self::Target {
                &self.value
            }
        }

        impl std::ops::DerefMut for Vocab {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.value
            }
        }

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct Vocabulary {
            inner: Vec<Token>,
            token_type: VocabTokenType,
            fallback: Option<Box<Vocab>>,
            index_offset: usize,
            seq_start_token: Option<GDTToken>,
        }

        impl Vocab {
            pub fn new_builder(which: VocabTokenType) -> VocabBuilder {
                VocabBuilder::new(which)
            }

            // remove
            pub fn dictionary(&self) -> HashMap<Token, usize> {
                self.encoder().as_ref().clone()
            }

            // rename, this is infallible
            pub fn try_dictionary(&self) -> Result<HashMap<Token, usize>> {
                let dict = self.encoder().as_ref().clone();

                Ok(dict)
            }

            pub fn token_type(&self) -> VocabTokenType {
                self.token_type
            }

            pub fn len(&self) -> usize {
                self.inner.len() + self.fallback.as_ref().map_or(0, |x| x.len())
            }

            pub fn control_token(&self) -> Token {
                self.inner.first().cloned().unwrap()
            }

            pub fn encode(&self, input: &str) -> Result<Vec<Token>> {
                self.encode_truncated(input, None)
            }

            pub fn encode_truncated(
                &self,
                input: &str,
                max_tokens: Option<usize>,
            ) -> Result<Vec<Token>> {
                let encoder = self.encoder();
                let start_token: Option<&Token> = self
                    .seq_start_token
                    .clone()
                    .and_then(|x| Some(encoder.get_key_value(&x.into())?.0));

                let tokens: Result<Vec<Vec<Token>>, String> = match self.token_type {
                    VocabTokenType::Word => input
                        .split_whitespace()
                        .map(|x| {
                            let token = GDTToken::Word(x.to_string()).into();
                            encoder
                                .get_key_value(&token)
                                .map(|(x, _)| vec![x.clone()])
                                .or_else(|| {
                                    Self::encode_word_token_iter(x)
                                        .map(|token| {
                                            encoder
                                                .get_key_value(&token.into())
                                                .map(|(x, _)| x.clone())
                                        })
                                        .collect::<Option<_>>()
                                })
                                .ok_or_else(|| token.to_string())
                                .or_else(|token| match &self.fallback {
                                    Some(fallback) => fallback.encode(&token).map_err(|_| token),
                                    None => Err(token),
                                })
                        })
                        .take(max_tokens.unwrap_or(usize::MAX))
                        .collect(),

                    VocabTokenType::BPE => {
                        let by_char_count = self.bpe_encoder(&encoder);
                        let mut encode_str = input.to_string();
                        let mut encoded_tokens = vec![];

                        while encode_str.len() > 0
                            && max_tokens
                                .clone()
                                .map_or(true, |max| encoded_tokens.len() < max)
                        {
                            match try_consume_token(&mut encode_str, &by_char_count) {
                                Some(token) => {
                                    encoded_tokens.push(Some(token));
                                }
                                None => {
                                    let discarded = encode_str.remove(0);
                                    match self.fallback.as_ref().and_then(|fallback| {
                                        fallback.encode(&discarded.to_string()).ok()
                                    }) {
                                        Some(tokens) => tokens.into_iter().for_each(|token| {
                                            encoded_tokens.push(Some(token.inner().clone()))
                                        }),
                                        None => {
                                            println!("unknown token: {:?}", discarded);
                                            encoded_tokens.push(None);
                                        }
                                    }
                                }
                            }
                        }

                        fn try_consume_token(
                            encode_input: &mut String,
                            by_char_count: &[(usize, HashMap<Token, usize>)],
                        ) -> Option<GDTToken> {
                            for (token_len, tokens) in by_char_count.iter().rev() {
                                let truncated: String =
                                    encode_input.chars().take(*token_len).collect();
                                if truncated.chars().count() != *token_len {
                                    continue;
                                }
                                let gdt_token = GDTToken::Subword(truncated);
                                if let Some((token, _)) = tokens.get_key_value(&gdt_token.into()) {
                                    *encode_input = encode_input.chars().skip(*token_len).collect();
                                    return Some(token.inner().clone());
                                }
                            }
                            None
                        }

                        encoded_tokens
                            .into_iter()
                            .map(|x| {
                                x.as_ref()
                                    .and_then(|token| encoder.get_key_value(&token.clone().into()))
                                    .ok_or_else(|| {
                                        x.map_or_else(|| "<unknown>".to_string(), |x| x.to_string())
                                    })
                                    .map(|(x, _)| vec![x.clone()])
                            })
                            .collect()
                    }

                    VocabTokenType::Char => input
                        .chars()
                        .map(|x| {
                            let token = GDTToken::Char(x).into();
                            encoder
                                .get_key_value(&token)
                                .ok_or_else(|| x.to_string())
                                .map(|(x, _)| vec![x.clone()])
                        })
                        .collect(),
                };

                match (tokens, start_token) {
                    (Ok(tokens), None) => Ok(tokens.into_iter().flatten().collect()),
                    (Ok(tokens), Some(start_token)) => {
                        let start_token = [start_token.clone()].into_iter();
                        Ok(start_token.chain(tokens.into_iter().flatten()).collect())
                    }
                    (Err(token), _) => Err(anyhow!(
                        "missing token = '{token}' (source text = '{input}')"
                    )),
                }
            }

            pub fn decode(&self, input: &[Token]) -> Result<String> {
                Ok(input
                    .iter()
                    .enumerate()
                    .map(|(i, x)| match (i, &x.inner()) {
                        (0, GDTToken::Word(token)) => token.to_string(),
                        (_, GDTToken::Word(token)) => " ".to_string() + token,
                        (_, GDTToken::Subword(token)) => token.to_string(),
                        (_, GDTToken::Char(token)) => token.to_string(),
                    })
                    .join(""))
            }

            pub fn token_encode(&self, c: &Token) -> Result<usize> {
                let encoder = self.encoder();

                encoder.get(c).copied().with_context(|| {
                    format!("failed to resolve vocab from dict: unknown token '{c:?}'")
                })
            }

            pub fn token_decode(&self, encoded: usize) -> Option<Token> {
                match self.inner.get(encoded - self.index_offset).cloned() {
                    Some(decoded) => Some(decoded),
                    None => match &self.fallback {
                        Some(fallback) => fallback.token_decode(encoded),
                        None => None,
                    },
                }
            }

            pub fn sample_batch(
                &self,
                corpus: &str,
                token_len: usize,
                sequence_count: usize,
                sample_from_pattern: Option<&str>,
                rng: &RngStrategy,
            ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
                let mut batch_samples: Vec<Vec<Token>> = vec![];

                match self.token_type {
                    VocabTokenType::Word => {
                        let corpus_len = corpus.chars().count();
                        for _ in 0..sequence_count {
                            let start_idx = sample_next_word_boundary_idx(corpus, corpus_len, rng);
                            let start_offset = sample_from_pattern
                                .and_then(|pattern| corpus[start_idx..].find(pattern))
                                .unwrap_or_default();
                            let indexed_corpus = &corpus[start_idx + start_offset..];
                            let chars: String = indexed_corpus
                                .lines()
                                .scan(token_len + 1, |budget, x| {
                                    let peekable = &mut x
                                        .split_whitespace()
                                        .take_while(|_| {
                                            let can_decrement = *budget > 0;
                                            if can_decrement {
                                                *budget -= 1;
                                            }
                                            can_decrement
                                        })
                                        .peekable();
                                    peekable.peek()?;
                                    Some(peekable.join(" "))
                                })
                                .join("\n");

                            let mut sequence =
                                self.encode_truncated(&chars, Some(token_len + 1))?;
                            let sequence = if sequence.len() > token_len + 1 {
                                if sample_from_pattern.is_none() {
                                    let max_start_idx =
                                        sequence.len().saturating_sub(token_len + 1);
                                    let start_idx = rng.rand_range(0, max_start_idx);
                                    let end_idx = start_idx + token_len + 1;

                                    sequence[start_idx..end_idx].to_vec()
                                } else {
                                    sequence.truncate(token_len + 1);
                                    sequence
                                }
                            } else if sequence.len() < token_len + 1 {
                                let control_token = self.control_token();
                                let pad_count = (token_len + 1).saturating_sub(sequence.len());
                                let mut padded_sequence = sequence;
                                for _ in 0..pad_count {
                                    padded_sequence.push(control_token.clone());
                                }

                                padded_sequence
                            } else {
                                sequence
                            };
                            batch_samples.push(sequence);
                        }
                    }
                    VocabTokenType::BPE => {
                        let corpus_len = corpus.chars().count();
                        for _ in 0..sequence_count {
                            let start_idx = sample_next_word_boundary_idx(corpus, corpus_len, rng);
                            let start_offset = sample_from_pattern
                                .and_then(|pattern| corpus[start_idx..].find(pattern))
                                .unwrap_or_default();
                            let indexed_corpus = &corpus[start_idx + start_offset..];
                            let chars: String = indexed_corpus
                                .lines()
                                .map(|x| x.split_whitespace().take(token_len + 1).join(" "))
                                .join("\n");

                            let sequence = self.encode_truncated(&chars, Some(token_len + 1))?; // needed?
                            let sequence = if sequence.len() > token_len + 1 {
                                let max_start_idx = sequence.len().saturating_sub(token_len + 1);
                                let start_idx = rng.rand_range(0, max_start_idx);
                                let end_idx = start_idx + token_len + 1;

                                sequence[start_idx..end_idx].to_vec()
                            } else if sequence.len() < token_len + 1 {
                                let control_token = self.control_token();
                                let pad_count = (token_len + 1).saturating_sub(sequence.len());
                                let mut padded_sequence = sequence;
                                for _ in 0..pad_count {
                                    padded_sequence.push(control_token.clone());
                                }

                                padded_sequence
                            } else {
                                sequence
                            };
                            batch_samples.push(sequence);
                        }
                    }
                    VocabTokenType::Char => {
                        let corpus_len = corpus.chars().count();
                        for _ in 0..sequence_count {
                            let exclusive_max = corpus_len - token_len;
                            let start_index = rng.rand_range(0, exclusive_max);
                            let substring = {
                                let mut chars = corpus.chars();
                                // TODO replace with `chars.advance_by(start_index)` once iter api is stable
                                chars
                                    .nth(start_index.saturating_sub(1))
                                    .expect("start index should be in range");
                                chars.as_str()
                            };
                            let position = sample_from_pattern
                                .and_then(|pattern| substring.find(pattern))
                                .unwrap_or_default();

                            let chars: String = substring
                                .chars()
                                .skip(position)
                                .take(token_len + 1)
                                .collect();
                            let sequence = self.encode(&chars)?;
                            batch_samples.push(sequence);
                        }
                    }
                }

                let samples = batch_samples
                    .into_iter()
                    .filter_map(|sequence| {
                        let (context, targets) =
                            (&sequence[0..token_len], &sequence[1..token_len + 1]);
                        Some((
                            self.encode_input_sequence(&context, token_len).ok()?,
                            self.encode_input_sequence(&targets, token_len).ok()?,
                        ))
                    })
                    .collect_vec();

                Ok(samples)
            }

            pub fn encode_input_sequence(
                &self,
                input_sequence: &[Token],
                context_length: usize,
            ) -> Result<Vec<usize>> {
                let offset = input_sequence.len().saturating_sub(context_length);
                input_sequence
                    .iter()
                    .skip(offset)
                    .take(context_length)
                    .map(|c| self.token_encode(c))
                    .collect()
            }

            fn encoder(&self) -> Arc<HashMap<Token, usize>> {
                self.encoder.clone().unwrap_or_else(|| {
                    Arc::new(Self::build_encoder_dictionary(
                        &self.inner,
                        self.fallback.as_deref(),
                        self.index_offset,
                    ))
                })
            }

            fn bpe_encoder(
                &self,
                encoder: &HashMap<Token, usize>,
            ) -> Arc<[(usize, HashMap<Token, usize>)]> {
                self.bpe_encoder
                    .clone()
                    .unwrap_or_else(|| Self::build_bpe_encoder_dictionary(encoder).into())
            }

            fn build_encoder_dictionary(
                vocab: &[Token],
                fallback: Option<&Vocab>,
                offset: usize,
            ) -> HashMap<Token, usize> {
                let dict: HashMap<Token, usize> = vocab
                    .iter()
                    .enumerate()
                    .map(|(idx, x)| (x.clone().with_id(idx + offset), idx + offset))
                    .chain(fallback.iter().flat_map(|x| x.dictionary()))
                    .collect();
                dict
            }

            fn build_bpe_encoder_dictionary(
                encoder: &HashMap<Token, usize>,
            ) -> Vec<(usize, HashMap<Token, usize>)> {
                let mut counts: Vec<(usize, HashMap<Token, usize>)> = encoder
                    .iter()
                    .group_by(|(x, _)| x.len())
                    .into_iter()
                    .map(|(k, v)| (k, v.map(|(c, x)| (c.clone(), *x)).collect()))
                    .collect_vec();
                counts.sort_by_key(|x| x.0);
                counts
            }

            fn strip_not_alphabetic(x: &str) -> (&str, Option<&str>, Option<&str>) {
                fn strip_prefix(x: &str) -> (Option<&str>, &str) {
                    if x.is_empty() {
                        return (None, x);
                    }

                    x.char_indices()
                        .find(|&(_, c)| char::is_alphabetic(c))
                        .map_or_else(
                            || (Some(x), &x[..0]),
                            |(pos, _)| (Some(&x[..pos]).filter(|x| !x.is_empty()), &x[pos..]),
                        )
                }
                fn strip_suffix(x: &str) -> (&str, Option<&str>) {
                    if x.is_empty() {
                        return (x, None);
                    }
                    x.char_indices()
                        .rev()
                        .find(|&(_, c)| char::is_alphabetic(c))
                        .map_or_else(
                            || (&x[..0], Some(x)),
                            |(pos, c)| {
                                let pos = pos + c.len_utf8();
                                (&x[..pos], Some(&x[pos..]).filter(|x| !x.is_empty()))
                            },
                        )
                }

                if let (x, Some(suffix)) = strip_suffix(x) {
                    if let (Some(prefix), x) = strip_prefix(x) {
                        (x, Some(prefix), Some(suffix))
                    } else {
                        (x, None, Some(suffix))
                    }
                } else if let (Some(prefix), x) = strip_prefix(x) {
                    (x, Some(prefix), None)
                } else {
                    (x, None, None)
                }
            }

            fn encode_word_token_iter(corpus: &str) -> impl Iterator<Item = GDTToken> + '_ {
                let (word, tokens): (Vec<_>, Vec<_>) =
                    corpus.split_whitespace().partition_map(|x| {
                        match Self::strip_not_alphabetic(x) {
                            (x, None, None) if !x.is_empty() => {
                                Either::Left(GDTToken::Word(x.to_string()))
                            }
                            (x, prefix, suffix) => Either::Right(
                                prefix
                                    .into_iter()
                                    .flat_map(|x| x.chars())
                                    .map(|x| GDTToken::Char(x))
                                    .chain(
                                        Some(x)
                                            .filter(|x| !x.is_empty())
                                            .map(|x| [GDTToken::Word(x.to_string())])
                                            .into_iter()
                                            .flatten(),
                                    )
                                    .chain(
                                        suffix
                                            .into_iter()
                                            .flat_map(|x| x.chars())
                                            .map(|x| GDTToken::Char(x)),
                                    ),
                            ),
                        }
                    });

                word.into_iter().chain(tokens.into_iter().flatten())
            }
        }

        fn sample_next_word_boundary_idx(
            corpus: &str,
            corpus_len: usize,
            rng: &RngStrategy,
        ) -> usize {
            let mut rand_index = None;
            while rand_index.map_or(true, |x| !corpus.is_char_boundary(x)) {
                rand_index = Some(rng.rand_range(0, corpus_len));
            }
            let rand_index = rand_index.unwrap();
            let byte_offset: usize = corpus[0..rand_index]
                .chars()
                .rev()
                .take_while(|x| !x.is_whitespace())
                .map(|x| x.len_utf8())
                .sum();
            rand_index - byte_offset
        }

        mod builder {
            use std::borrow::Cow;

            use anyhow::{anyhow, Result};
            use itertools::Itertools;

            use crate::ml::{bpe::BytePairEncoder, gdt::token::hash};

            use super::{GDTToken, Token, Vocab, VocabTokenType};

            #[derive(Debug, Clone)]
            pub struct VocabBpeConfig {
                num_merges: usize,
                iters: usize,
            }

            impl Default for VocabBpeConfig {
                fn default() -> Self {
                    Self {
                        num_merges: 1000,
                        iters: 5,
                    }
                }
            }

            #[derive(Debug, Clone)]
            pub struct VocabBuilder {
                token_type: VocabTokenType,
                control_token: GDTToken,
                tokens: hash::DeterministicHashMap<GDTToken, usize>,
                manual_tokens: hash::DeterministicHashMap<GDTToken, usize>,
                max_vocab_size: Option<usize>,
                bpe_config: Option<VocabBpeConfig>,
                fallback: Option<Box<Vocab>>,
            }

            impl VocabBuilder {
                pub fn new(which: VocabTokenType) -> Self {
                    Self {
                        token_type: which,
                        control_token: match which {
                            VocabTokenType::Word => GDTToken::Word("<CTRL>".to_string()),
                            VocabTokenType::BPE => GDTToken::Subword("<CTRL>".to_string()),
                            VocabTokenType::Char => GDTToken::Char('\x00'),
                        },
                        tokens: Default::default(),
                        manual_tokens: Default::default(),
                        bpe_config: Default::default(),
                        max_vocab_size: None,
                        fallback: None,
                    }
                }
                pub fn from_corpus(mut self, corpus: &str) -> Self {
                    match self.token_type {
                        VocabTokenType::Word => Vocab::encode_word_token_iter(corpus)
                            .for_each(|token| self.push_token(token)),
                        VocabTokenType::BPE => {
                            let VocabBpeConfig { num_merges, iters } =
                                self.bpe_config.clone().unwrap_or_default();

                            let vocab = BytePairEncoder::build(
                                corpus,
                                num_merges,
                                iters,
                                self.max_vocab_size.unwrap_or(1000),
                                false,
                            );

                            vocab
                                .into_iter()
                                .map(|x| GDTToken::Subword(x))
                                .for_each(|token| self.push_token(token));
                        }
                        VocabTokenType::Char => corpus
                            .chars()
                            .map(|x| GDTToken::Char(x))
                            .for_each(|token| self.push_token(token)),
                    }
                    self
                }
                pub fn from_gdt(mut self, tokens: &[GDTToken]) -> Self {
                    for token in tokens {
                        self.push_token(token.clone())
                    }
                    self
                }
                pub fn with_char_fallback(mut self) -> Self {
                    match self.token_type {
                        VocabTokenType::Word | VocabTokenType::BPE => {
                            let space_token = GDTToken::Char(' ');
                            let newline_token = GDTToken::Char('\n');

                            let mut tokens = ('a'..='z')
                                .map(|x| GDTToken::Char(x))
                                .chain([space_token.clone()])
                                .chain([newline_token.clone()])
                                .collect_vec();

                            let mut subword_chars: hash::DeterministicHashSet<char> = self
                                .tokens
                                .keys()
                                .filter_map(|x| match x {
                                    GDTToken::Word(x) => Some(Cow::from(x)),
                                    GDTToken::Subword(x) => Some(Cow::from(x)),
                                    GDTToken::Char(x) => Some(Cow::from(x.to_string())),
                                })
                                .collect_vec()
                                .iter()
                                .flat_map(|x| x.chars())
                                .collect();

                            let char_tokens = self
                                .tokens
                                .keys()
                                .filter(|x| matches!(x, GDTToken::Char(_)))
                                .cloned()
                                .collect_vec();

                            char_tokens.iter().for_each(|x| _ = self.tokens.remove(x));

                            tokens.iter().for_each(|x| match x {
                                GDTToken::Char(x) => {
                                    subword_chars.remove(x);
                                }
                                _ => (),
                            });

                            tokens.extend(subword_chars.into_iter().map(|x| GDTToken::Char(x)));

                            let mut fallback = Self::new(VocabTokenType::Char)
                                .from_gdt(&tokens)
                                .build()
                                .unwrap();

                            fallback.seq_start_token = Some(space_token);
                            self.fallback = Some(Box::new(fallback));
                        }
                        VocabTokenType::Char => {
                            self.fallback = None;
                        }
                    }
                    self
                }
                pub fn with_max_vocab_size(mut self, max_vocab_size: usize) -> Self {
                    self.max_vocab_size = Some(max_vocab_size);
                    self
                }
                pub fn add_word_token_literal(mut self, token_literal: &str) -> Self {
                    let token = GDTToken::Word(token_literal.to_string());
                    self.manual_tokens.insert(token, 0);
                    self
                }
                fn push_token(&mut self, token: GDTToken) {
                    self.tokens
                        .entry(token)
                        .and_modify(|x| *x += 1)
                        .or_insert(1);
                }
                pub fn build(self) -> Result<Vocab> {
                    let control_token: Token = self.control_token.into();
                    let capacity = self.max_vocab_size.unwrap_or(self.tokens.len());
                    let mut vocab = {
                        let mut init_vocab = Vec::with_capacity(capacity);
                        init_vocab.push(control_token.with_id(init_vocab.len()));
                        for (token, _) in self.manual_tokens {
                            let value: Token = token.into();
                            init_vocab.push(value.with_id(init_vocab.len()));
                        }
                        init_vocab
                    };

                    let vocab_size = self.max_vocab_size.map_or(self.tokens.len(), |x| {
                        x - self.fallback.as_ref().map_or(0, |x| x.len()) - vocab.len()
                    });
                    let offset = vocab.len();
                    for (i, (token, _)) in self
                        .tokens
                        .into_iter()
                        .sorted_by_key(|x| -(x.1 as i64))
                        .take(vocab_size)
                        .enumerate()
                    {
                        match (&token, self.token_type) {
                            (&GDTToken::Word(_), VocabTokenType::Word) => {
                                let value: Token = token.into();
                                vocab.push(value.with_id(i + offset))
                            }
                            (&GDTToken::Subword(_), VocabTokenType::BPE) => {
                                let value: Token = token.into();
                                vocab.push(value.with_id(i + offset))
                            }
                            (&GDTToken::Char(_), VocabTokenType::Char) => {
                                let value: Token = token.into();
                                vocab.push(value.with_id(i + offset))
                            }
                            _ => Err(anyhow!("Invalid token/token type pair"))?,
                        }
                    }
                    let fallback = match self.fallback {
                        Some(mut fallback) => {
                            let index_offset = vocab.len();
                            fallback.encoder.take();
                            fallback.index_offset = index_offset;
                            for (i, token) in fallback.inner.iter_mut().enumerate() {
                                *token = token.clone().with_id(index_offset + i);
                            }
                            Some(fallback)
                        }
                        None => None,
                    };

                    let vocab = super::Vocabulary {
                        inner: vocab,
                        token_type: self.token_type,
                        fallback,
                        index_offset: 0,
                        seq_start_token: None,
                    };

                    let vocab: Vocab = vocab.into();
                    Ok(vocab)
                }
            }
        }
    }

    mod hash {
        use std::collections::{HashMap, HashSet};

        pub type DeterministicHashSet<T> =
            HashSet<T, std::hash::BuildHasherDefault<std::collections::hash_map::DefaultHasher>>;

        pub type DeterministicHashMap<T, V> =
            HashMap<T, V, std::hash::BuildHasherDefault<std::collections::hash_map::DefaultHasher>>;
    }
}

#[derive(Serialize, Deserialize)]
pub struct GenerativeDecoderTransformer {
    network: Decoder,
    vocab: token::Vocab,
    rng: RngStrategy,
    context_length: usize,
}

impl GenerativeDecoderTransformer {
    pub fn from_builder(
        builder: DecoderBuilder,
        vocab: token::Vocab,
        rng: RngStrategy,
    ) -> Result<Self> {
        let context_length = builder.sequence_len();
        let builder = builder.with_target_vocab_size(vocab.len());
        Ok(Self {
            network: builder.build()?,
            vocab,
            rng,
            context_length,
        })
    }

    pub fn snapshot(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn snapshot_pretty(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    #[instrument(level = "info", name = "embed_train", skip_all)]
    pub fn train<T: Deref<Target = O>, O: OptimizerSource>(
        &mut self,
        corpus: &str,
        optimizer: T,
        train_context: &TrainContext,
    ) -> Result<NodeValue> {
        let batches = self.into_batches(corpus, train_context)?;
        let (encoder_output, encoder_mask) = (None, None);

        #[cfg(feature = "threadpool")]
        use rayon::prelude::*;

        let batches_iter = {
            #[cfg(feature = "threadpool")]
            {
                batches.par_iter()
            }
            #[cfg(not(feature = "threadpool"))]
            {
                batches.iter()
            }
        };
        let batches_forward_pass_costs = batches_iter
            .map(|(input_sequence, targets)| {
                let decoder_outputs =
                    self.network
                        .forward_training(&input_sequence, encoder_output, encoder_mask)?;

                let output_gradients =
                    decoder_outputs
                        .rows_iter()
                        .zip_eq(targets)
                        .map(|(decoder_output, target)| {
                            let vocab_len = self.vocab.len();
                            let expected_outputs = (0..vocab_len)
                                .map(move |i| if i == *target { 1.0 } else { 0.0 })
                                .collect();

                            let post_apply_mode = NetworkActivationMode::SoftMaxCrossEntropy;
                            let probabilities = post_apply_mode.apply(&decoder_output.into());

                            let train_error =
                                probabilities.cross_entropy_error(&expected_outputs)?;

                            let output_gradients =
                                probabilities.cross_entropy_error_d(&expected_outputs)?;

                            Ok((output_gradients, train_error.ave()))
                        });

                let (output_gradients, train_error): (Vec<_>, Vec<_>) = output_gradients
                    .collect::<Result<Vec<(LayerValues, NodeValue)>>>()?
                    .into_iter()
                    .unzip();
                let decoder_gradients = Linear::from_values(&output_gradients)?;
                self.network.backward(
                    &input_sequence,
                    encoder_output,
                    encoder_mask,
                    decoder_gradients,
                )?;

                Ok(train_error)
            })
            .collect::<Result<Vec<_>>>()?;

        self.network.apply_gradients(&*optimizer)?;

        let costs: LayerValues = batches_forward_pass_costs.into_iter().flatten().collect();
        let ave_cost = costs.ave();
        Ok(ave_cost)
    }

    #[instrument(level = "info", skip_all)]
    fn into_batches(
        &self,
        corpus: &str,
        ctx: &TrainContext,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        match &ctx.batch_size {
            &TrainBatchConfig::SingleBatch(batch_size) | &TrainBatchConfig::Batches(batch_size) => {
                self.vocab.sample_batch(
                    corpus,
                    self.context_length,
                    batch_size,
                    ctx.sample_from_pattern,
                    &self.rng,
                )
            }
        }
    }

    pub fn sample_probabilities(&self, input_sequence: &[token::Token]) -> Result<LayerValues> {
        let network_input = self.encode_input_sequence(input_sequence)?;
        let output = self.network.forward_inference(&network_input, None, None)?;
        let probabilities = self.compute_probabilities(&output, network_input.len())?;
        Ok(probabilities)
    }

    fn compute_probabilities(
        &self,
        output: &Linear,
        network_input_len: usize,
    ) -> Result<LayerValues> {
        let output_idx = network_input_len
            .saturating_sub(1)
            .min(self.context_length - 1);
        let row: LayerValues = output.rows_iter().nth(output_idx).unwrap().into();
        self.compute_probabilities_row(&row)
    }

    fn compute_probabilities_row(&self, output: &LayerValues) -> Result<LayerValues> {
        let post_apply_mode = NetworkActivationMode::SoftMaxCrossEntropy;
        let probabilities = post_apply_mode.apply(&output);
        Ok(probabilities)
    }

    fn one_hot(&self, idx: usize) -> impl Iterator<Item = NodeValue> {
        let vocab_len = self.vocab.len();
        (0..vocab_len).map(move |i| if i == idx { 1.0 } else { 0.0 })
    }

    pub fn predict_arg_max(
        &self,
        last_tokens: &[token::Token],
    ) -> Result<(token::Token, NodeValue)> {
        let probabilities = self.sample_probabilities(last_tokens)?;
        let arg_max_idx = probabilities
            .position_max()
            .context("failed to perform argmax on probabilities")?;

        let token = self.vocab.token_decode(arg_max_idx);
        let prob = probabilities
            .get(arg_max_idx)
            .copied()
            .expect("arg_max_idx should be within bounds");

        Ok((
            token.context("sampled token should be in vocab dict")?,
            prob,
        ))
    }

    pub fn predict_next(&self, last_tokens: &[token::Token]) -> Result<token::Token> {
        let probabilities = self.sample_probabilities(last_tokens)?;
        let sampled_idx = self.rng.sample_uniform(&probabilities)?;
        let token = self.vocab.token_decode(sampled_idx);

        token.context("sampled token should be in vocab dict")
    }

    pub fn predict_next_with_temperature(
        &self,
        last_tokens: &[token::Token],
        temperature: NodeValue,
    ) -> Result<token::Token> {
        let logits = {
            let mut probabilities = self.sample_probabilities(last_tokens)?;
            let logit_factor = 1.0 / (temperature + 1e-8);
            probabilities
                .iter_mut()
                .for_each(|p| *p = p.ln() * logit_factor);
            probabilities
        };
        let scaled_probabilities = self.compute_probabilities_row(&logits)?;
        let sampled_idx = self.rng.sample_uniform(&scaled_probabilities)?;
        let token = self.vocab.token_decode(sampled_idx);

        token.context("sampled token should be in vocab dict")
    }

    pub fn predict_from(&self, last_word: &str) -> Result<token::Token> {
        let last_tokens = self.vocab.encode(last_word)?;
        self.predict_next(&last_tokens)
    }

    pub fn predict_arg_max_from(&self, last_word: &str) -> Result<token::Token> {
        let input_sequence = self.vocab.encode(last_word)?;
        let probabilities = self.sample_probabilities(&input_sequence)?;
        let arg_max_idx = probabilities
            .position_max()
            .context("failed to perform argmax on probabilities")?;
        let token = self.vocab.token_decode(arg_max_idx);

        token.context("sampled token should be in vocab dict")
    }

    pub fn nll(
        &self,
        context_tokens: &[token::Token],
        expected_next_token: token::Token,
    ) -> Result<NodeValue> {
        let expected_encoded_token = self.vocab.token_encode(&expected_next_token)?;
        let probabilities = self.sample_probabilities(context_tokens)?;
        let log_logits = probabilities
            .get(expected_encoded_token)
            .expect("output should have same count as vocab")
            .ln();

        Ok(-log_logits)
    }

    pub fn nll_each(&self, context_tokens: &[token::Token]) -> Result<Vec<NodeValue>> {
        let expected_encoded_tokens = context_tokens
            .iter()
            .skip(1)
            .map(|token| self.vocab.token_encode(&token))
            .collect::<Result<Vec<_>>>()?;

        let network_input = self.encode_input_sequence(context_tokens)?;
        let output = self.network.forward_inference(&network_input, None, None)?;

        let probabilities = output
            .rows_iter()
            .take(network_input.len() - 1)
            .map(|row| self.compute_probabilities_row(&row.into()))
            .collect::<Result<Vec<_>>>()?;

        let log_logits = probabilities
            .iter()
            .zip_eq(expected_encoded_tokens.iter())
            .map(|(probabilities, expected_encoded_token)| {
                -probabilities
                    .get(*expected_encoded_token)
                    .expect("output should have same count as vocab")
                    .ln()
            })
            .collect::<Vec<_>>();

        Ok(log_logits)
    }

    pub fn encode_input_sequence(&self, input_sequence: &[token::Token]) -> Result<Vec<usize>> {
        let offset = input_sequence.len().saturating_sub(self.context_length);
        input_sequence
            .iter()
            .skip(offset)
            .take(self.context_length)
            .map(|c| self.vocab.token_encode(c))
            .collect()
    }

    pub fn generate_sequence_string(&self, seed_token: Option<token::Token>) -> Result<String> {
        let sequence = match seed_token {
            Some(seed_token) => self.predict_from_iter(&[seed_token]),
            None => self.predict_from_iter(&[]),
        }
        .collect_vec();

        self.vocab.decode(&sequence)
    }

    pub fn input_stride_width(&self) -> usize {
        self.context_length
    }

    pub fn control_vocab(&self) -> token::Token {
        self.vocab.control_token()
    }

    pub fn control_vocab_index(&self) -> usize {
        0
    }

    pub fn predict_from_iter<'a>(
        &'a self,
        seed_tokens: &[token::Token],
    ) -> impl Iterator<Item = token::Token> + 'a {
        self.generate_sequence_iter(
            seed_tokens,
            InferContext {
                sampling: InferSampling::Uniform,
                max_len: 40,
            },
        )
    }

    pub fn generate_sequence_iter<'a>(
        &'a self,
        seed_tokens: &[token::Token],
        context: InferContext,
    ) -> impl Iterator<Item = token::Token> + 'a {
        let mut token_counts = HashMap::new();
        let (curr_token, seed_tokens) = match seed_tokens.split_last() {
            Some((curr_token, seed_tokens)) => (Some(curr_token), seed_tokens),
            None => (None, seed_tokens),
        };

        let mut curr_token = curr_token.cloned();
        let mut tokens_generated = 0;
        let mut context_tokens = VecDeque::from_iter(seed_tokens.iter().cloned());

        let input_stride_width = self.input_stride_width();
        let control_vocab = self.control_vocab();
        let next_token: Box<dyn Fn(&[token::Token]) -> Option<token::Token>> =
            match context.sampling {
                InferSampling::Uniform => {
                    Box::new(|context_tokens| self.predict_next(&context_tokens).ok())
                }
                InferSampling::Temperature(t) => Box::new(move |context_tokens| {
                    self.predict_next_with_temperature(&context_tokens, t).ok()
                }),
            };

        std::iter::from_fn(move || -> Option<token::Token> {
            if curr_token
                .as_ref()
                .map(|x| x == &control_vocab)
                .unwrap_or(false)
                && tokens_generated > 0
            {
                None
            } else if *token_counts
                .get(curr_token.as_ref().unwrap_or(&control_vocab))
                .unwrap_or(&0)
                > context.max_len
            {
                None
            } else {
                match &curr_token {
                    Some(last_token) => {
                        context_tokens.push_back(last_token.clone());
                        let skip_count = !last_token.is_whitespace_or_control();

                        if !skip_count {
                            *token_counts.entry(last_token.clone()).or_insert(0) += 1;
                        }
                    }
                    None => {}
                };

                let context_tokens_slice = context_tokens.iter().cloned().collect::<Vec<_>>();
                let token = next_token(&context_tokens_slice)?;
                curr_token = Some(token);
                tokens_generated += 1;

                while context_tokens.len() > input_stride_width {
                    context_tokens.pop_front();
                }

                curr_token.clone().filter(|token| token != &control_vocab)
            }
        })
    }

    pub fn compute_error(&self, tokens: &[token::Token]) -> Result<NodeValue> {
        let mut errors: Vec<NodeValue> = vec![];

        for sequence_segment in tokens.windows(self.context_length + 1) {
            let input_sequence = &sequence_segment[0..self.context_length];
            let probabilities = self.sample_probabilities(input_sequence)?;

            let target = &sequence_segment[self.context_length];
            let target_token = self.vocab.token_encode(target)?;
            let expected_outputs = self.one_hot(target_token).collect();

            let train_error = probabilities.cross_entropy_error(&expected_outputs)?;
            errors.push(train_error.ave());
        }

        let errors = LayerValues::new(errors);
        Ok(errors.ave())
    }

    pub fn vocab(&self) -> &token::Vocab {
        &self.vocab
    }
}

pub struct InferContext {
    pub sampling: InferSampling,
    pub max_len: usize,
}

pub enum InferSampling {
    Uniform,
    Temperature(NodeValue),
}

pub struct TrainContext<'a> {
    batch_size: TrainBatchConfig,
    sample_from_pattern: Option<&'a str>,
}

impl<'a> TrainContext<'a> {
    pub fn new<B: Into<TrainBatchConfig>>(
        batch_size: B,
        sample_from_pattern: Option<&'a str>,
    ) -> Self {
        Self {
            batch_size: batch_size.into(),
            sample_from_pattern,
        }
    }
}

#[cfg(test)]
mod tests {
    use test_log::test;

    use crate::ml::{gdt::token::Token, layer::LayerInitStrategy, transformer::solver};

    use super::{token::Vocab, *};

    #[test]
    fn transformer_work_build_word_vocab_strips_nonalphabetic() {
        let corpus = "this is an example, we only want word tokens! that means no 'numbers' like: 1, 1234 or emoji 😢, but maybe ###chat";
        let vocab = Vocab::new_builder(token::VocabTokenType::Word)
            .from_corpus(&corpus)
            .add_word_token_literal("###chat")
            .with_char_fallback()
            .build()
            .unwrap();

        assert!(vocab.dictionary().contains_key(&Token::word("example")));
        assert!(vocab.dictionary().contains_key(&Token::word("tokens")));
        assert!(!vocab.dictionary().contains_key(&Token::word("tokens!")));
        assert!(!vocab.dictionary().contains_key(&Token::word("'numbers''")));
        assert!(!vocab.dictionary().contains_key(&Token::word("1234")));

        assert_eq!(vec![Token::word("this")], vocab.encode("this").unwrap());
        assert_eq!(vec![Token::word("is")], vocab.encode("is").unwrap());
        assert_eq!(vec![Token::word("an")], vocab.encode("an").unwrap());
        assert_eq!(
            vec![Token::word("example"), Token::char(',')],
            vocab.encode("example,").unwrap()
        );

        assert_eq!(
            vec![Token::word("tokens"), Token::char('!')],
            vocab.encode("tokens!").unwrap()
        );

        assert_eq!(
            vec![Token::char('\''), Token::word("numbers"), Token::char('\'')],
            vocab.encode("'numbers'").unwrap()
        );
        assert_eq!(
            vec![Token::word("like"), Token::char(':')],
            vocab.encode("like:").unwrap()
        );

        assert_eq!(
            vec![Token::char('2'), Token::char('3')],
            vocab.encode("23").unwrap()
        );
        assert_eq!(vec![Token::char('😢')], vocab.encode("😢").unwrap());
        assert_eq!(
            vec![Token::word("###chat")],
            vocab.encode("###chat").unwrap()
        );
    }

    #[ignore = "fix test"]
    #[test]
    fn transformer_work_build_bpe_vocab() {
        let corpus_part1 = "this is an example corpus that includes a number of words that comprise only of lowercase alphabetic";
        let corpus_part2 = " characters";
        let corpus = format!("{}{}", corpus_part1, corpus_part2);
        let vocab = Vocab::new_builder(token::VocabTokenType::BPE)
            .from_corpus(&corpus)
            // .with_char_fallback()
            // .with_max_vocab_size(100)
            .build()
            .unwrap();

        let all_tokens = vocab.encode(&corpus).unwrap();
        let ctrl_vocab = vocab.control_token();
        let part1_tokens = vocab.encode(&corpus_part1).unwrap();
        let part1_tokens_iter = part1_tokens.iter().chain(std::iter::repeat(&ctrl_vocab));

        let next_token_start = all_tokens
            .iter()
            .zip(part1_tokens_iter)
            .position(|(all, ctx)| all != ctx);

        let (ctx_tokens, next_tokens) = all_tokens.split_at(next_token_start.unwrap());

        let optimizer = solver::SGDOptimizer::new_cache(0.1);
        let sequence_len = all_tokens.len() - 1;
        // let sequence_len = 8;
        let builder = DecoderBuilder::new(sequence_len, 12, vocab.len())
            // .with_head_count(2)
            .with_block_count(1)
            // .with_feed_forward_hidden_dimension(8)
            .with_dropout_rate(0.0);

        let transformer = training::setup_and_train_transformer(training::TestGdtConfig {
            builder,
            training_rounds: 100,
            batch_size: 1,
            sample_from_pattern: None,
            optimizer,
            vocab,
            training_corpus: corpus.clone(),
            testing_corpus: corpus,
        });

        let (pred_next, prob) = transformer.predict_arg_max(ctx_tokens).unwrap();

        let actual_next = next_tokens.first().unwrap();

        dbg!((actual_next, &pred_next));
        let prob = dbg!(prob);
        assert!(prob > 0.95);
        assert_eq!(actual_next, &pred_next);
    }

    #[test]
    fn transformer_work_single_simple_pattern() {
        let corpus = "abcd".to_string();
        let vocab = Vocab::new_builder(token::VocabTokenType::Char)
            .from_corpus(&corpus)
            .build()
            .unwrap();
        let optimizer = solver::SGDOptimizer::new_cache(0.1);
        let builder = DecoderBuilder::new(corpus.len() - 1, 12, vocab.len())
            // .with_head_count(2)
            .with_block_count(2)
            // .with_feed_forward_hidden_dimension(8)
            .with_dropout_rate(0.0);

        let transformer = training::setup_and_train_transformer(training::TestGdtConfig {
            builder,
            training_rounds: 25,
            batch_size: 1,
            sample_from_pattern: None,
            optimizer,
            vocab,
            training_corpus: corpus.clone(),
            testing_corpus: corpus,
        });

        let probabilities = sample_probabilities(&transformer, &['a', 'b', 'c']);
        let token = transformer.vocab().token_encode(&Token::char('d')).unwrap();
        let prob = dbg!(probabilities[token]);
        assert!(prob > 0.95);
    }

    #[test]
    fn transformer_works_on_partial_simple_patterns() {
        let corpus = "abcd".to_string();
        let vocab = Vocab::new_builder(token::VocabTokenType::Char)
            .from_corpus(&corpus)
            .build()
            .unwrap();
        let optimizer = solver::AdamOptimizer::new_cache(0.001);
        let builder = DecoderBuilder::new(3, 12, vocab.len())
            .with_block_count(6)
            .with_dropout_rate(0.0)
            .with_embedding_init_strategy(LayerInitStrategy::KaimingZeroBias)
            // .with_output_dense_init_strategy(LayerInitStrategy::FullRandom)
            .with_output_dense_init_strategy(LayerInitStrategy::KaimingZeroBias)
            .with_attention_kqv_weights_init_strategy(LayerInitStrategy::ScaledFullRandom)
            .with_attention_dense_layer_init_strategy(LayerInitStrategy::KaimingZeroBias)
            .with_feed_forward_init_strategy(LayerInitStrategy::KaimingZeroBias);

        let transformer = training::setup_and_train_transformer(training::TestGdtConfig {
            builder,
            training_rounds: 750,
            batch_size: 1,
            sample_from_pattern: None,
            optimizer,
            vocab,
            training_corpus: corpus.clone(),
            testing_corpus: corpus,
        });

        let probabilities = sample_probabilities(&transformer, &['a', 'b']);
        let token1 = transformer.vocab().token_encode(&Token::char('c')).unwrap();
        let prob1 = dbg!(probabilities[token1]);
        assert!(prob1 > 0.95);
    }

    #[test]
    fn transformer_works_on_regular_prefix_patterns() {
        let corpus = "|-->1|-->2|-->3|-->4".to_string();
        let vocab = Vocab::new_builder(token::VocabTokenType::Char)
            .from_corpus(&corpus)
            .build()
            .unwrap();
        let optimizer = solver::AdamOptimizer::new_cache(0.001);
        let builder = DecoderBuilder::new(4, 12, vocab.len())
            .with_block_count(6)
            .with_dropout_rate(0.0)
            .with_embedding_init_strategy(LayerInitStrategy::KaimingZeroBias)
            // .with_output_dense_init_strategy(LayerInitStrategy::FullRandom)
            .with_output_dense_init_strategy(LayerInitStrategy::KaimingZeroBias)
            .with_attention_kqv_weights_init_strategy(LayerInitStrategy::ScaledFullRandom)
            .with_attention_dense_layer_init_strategy(LayerInitStrategy::KaimingZeroBias)
            .with_feed_forward_init_strategy(LayerInitStrategy::KaimingZeroBias);

        let transformer = training::setup_and_train_transformer(training::TestGdtConfig {
            builder,
            training_rounds: 750,
            batch_size: 1,
            sample_from_pattern: Some("|-->"),
            optimizer,
            vocab,
            training_corpus: corpus.clone(),
            testing_corpus: corpus,
        });

        let probabilities = sample_probabilities(&transformer, &['|', '-']);
        let token1 = transformer.vocab().token_encode(&Token::char('-')).unwrap();
        let prob1 = dbg!(probabilities[token1]);
        assert!(prob1 > 0.95);

        let probabilities = sample_probabilities(&transformer, &['|', '-', '-']);
        let token2 = transformer.vocab().token_encode(&Token::char('>')).unwrap();
        let prob2 = dbg!(probabilities[token2]);
        assert!(prob2 > 0.95);
    }

    #[test]
    #[ignore = "takes too long"]
    fn transformer_works_on_simultaneous_simple_patterns() {
        let corpus = "abcdefghijklmnopqrstuvwxyz".to_string();
        let vocab = Vocab::new_builder(token::VocabTokenType::Char)
            .from_corpus(&corpus)
            .build()
            .unwrap();
        println!("[(0,BLOCKS),(1,ROUNDS),(2,---),(3,EMBEDIM),(4,HIDNDIM),(5,HEADS)]");
        let inject_env_values = inject_env_values();
        // let optimizer = solver::AdamOptimizer::new_cache(0.001);
        let optimizer = solver::source::DefaultOptimizerCache::new(
            solver::source::DynamicOptimizerFactory::new(move |param_count, param_dimension| {
                solver::AdamOptimizer::new_builder(param_count, param_dimension)
                    .with_beta(0.9, 0.98)
                    .with_epsilon(1e-9)
                    .build()
            }),
        );

        let model_dimension = *inject_env_values.get(3).unwrap_or(&12.0) as usize;
        let builder = DecoderBuilder::new(3, model_dimension, vocab.len())
            .with_block_count(*inject_env_values.get(0).unwrap_or(&1.0) as usize)
            .with_feed_forward_hidden_dimension(*inject_env_values.get(4).unwrap_or(&64.0) as usize)
            .with_head_count(*inject_env_values.get(5).unwrap_or(&3.0) as usize)
            .with_dropout_rate(0.0)
            .with_embedding_init_strategy(LayerInitStrategy::KaimingZeroBias)
            .with_output_dense_init_strategy(LayerInitStrategy::KaimingZeroBias)
            .with_attention_kqv_weights_init_strategy(LayerInitStrategy::ScaledFullRandom)
            .with_attention_dense_layer_init_strategy(LayerInitStrategy::KaimingZeroBias)
            .with_feed_forward_init_strategy(LayerInitStrategy::KaimingZeroBias);

        let transformer = training::setup_and_train_transformer(training::TestGdtConfig {
            builder,
            training_rounds: *inject_env_values.get(1).unwrap_or(&2000.0) as usize,
            batch_size: 1,
            sample_from_pattern: None,
            optimizer,
            vocab,
            training_corpus: corpus.clone(),
            testing_corpus: corpus,
        });

        let probabilities = sample_probabilities(&transformer, &['a', 'b', 'c']);
        let token1 = dbg!(transformer.vocab().token_encode(&Token::char('d')).unwrap());
        let prob1 = dbg!(probabilities[token1]);
        assert!(prob1 > 0.95);

        let probabilities = sample_probabilities(&transformer, &['b', 'c', 'd']);
        let token2 = transformer.vocab().token_encode(&Token::char('e')).unwrap();
        let prob2 = dbg!(probabilities[token2]);
        assert!(prob2 > 0.95);

        let probabilities = sample_probabilities(&transformer, &['c', 'd', 'e']);
        let token3 = transformer.vocab().token_encode(&Token::char('f')).unwrap();
        let prob3 = dbg!(probabilities[token3]);
        assert!(prob3 > 0.95);
    }

    #[test]
    #[ignore = "takes too long"]
    fn transformer_work_simple_counter_prediction_words() {
        let phrases = [
            // used only for word vocab building
            "six seven eight nine tho this may been seen less often",
            "zero one two three four five six seven eight nine ten",
            "cero uno dos tres cuatro cinco seis siete ocho nueve diez",
            "zero uno due tre quattro cinque sei sette otto nove dieci",
            // also used in training, relying on fallback char encoder
            "nul een twee drie vier vijf zes zeven acht negen tien",
            "null eins zwei drei vier funf sechs sieben acht neun zehn",
            "sero un dau tri pedwar pump chwech saith wyth naw deg",
        ];

        let vocab_corpus = phrases[0..4].join("\n");
        let corpus = phrases.join("\n");
        let vocab = Vocab::new_builder(token::VocabTokenType::Word)
            .from_corpus(&vocab_corpus)
            .with_char_fallback()
            .build()
            .unwrap();
        let optimizer = solver::AdamOptimizer::new_cache(0.001);

        let builder = DecoderBuilder::new(3, 12, vocab.len())
            .with_block_count(1) // multiple blocks cause lower accuracy, check init strats and back prop grads?
            .with_dropout_rate(0.0)
            .with_embedding_init_strategy(LayerInitStrategy::KaimingZeroBias)
            .with_output_dense_init_strategy(LayerInitStrategy::KaimingZeroBias)
            .with_attention_kqv_weights_init_strategy(LayerInitStrategy::ScaledFullRandom)
            .with_attention_dense_layer_init_strategy(LayerInitStrategy::KaimingZeroBias)
            .with_feed_forward_init_strategy(LayerInitStrategy::KaimingZeroBias);

        let t = training::setup_and_train_transformer(training::TestGdtConfig {
            builder,
            training_rounds: 3000,
            // training_rounds: 10000,
            batch_size: 16,
            sample_from_pattern: None,
            optimizer,
            vocab,
            training_corpus: corpus.clone(),
            testing_corpus: corpus,
        });

        let check_corpus_stride = |target: Token, context: &str| {
            let input_sequence = t.vocab().encode(context).unwrap();
            let probabilities = t.sample_probabilities(&input_sequence).unwrap();

            let token = t.vocab().token_encode(&target).unwrap();
            let prob = probabilities[token];
            let prob_ranks = probabilities.rank_iter().collect_vec();
            let rank = prob_ranks[token];
            let prob_pct = prob * 100.0;

            println!("ctx: '{context}', target: '{target}' -> prob: {prob_pct:.2}% (rank #{rank})");

            assert!(prob.is_finite());
        };

        check_corpus_stride(Token::word("nine"), "six seven eight");
        check_corpus_stride(Token::word("this"), "eight nine tho");
        check_corpus_stride(Token::word("may"), "nine tho this");
        check_corpus_stride(Token::char('r'), "vie");

        assert_eq!(
            Token::word("nine"),
            t.predict_arg_max_from("six seven eight").unwrap()
        );
        assert_eq!(
            Token::word("this"),
            t.predict_arg_max_from("eight nine tho").unwrap()
        );
        assert_eq!(
            Token::word("may"),
            t.predict_arg_max_from("nine tho this").unwrap()
        );
    }

    #[test]
    #[ignore = "takes too long"]
    fn transformer_work_simple_counter_prediction() {
        let phrases = [
            "six seven eight nine tho this may been seen less often",
            "zero one two three four five six seven eight nine ten",
            "cero uno dos tres cuatro cinco seis siete ocho nueve diez",
            "zero uno due tre quattro cinque sei sette otto nove dieci",
            // "nul een twee drie vier vijf zes zeven acht negen tien",
            // "null eins zwei drei vier fünf sechs sieben acht neun zehn",
            // "sero un dau tri pedwar pump chwech saith wyth naw deg",
        ];

        let corpus = phrases.join("\n");
        let vocab = Vocab::new_builder(token::VocabTokenType::Char)
            .from_corpus(&corpus)
            .build()
            .unwrap();
        let optimizer = solver::AdamOptimizer::new_cache(0.001);

        let builder = DecoderBuilder::new(12, 12, vocab.len())
            .with_block_count(1) // multiple blocks cause lower accuracy, check init strats and back prop grads?
            .with_dropout_rate(0.0)
            .with_embedding_init_strategy(LayerInitStrategy::KaimingZeroBias)
            .with_output_dense_init_strategy(LayerInitStrategy::KaimingZeroBias)
            .with_attention_kqv_weights_init_strategy(LayerInitStrategy::ScaledFullRandom)
            .with_attention_dense_layer_init_strategy(LayerInitStrategy::KaimingZeroBias)
            .with_feed_forward_init_strategy(LayerInitStrategy::KaimingZeroBias);

        let t = training::setup_and_train_transformer(training::TestGdtConfig {
            builder,
            training_rounds: 3000,
            // training_rounds: 10000,
            batch_size: 5,
            sample_from_pattern: None,
            optimizer,
            vocab,
            training_corpus: corpus.clone(),
            testing_corpus: corpus,
        });

        let check_corpus_stride = |target: char, context: &str| {
            let input_sequence = t.vocab().encode(context).unwrap();
            let probabilities = t.sample_probabilities(&input_sequence).unwrap();

            let target = Token::char(target);
            let token = t.vocab().token_encode(&target).unwrap();
            let prob = probabilities[token];
            let prob_ranks = probabilities.rank_iter().collect_vec();
            let rank = prob_ranks[token];
            let prob_pct = prob * 100.0;

            println!("ctx: '{context}', target: '{target}' -> prob: {prob_pct:.2}% (rank #{rank})");

            assert!(prob.is_finite());
        };

        check_corpus_stride('n', "seven eight ");
        check_corpus_stride('i', "even eight n");
        check_corpus_stride('n', "ven eight ni");
        check_corpus_stride('e', "en eight nin");
        check_corpus_stride(' ', "n eight nine");

        assert_eq!(
            Token::char('n'),
            t.predict_arg_max_from("seven eight ").unwrap()
        );
        assert_eq!(
            Token::char('i'),
            t.predict_arg_max_from("even eight n").unwrap()
        );
        assert_eq!(
            Token::char('n'),
            t.predict_arg_max_from("ven eight ni").unwrap()
        );
        assert_eq!(
            Token::char('e'),
            t.predict_arg_max_from("en eight nin").unwrap()
        );
        assert_eq!(
            Token::char(' '),
            t.predict_arg_max_from("n eight nine").unwrap()
        );
    }

    fn sample_probabilities(
        transformer: &GenerativeDecoderTransformer,
        input_sequence: &[char],
    ) -> LayerValues {
        let joined_sequence: String = input_sequence.iter().collect();
        let input_sequence = transformer.vocab().encode(&joined_sequence).unwrap();
        transformer.sample_probabilities(&input_sequence).unwrap()
    }

    fn inject_env_values() -> Vec<f64> {
        let inject_env_var = std::env::var("RUST_INJECT")
            .iter()
            .flat_map(|x| x.split(',').map(|x| x.parse::<f64>()))
            .flatten()
            .collect_vec();

        dbg!(inject_env_var)
    }

    mod training {
        use crate::ml::{
            gdt::{token::Vocab, GenerativeDecoderTransformer, TrainContext},
            transformer::{decoder::builder::DecoderBuilder, solver::source::OptimizerSource},
            RngStrategy,
        };

        #[derive(Clone)]
        pub struct TestGdtConfig<O> {
            pub builder: DecoderBuilder,
            pub training_rounds: usize,
            pub batch_size: usize,
            pub sample_from_pattern: Option<&'static str>,
            pub optimizer: O,
            pub vocab: Vocab,
            pub training_corpus: String,
            pub testing_corpus: String,
        }

        pub fn setup_and_train_transformer<O: OptimizerSource>(
            config: TestGdtConfig<O>,
        ) -> GenerativeDecoderTransformer {
            let rng = RngStrategy::testable(1234);
            let builder = config.builder.with_rng(rng);
            let rng = RngStrategy::testable(12345);
            let mut transformer =
                GenerativeDecoderTransformer::from_builder(builder, config.vocab, rng).unwrap();

            for _ in 0..config.training_rounds {
                transformer
                    .train(
                        &config.training_corpus,
                        &config.optimizer,
                        &TrainContext::new(config.batch_size, config.sample_from_pattern),
                    )
                    .unwrap();
            }

            transformer
        }
    }
}
