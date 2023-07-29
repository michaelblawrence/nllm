use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use tracing::instrument;

use plane::ml::{embeddings::Embedding, JsRng, NodeValue, ShuffleRng, RNG};

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainEmbeddingConfig {
    pub embedding_size: usize,
    pub hidden_layer_nodes: usize,
    pub training_rounds: usize,
    pub max_phrases_count: usize,
    pub max_vocab_words_count: usize,
    pub input_stride_width: usize,
    pub batch_size: usize,
    pub train_rate: NodeValue,
    pub process_all_batches: bool,
    pub test_phrases_pct: NodeValue,
}

#[instrument(skip_all)]
pub(crate) fn train_embedding(
    embedding: &mut Embedding,
    phrases: &Vec<Vec<String>>,
    train_rate: f64,
    batch_size: usize,
    process_all_batches: bool,
) -> f64 {
    let train_timer_label = &"embedding train";
    web_sys::console::time_with_label(train_timer_label);
    let rng: &dyn RNG = &JsRng::default();

    let processed_phrases = (!process_all_batches).then(|| {
        let batch_size = batch_size * batch_size;
        let max_len = phrases.len() - batch_size;
        let max_len = max_len.max(batch_size);
        let n = rng.rand_range(0, max_len);
        let mut phrases = phrases[n..n + batch_size]
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        rng.shuffle_vec(&mut phrases);
        
        phrases
    });
    let phrases = processed_phrases.as_ref().unwrap_or(phrases);

    let loss = embedding
        .train(&phrases, train_rate, batch_size)
        .unwrap();

    web_sys::console::time_end_with_label(train_timer_label);

    loss
}

pub(crate) fn split_training_and_testing(
    phrases: &mut Vec<Vec<String>>,
    max_phrases_count: usize,
    test_phrases_pct: f64,
) -> Vec<Vec<String>> {
    _ = phrases.split_off(max_phrases_count.min(phrases.len()));

    let testing_count = phrases.len() as NodeValue * (test_phrases_pct as NodeValue / 100.0);
    let testing_count = testing_count as usize;

    phrases.split_off(phrases.len() - testing_count)
}

pub(crate) fn parse_vocab_and_phrases(
    max_vocab: Option<usize>,
) -> (HashSet<String>, Vec<Vec<String>>) {
    let corpus = include_str!("../../../../res/tinytester.txt");
    let mut phrases: Vec<Vec<String>> = corpus
        .lines()
        .map(|line| {
            line.split_ascii_whitespace()
                .into_iter()
                .map(|value| value.to_owned())
                .collect()
        })
        .collect();

    let rng: &dyn RNG = &JsRng::default();
    rng.shuffle_vec(&mut phrases);

    let vocab = phrases
        .iter()
        .map(|phrase| phrase.iter().cloned())
        .flatten()
        .fold(HashMap::new(), |mut map, c| {
            *map.entry(c).or_insert(0_usize) += 1;
            map
        })
        .into_iter();
    let vocab: HashSet<(String, usize)> = vocab.collect();
    let mut vocab: Vec<_> = vocab.into_iter().collect();

    rng.shuffle_vec(&mut vocab);
    vocab.sort_by_key(|(_, count)| -(*count as i64));

    let vocab = vocab.into_iter().map(|(v, _)| v);
    let vocab: HashSet<String> = match max_vocab {
        Some(max_vocab) => vocab.take(max_vocab).collect(),
        None => vocab.collect(),
    };

    if max_vocab.is_some() {
        phrases = phrases
            .into_iter()
            .map(|phrase| {
                phrase
                    .into_iter()
                    .take_while(|word| vocab.contains(word))
                    .collect::<Vec<_>>()
            })
            .filter(|x| !x.is_empty())
            .collect();
        phrases.sort_by_key(|phrase| -(phrase.len() as i64));
    }

    (vocab, phrases)
}
