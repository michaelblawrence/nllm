use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use super::{LayerInitStrategy, LayerValues, Network, NetworkShape, NodeValue, RNG};

pub struct Embedding {
    network: Network,
    vocab: HashMap<String, usize>,
    rng: Rc<dyn RNG>,
}

impl Embedding {
    pub fn new(vocab: HashSet<String>, size: usize, rng: Rc<dyn RNG>) -> Self {
        let vocab_len = vocab.len();
        let mut ordered_vocab = vocab.into_iter().collect::<Vec<_>>();
        ordered_vocab.sort();

        let vocab = ordered_vocab
            .into_iter()
            .enumerate()
            .map(|(i, word)| (word, i))
            .collect();

        let mut network = Network::new(
            &NetworkShape::new(vocab_len, vocab_len, vec![size]),
            LayerInitStrategy::Random(rng.clone()),
        );

        network.set_activation_mode(super::NetworkActivationMode::Sigmoid);
        network.set_softmax_output_enabled(true);

        Self {
            network,
            vocab,
            rng,
        }
    }

    pub fn train(
        &mut self,
        phrases: &Vec<Vec<String>>,
        learn_rate: NodeValue,
        word_locality_factor: usize,
    ) -> Result<NodeValue, ()> {
        let indexed_phrases = phrases.iter().map(|phrase| {
            phrase
                .iter()
                .map(|word| self.vocab.get(&word.to_string()))
                .take_while(|word| word.is_some())
                .flatten()
        });

        let vectored_phrases = indexed_phrases
            .map(|phrase| {
                phrase
                    .map(|&vocab_idx| self.one_hot(vocab_idx).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut counter = 0;
        let mut cost_sum = 0.0;

        for phrase in vectored_phrases {
            let random_word_vectors = phrase.windows(word_locality_factor).map(|word_vectors| {
                let index_0 = self.rng.rand_range(0, word_vectors.len());
                let index_1 = self.rng.rand_range(0, word_vectors.len() - 1);

                if index_1 >= index_0 {
                    (&word_vectors[index_0], &word_vectors[index_1 + 1])
                } else {
                    (&word_vectors[index_1], &word_vectors[index_0])
                }
            });

            let cost = self
                .network
                .learn(super::NetworkLearnStrategy::BatchGradientDecent {
                    training_pairs: random_word_vectors
                        .map(|(x, y)| (LayerValues::new(x.clone()), LayerValues::new(y.clone())))
                        .collect(),
                    batch_size: 100,
                    learn_rate,
                })?;

            let cost = cost.unwrap_or_default();
            cost_sum += cost;
            counter += 1;
        }
        let cost = cost_sum / counter as NodeValue;
        println!(" -- net train iter cost {cost}");

        Ok(cost)
    }

    pub fn predict_next(&self, last_word: &str) -> Result<String, ()> {
        let vocab_idx = self.vocab.get(&last_word.to_string()).ok_or(())?;
        let network_input = self.one_hot(*vocab_idx).collect();
        let output = self.network.compute(network_input)?;

        let (predicted_vocab_idx, _) = output
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or(())?;

        Ok(self
            .vocab
            .iter()
            .find(|(_, vocab_idx)| **vocab_idx == predicted_vocab_idx)
            .ok_or(())?
            .0
            .clone())
    }

    pub fn predict_iter<'a>(&'a self, seed_word: &str) -> impl Iterator<Item = String> + 'a {
        let mut seen_words = HashSet::new();
        let mut curr_word = seed_word.to_string();
        std::iter::from_fn(move || {
            if seen_words.contains(&curr_word) {
                None
            } else {
                let last_word = curr_word.clone();

                seen_words.insert(last_word.clone());
                curr_word = self.predict_next(&last_word).ok()?;

                Some(last_word)
            }
        })
    }

    pub fn compute_error_batch(&self, phrases: &Vec<Vec<String>>) -> Result<NodeValue, ()> {
        let mut errors = vec![];

        for testing_phrase in phrases.iter() {
            let error = self.compute_error(testing_phrase)?;
            errors.push(error);
        }
        let error = errors
            .into_iter()
            .filter(|x| !x.is_nan())
            .sum::<NodeValue>()
            / phrases.len() as NodeValue;

        Ok(error)
    }

    pub fn compute_error(&self, phrase: &Vec<String>) -> Result<NodeValue, ()> {
        let indexed_phrase = phrase
            .iter()
            .map(|word| self.vocab.get(&word.to_string()))
            .take_while(|word| word.is_some())
            .flatten();

        let vectored_phrase = indexed_phrase
            .map(|&vocab_idx| self.one_hot(vocab_idx).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let mut errors = vec![];

        for word_vectors in vectored_phrase.chunks_exact(2) {
            if let [first_word_vector, second_word_vector] = word_vectors {
                let error = self.network.compute_error(
                    LayerValues::new(first_word_vector.clone()),
                    &LayerValues::new(second_word_vector.clone()),
                )?;
                let error: NodeValue = error.iter().sum();
                errors.push(error);
            }
        }

        Ok(errors.iter().sum::<NodeValue>() / errors.len() as NodeValue)
    }

    pub fn embeddings(&self, vocab_entry: &str) -> Result<LayerValues, ()> {
        let vocab_index = self.vocab.get(vocab_entry).ok_or(())?;
        Ok(self
            .network
            .node_weights(0, *vocab_index)?
            // .expect("should have valid layer strcture")
            .into())
    }

    pub fn nearest(&self, vocab_entry: &str) -> Result<(String, NodeValue), ()> {
        let candidate_embedding = self.embeddings(vocab_entry)?;
        let mut dot_products = vec![];

        for searched_vocab in self.vocab.keys().filter(|x| x.as_str() != vocab_entry) {
            let searched_embedding = self.embeddings(searched_vocab)?;
            let dot_product = candidate_embedding.normalized_dot_product(&searched_embedding);
            dot_products.push((searched_vocab, dot_product.unwrap_or_default()));
        }

        let (nearest_vocab, nearest_dot_product) = dot_products
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .ok_or(())?;

        Ok((nearest_vocab.clone(), nearest_dot_product))
    }

    fn one_hot(&self, idx: usize) -> impl Iterator<Item = NodeValue> {
        let vocab_len = self.vocab.len();
        (0..vocab_len).map(move |i| if i == idx { 1.0 } else { 0.0 })
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use serde_json::Value;

    use crate::ml::network::{JsRng, ShuffleRng};

    use super::*;

    #[test]
    #[ignore = "simple but not enough data?"]
    fn embeddings_work_simple() {
        let phrases = [
            "bottle wine bottle wine bottle wine bottle wine",
            "piano violin piano violin piano violin piano violin",
            "piano violin piano violin piano violin piano violin",
            "piano violin piano violin piano violin piano violin",
            "pizza pasta pizza pasta pizza pasta pizza pasta",
            "coffee soda coffee soda coffee soda coffee soda",
            "bottle soda bottle soda bottle soda bottle soda",
        ];

        let phrases: Vec<Vec<String>> = iter::repeat(phrases.into_iter())
            .flatten()
            .take(250)
            .map(|phrase| {
                phrase
                    .split_whitespace()
                    .map(|word| word.to_string())
                    .collect()
            })
            .collect();

        let vocab: HashSet<String> = phrases
            .iter()
            .flat_map(|phrase| phrase.iter())
            .cloned()
            .collect();

        let embedding = setup_embeddings(
            (vocab, phrases),
            TestEmbeddingConfig {
                embedding_size: 8,
                training_rounds: 1000,
                max_phrases_count: 500,
                word_locality_factor: 2,
                train_rate: 1e-3,
                test_phrases_pct: 20.0,
            },
        );

        // assert_eq!("wine", embedding.nearest("bottle").unwrap().0.as_str());
        assert_eq!("piano", embedding.nearest("violin").unwrap().0.as_str());
    }

    #[test]
    #[ignore = "takes too long"]
    fn embeddings_work1() {
        let (vocab, phrases) = parse_vocab_and_phrases();
        setup_embeddings(
            (vocab, phrases),
            TestEmbeddingConfig {
                embedding_size: 80,
                training_rounds: 50,
                max_phrases_count: 250,
                word_locality_factor: 5,
                train_rate: 1e-3,
                test_phrases_pct: 20.0,
            },
        );
        todo!("finish test case")
    }

    struct TestEmbeddingConfig {
        embedding_size: usize,
        training_rounds: usize,
        max_phrases_count: usize,
        word_locality_factor: usize,
        train_rate: NodeValue,
        test_phrases_pct: NodeValue,
    }

    fn setup_embeddings(
        (vocab, mut phrases): (HashSet<String>, Vec<Vec<String>>),
        TestEmbeddingConfig {
            embedding_size,
            training_rounds,
            max_phrases_count,
            word_locality_factor,
            train_rate,
            test_phrases_pct,
        }: TestEmbeddingConfig,
    ) -> Embedding {
        let rng = Rc::new(JsRng::default());
        let mut embedding = Embedding::new(vocab, embedding_size, rng.clone());
        rng.shuffle_vec(&mut phrases);

        let testing_phrases =
            split_training_and_testing(&mut phrases, max_phrases_count, test_phrases_pct);

        let mut seen_pairs = HashSet::new();

        for round in 0..training_rounds {
            embedding
                .train(&phrases, train_rate, word_locality_factor)
                .unwrap();

            let mut errors = vec![];
            let mut correct_first_word_predictions = 0;

            for testing_phrase in testing_phrases.iter() {
                let mut testing_phrase_iter = testing_phrase.iter();

                let last_word = &testing_phrase_iter.next().unwrap();
                let predicted = embedding.predict_next(last_word).unwrap();
                let actual = testing_phrase_iter.next().unwrap();

                if &predicted == actual {
                    correct_first_word_predictions += 1;
                } else {
                    let error = embedding.compute_error(testing_phrase).unwrap();
                    errors.push(error);

                    if seen_pairs.insert((predicted.clone(), actual.clone())) {
                        println!(
                            "embed! round = {}, found new messed up prediction pairing.. '{last_word} {predicted}' was predicted instead of '{last_word} {actual}'",
                            round + 1
                        );
                    }
                }
            }

            if round < 50
                || (round < 1000 && round % 100 == 0)
                || (round < 10000 && round % 1000 == 0)
            {
                let error = errors.iter().filter(|x| !x.is_nan()).sum::<NodeValue>()
                    / errors.len() as NodeValue;

                let correct_first_word_predictions_pct =
                    correct_first_word_predictions as NodeValue * 100.0
                        / testing_phrases.len() as NodeValue;

                println!(
                    "embed! round = {}, prediction accuracy: {}%, network_cost = {error}",
                    round + 1,
                    correct_first_word_predictions_pct
                );
            }
        }

        embedding
    }

    fn split_training_and_testing(
        phrases: &mut Vec<Vec<String>>,
        max_phrases_count: usize,
        test_phrases_pct: f64,
    ) -> Vec<Vec<String>> {
        _ = phrases.split_off(max_phrases_count.min(phrases.len()));
        let testing_phrases = phrases.split_off(
            phrases.len()
                - (phrases.len() as NodeValue * (test_phrases_pct as NodeValue / 100.0)) as usize,
        );
        testing_phrases
    }

    fn parse_vocab_and_phrases() -> (HashSet<String>, Vec<Vec<String>>) {
        let vocab_json = include_str!("../../res/vocab_set.json");
        let vocab_json: Value = serde_json::from_str(vocab_json).unwrap();
        let vocab: HashSet<String> = vocab_json
            .as_array()
            .unwrap()
            .into_iter()
            .map(|value| value.as_str().unwrap().to_string())
            .collect();

        let phrase_json = include_str!("../../res/phrase_list.json");
        let phrase_json: Value = serde_json::from_str(phrase_json).unwrap();
        let phrases: Vec<Vec<String>> = phrase_json
            .as_array()
            .unwrap()
            .into_iter()
            .map(|value| {
                value
                    .as_array()
                    .unwrap()
                    .into_iter()
                    .map(|value| value.as_str().unwrap().to_owned())
                    .collect()
            })
            .collect();
        (vocab, phrases)
    }
}
