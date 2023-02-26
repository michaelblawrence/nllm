use std::{
    collections::{BTreeSet, HashMap, HashSet},
    rc::Rc,
};

use super::{network, LayerInitStrategy, LayerValues, Network, NetworkShape, NodeValue, RNG};

pub struct Embedding {
    network: Network,
    vocab: HashMap<String, usize>,
    size: usize,
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
        network.set_activation_mode(super::NetworkActivationMode::Linear);

        Self {
            network,
            vocab,
            size,
            rng,
        }
    }

    pub fn train(&mut self, phrases: &Vec<Vec<String>>, learn_rate: NodeValue) -> Result<(), ()> {
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

        for phrase in vectored_phrases {
            for word_vectors in phrase.windows(5) {
                // TODO: param
                let indexes = word_vectors.iter();
                let indexes = indexes.enumerate();
                let index_1 = (self.rng.rand() * indexes.len() as NodeValue) as usize;
                let index_2 = (self.rng.rand() * (indexes.len() - 1) as NodeValue) as usize;
                let mut indexes = (
                    indexes.clone().nth(index_1),
                    indexes.skip_while(|(i, ..)| *i == index_1).nth(index_2),
                );

                if let (Some((_, input_vector)), Some((_, output_vector))) =
                    (indexes.0.take(), indexes.1.take())
                {
                    self.network
                        .learn(super::NetworkLearnStrategy::GradientDecent {
                            inputs: LayerValues::new(input_vector.clone()),
                            target_outputs: LayerValues::new(output_vector.clone()),
                            learn_rate,
                        })?;
                } else {
                    panic!("should have two vectors here");
                }
            }
        }
        Ok(())
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

    pub fn shuffle_vec<T>(&self, vec: &mut Vec<T>) {
        let len = vec.len();

        for i in 0..len {
            let j = self.rng.rand_range(i, len);
            vec.swap(i, j);
        }
    }

    pub fn embeddings(&self, vocab_entry: &str) -> Result<LayerValues, ()> {
        let vocab_index = self.vocab.get(vocab_entry).ok_or(())?;
        Ok(self
            .network
            .node_weights(0, *vocab_index)?
            // .expect("should have valid layer strcture")
            .into())
    }

    pub fn nearest(&self, vocab_entry: &str) -> Result<String, ()> {
        let candidate_embedding = self.embeddings(vocab_entry)?;
        let mut distances = vec![];

        for searched_vocab in self.vocab.keys().filter(|x| x.as_str() != vocab_entry) {
            let searched_embedding = self.embeddings(searched_vocab)?;
            let distance = candidate_embedding.normalized_dot_product(&searched_embedding);
            distances.push((searched_vocab, distance));
        }

        let nearest = distances
            .into_iter()
            .min_by(|(_, distance_a), (_, distance_b)| distance_a.partial_cmp(distance_b).unwrap())
            .ok_or(())?;

        Ok(nearest.0.clone())
    }

    fn one_hot(&self, idx: usize) -> impl Iterator<Item = NodeValue> {
        let vocab_len = self.vocab.len();
        (0..vocab_len).map(move |i| if i == idx { 1.0 } else { 0.0 })
    }
}

#[cfg(test)]
mod tests {
    use network::JsRng;
    use serde_json::Value;

    use super::*;

    #[test]
    #[ignore = "takes too long"]
    fn embeddings_work() {
        let (vocab, phrases) = parse_vocab_and_phrases();
        let mut embedding = setup_embeddings(
            (vocab, phrases),
            TestEmbeddingConfig {
                embedding_size: 2,
                training_rounds: 100,
                max_vocab_count: 500,
                train_rate: 1e-4,
                test_phrases_pct: 20.0,
            },
        );

        // let error_1 = embedding.compute_error(phrases.first().unwrap());
        todo!("finish test case")
    }
    #[test]
    #[ignore = "takes too long"]
    fn embeddings_work1() {
        let (vocab, phrases) = parse_vocab_and_phrases();
        let mut embedding = setup_embeddings(
            (vocab, phrases),
            TestEmbeddingConfig {
                embedding_size: 80,
                training_rounds: 50,
                max_vocab_count: 250,
                train_rate: 1e-3,
                test_phrases_pct: 20.0,
            },
        );

        // let error_1 = embedding.compute_error(phrases.first().unwrap());
        todo!("finish test case")
    }

    struct TestEmbeddingConfig {
        embedding_size: usize,
        training_rounds: usize,
        max_vocab_count: usize,
        train_rate: NodeValue,
        test_phrases_pct: NodeValue,
    }

    fn setup_embeddings(
        (vocab, mut phrases): (HashSet<String>, Vec<Vec<String>>),
        TestEmbeddingConfig {
            embedding_size,
            training_rounds,
            max_vocab_count,
            train_rate,
            test_phrases_pct,
        }: TestEmbeddingConfig,
    ) -> Embedding {
        let mut embedding = Embedding::new(vocab, embedding_size, Rc::new(JsRng::default()));
        embedding.shuffle_vec(&mut phrases);

        let testing_phrases =
            split_training_and_testing(&mut phrases, max_vocab_count, test_phrases_pct);

        for round in 0..training_rounds {
            embedding.train(&phrases, train_rate).unwrap();
            let mut errors = vec![];
            for testing_phrase in testing_phrases.iter() {
                let error = embedding.compute_error(testing_phrase).unwrap();
                errors.push(error);
            }
            let error = errors
                .into_iter()
                .filter(|x| !x.is_nan())
                .sum::<NodeValue>()
                / testing_phrases.len() as NodeValue;

            if round < 50
                || (round < 1000 && round % 100 == 0)
                || (round < 10000 && round % 1000 == 0)
            {
                println!("embed! round = {}, network_cost = {error}", round + 1);
            }
        }

        embedding
    }

    fn split_training_and_testing(
        phrases: &mut Vec<Vec<String>>,
        max_vocab_count: usize,
        test_phrases_pct: f64,
    ) -> Vec<Vec<String>> {
        _ = phrases.split_off(max_vocab_count);
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
