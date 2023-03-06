use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    iter,
    rc::Rc,
};

use super::{
    BatchSamplingStrategy, LayerInitStrategy, LayerValues, Network, NetworkActivationMode,
    NetworkLearnStrategy, NetworkShape, NodeValue, RNG, SamplingRng,
};

pub struct Embedding {
    network: Network,
    vocab: HashMap<String, usize>,
    rng: Rc<dyn RNG>,
}
const CONTROL_VOCAB: &str = &"<BREAK>";

impl Embedding {
    pub fn new(
        vocab: HashSet<String>,
        size: usize,
        hidden_layer_shape: Vec<usize>,
        rng: Rc<dyn RNG>,
    ) -> Self {
        let mut ordered_vocab = vocab.into_iter().collect::<Vec<_>>();
        ordered_vocab.sort();

        let vocab: HashMap<_, _> = [CONTROL_VOCAB.to_string()]
            .into_iter()
            .chain(ordered_vocab.into_iter())
            .enumerate()
            .map(|(i, word)| (word, i))
            .collect();

        let hidden_layer_init_stratergy = LayerInitStrategy::ScaledFullRandom(rng.clone());

        let network = {
            let mut network = Network::new(&NetworkShape::new_embedding(
                vocab.len(),
                size,
                hidden_layer_shape,
                hidden_layer_init_stratergy,
                rng.clone(),
            ));

            network.set_activation_mode(super::NetworkActivationMode::Sigmoid);
            network.set_softmax_output_enabled(false);
            network.set_first_layer_activation_enabled(false);
            network
        };

        Self {
            network,
            vocab,
            rng,
        }
    }

    pub fn set_activation_mode(&mut self, activation_mode: NetworkActivationMode) {
        self.network.set_activation_mode(activation_mode);
    }

    pub fn train_v2(
        &mut self,
        phrases: &Vec<Vec<String>>,
        learn_rate: NodeValue,
        word_locality_factor: usize,
        batch_size: usize,
    ) -> Result<NodeValue, ()> {
        let control_vocab_idx = self
            .vocab
            .get(&CONTROL_VOCAB.to_string())
            .expect("CONTROL_VOCAB should be present in vocab");

        if word_locality_factor < 1 {
            return Err(()); // need at least one sample
        }
        if word_locality_factor > 1 {
            todo!("cant pass in multiple embedding vectors to hidden layers yet");
        }

        let indexed_phrases = phrases.iter().flat_map(|phrase| {
            phrase
                .iter()
                .flat_map(|word| self.vocab.get(&word.to_string()))
                .chain([control_vocab_idx])
        });

        let vectored_phrases = iter::repeat(control_vocab_idx)
            .take(word_locality_factor)
            .chain(indexed_phrases)
            .map(|&vocab_idx| self.one_hot(vocab_idx).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let windowed_word_vectors = vectored_phrases.windows(word_locality_factor + 1).map(
            |word_vectors| match word_vectors.split_last() {
                Some((last, rest)) => (rest, last),
                None => panic!("should have "),
            },
        );

        // TODO: remove workaround contrained by word_locality_factor
        let windowed_word_vectors = windowed_word_vectors.map(|(x, y)| (x.first().unwrap(), y));

        let training_pairs: Vec<(LayerValues, LayerValues)> = windowed_word_vectors
            .map(|(x, y)| (LayerValues::new(x.clone()), LayerValues::new(y.clone())))
            .collect();

        let mut costs = vec![];

        for training_pairs in training_pairs.chunks(batch_size.pow(2)) {
            let cost = self
                .network
                .learn(NetworkLearnStrategy::BatchGradientDecent {
                    training_pairs: training_pairs.into_iter().cloned().collect::<Vec<_>>(),
                    batch_size,
                    learn_rate,
                    batch_sampling: BatchSamplingStrategy::Shuffle(self.rng.clone()),
                })?;
            costs.push(cost);
            // println!(
            //     " -- net train iter cost {}",
            //     cost.clone().unwrap_or_default()
            // );
        }

        let cost = costs.iter().flatten().sum::<NodeValue>() / costs.len() as NodeValue;

        Ok(cost)
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

            let training_pairs = random_word_vectors
                .map(|(x, y)| (LayerValues::new(x.clone()), LayerValues::new(y.clone())))
                .collect();

            let cost = self
                .network
                .learn(NetworkLearnStrategy::BatchGradientDecent {
                    training_pairs,
                    batch_size: 100,
                    learn_rate,
                    batch_sampling: BatchSamplingStrategy::Sequential,
                })?;

            let cost = cost.unwrap_or_default();
            cost_sum += cost;
            counter += 1;
        }
        let cost = cost_sum / counter as NodeValue;
        // println!(" -- net train iter cost {cost}");

        Ok(cost)
    }

    pub fn predict_next(&self, last_word: &str) -> Result<String, ()> {
        let vocab_idx = self.vocab.get(&last_word.to_string()).ok_or(())?;
        let network_input = self.one_hot(*vocab_idx).collect();
        let output = self.network.compute(network_input)?;

        let probabilities = NetworkActivationMode::SoftMax.apply(&output);
        let sampled_idx = self.rng.sample_uniform(&probabilities)?;

        Ok(self
            .vocab
            .iter()
            .find(|(_, vocab_idx)| **vocab_idx == sampled_idx)
            .ok_or(())?
            .0
            .clone())
    }

    pub fn nll(&self, last_word: &str, expected_next_word: &str) -> Result<NodeValue, ()> {
        let vocab_idx = self.vocab.get(&last_word.to_string()).ok_or(())?;
        let network_input = self.one_hot(*vocab_idx).collect();
        let output = self.network.compute(network_input)?;

        // TODO: do softmax in net?
        let expected_vocab_idx = *self.vocab.get(&expected_next_word.to_string()).ok_or(())?;
        let probabilities = NetworkActivationMode::SoftMax.apply(&output);

        let log_logits = probabilities
            .get(expected_vocab_idx)
            .expect("output should have same count as vocab")
            .ln();

        Ok(-log_logits)
    }

    pub fn nll_batch(&self, phrases: &Vec<Vec<String>>) -> Result<NodeValue, ()> {
        let word_locality_factor = 1;
        let control_vocab = CONTROL_VOCAB.to_string();
        let indexed_phrases = phrases
            .iter()
            .flat_map(|phrase| phrase.iter().chain([&control_vocab]));

        let all_words = iter::repeat(&control_vocab)
            .take(word_locality_factor)
            .chain(indexed_phrases)
            .collect::<Vec<_>>();

        let windowed_word_vectors =
            all_words
                .windows(word_locality_factor + 1)
                .map(|word_vectors| match word_vectors.split_last() {
                    Some((last, rest)) => (rest, last),
                    None => panic!("should have "),
                });

        // TODO: remove workaround contrained by word_locality_factor
        let testing_pairs = windowed_word_vectors.map(|(x, y)| (x.first().unwrap(), y));

        let mut nlls = vec![];

        for (prev, actual) in testing_pairs {
            let nll = self.nll(prev, actual)?;
            nlls.push(nll);
        }

        Ok(nlls.iter().sum::<NodeValue>() / nlls.len() as NodeValue)
    }

    pub fn predict_iter<'a>(&'a self, seed_word: &str) -> impl Iterator<Item = String> + 'a {
        let mut seen_words = HashMap::new();
        let mut curr_word = seed_word.to_string();
        std::iter::from_fn(move || {
            if curr_word.as_str() == CONTROL_VOCAB {
                None
            } else if seen_words.get(&curr_word).cloned().unwrap_or_default() > 3 {
                None
            } else {
                let last_word = curr_word.clone();

                *seen_words.entry(last_word.clone()).or_insert(0) += 1;
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
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .ok_or(())?;

        let (furthest_vocab, furthest_dot_product) = dot_products
            .iter()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .ok_or(())?;

        let available_range = 1.0 - furthest_dot_product;
        let similarity_fact = (nearest_dot_product - furthest_dot_product) / available_range;

        Ok((nearest_vocab.to_string(), similarity_fact))
    }

    fn one_hot(&self, idx: usize) -> impl Iterator<Item = NodeValue> {
        let vocab_len = self.vocab.len();
        (0..vocab_len).map(move |i| if i == idx { 1.0 } else { 0.0 })
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use self::training::{TestEmbeddingConfig, TestEmbeddingConfigV2};

    use super::*;

    #[test]
    #[ignore = "takes too long"]
    fn embeddings_work_simple_prediction_v2() {
        let phrases = [
            "bottle wine bottle wine bottle wine bottle wine",
            "piano violin piano violin piano violin piano violin",
            "pizza pasta pizza pasta pizza pasta pizza pasta",
            "coffee soda coffee soda coffee soda coffee soda",
            "tin can tin can tin can tin can",
        ];

        let phrases: Vec<Vec<String>> = phrases
            .into_iter()
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

        let testing_phrases = phrases.clone();

        let embedding = training::setup_and_train_embeddings_v2(
            (vocab, phrases, testing_phrases),
            TestEmbeddingConfigV2 {
                embedding_size: 2,
                hidden_layer_nodes: 100,
                batch_size: 32,
                training_rounds: 100000,
                train_rate: 1e-3,
                activation_mode: NetworkActivationMode::Tanh,
                // activation_mode: NetworkActivationMode::Sigmoid,
            },
        );

        assert_eq!("violin", embedding.predict_next("piano").unwrap().as_str());
        assert_eq!("pasta", embedding.predict_next("pizza").unwrap().as_str());
    }

    #[test]
    #[ignore = "takes too long"]
    fn embeddings_work_real_prediction_v2() {
        let (_, phrases) = training::parse_vocab_and_phrases();

        let phrases: Vec<Vec<String>> = phrases
            .into_iter()
            .filter(|phrase| phrase.len() > 6)
            .take(100)
            .collect();

        let vocab: HashSet<String> = phrases
            .iter()
            .flat_map(|phrase| phrase.iter())
            .cloned()
            .collect();

        dbg!((phrases.len(), vocab.len()));

        let testing_phrases = phrases.clone();

        let embedding = training::setup_and_train_embeddings_v2(
            (vocab.clone(), phrases, testing_phrases),
            TestEmbeddingConfigV2 {
                embedding_size: 15,
                hidden_layer_nodes: 85,
                batch_size: 32,
                training_rounds: 100,
                train_rate: 1e-4,
                // activation_mode: NetworkActivationMode::Tanh,
                activation_mode: NetworkActivationMode::Sigmoid,
            },
        );

        let systime = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis();

        std::fs::File::create(format!("out-{systime}_nearest.json"))
            .unwrap()
            .write_all(
                serde_json::to_string_pretty(&{
                    let mut map = HashMap::new();

                    for v in vocab.iter() {
                        let nearest = embedding
                            .nearest(&v)
                            .map(|(x, _)| x)
                            .unwrap_or_else(|_| "<none>".to_string());
                        map.insert(v, nearest);
                    }
                    map
                })
                .unwrap()
                .as_bytes(),
            )
            .unwrap();

        std::fs::File::create(format!("out-{systime}_predictions.json"))
            .unwrap()
            .write_all(
                serde_json::to_string_pretty(&{
                    let mut map = HashMap::new();

                    for v in vocab.iter() {
                        let predict = embedding
                            .predict_next(&v)
                            .unwrap_or_else(|_| "<none>".to_string());
                        map.insert(v, predict);
                    }
                    map
                })
                .unwrap()
                .as_bytes(),
            )
            .unwrap();

        assert_eq!("violin", embedding.predict_next("piano").unwrap().as_str());
        assert_eq!("pasta", embedding.predict_next("pizza").unwrap().as_str());
    }

    #[test]
    #[ignore = "takes too long"]
    fn embeddings_work_simple_prediction() {
        let phrases = [
            "bottle wine bottle wine bottle wine bottle wine",
            // "piano violin piano violin piano violin piano violin",
            // "piano violin piano violin piano violin piano violin",
            "piano violin piano violin piano violin piano violin",
            "pizza pasta pizza pasta pizza pasta pizza pasta",
            "coffee soda coffee soda coffee soda coffee soda",
            "tin can tin can tin can tin can",
        ];

        let phrases: Vec<Vec<String>> = phrases
            .into_iter()
            .cycle()
            .take(50)
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

        let embedding = training::setup_and_train_embeddings(
            (vocab, phrases),
            TestEmbeddingConfig {
                embedding_size: 5,
                hidden_layer_nodes: 4,
                training_rounds: 2500,
                max_phrases_count: 500,
                word_locality_factor: 2,
                train_rate: 1e-1,
                test_phrases_pct: 10.0,
            },
        );

        assert_eq!("violin", embedding.predict_next("piano").unwrap().as_str());
        assert_eq!("pasta", embedding.predict_next("pizza").unwrap().as_str());
    }

    #[test]
    fn can_network_learn_every_layers_grad_descent_sigmoid() {
        let phrases = [
            "bottle wine bottle wine bottle wine bottle wine",
            "piano violin piano violin piano violin piano violin",
            "piano violin piano violin piano violin piano violin",
            "piano violin piano violin piano violin piano violin",
            "pizza pasta pizza pasta pizza pasta pizza pasta",
            "coffee soda coffee soda coffee soda coffee soda",
            "bottle soda bottle soda bottle soda bottle soda",
        ];

        // let phrases = phrases.iter().cycle().take(250);
        let phrases = phrases.iter();

        let phrases: Vec<Vec<String>> = phrases
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

        let test_embedding_config = TestEmbeddingConfig {
            embedding_size: 8,
            hidden_layer_nodes: 10,
            training_rounds: 0,
            max_phrases_count: 500,
            word_locality_factor: 2,
            train_rate: 1e-3,
            test_phrases_pct: 0.0,
        };
        let mut embedding = training::setup_and_train_embeddings(
            (vocab, phrases.clone()),
            test_embedding_config.clone(),
        );

        let each_layer_weights = (*embedding.embeddings("piano").unwrap()).clone();
        let init_nearest = embedding.nearest("piano");

        embedding
            .train(
                &phrases,
                test_embedding_config.train_rate,
                test_embedding_config.word_locality_factor,
            )
            .unwrap();

        let current_nearest = embedding.nearest("piano");
        let current_layer_weights = (*embedding.embeddings("piano").unwrap()).clone();

        let diffs = current_layer_weights
            .iter()
            .zip(each_layer_weights.iter())
            .map(|(c, i)| c - i)
            .collect::<Vec<_>>();

        dbg!(diffs);
        dbg!((&init_nearest, &current_nearest));
        assert_ne!(&each_layer_weights, &current_layer_weights);
        assert_ne!(&init_nearest, &current_nearest);
    }

    #[test]
    #[ignore = "takes too long"]
    fn embeddings_work1() {
        let (vocab, phrases) = training::parse_vocab_and_phrases();
        training::setup_and_train_embeddings(
            (vocab, phrases),
            TestEmbeddingConfig {
                embedding_size: 80,
                hidden_layer_nodes: 100,
                training_rounds: 50,
                max_phrases_count: 250,
                word_locality_factor: 2,
                train_rate: 1e-2,
                test_phrases_pct: 10.0,
            },
        );
        todo!("finish test case")
    }

    mod training {
        use serde_json::Value;

        use crate::ml::network::{JsRng, ShuffleRng};

        use super::*;

        #[derive(Clone)]
        pub struct TestEmbeddingConfig {
            pub embedding_size: usize,
            pub hidden_layer_nodes: usize,
            pub training_rounds: usize,
            pub max_phrases_count: usize,
            pub word_locality_factor: usize,
            pub train_rate: NodeValue,
            pub test_phrases_pct: NodeValue,
        }

        pub fn setup_and_train_embeddings(
            (vocab, mut phrases): (HashSet<String>, Vec<Vec<String>>),
            config: TestEmbeddingConfig,
        ) -> Embedding {
            let rng: Rc<dyn RNG> = Rc::new(JsRng::default());
            let mut embedding = Embedding::new(
                vocab,
                config.embedding_size,
                vec![config.hidden_layer_nodes],
                rng.clone(),
            );
            rng.shuffle_vec(&mut phrases);

            let testing_phrases = split_training_and_testing(
                &mut phrases,
                config.max_phrases_count,
                config.test_phrases_pct,
            );

            let mut seen_pairs = HashSet::new();

            for round in 0..config.training_rounds {
                let training_error = embedding
                    .train(&phrases, config.train_rate, config.word_locality_factor)
                    .unwrap();

                let (validation_errors, predictions_pct) =
                    validate_embeddings(&embedding, &testing_phrases, &mut seen_pairs, round);

                report_training_round(
                    round,
                    config.training_rounds,
                    training_error,
                    validation_errors,
                    predictions_pct,
                );
            }

            embedding
        }

        #[derive(Clone)]
        pub struct TestEmbeddingConfigV2 {
            pub embedding_size: usize,
            pub hidden_layer_nodes: usize,
            pub training_rounds: usize,
            pub batch_size: usize,
            pub train_rate: NodeValue,
            pub activation_mode: NetworkActivationMode,
        }

        pub fn setup_and_train_embeddings_v2(
            (vocab, phrases, testing_phrases): (
                HashSet<String>,
                Vec<Vec<String>>,
                Vec<Vec<String>>,
            ),
            config: TestEmbeddingConfigV2,
        ) -> Embedding {
            let rng = Rc::new(JsRng::default());
            let mut embedding = Embedding::new(
                vocab,
                config.embedding_size,
                vec![config.hidden_layer_nodes],
                rng.clone(),
            );
            embedding.set_activation_mode(config.activation_mode);

            let mut seen_pairs = HashSet::new();

            for round in 0..config.training_rounds {
                let training_error = embedding
                    .train_v2(&phrases, config.train_rate, 1, config.batch_size)
                    .unwrap();

                let (validation_errors, predictions_pct) =
                    validate_embeddings(&embedding, &testing_phrases, &mut seen_pairs, round);

                report_training_round(
                    round,
                    config.training_rounds,
                    training_error,
                    validation_errors,
                    predictions_pct,
                );
            }

            embedding
        }

        pub fn parse_vocab_and_phrases() -> (HashSet<String>, Vec<Vec<String>>) {
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

        fn report_training_round(
            round: usize,
            training_rounds: usize,
            training_error: f64,
            validation_errors: Vec<(f64, f64)>,
            predictions_pct: f64,
        ) {
            let round_1based = round + 1;
            if round_1based <= 50
                || (round_1based < 1000 && round_1based % 100 == 0)
                || (round_1based < 10000 && round_1based % 1000 == 0)
                || round_1based == training_rounds
            {
                let (validation_error, nll) = validation_errors.last().unwrap();

                println!(
                "embed! round = {}, prediction accuracy: {}%, training_loss = {} validation_cost = {}, nll = {}",
                round_1based, predictions_pct, training_error, validation_error, nll
            );
            }
        }

        fn validate_embeddings(
            embedding: &Embedding,
            testing_phrases: &Vec<Vec<String>>,
            seen_pairs: &mut HashSet<(String, String)>,
            round: usize,
        ) -> (Vec<(f64, f64)>, f64) {
            let mut validation_errors = vec![];
            let mut correct_first_word_predictions = 0;

            for testing_phrase in testing_phrases.iter() {
                let mut testing_phrase_iter = testing_phrase.iter();

                let last_word = &testing_phrase_iter.next().unwrap();
                let predicted = embedding.predict_next(last_word).unwrap();
                let actual = testing_phrase_iter.next().unwrap();

                if &predicted == actual {
                    correct_first_word_predictions += 1;
                    for word in embedding.vocab.keys() {
                        if word != actual {
                            seen_pairs.remove(&(word.clone(), actual.clone()));
                        }
                    }
                    if seen_pairs.insert((predicted.clone(), actual.clone())) {
                        println!(
                        "embed! round = {}, ✅ correctly identified new prediction pairing.. '{last_word} {predicted}' was predicted correctly",
                        round + 1
                    );
                    }
                } else {
                    if seen_pairs.insert((predicted.clone(), actual.clone())) {
                        // println!(
                        //     "  embed! round = {}, ❌ found new messed up prediction pairing.. '{last_word} {predicted}' was predicted instead of '{last_word} {actual}'",
                        //     round + 1
                        // );
                    }
                }

                let error = embedding.compute_error(testing_phrase).unwrap();
                let nll = embedding.nll(&last_word, &actual).unwrap();
                validation_errors.push((error, nll));
            }

            let predictions_pct = correct_first_word_predictions as NodeValue * 100.0
                / testing_phrases.len() as NodeValue;
            (validation_errors, predictions_pct)
        }

        fn split_training_and_testing(
            phrases: &mut Vec<Vec<String>>,
            max_phrases_count: usize,
            test_phrases_pct: f64,
        ) -> Vec<Vec<String>> {
            _ = phrases.split_off(max_phrases_count.min(phrases.len()));
            let testing_phrases = phrases.split_off(
                phrases.len()
                    - (phrases.len() as NodeValue * (test_phrases_pct as NodeValue / 100.0))
                        as usize,
            );
            testing_phrases
        }
    }
}
