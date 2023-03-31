use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet, VecDeque},
    iter,
    rc::Rc,
};

use anyhow::{anyhow, Context, Result};
use serde::{ser::SerializeStruct, Serialize};
use serde_json::Value;
use tracing::debug;

use super::{
    BatchSamplingStrategy, JsRng, LayerInitStrategy, LayerShape, LayerValues, Network,
    NetworkActivationMode, NetworkLearnStrategy, NetworkShape, NodeValue, SamplingRng, RNG,
};

pub struct Embedding {
    network: Network,
    vocab: HashMap<String, usize>,
    rng: Rc<dyn RNG>,
}

const CONTROL_VOCAB: &str = &"<CTRL>";

impl Embedding {
    pub fn new(
        vocab: HashSet<String>,
        embedding_dimensions: usize,
        input_stride_width: usize,
        hidden_layer_shape: Vec<usize>,
        rng: Rc<dyn RNG>,
    ) -> Self {
        EmbeddingBuilder::new(vocab, rng)
            .with_embedding_dimensions(embedding_dimensions)
            .with_input_stride_width(input_stride_width)
            .with_hidden_layer_custom_shape(hidden_layer_shape)
            .build()
    }

    pub fn new_builder(vocab: HashSet<String>, rng: Rc<dyn RNG>) -> EmbeddingBuilder {
        EmbeddingBuilder::new(vocab, rng)
    }

    pub fn from_snapshot(json: &str, rng: Rc<dyn RNG>) -> Result<Self> {
        let value: Value = serde_json::from_str(json)?;

        let network = value
            .get("network")
            .context("provided json should contain network field")?;
        let network = serde_json::from_value(network.clone())?;

        let vocab = value
            .get("vocab")
            .context("provided json should contain vocab field")?;
        let vocab = serde_json::from_value(vocab.clone())?;

        Ok(Self {
            network,
            vocab,
            rng,
        })
    }

    pub fn snapshot(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn snapshot_pretty(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn train_v2(
        &mut self,
        phrases: &Vec<Vec<String>>,
        learn_rate: NodeValue,
        batch_size: usize,
    ) -> Result<NodeValue> {
        let control_vocab_idx = self
            .vocab
            .get(&CONTROL_VOCAB.to_string())
            .expect("CONTROL_VOCAB should be present in vocab");
        let word_locality_factor = self.input_stride_width();

        // TODO: validate this in a better place
        if word_locality_factor < 1 {
            dbg!(word_locality_factor);
            return Err(anyhow!("need at least one sample"));
        }

        let indexed_phrases = phrases.iter().flat_map(|phrase| {
            phrase
                .iter()
                .flat_map(|word| self.vocab.get(&word.to_string()))
                .chain(iter::repeat(control_vocab_idx).take(word_locality_factor - 1))
        });

        let vectored_phrases = iter::repeat(control_vocab_idx)
            .take(word_locality_factor - 1)
            .chain(indexed_phrases)
            .map(|&vocab_idx| self.one_hot(vocab_idx).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let windowed_word_vectors = vectored_phrases.windows(word_locality_factor + 1).map(
            |word_vectors| match word_vectors.split_last() {
                Some((last, rest)) => (rest, last),
                None => panic!("should have last"),
            },
        );

        let training_pairs: Vec<(LayerValues, LayerValues)> = windowed_word_vectors
            .map(|(x, y)| {
                (
                    x.into_iter().flatten().cloned().collect(),
                    LayerValues::new(y.clone()),
                )
            })
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
            debug!(
                " -- net train iter cost {} (N = {})",
                cost.clone().unwrap_or_default(),
                training_pairs.len()
            );
        }

        let cost = costs.iter().flatten().sum::<NodeValue>() / costs.len() as NodeValue;

        Ok(cost)
    }

    pub fn train(
        &mut self,
        phrases: &Vec<Vec<String>>,
        learn_rate: NodeValue,
        word_locality_factor: usize,
    ) -> Result<NodeValue> {
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
        debug!(" -- net train iter cost {cost}");

        Ok(cost)
    }

    pub fn predict_next(&self, last_words: &[&str]) -> Result<String> {
        let network_input = self.get_padded_network_input(last_words)?;
        let probabilities = self.compute_probabilities(network_input)?;
        let sampled_idx = self.rng.sample_uniform(&probabilities)?;

        Ok(self
            .vocab
            .iter()
            .find(|(_, vocab_idx)| **vocab_idx == sampled_idx)
            .context("sampled vocab word should be in vocab dict")?
            .0
            .clone())
    }

    pub fn predict_from(&self, last_word: &str) -> Result<String> {
        let last_words: Vec<_> = iter::repeat(CONTROL_VOCAB)
            .take(self.input_stride_width() - 1)
            .chain([last_word].into_iter())
            .collect();
        let network_input = self.get_padded_network_input(&last_words[..])?;
        let probabilities = self.compute_probabilities(network_input)?;
        let sampled_idx = self.rng.sample_uniform(&probabilities)?;

        Ok(self
            .vocab
            .iter()
            .find(|(_, vocab_idx)| **vocab_idx == sampled_idx)
            .context("sampled vocab word should be in vocab dict")?
            .0
            .clone())
    }

    pub fn nll(&self, last_words: &Vec<&str>, expected_next_word: &str) -> Result<NodeValue> {
        let network_input = self.get_padded_network_input(&last_words[..])?;

        let expected_vocab_idx = *self
            .vocab
            .get(&expected_next_word.to_string())
            .context("provided vocab word should be in vocab dict")?;

        let probabilities = self.compute_probabilities(network_input)?;

        let log_logits = probabilities
            .get(expected_vocab_idx)
            .expect("output should have same count as vocab")
            .ln();

        Ok(-log_logits)
    }

    fn compute_probabilities(
        &self,
        network_input: LayerValues,
    ) -> Result<LayerValues, anyhow::Error> {
        let network_shape = &self.network.shape();
        let last_layer_shape = network_shape
            .iter()
            .last()
            .context("should have final layer defined")?;

        let post_apply_mode = match last_layer_shape.mode_override() {
            Some(NetworkActivationMode::SoftMax) => NetworkActivationMode::Linear,
            _ => NetworkActivationMode::SoftMax,
        };

        let output = self.network.compute(network_input)?;
        let probabilities = post_apply_mode.apply(&output);
        Ok(probabilities)
    }

    pub fn nll_batch(&self, phrases: &Vec<Vec<String>>) -> Result<NodeValue> {
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

        let testing_pairs =
            windowed_word_vectors.map(|(x, y)| (x.into_iter().map(|x| x.as_str()), y));

        let mut nlls = vec![];

        for (prev, actual) in testing_pairs {
            let nll = self.nll(&prev.collect(), actual)?;
            nlls.push(nll);
        }

        Ok(nlls.iter().sum::<NodeValue>() / nlls.len() as NodeValue)
    }

    pub fn predict_iter<'a>(&'a self, seed_word: &str) -> impl Iterator<Item = String> + 'a {
        let mut seen_words = HashMap::new();
        let mut recent_generated_words = VecDeque::new();
        let mut curr_word = seed_word.to_string();
        std::iter::from_fn(move || {
            if curr_word.as_str() == CONTROL_VOCAB {
                None
            } else if seen_words.get(&curr_word).cloned().unwrap_or_default() > 3 {
                None
            } else {
                let last_word = curr_word.clone();

                recent_generated_words.push_back(last_word.clone());
                *seen_words.entry(last_word.clone()).or_insert(0) += 1;

                curr_word = self
                    .predict_next(
                        &recent_generated_words
                            .iter()
                            .map(|a| a.as_str())
                            .collect::<Vec<_>>()[..],
                    )
                    .ok()?;
                let input_stride_width = self.input_stride_width();
                while recent_generated_words.len() > input_stride_width {
                    recent_generated_words.pop_front();
                }

                Some(last_word)
            }
        })
    }

    fn get_padded_network_input(&self, last_words: &[&str]) -> Result<LayerValues> {
        let input_stride_width = self.input_stride_width();
        let vocab_idxs = iter::repeat(&CONTROL_VOCAB)
            .take(input_stride_width.saturating_sub(last_words.len()))
            .chain(last_words.iter())
            .skip(last_words.len().saturating_sub(input_stride_width))
            .map(|vocab_word| {
                self.vocab
                    .get(&vocab_word.to_string())
                    .context("can not find vocab word")
            })
            .collect::<Vec<_>>();

        if let Some(Err(e)) = vocab_idxs.iter().find(|v| v.is_err()) {
            return Err(anyhow!("failed to resolve vocab from dict: {e}"));
        }

        let network_input = vocab_idxs
            .into_iter()
            .flatten()
            .flat_map(|vocab_idx| self.one_hot(*vocab_idx))
            .collect();
        Ok(network_input)
    }

    pub fn compute_error_batch(&self, phrases: &Vec<Vec<String>>) -> Result<NodeValue> {
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

    pub fn compute_error(&self, phrase: &Vec<String>) -> Result<NodeValue> {
        let indexed_phrase = phrase
            .iter()
            .map(|word| self.vocab.get(&word.to_string()))
            .take_while(|word| word.is_some())
            .flatten();

        let vectored_phrase = indexed_phrase
            .map(|&vocab_idx| self.one_hot(vocab_idx).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let mut errors = vec![];

        for word_vectors in vectored_phrase.chunks_exact(self.input_stride_width() + 1) {
            let (last_word_vector, context_word_vectors) = word_vectors
                .split_last()
                .context("should have last element")?;

            let context_word_vectors = context_word_vectors.into_iter().flatten();
            let inputs = context_word_vectors.cloned().collect();
            let target_outputs = LayerValues::new(last_word_vector.clone());

            let error = self.network.compute_error(inputs, &target_outputs)?;
            let error: NodeValue = error.ave();
            errors.push(error);
        }

        Ok(errors.iter().sum::<NodeValue>() / errors.len() as NodeValue)
    }

    pub fn embeddings(&self, vocab_entry: &str) -> Result<LayerValues> {
        let vocab_index = self
            .vocab
            .get(vocab_entry)
            .context("provided vocab word should be in vocab dict")?;

        Ok(self
            .network
            .node_weights(0, *vocab_index)
            .context("should have valid layer strcture")?
            .into())
    }

    pub fn nearest(&self, vocab_entry: &str) -> Result<(String, NodeValue)> {
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
            .context("can not sort dot product collection")?;

        let (_furthest_vocab, furthest_dot_product) = dot_products
            .iter()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .context("can not sort dot product collection")?;

        let available_range = 1.0 - furthest_dot_product;
        let similarity_fact = (nearest_dot_product - furthest_dot_product) / available_range;

        Ok((nearest_vocab.to_string(), similarity_fact))
    }

    fn one_hot(&self, idx: usize) -> impl Iterator<Item = NodeValue> {
        let vocab_len = self.vocab.len();
        (0..vocab_len).map(move |i| if i == idx { 1.0 } else { 0.0 })
    }

    pub fn input_stride_width(&self) -> usize {
        let layer_shape = self.network.shape().iter().nth(1).unwrap();
        layer_shape.stride_count()
    }

    pub fn vocab(&self) -> &HashMap<String, usize> {
        &self.vocab
    }
}

impl Default for Embedding {
    fn default() -> Self {
        let vocab = Default::default();
        let hidden_layer_shape = vec![];
        let rng = Rc::new(JsRng::default());
        let size = 0;
        let input_stride_width = 1;

        Embedding::new(vocab, size, input_stride_width, hidden_layer_shape, rng)
    }
}

impl Serialize for Embedding {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let mut embedding = serializer.serialize_struct("Embedding", 2)?;
        embedding.serialize_field("network", &self.network)?;
        embedding.serialize_field("vocab", &self.vocab)?;
        embedding.end()
    }
}

#[derive(Clone)]
pub struct EmbeddingBuilder {
    vocab: HashSet<String>,
    activation_mode: NetworkActivationMode,
    embedding_dimensions: usize,
    input_stride_width: usize,
    hidden_layer_shape: Vec<usize>,
    hidden_layer_init_stratergy: LayerInitStrategy,
    rng: Rc<dyn RNG>,
}

impl Default for EmbeddingBuilder {
    fn default() -> Self {
        let rng = Rc::new(JsRng::default());
        Self {
            vocab: Default::default(),
            activation_mode: NetworkActivationMode::Tanh,
            embedding_dimensions: 2,
            input_stride_width: 1,
            hidden_layer_shape: vec![10],
            hidden_layer_init_stratergy: LayerInitStrategy::ScaledFullRandom,
            rng,
        }
    }
}

impl EmbeddingBuilder {
    pub fn new(vocab: HashSet<String>, rng: Rc<dyn RNG>) -> Self {
        Self {
            vocab,
            rng,
            ..Default::default()
        }
    }

    pub fn build(self) -> Embedding {
        let mut ordered_vocab = self.vocab.into_iter().collect::<Vec<_>>();
        ordered_vocab.sort();

        let vocab: HashMap<_, _> = [CONTROL_VOCAB.to_string()]
            .into_iter()
            .chain(ordered_vocab.into_iter())
            .enumerate()
            .map(|(i, word)| (word, i))
            .collect();

        let embedding_layer_shape =
            Self::build_embedding_layer_shape(self.embedding_dimensions, self.input_stride_width);

        let network_shape = Self::build_shape(
            vocab.len(),
            embedding_layer_shape,
            self.hidden_layer_shape,
            self.hidden_layer_init_stratergy,
            self.activation_mode,
        );

        let network = Network::new(network_shape);

        Embedding {
            network,
            vocab,
            rng: self.rng,
        }
    }

    fn build_shape(
        token_count: usize,
        embedding_layer_shape: LayerShape,
        hidden_layer_shape: Vec<usize>,
        hidden_layer_init_strategy: LayerInitStrategy,
        activation_mode: NetworkActivationMode,
    ) -> NetworkShape {
        let strategy = hidden_layer_init_strategy;
        let final_layer: LayerShape = (token_count, strategy.clone()).into();
        let hidden_layer_shape = hidden_layer_shape.into_iter();

        let layers_shape = [embedding_layer_shape]
            .into_iter()
            .chain(hidden_layer_shape.map(|n| (n, strategy.clone()).into()))
            .chain([final_layer.with_activation_mode(NetworkActivationMode::SoftMax)])
            .collect();

        NetworkShape::new_custom(token_count, layers_shape, activation_mode)
    }

    fn build_embedding_layer_shape(embedding_size: usize, embedding_stride: usize) -> LayerShape {
        LayerShape::new(
            embedding_size,
            LayerInitStrategy::NoBias,
            Some(embedding_stride),
            Some(NetworkActivationMode::Linear),
        )
    }

    pub fn with_activation_mode(mut self, activation_mode: NetworkActivationMode) -> Self {
        self.activation_mode = activation_mode;
        self
    }

    pub fn with_embedding_dimensions(mut self, embedding_dimensions: usize) -> Self {
        self.embedding_dimensions = embedding_dimensions;
        self
    }

    pub fn with_input_stride_width(mut self, input_stride_width: usize) -> Self {
        self.input_stride_width = input_stride_width;
        self
    }

    pub fn with_hidden_layer_custom_shape(mut self, hidden_layer_shape: Vec<usize>) -> Self {
        self.hidden_layer_shape = hidden_layer_shape;
        self
    }

    pub fn with_hidden_layer_size(mut self, hidden_layer_size: usize) -> Self {
        self.hidden_layer_shape = vec![hidden_layer_size];
        self
    }

    pub fn with_hidden_layer_init_stratergy(
        mut self,
        hidden_layer_init_stratergy: LayerInitStrategy,
    ) -> Self {
        self.hidden_layer_init_stratergy = hidden_layer_init_stratergy;
        self
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use test_log::test;
    use tracing::info;

    use self::training::{TestEmbeddingConfig, TestEmbeddingConfigV2};

    use super::*;

    #[test]
    #[ignore = "takes too long"]
    fn embeddings_work_medium_len_phrases_prediction_v2() {
        let phrases = training::parse_phrases();
        let rng: &dyn RNG = &JsRng::default();

        let mut phrases: Vec<Vec<String>> = phrases
            .into_iter()
            .filter(|phrase| phrase.len() > 12 && phrase.len() < 20)
            .collect();

        let vocab_counts = phrases
            .iter()
            .flat_map(|phrase| phrase.iter())
            .cloned()
            .fold(HashMap::new(), |mut counts, word| {
                counts.entry(word).and_modify(|x| *x += 1).or_insert(1_i32);
                counts
            });

        phrases.sort_by_key(|phrase| {
            phrase
                .iter()
                .map(|word| vocab_counts[word].pow(2))
                .sum::<i32>()
        });
        use itertools::Itertools;

        phrases.truncate(125);
        info!(
            "First 3 phrases: {:#?}",
            phrases[0..3]
                .iter()
                .map(|phrase| (
                    phrase.join(" "),
                    phrase
                        .iter()
                        .map(|word| vocab_counts[word].pow(2))
                        .join("|")
                ))
                .collect::<Vec<_>>()
        );

        let vocab: HashSet<String> = phrases
            .iter()
            .flat_map(|phrase| phrase.iter())
            .cloned()
            .collect();

        dbg!((phrases.len(), vocab.len()));

        crate::ml::ShuffleRng::shuffle_vec(&rng, &mut phrases);

        // let testing_phrases = phrases.split_off(phrases.len() / 10);
        let testing_phrases = phrases.clone();

        let embedding = training::setup_and_train_embeddings_v2(
            (vocab.clone(), phrases, testing_phrases),
            TestEmbeddingConfigV2 {
                embedding_size: 2,
                hidden_layer_nodes: 35,
                input_stride_width: 3,
                batch_size: 8,
                training_rounds: 25_000,
                train_rate: 1e-2,
                activation_mode: NetworkActivationMode::Sigmoid,
            },
        );

        training::write_results_to_disk(&embedding, &vocab, "convo-midlen");
    }

    #[test]
    #[ignore = "takes too long"]
    fn embeddings_work_simple_counter_prediction_v2() {
        let phrases = [
            "zero one two three four five six seven eight nine ten",
            "cero uno dos tres cuatro cinco seis siete ocho nueve diez",
            "zero uno due tre quattro cinque sei sette otto nove dieci",
            "nul een twee drie vier vijf zes zeven acht negen tien",
            "null eins zwei drei vier fünf sechs sieben acht neun zehn",
            "sero un dau tri pedwar pump chwech saith wyth naw deg",
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
            (vocab.clone(), phrases, testing_phrases),
            TestEmbeddingConfigV2 {
                embedding_size: 2,
                hidden_layer_nodes: 30,
                input_stride_width: 3,
                batch_size: 3,
                training_rounds: 100_000,
                train_rate: 1e-2,
                activation_mode: NetworkActivationMode::Tanh,
            },
        );

        training::write_results_to_disk(&embedding, &vocab, "counter-lang");

        assert_eq!("one", embedding.predict_from("zero").unwrap().as_str());
        assert_eq!(
            "six",
            embedding
                .predict_next(&["three", "four", "five"])
                .unwrap()
                .as_str()
        );
    }

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
                embedding_size: 6,
                hidden_layer_nodes: 100,
                input_stride_width: 3,
                batch_size: 1,
                training_rounds: 100000,
                train_rate: 1e-4,
                activation_mode: NetworkActivationMode::Tanh,
                // activation_mode: NetworkActivationMode::Sigmoid,
            },
        );

        assert_eq!("violin", embedding.predict_from("piano").unwrap().as_str());
        assert_eq!("pasta", embedding.predict_from("pizza").unwrap().as_str());
        // assert_eq!("violin", embedding.predict_next(&vec!["piano"]).unwrap().as_str());
        // assert_eq!("pasta", embedding.predict_next(&vec!["pizza"]).unwrap().as_str());
    }

    #[test]
    #[ignore = "takes too long"]
    fn embeddings_work_real_prediction_v2() {
        let phrases = training::parse_phrases();

        let phrases: Vec<Vec<String>> = phrases
            .into_iter()
            .filter(|phrase| phrase.len() > 6)
            .take(1000)
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
                embedding_size: 5,
                hidden_layer_nodes: 125,
                input_stride_width: 3,
                batch_size: 16,
                training_rounds: 1000,
                train_rate: 1e-4,
                activation_mode: NetworkActivationMode::Tanh,
                // activation_mode: NetworkActivationMode::Sigmoid,
            },
        );

        training::write_results_to_disk(&embedding, &vocab, "convo-big");

        let seed_word = "first";
        info!(
            "Lets see what we can generate starting with the word '{seed_word}'\n\t{}",
            embedding
                .predict_iter(seed_word)
                .collect::<Vec<_>>()
                .join(" ")
        );
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

        assert_eq!("violin", embedding.predict_from("piano").unwrap().as_str());
        assert_eq!("pasta", embedding.predict_from("pizza").unwrap().as_str());
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
        assert_ne!(init_nearest.unwrap(), current_nearest.unwrap());
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
        use std::time::{Duration, Instant};

        use serde_json::Value;

        use crate::ml::{JsRng, ShuffleRng};

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
                1,
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

                if should_report_round(round, config.training_rounds) {
                    report_training_round(
                        round,
                        training_error,
                        validation_errors,
                        predictions_pct,
                        None,
                    );
                }
            }

            embedding
        }

        #[derive(Clone)]
        pub struct TestEmbeddingConfigV2 {
            pub embedding_size: usize,
            pub hidden_layer_nodes: usize,
            pub training_rounds: usize,
            pub input_stride_width: usize,
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
            let rng: Rc<dyn RNG> = Rc::new(JsRng::default());

            let mut embedding = Embedding::new_builder(vocab, rng.clone())
                .with_embedding_dimensions(config.embedding_size)
                .with_hidden_layer_size(config.hidden_layer_nodes)
                .with_input_stride_width(config.input_stride_width)
                .with_activation_mode(config.activation_mode)
                .build();

            let mut seen_pairs = HashSet::new();
            let mut last_report_time: Option<(Instant, usize)> = None;

            for round in 0..config.training_rounds {
                let training_error = embedding
                    .train_v2(&phrases, config.train_rate, config.batch_size)
                    .unwrap();

                if should_report_round(round, config.training_rounds) {
                    let (validation_errors, predictions_pct) =
                        validate_embeddings(&embedding, &testing_phrases, &mut seen_pairs, round);

                    report_training_round(
                        round,
                        training_error,
                        validation_errors,
                        predictions_pct,
                        last_report_time.map(|(last_dt, round)| (last_dt.elapsed(), round)),
                    );

                    last_report_time = Some((std::time::Instant::now(), round));
                }
            }

            embedding
        }

        pub fn parse_vocab_and_phrases() -> (HashSet<String>, Vec<Vec<String>>) {
            let vocab = parse_vocab();
            let phrases = parse_phrases();

            (vocab, phrases)
        }

        pub fn parse_vocab() -> HashSet<String> {
            let vocab_json = include_str!("../../res/vocab_set.json");
            let vocab_json: Value = serde_json::from_str(vocab_json).unwrap();
            let vocab: HashSet<String> = vocab_json
                .as_array()
                .unwrap()
                .into_iter()
                .map(|value| value.as_str().unwrap().to_string())
                .collect();
            vocab
        }

        pub fn parse_phrases() -> Vec<Vec<String>> {
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
            phrases
        }

        fn should_report_round(round: usize, training_rounds: usize) -> bool {
            let round_1based = round + 1;

            round_1based <= 50
                || (round_1based <= 1000 && round_1based % 100 == 0)
                || (round_1based <= 10000 && round_1based % 1000 == 0)
                || (round_1based <= 100000 && round_1based % 10000 == 0)
                || (round_1based <= 1000000 && round_1based % 100000 == 0)
                || round_1based == training_rounds
        }

        fn report_training_round(
            round: usize,
            training_error: f64,
            validation_errors: Vec<(f64, f64)>,
            predictions_pct: f64,
            last_report_time: Option<(Duration, usize)>,
        ) {
            let val_count = validation_errors.len() as NodeValue;
            let (validation_error, nll) =
                validation_errors
                    .iter()
                    .fold((0.0, 0.0), |sum, (validation_error, nll)| {
                        (
                            sum.0 + (validation_error / val_count),
                            sum.1 + (nll / val_count),
                        )
                    });

            let ms_per_round = last_report_time
                .map(|(duration, last_round)| duration.as_millis() / (round - last_round) as u128);
            let ms_per_round = ms_per_round
                .map(|ms_per_round| format!("(ms/round={ms_per_round:<4.1})"))
                .unwrap_or_default();

            info!(
                    "round = {:<6} |  train_loss = {:<12.10}, val_pred_acc: {:0>4.1}%, val_loss = {:<2.6e}, val_nll = {:<6.3} {ms_per_round}",
                    round + 1, training_error, predictions_pct, validation_error, nll
            );
        }

        fn validate_embeddings(
            embedding: &Embedding,
            testing_phrases: &Vec<Vec<String>>,
            seen_pairs: &mut HashSet<(String, String)>,
            round: usize,
        ) -> (Vec<(f64, f64)>, f64) {
            let mut validation_errors = vec![];
            let mut correct_first_word_predictions = 0;
            let mut total_first_word_predictions = 0;

            for testing_phrase in testing_phrases.iter() {
                for testing_phrase_window in
                    testing_phrase.windows(embedding.input_stride_width() + 1)
                {
                    let (last_word_vector, context_word_vectors) =
                        testing_phrase_window.split_last().unwrap();

                    let last_words = context_word_vectors
                        .into_iter()
                        .map(|x| x.as_str())
                        .collect::<Vec<_>>();

                    let predicted = embedding.predict_next(&last_words[..]).unwrap();
                    let actual = last_word_vector;

                    if &predicted == actual {
                        correct_first_word_predictions += 1;
                        for word in embedding.vocab.keys() {
                            if word != actual {
                                seen_pairs.remove(&(word.clone(), actual.clone()));
                            }
                        }
                        if seen_pairs.insert((predicted.clone(), actual.clone())) {
                            debug!(
                                "round = {:<6} |   ✅ correctly identified new prediction pairing.. '{last_word} {predicted}' was predicted correctly",
                                round + 1,
                                last_word = last_words.join(" "),
                            );
                        }
                    } else {
                        if seen_pairs.insert((predicted.clone(), actual.clone())) {
                            debug!(
                                "round = {:<6} |  ❌ found new messed up prediction pairing.. '{last_word} {predicted}' was predicted instead of '{last_word} {actual}'",
                                round + 1,
                                last_word = last_words.join(" "),
                            );
                        }
                    }

                    let error = embedding.compute_error(testing_phrase).unwrap();
                    let nll = embedding.nll(&last_words, &actual).unwrap();
                    validation_errors.push((error, nll));
                    total_first_word_predictions += 1;
                }
            }

            let predictions_pct = correct_first_word_predictions as NodeValue * 100.0
                / total_first_word_predictions as NodeValue;
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

        pub fn write_results_to_disk(embedding: &Embedding, vocab: &HashSet<String>, label: &str) {
            use itertools::Itertools;

            let systime = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis();

            std::fs::File::create(format!("out/out-{label}-{systime}_nearest.json"))
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

            std::fs::File::create(format!("out/out-{label}-{systime}_predictions.json"))
                .unwrap()
                .write_all(
                    serde_json::to_string_pretty(&{
                        let mut map = HashMap::new();

                        for v in vocab.iter() {
                            let predict = embedding
                                .predict_from(&v)
                                .unwrap_or_else(|_| "<none>".to_string());
                            map.insert(v, predict);
                        }
                        map
                    })
                    .unwrap()
                    .as_bytes(),
                )
                .unwrap();

            std::fs::File::create(format!("out/out-{label}-{systime}_embeddings.csv"))
                .unwrap()
                .write_fmt(format_args!(
                    "{}",
                    vocab
                        .iter()
                        .map(|word| {
                            format!(
                                "{word},{}",
                                embedding.embeddings(&word).unwrap().iter().join(",")
                            )
                        })
                        .join("\n")
                ))
                .unwrap();
        }
    }
}
