use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet, VecDeque},
    iter,
    ops::Deref,
};

use anyhow::{Context, Result};
use serde::Serialize;

use tracing::{debug, instrument};

use super::{
    BatchSamplingStrategy, LayerValues, Network, NetworkActivationMode, NetworkLearnStrategy,
    NetworkShape, NodeValue, RngStrategy, SamplingRng, ShuffleRng,
};

#[derive(Serialize)]
pub struct Embedding {
    network: Network,
    vocab: HashMap<String, usize>,

    #[serde(default)]
    rng: RngStrategy,
}

const CONTROL_VOCAB: &str = &"<CTRL>";

impl Embedding {
    pub fn new(
        vocab: HashSet<String>,
        embedding_dimensions: usize,
        input_stride_width: usize,
        hidden_layer_shape: Vec<usize>,
    ) -> Self {
        builder::EmbeddingBuilder::new(vocab)
            .with_embedding_dimensions(embedding_dimensions)
            .with_input_stride_width(input_stride_width)
            .with_hidden_layer_custom_shape(hidden_layer_shape)
            .build()
            .unwrap()
    }

    pub fn new_builder(vocab: HashSet<String>) -> builder::EmbeddingBuilder {
        builder::EmbeddingBuilder::new(vocab)
    }

    pub fn new_builder_ordered(vocab: &[String]) -> builder::EmbeddingBuilder {
        builder::EmbeddingBuilder::new_ordered(vocab)
    }

    pub fn snapshot(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn snapshot_pretty(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    #[instrument(level = "info", name = "embed_train", skip_all)]
    pub fn train<B: Into<TrainBatchConfig>>(
        &mut self,
        phrases: &Vec<Vec<String>>,
        learn_rate: NodeValue,
        batch_size: B,
    ) -> Result<NodeValue> {
        let batch_size: TrainBatchConfig = batch_size.into();
        let batches = self.into_batches(phrases, batch_size);
        let mut costs = vec![];

        for training_pairs in batches {
            let rng = self.rng.clone();
            let batch_sampling = BatchSamplingStrategy::Shuffle(*batch_size, rng);
            let training_pairs_len = training_pairs.len();

            let cost = self
                .network
                .learn(NetworkLearnStrategy::BatchGradientDecent {
                    training_pairs,
                    learn_rate,
                    batch_sampling,
                })?;

            costs.push(cost);

            debug!(
                " -- net train iter cost {} (N = {})",
                cost.clone().unwrap_or_default(),
                training_pairs_len
            );
        }

        let costs: LayerValues = costs.into_iter().flatten().collect();
        let ave_cost = costs.ave();
        Ok(ave_cost)
    }

    #[instrument(level = "info", skip_all)]
    fn into_batches(
        &self,
        phrases: &Vec<Vec<String>>,
        batch_size: TrainBatchConfig,
    ) -> Vec<Vec<(LayerValues, LayerValues)>> {
        // reduce phrase count for processing on small batches
        let phrases: Box<dyn Iterator<Item = &Vec<String>>> = match &batch_size {
            &TrainBatchConfig::SingleBatch(batch_size) => {
                let ave_len = phrases.iter().map(|x| x.len()).sum::<usize>() / phrases.len();
                let phrases = {
                    let mut tokens = 0;
                    let mut my_vec = vec![];
                    while tokens < batch_size {
                        let take = batch_size / ave_len.max(1);
                        self.rng
                            .take_rand(phrases, take.max(1))
                            .into_iter()
                            .for_each(|phrase| {
                                tokens += phrase.len();
                                my_vec.push(phrase);
                            })
                    }
                    my_vec
                };
                Box::new(phrases.into_iter())
            }
            _ => Box::new(phrases.iter()),
        };

        let training_pairs = self.into_context_target_pairs(phrases);

        match batch_size {
            TrainBatchConfig::Batches(batch_size) => {
                let available_parallelism = std::thread::available_parallelism();
                let available_parallelism = available_parallelism.map_or(0, |x| x.get());

                training_pairs
                    .chunks(batch_size * available_parallelism)
                    .map(|training_pairs| training_pairs.into_iter().cloned().collect::<Vec<_>>())
                    .collect()
            }
            TrainBatchConfig::SingleBatch(batch_size) => {
                let mut training_pairs = training_pairs;
                self.rng.shuffle_vec(&mut training_pairs);
                training_pairs.truncate(batch_size);

                vec![training_pairs]
            }
        }
    }

    fn into_context_target_pairs<'a, P>(&self, phrases: P) -> Vec<(LayerValues, LayerValues)>
    where
        P: Iterator<Item = &'a Vec<String>>,
    {
        let vectored_phrases: Vec<Vec<f64>> = phrases
            .flat_map(|phrase| self.get_padded_network_input_iter(&phrase[..]))
            .flatten() // flatten here ignores any error variants (missing vocab etc.)
            .map(|one_hot| one_hot.collect::<Vec<_>>())
            .collect();

        let word_locality_factor = self.input_stride_width();
        let windowed_word_vectors = vectored_phrases.windows(word_locality_factor + 1);

        windowed_word_vectors
            .map(|context_token_vectors| {
                let (last, rest) = context_token_vectors.split_last().unwrap();
                let expected_output = LayerValues::new(last.clone());
                let network_inputs: LayerValues = rest.into_iter().flatten().cloned().collect();

                (network_inputs, expected_output)
            })
            .collect()
    }

    pub fn sample_probabilities(&self, context_tokens: &[&str]) -> Result<LayerValues> {
        let network_input = self.get_padded_network_input(context_tokens)?;
        let output = self.network.compute(network_input)?;
        let probabilities = self.compute_probabilities(&output)?;
        Ok(probabilities)
    }

    fn compute_probabilities(&self, output: &LayerValues) -> Result<LayerValues> {
        let last_layer_shape = self.last_layer_shape()?;
        let post_apply_mode = match last_layer_shape.mode_override() {
            Some(NetworkActivationMode::SoftMaxCrossEntropy) => NetworkActivationMode::Linear,
            _ => NetworkActivationMode::SoftMaxCrossEntropy,
        };
        let probabilities = post_apply_mode.apply(&output);
        Ok(probabilities)
    }

    pub fn predict_next(&self, last_words: &[&str]) -> Result<String> {
        let probabilities = self.sample_probabilities(last_words)?;
        let sampled_idx = self.rng.sample_uniform(&probabilities)?;
        let token = self.query_token(sampled_idx).cloned();

        token.context("sampled vocab word should be in vocab dict")
    }

    pub fn predict_from(&self, last_word: &str) -> Result<String> {
        self.predict_next(&[last_word])
    }

    pub fn nll(&self, context_tokens: &Vec<&str>, expected_next_word: &str) -> Result<NodeValue> {
        let expected_vocab_idx = *self
            .vocab
            .get(&expected_next_word.to_string())
            .context("provided vocab word should be in vocab dict")?;
        let expected_ouput = self.one_hot(expected_vocab_idx);
        let context_vectors = self.get_padded_network_input(context_tokens)?;

        self.nll_vector(context_vectors, expected_ouput.collect())
    }

    pub fn nll_batch(&self, phrases: &Vec<Vec<String>>) -> Result<NodeValue> {
        let testing_pairs = self.into_context_target_pairs(phrases.iter());
        let mut nlls = vec![];

        for (prev, actual) in testing_pairs {
            let nll = self.nll_vector(prev, actual)?;
            nlls.push(nll);
        }

        let nlls = LayerValues::new(nlls);
        Ok(nlls.ave())
    }

    pub fn nll_vector(
        &self,
        context_vectors: LayerValues,
        expected_ouput: LayerValues,
    ) -> Result<NodeValue> {
        let expected_vocab_idx = expected_ouput
            .iter()
            .position(|x| *x == 1.0)
            .context("provided vocab word should be in vocab dict")?;

        let output = self.network.compute(context_vectors)?;
        let probabilities = self.compute_probabilities(&output)?;
        let log_logits = probabilities
            .get(expected_vocab_idx)
            .expect("output should have same count as vocab")
            .ln();

        Ok(-log_logits)
    }

    pub fn generate_sequence_string(&self, token_separator: &str) -> String {
        let sequence: Vec<String> = self.predict_from_iter(&[CONTROL_VOCAB]).collect();
        sequence.join(token_separator)
    }

    pub fn predict_iter<'a>(&'a self, seed_word: &str) -> impl Iterator<Item = String> + 'a {
        self.predict_from_iter(&[seed_word])
    }

    pub fn predict_from_iter<'a>(
        &'a self,
        seed_tokens: &[&str],
    ) -> impl Iterator<Item = String> + 'a {
        let mut token_counts = HashMap::new();
        let (curr_token, seed_tokens) = seed_tokens
            .split_last()
            .expect("should have at lease one element");

        let mut curr_token = curr_token.to_string();
        let mut context_tokens = VecDeque::from_iter(seed_tokens.iter().map(|x| x.to_string()));

        let input_stride_width = self.input_stride_width();
        let max_len = |token: &str| if token.len() > 1 { 4 } else { 40 };

        std::iter::from_fn(move || {
            if curr_token.as_str() == CONTROL_VOCAB {
                None
            } else if token_counts.get(&curr_token).unwrap_or(&0) > &max_len(&curr_token) {
                None
            } else {
                let last_token = curr_token.clone();

                context_tokens.push_back(last_token.clone());
                let skip_count =
                    last_token.len() == 1 && !last_token.chars().next().unwrap().is_alphabetic();

                if !skip_count {
                    *token_counts.entry(last_token.clone()).or_insert(0) += 1;
                }

                let context_tokens_slice = &context_tokens
                    .iter()
                    .map(|a| a.as_str())
                    .collect::<Vec<_>>()[..];

                curr_token = self.predict_next(&context_tokens_slice).ok()?;

                while context_tokens.len() > input_stride_width {
                    context_tokens.pop_front();
                }

                Some(curr_token.clone()).filter(|token| &token != &CONTROL_VOCAB)
            }
        })
    }

    fn get_padded_network_input<T: AsRef<str>>(&self, input_tokens: &[T]) -> Result<LayerValues> {
        let input_stride_width = self.input_stride_width();
        let padded_network_input = self
            .get_padded_network_input_iter(input_tokens)
            .skip(input_tokens.len())
            .take(input_stride_width)
            .collect::<Result<Vec<_>>>()?;

        Ok(padded_network_input.into_iter().flatten().collect())
    }

    fn get_padded_network_input_iter<'a, T: AsRef<str>>(
        &'a self,
        input_tokens: &'a [T],
    ) -> impl Iterator<Item = Result<impl Iterator<Item = f64>>> + 'a {
        iter::repeat(CONTROL_VOCAB)
            .take(self.input_stride_width())
            .chain(input_tokens.iter().map(|x| x.as_ref()))
            .chain([CONTROL_VOCAB].into_iter())
            .map(|vocab_word| {
                self.vocab.get(&vocab_word.to_string()).with_context(|| {
                    format!("failed to resolve vocab from dict: unknown token '{vocab_word}'")
                })
            })
            .map(|vocab_idx| vocab_idx.map(|&vocab_idx| self.one_hot(vocab_idx)))
    }

    pub fn compute_error_batch(&self, phrases: &Vec<Vec<String>>) -> Result<NodeValue> {
        let mut errors: Vec<NodeValue> = vec![];

        let pairs = self.into_context_target_pairs(phrases.iter());
        for (inputs, target_outputs) in pairs {
            let error = self.network.compute_error(inputs, &target_outputs)?;
            errors.push(error.ave());
        }

        let errors = LayerValues::new(errors);
        Ok(errors.ave())
    }

    pub fn compute_error(&self, phrase: &Vec<String>) -> Result<NodeValue> {
        let mut errors: Vec<NodeValue> = vec![];

        let pairs = self.into_context_target_pairs([phrase].into_iter());
        for (inputs, target_outputs) in pairs {
            let error = self.network.compute_error(inputs, &target_outputs)?;
            errors.push(error.ave());
        }

        let errors = LayerValues::new(errors);
        Ok(errors.ave())
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
            .into_iter()
            .into())
    }

    pub fn flatten_embeddings(&self) -> Result<LayerValues> {
        Self::flatten_embeddings_from_network(&self.network)
    }

    pub fn flatten_embeddings_from_network(network: &Network) -> Result<LayerValues> {
        Ok(network
            .layer_weights(0)
            .context("should have valid layer strcture")?
            .into_iter()
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

        let (_, furthest_dot_product) = dot_products
            .iter()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .context("can not sort dot product collection")?;

        let available_range = 1.0 - furthest_dot_product;
        let similarity_fact = (nearest_dot_product - furthest_dot_product) / available_range;

        Ok((nearest_vocab.to_string(), similarity_fact))
    }

    fn last_layer_shape(&self) -> Result<super::LayerShape> {
        let network_shape = &self.network.shape();
        let last_layer_shape = network_shape.iter().last();
        last_layer_shape.context("should have final layer defined")
    }

    fn one_hot(&self, idx: usize) -> impl Iterator<Item = NodeValue> {
        let vocab_len = self.vocab.len();
        (0..vocab_len).map(move |i| if i == idx { 1.0 } else { 0.0 })
    }

    pub fn query_token(&self, token_idx: usize) -> Option<&String> {
        self.vocab
            .iter()
            .find(|(_, vocab_idx)| **vocab_idx == token_idx)
            .map(|(token, _)| token)
    }

    pub fn input_stride_width(&self) -> usize {
        let layer_shape = self.network.shape().iter().nth(1).unwrap();
        layer_shape.stride_count()
    }

    pub fn control_vocab(&self) -> &'static str {
        CONTROL_VOCAB
    }

    pub fn control_vocab_index(&self) -> usize {
        *self
            .vocab
            .get(&CONTROL_VOCAB.to_string())
            .expect("CONTROL_VOCAB should be present in vocab")
    }

    pub fn vocab(&self) -> &HashMap<String, usize> {
        &self.vocab
    }

    pub fn shape(&self) -> &NetworkShape {
        self.network.shape()
    }
}

impl Default for Embedding {
    fn default() -> Self {
        let vocab = Default::default();
        let hidden_layer_shape = vec![];
        let size = 0;
        let input_stride_width = 1;

        Embedding::new(vocab, size, input_stride_width, hidden_layer_shape)
    }
}

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

pub mod builder {
    use std::collections::{HashMap, HashSet};

    use anyhow::{anyhow, Context, Result};
    use serde_json::Value;

    use crate::ml::{
        layer::LayerInitStrategy, LayerShape, LayerValues, Network, NetworkActivationMode,
        NetworkShape,
    };

    use super::{Embedding, RngStrategy, CONTROL_VOCAB};

    #[derive(Clone)]
    pub struct EmbeddingBuilder {
        vocab: HashSet<String>,
        override_vocab: Option<HashMap<String, usize>>,
        override_network: Option<Network>,
        override_embeddings_layer: Option<LayerValues>,
        activation_mode: NetworkActivationMode,
        embedding_dimensions: usize,
        input_stride_width: usize,
        hidden_layer_shape: Vec<LayerShapeConfig>,
        hidden_layer_init_strategy: LayerInitStrategy,
        rng: RngStrategy,
    }

    impl Default for EmbeddingBuilder {
        fn default() -> Self {
            Self {
                vocab: Default::default(),
                override_vocab: None,
                override_network: None,
                override_embeddings_layer: None,
                activation_mode: NetworkActivationMode::Tanh,
                embedding_dimensions: 2,
                input_stride_width: 1,
                hidden_layer_shape: vec![10.into()],
                hidden_layer_init_strategy: LayerInitStrategy::ScaledFullRandom,
                rng: Default::default(),
            }
        }
    }

    impl EmbeddingBuilder {
        pub fn new(vocab: HashSet<String>) -> Self {
            Self {
                vocab,
                ..Default::default()
            }
        }

        pub fn new_ordered(vocab: impl AsRef<[String]>) -> EmbeddingBuilder {
            let override_vocab = [&CONTROL_VOCAB.to_string()]
                .into_iter()
                .chain(vocab.as_ref())
                .enumerate()
                .map(|(i, word)| (word.clone(), i))
                .collect();
            Self {
                override_vocab: Some(override_vocab),
                ..Default::default()
            }
        }

        pub fn from_snapshot(json: &str) -> Result<Self> {
            let value: Value = serde_json::from_str(json)?;

            let network = value
                .get("network")
                .context("provided json should contain network field")?;
            let network = serde_json::from_value(network.clone())?;

            let vocab = value
                .get("vocab")
                .context("provided json should contain vocab field")?;
            let vocab = serde_json::from_value(vocab.clone())?;

            let input_stride_width = Self::input_stride_width(&network);
            let hidden_layer_shape = Self::hidden_layer_shape_from_network(&network);
            let hidden_layer_shape = hidden_layer_shape.map(|shape| shape.into()).collect();

            let default_builder = Self::default();
            let embedding_dimensions = Self::embedding_dimensions(&network);
            let embedding_dimensions =
                embedding_dimensions.unwrap_or(default_builder.embedding_dimensions);

            Ok(Self {
                override_vocab: Some(vocab),
                override_network: Some(network),
                input_stride_width: input_stride_width.unwrap_or(1),
                embedding_dimensions,
                hidden_layer_shape,
                ..default_builder
            })
        }

        pub fn from_existing(embedding: &Embedding) -> Result<Self> {
            let vocab = embedding.vocab.clone();
            let embeddings_layer = embedding.flatten_embeddings()?;
            let input_stride_width = Self::input_stride_width(&embedding.network);
            let hidden_layer_shape = Self::hidden_layer_shape_from_network(&embedding.network);
            let hidden_layer_shape = hidden_layer_shape.map(|shape| shape.into()).collect();
            let rng = embedding.rng.clone();

            let default_builder = Self::default();
            let embedding_dimensions = Self::embedding_dimensions(&embedding.network);
            let embedding_dimensions =
                embedding_dimensions.unwrap_or(default_builder.embedding_dimensions);

            Ok(Self {
                override_vocab: Some(vocab),
                override_embeddings_layer: Some(embeddings_layer),
                input_stride_width: input_stride_width.unwrap_or(1),
                embedding_dimensions,
                hidden_layer_shape,
                rng,
                ..default_builder
            })
        }

        pub fn build(self) -> Result<Embedding> {
            let (embedding, _) = self.build_advanced()?;
            Ok(embedding)
        }

        pub fn build_advanced(self) -> Result<(Embedding, BuilderContext)> {
            let mut context = BuilderContext::default();

            let vocab: HashMap<_, _> = match self.override_vocab {
                Some(vocab) => {
                    context.restored_exact_vocab = true;
                    vocab
                        .get(&CONTROL_VOCAB.to_string())
                        .filter(|&id| *id == 0)
                        .context(format!(
                            "restored vocab must contain control token '{}'",
                            CONTROL_VOCAB
                        ))?;
                    vocab
                }
                None => {
                    self.vocab
                        .get(&CONTROL_VOCAB.to_string())
                        .ok_or(())
                        .err()
                        .context(format!(
                            "provided vocab must not contain control token '{}'",
                            CONTROL_VOCAB
                        ))?;

                    let ordered_vocab = {
                        let mut v = self.vocab.into_iter().collect::<Vec<_>>();
                        v.sort();
                        v
                    };

                    [CONTROL_VOCAB.to_string()]
                        .into_iter()
                        .chain(ordered_vocab.into_iter())
                        .enumerate()
                        .map(|(i, word)| (word, i))
                        .collect()
                }
            };

            let (override_network, override_embeddings_layer) = match self.override_network {
                Some(network)
                    if Self::is_distinct_layer_shape(
                        &self.hidden_layer_shape,
                        self.input_stride_width,
                        &network,
                    ) =>
                {
                    let embeddings_layer = Embedding::flatten_embeddings_from_network(&network);
                    let embeddings_layer =
                        embeddings_layer.context("should get embedding layer")?;
                    context.rebuilt_network = true;

                    (None, Some(embeddings_layer))
                }
                Some(network) => (Some(network), None),
                None => (None, self.override_embeddings_layer),
            };

            let network = match override_network {
                Some(network) => {
                    context.restored_exact_network = true;
                    network
                }
                None => {
                    let embedding_layer_shape = Self::build_embedding_layer_shape(
                        self.embedding_dimensions,
                        self.input_stride_width,
                        override_embeddings_layer,
                    );

                    let network_shape = Self::build_shape(
                        vocab.len(),
                        embedding_layer_shape,
                        self.hidden_layer_shape,
                        self.hidden_layer_init_strategy,
                        self.activation_mode,
                    )?;
                    Network::new(network_shape)
                }
            };

            let embedding = Embedding {
                network,
                vocab,
                rng: self.rng.upgrade(),
            };

            Ok((embedding, context))
        }

        fn build_shape(
            token_count: usize,
            embedding_layer_shape: LayerShape,
            hidden_layer_shape: Vec<LayerShapeConfig>,
            hidden_layer_init_strategy: LayerInitStrategy,
            activation_mode: NetworkActivationMode,
        ) -> Result<NetworkShape> {
            let to_hidden_layer = |n: LayerShapeConfig| -> Result<LayerShape> {
                let shape = (n.dimensions, hidden_layer_init_strategy.clone());
                let shape: LayerShape = shape.into();

                if n.residual {
                    Err(anyhow!("residual_connections not yet supported"))?;
                    Ok(shape.with_residual_connections(true))
                } else {
                    Ok(shape)
                }
            };

            let output_layer = (token_count, LayerInitStrategy::KaimingZeroBias);
            let output_layer: LayerShape = output_layer.into();
            let output_layer_shape =
                output_layer.with_activation_mode(NetworkActivationMode::SoftMaxCrossEntropy);

            let layers_shape = [Ok(embedding_layer_shape)]
                .into_iter()
                .chain(hidden_layer_shape.into_iter().map(to_hidden_layer))
                .chain([Ok(output_layer_shape)])
                .collect::<Result<_>>()?;

            NetworkShape::new_custom(token_count, layers_shape, activation_mode)
        }

        fn build_embedding_layer_shape(
            embedding_size: usize,
            embedding_stride: usize,
            embedding_layer_weights: Option<LayerValues>,
        ) -> LayerShape {
            let init_strategy = match embedding_layer_weights {
                Some(weights) => LayerInitStrategy::NoBiasCopied(weights),
                None => LayerInitStrategy::NoBias,
            };

            LayerShape::new(
                embedding_size,
                init_strategy,
                Some(embedding_stride),
                Some(NetworkActivationMode::Linear),
            )
        }

        pub fn with_rng(mut self, rng: RngStrategy) -> Self {
            self.rng = rng;
            self
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

        pub fn with_hidden_layer_custom_shape<S: Into<LayerShapeConfig>>(
            mut self,
            hidden_layer_shape: Vec<S>,
        ) -> Self {
            self.hidden_layer_shape = hidden_layer_shape.into_iter().map(|x| x.into()).collect();
            self
        }

        pub fn with_hidden_layer_size(mut self, hidden_layer_size: usize) -> Self {
            self.hidden_layer_shape = vec![hidden_layer_size.into()];
            self
        }

        pub fn with_hidden_layer_init_strategy(
            mut self,
            hidden_layer_init_strategy: LayerInitStrategy,
        ) -> Self {
            self.hidden_layer_init_strategy = hidden_layer_init_strategy;
            self
        }

        fn is_distinct_layer_shape(
            self_hidden_layer_shape: &Vec<LayerShapeConfig>,
            input_stride_width: usize,
            network: &Network,
        ) -> bool {
            match Self::input_stride_width(&network) {
                Some(stride_count) if stride_count != input_stride_width => true,
                _ => {
                    let ref_hidden_layer_shape = Self::hidden_layer_shape_from_network(network);
                    let ref_hidden_layer_shape =
                        ref_hidden_layer_shape.take(self_hidden_layer_shape.len());
                    let ref_hidden_layer_shape =
                        ref_hidden_layer_shape.map(|shape| shape.node_count());

                    !ref_hidden_layer_shape.eq(self_hidden_layer_shape.iter().map(|x| x.dimensions))
                }
            }
        }

        fn input_stride_width(network: &Network) -> Option<usize> {
            network.shape().iter().nth(1).map(|x| x.stride_count())
        }

        fn embedding_dimensions(network: &Network) -> Option<usize> {
            network.shape().iter().nth(1).map(|x| x.node_count())
        }

        fn hidden_layer_shape_from_network<'a>(
            network: &'a Network,
        ) -> impl Iterator<Item = LayerShape> + 'a {
            let shape = &network.shape();
            let hidden_layer_len = shape.len() - 1;
            shape.iter().skip(2).take(hidden_layer_len)
        }
    }

    #[derive(Clone, Debug)]
    pub struct LayerShapeConfig {
        dimensions: usize,
        residual: bool,
    }

    impl LayerShapeConfig {
        pub fn with_residual_connections(self) -> Self {
            Self {
                residual: true,
                ..self
            }
        }
    }

    impl From<usize> for LayerShapeConfig {
        fn from(value: usize) -> Self {
            Self {
                dimensions: value,
                residual: false,
            }
        }
    }

    impl From<LayerShape> for LayerShapeConfig {
        fn from(value: LayerShape) -> Self {
            Self {
                dimensions: value.node_count(),
                residual: value.residual_connections(),
            }
        }
    }

    #[derive(Clone, Default, Debug)]
    pub struct BuilderContext {
        pub rebuilt_network: bool,
        pub restored_exact_network: bool,
        pub restored_exact_vocab: bool,
    }
}

#[cfg(test)]
mod tests {
    use test_log::test;
    use tracing::info;

    use self::training::TestEmbeddingConfig;

    use super::*;

    #[test]
    fn embeddings_phrases_can_transform_to_training_pairs() {
        let phrase_str = "one two three four".to_string();
        let phrase: Vec<String> = phrase_str
            .split_whitespace()
            .map(|word| word.to_string())
            .collect();

        let rng = RngStrategy::testable(12345);
        let embedding = Embedding::new_builder(phrase.iter().cloned().collect())
            .with_input_stride_width(2)
            .with_rng(rng)
            .build()
            .unwrap();

        let vocab = embedding.vocab();
        let ctrl_token = embedding.control_vocab();
        let one_hot = |token: &str| embedding.one_hot(vocab[&token.to_string()]);

        let expected_pairs = [
            ([ctrl_token, ctrl_token], "one"),
            ([ctrl_token, "one"], "two"),
            (["one", "two"], "three"),
            (["two", "three"], "four"),
            (["three", "four"], ctrl_token),
        ];

        let expected_context_target_pairs: Vec<(LayerValues, LayerValues)> = expected_pairs
            .iter()
            .map(|(ctx, target)| {
                let ctx = ctx.iter().map(|x| one_hot(&x)).flatten();
                let target = one_hot(&target);
                (ctx.collect(), target.collect())
            })
            .collect();

        let context_target_pairs = embedding.into_context_target_pairs([phrase].iter());

        assert_eq!(context_target_pairs, expected_context_target_pairs);
    }

    #[test]
    fn embeddings_phrases_can_pad_tokens_to_network_input() {
        let vocab: HashSet<String> = ["one", "two", "three", "four"]
            .into_iter()
            .map(|word| word.to_string())
            .collect();

        let rng = RngStrategy::testable(12345);
        let embedding = Embedding::new_builder(vocab)
            .with_input_stride_width(3)
            .with_rng(rng)
            .build()
            .unwrap();

        let vocab = embedding.vocab();
        let ctrl_token = embedding.control_vocab();
        let vocab_idx = |vector: LayerValues| {
            vector
                .chunks_exact(vocab.len())
                .flat_map(|x| x.iter().position(|&z| z == 1.0))
                .flat_map(|idx| embedding.query_token(idx).map(|x| x.as_str()))
                .collect::<Vec<_>>()
        };

        let expected_padded_vectors = &[ctrl_token, ctrl_token, "two"];
        let context_padded_vectors =
            vocab_idx(embedding.get_padded_network_input(&["two"]).unwrap());
        assert_eq!(&context_padded_vectors[..], expected_padded_vectors);

        let expected_padded_vectors = &[ctrl_token, "one", "two"];
        let context_padded_vectors =
            vocab_idx(embedding.get_padded_network_input(&["one", "two"]).unwrap());
        assert_eq!(&context_padded_vectors[..], expected_padded_vectors);

        let expected_padded_vectors = &["one", "two", "three"];
        let context_padded_vectors = vocab_idx(
            embedding
                .get_padded_network_input(&["one", "two", "three"])
                .unwrap(),
        );
        assert_eq!(&context_padded_vectors[..], expected_padded_vectors);

        let expected_padded_vectors = &["two", "three", "four"];
        let context_padded_vectors = vocab_idx(
            embedding
                .get_padded_network_input(&["one", "two", "three", "four"])
                .unwrap(),
        );
        assert_eq!(&context_padded_vectors[..], expected_padded_vectors);
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

        let embedding = training::setup_and_train_embeddings(
            (vocab.clone(), phrases, testing_phrases),
            TestEmbeddingConfig {
                embedding_size: 2,
                hidden_layer_nodes: 30,
                input_stride_width: 3,
                batch_size: 3,
                training_rounds: 100_000,
                train_rate: 1e-2,
                activation_mode: NetworkActivationMode::Tanh,
            },
        );

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

        let embedding = training::setup_and_train_embeddings(
            (vocab, phrases, testing_phrases),
            TestEmbeddingConfig {
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
    }

    #[test]
    #[ignore = "takes too long"]
    fn embeddings_work_simple_prediction() {
        let phrases = [
            "bottle wine bottle wine bottle wine bottle wine",
            "piano violin piano violin piano violin piano violin",
            "pizza pasta pizza pasta pizza pasta pizza pasta",
            "coffee soda coffee soda coffee soda coffee soda",
            "tin can tin can tin can tin can",
        ];

        let mut phrases: Vec<Vec<String>> = phrases
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

        let testing_phrases = training::split_training_and_testing(&mut phrases, 500, 10.0);

        let embedding = training::setup_and_train_embeddings(
            (vocab, phrases, testing_phrases),
            TestEmbeddingConfig {
                embedding_size: 5,
                hidden_layer_nodes: 4,
                training_rounds: 2500,
                input_stride_width: 2,
                train_rate: 1e-1,
                batch_size: 1,
                activation_mode: NetworkActivationMode::Tanh,
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
            input_stride_width: 2,
            train_rate: 1e-3,
            batch_size: 1,
            activation_mode: NetworkActivationMode::Sigmoid,
        };
        let mut embedding = training::setup_and_train_embeddings(
            (vocab, phrases.clone(), phrases.clone()),
            test_embedding_config.clone(),
        );

        let each_layer_weights = (*embedding.embeddings("piano").unwrap()).clone();
        let init_nearest = embedding.nearest("piano");

        embedding
            .train(
                &phrases,
                test_embedding_config.train_rate,
                test_embedding_config.batch_size,
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

    mod training {
        use std::time::{Duration, Instant};

        use super::*;

        #[derive(Clone)]
        pub struct TestEmbeddingConfig {
            pub embedding_size: usize,
            pub hidden_layer_nodes: usize,
            pub training_rounds: usize,
            pub input_stride_width: usize,
            pub batch_size: usize,
            pub train_rate: NodeValue,
            pub activation_mode: NetworkActivationMode,
        }

        pub fn setup_and_train_embeddings(
            (vocab, phrases, testing_phrases): (
                HashSet<String>,
                Vec<Vec<String>>,
                Vec<Vec<String>>,
            ),
            config: TestEmbeddingConfig,
        ) -> Embedding {
            let rng = RngStrategy::testable(1234);
            let mut embedding = Embedding::new_builder(vocab)
                .with_embedding_dimensions(config.embedding_size)
                .with_hidden_layer_size(config.hidden_layer_nodes)
                .with_input_stride_width(config.input_stride_width)
                .with_activation_mode(config.activation_mode)
                .with_rng(rng)
                .build()
                .unwrap();

            let mut seen_pairs = HashSet::new();
            let mut last_report_time: Option<(Instant, usize)> = None;

            for round in 0..config.training_rounds {
                let training_error = embedding
                    .train(&phrases, config.train_rate, config.batch_size)
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

        fn should_report_round(round: usize, training_rounds: usize) -> bool {
            let round_1based = round + 1;

            round_1based <= 3
                || (round_1based <= 100 && round_1based % 10 == 0)
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
                let error = embedding.compute_error(testing_phrase).unwrap();

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

                    let nll = embedding.nll(&last_words, &actual).unwrap();
                    validation_errors.push((error, nll));
                    total_first_word_predictions += 1;
                }
            }

            let predictions_pct = correct_first_word_predictions as NodeValue * 100.0
                / total_first_word_predictions as NodeValue;
            (validation_errors, predictions_pct)
        }

        pub fn split_training_and_testing(
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
