use std::{
    collections::{HashMap, HashSet},
    fs::File,
    sync::Arc,
    time::{Duration, Instant},
};

use plane::ml::{
    embeddings::{Embedding, TrainBatchConfig},
    gdt::{self, GenerativeDecoderTransformer},
    seq2seq::transformer::CharacterTransformer,
    transformer::decoder::builder::DecoderBuilder,
    NodeValue, RngStrategy, RNG,
};

use anyhow::Result;
use itertools::Itertools;
use serde_json::Value;
use tracing::{debug, error, info};

use crate::{
    bounded::BoundedValueLogger,
    config::TrainEmbeddingConfig,
    messages::{
        TrainerHandle, TrainerHandleSender, TrainerMessage, TrainerReport, TrainerStateMetadata,
        YieldState,
    },
    model::MLModel,
};

pub struct TrainerState<M> {
    pub model: M,
    pub handle_tx: TrainerHandleSender<TrainerMessage>,
    pub output_label: Option<String>,
    pub training_rounds: usize,
    pub print_round_number: bool,
    pub supress_auto_report: bool,
    pub force_report_all: bool,
    pub sample_from_pattern: Option<String>,
    pub snapshot: (Option<String>, Instant),
    pub last_report: Option<(Instant, TrainerReport)>,
    paused: bool,
    batches_trained: usize,
    pub round: usize,
    learn_rate: NodeValue,
    training_loss_logger: BoundedValueLogger<(usize, NodeValue)>,
    rounds_until_pause_enabled: usize,
    halt: bool,
    force_report_test_set: bool,
    force_report_train_set: bool,
    round_reported: bool,
    last_autosave: Instant,
    autosave_interval: Duration,
    total_train_time: Duration,
    inital_config: TrainEmbeddingConfig,
}

impl<M: MLModel> TrainerState<M> {
    fn new(
        model: M,
        config: &TrainEmbeddingConfig,
        handle_tx: &TrainerHandleSender<TrainerMessage>,
    ) -> Self {
        Self {
            model,
            handle_tx: handle_tx.clone(),
            inital_config: config.clone(),
            output_label: config.output_label.clone(),
            learn_rate: config.train_rate,
            training_rounds: config.training_rounds,
            paused: config.pause_on_start,
            sample_from_pattern: config.sample_from_pattern.clone(),
            snapshot: (None, Instant::now()),
            training_loss_logger: BoundedValueLogger::new(1000),
            round_reported: false,
            print_round_number: false,
            supress_auto_report: false,
            force_report_test_set: false,
            force_report_train_set: false,
            force_report_all: false,
            last_report: None,
            rounds_until_pause_enabled: 0,
            halt: false,
            last_autosave: Instant::now(),
            autosave_interval: Duration::from_secs(config.autosave_interval_mins * 60),
            total_train_time: Duration::ZERO,
            batches_trained: 0,
            round: 0,
        }
    }

    fn train(
        &mut self,
        phrases: &Vec<Vec<String>>,
        batch_size: TrainBatchConfig,
    ) -> Option<NodeValue> {
        if self.training_paused() {
            return None;
        }
        let started = Instant::now();

        let train_error = self.model.train(phrases, batch_size);

        let train_duration = started.elapsed();
        self.total_train_time += train_duration;
        self.batches_trained += self.inital_config.batch_size;

        self.try_record_train_iteration(train_error)
    }

    fn try_record_train_iteration(&mut self, train_error: Result<NodeValue>) -> Option<NodeValue> {
        match train_error {
            Ok(error) if error.is_finite() => {
                if self.should_perform_autosave() {
                    self.perform_snapshot();
                }
                self.training_loss_logger.push((self.round, error));
                Some(error)
            }
            Ok(non_finite_error) => {
                error!(
                    "Failed perform embedding model training iteration: training loss = {}",
                    non_finite_error
                );
                if let (Some(_), when) = self.snapshot {
                    info!("Snapshot available from {when:?}. See help menu ('h') to restore... pausing")
                }
                self.pause();
                None
            }
            Err(e) => {
                error!("Failed to train embedding model {e}");
                if let (Some(_), when) = self.snapshot {
                    info!("Snapshot available from {when:?}. See help menu ('h') to restore... pausing")
                }
                self.pause();
                None
            }
        }
    }

    fn complete_round(&mut self) {
        self.force_report_test_set = false;
        self.force_report_train_set = false;
        self.round_reported = false;

        if self.rounds_until_pause_enabled > 0 {
            self.rounds_until_pause_enabled -= 1;

            if self.rounds_until_pause_enabled == 0 && self.paused {
                info!(
                    "Queued rounds completed (round = {}), pausing...",
                    self.round + 1
                );
            }
        }

        let quit_on_complete = self.inital_config.quit_on_complete;
        let complete_entire_run = self.round == self.training_rounds - 1
            || (quit_on_complete && self.round >= self.training_rounds);

        if complete_entire_run && !self.paused {
            if self.inital_config.quit_on_complete {
                info!("Completed rounds run, exiting...");
                self.halt();
            } else {
                info!("Completed rounds run, pausing...");
            }

            self.trigger_round_report_test_set();
            self.pause();
            return;
        }

        if !self.paused && self.last_autosave.elapsed() > self.autosave_interval {
            let metadata = self.metadata();
            let msg = TrainerMessage::WriteModelAndMetadataToDisk(
                metadata,
                crate::messages::TrainerModelCheckpointSource::Autosave,
            );
            match self.handle_tx.send(msg) {
                Ok(_) => self.last_autosave = Instant::now(),
                Err(_) => {}
            };
        }

        if !self.training_paused() {
            self.round += 1;
        }
        _ = self.handle_tx.send(TrainerMessage::Yield(YieldState {
            round: self.round + 1,
        }));
    }

    pub fn pause(&mut self) -> bool {
        let was_paused = self.paused;

        self.paused = true;
        self.rounds_until_pause_enabled = 0;

        was_paused != self.paused
    }

    pub fn unpause(&mut self) -> bool {
        let was_paused = self.paused;

        self.paused = false;
        self.rounds_until_pause_enabled = 0;

        was_paused != self.paused
    }

    pub fn defer_pause(&mut self, rounds: usize) {
        self.paused = true;

        let rounds_until_pause_enabled = self.rounds_until_pause_enabled.max(1);
        self.rounds_until_pause_enabled = rounds_until_pause_enabled + rounds;
    }

    pub fn training_paused(&self) -> bool {
        self.paused && self.rounds_until_pause_enabled == 0
    }

    fn perform_snapshot(&mut self) {
        let json = self.model.snapshot().unwrap();
        self.snapshot = (Some(json), Instant::now());
    }

    fn should_perform_autosave(&self) -> bool {
        !self.paused
            && self.snapshot.1.elapsed().as_secs() > self.inital_config.snapshot_interval_secs
    }

    fn should_report_round(&mut self) -> bool {
        let force_report_all = self.force_report_all && !self.paused;
        let should_report = if force_report_all || self.force_report_test_set {
            true
        } else {
            !self.paused
                && !self.supress_auto_report
                && validate::should_report_round(self.round, self.training_rounds)
        };

        if should_report {
            self.round_reported = true;
        }

        should_report
    }

    fn should_report_test_set_round(&mut self) -> bool {
        let should_report = self.force_report_train_set;
        if should_report {
            self.round_reported = true;
        }
        should_report
    }

    fn should_report_round_number_only(&mut self) -> bool {
        let should_report = self.print_round_number && !self.paused && !self.round_reported;
        if should_report {
            self.round_reported = true;
        }
        should_report
    }

    pub fn metadata(&self) -> TrainerStateMetadata {
        TrainerStateMetadata {
            learn_rate: self.learn_rate,
            training_rounds: self.training_rounds,
            current_round: self.round,
            output_label: self.output_label.clone(),
            total_train_seconds: self.total_train_time.as_secs(),
            training_report: self.last_report.as_ref().map(|(_, report)| report.clone()),
            training_error_history: self.training_loss_logger.iter().cloned().collect(),
        }
    }

    pub fn input_stride_width(&self) -> usize {
        self.inital_config.input_stride_width
    }

    pub fn batch_size(&self) -> TrainBatchConfig {
        if self.inital_config.single_batch_iterations {
            TrainBatchConfig::SingleBatch(self.inital_config.batch_size)
        } else {
            TrainBatchConfig::Batches(self.inital_config.batch_size)
        }
    }

    pub fn hidden_layer_shape(&self) -> Vec<usize> {
        Self::build_hidden_layer_shape(&self.inital_config)
    }

    pub fn set_model(&mut self, model: M, metadata: TrainerStateMetadata) {
        self.model = model;
        self.round = metadata.current_round;
        self.training_rounds = metadata.training_rounds;
        self.output_label = metadata.output_label;
        self.total_train_time = Duration::from_secs(metadata.total_train_seconds);

        self.learn_rate = metadata.learn_rate;
        self.model.set_learn_rate(metadata.learn_rate);

        self.training_loss_logger = {
            let logger_capacity = self.training_loss_logger.capacity();
            let mut logger: BoundedValueLogger<(usize, NodeValue)> = metadata
                .training_error_history
                .into_iter()
                .map(|(round, loss)| (round, (round, loss)))
                .collect();
            logger.set_capacity(logger_capacity);
            logger
        };

        self.last_report = None;
    }

    pub fn set_output_label(&mut self, output_label: String) {
        self.output_label = Some(output_label);
    }

    pub fn trigger_round_report_test_set(&mut self) {
        self.force_report_test_set = true;
    }

    pub fn trigger_round_report_train_set(&mut self) {
        self.force_report_train_set = true;
    }

    pub fn halt(&mut self) {
        let quit_on_complete = self.inital_config.quit_on_complete;
        let run_completed = (self.round + 1) >= self.training_rounds;

        if quit_on_complete && run_completed && !self.halt {
            let metadata = self.metadata();
            let msg = TrainerMessage::WriteModelAndMetadataToDisk(
                metadata,
                crate::messages::TrainerModelCheckpointSource::User,
            );

            self.handle_tx
                .send(msg)
                .expect("failed to invoke model write to disk");

            info!("Triggered action: save model to file");
        }

        self.halt = true;
    }

    pub fn batches_trained_since_process_started(&self) -> usize {
        self.batches_trained
    }

    pub fn learn_rate(&self) -> NodeValue {
        self.learn_rate
    }

    pub fn set_learn_rate(&mut self, learn_rate: NodeValue) {
        self.model.set_learn_rate(learn_rate);
        self.learn_rate = learn_rate;
    }
}

impl<M> TrainerState<M> {
    pub fn build_hidden_layer_shape(config: &TrainEmbeddingConfig) -> Vec<usize> {
        let layer_node_counts = [config.hidden_layer_nodes];
        let layer_node_counts = layer_node_counts.into_iter().chain(
            config
                .hidden_deep_layer_nodes
                .iter()
                .flat_map(|csv| csv.split(','))
                .filter_map(|x| x.trim().parse::<usize>().ok()),
        );

        layer_node_counts.collect()
    }

    pub fn join_phrases(phrases: &Vec<Vec<String>>, char_tokens: Option<bool>) -> String {
        let char_tokens =
            char_tokens.unwrap_or_else(|| phrases.first().unwrap().iter().all(|x| x.len() <= 1));
        {
            use std::fmt::Write;

            if char_tokens {
                let vec = phrases
                    .iter()
                    .flat_map(|phrase| phrase.iter().flat_map(|p| p.chars().next()).chain(['\n']))
                    .collect_vec();

                vec.into_iter().collect()
            } else {
                let ref mut iter = phrases
                    .iter()
                    .flat_map(|phrase| {
                        let separator = " ";
                        Itertools::intersperse(phrase.iter().map(|p| p.as_str()), separator)
                            .chain(["\n"])
                    })
                    .filter(|x| x.len() > 0);

                match iter.next() {
                    None => String::new(),
                    Some(first_elt) => {
                        // estimate lower bound of capacity needed
                        let (lower, _) = iter.size_hint();
                        let mut result = String::with_capacity(lower);
                        write!(&mut result, "{}", first_elt).unwrap();
                        iter.for_each(|elt| {
                            write!(&mut result, "{}", elt).unwrap();
                        });
                        result
                    }
                }
            }
        }
    }
}

pub fn setup_and_train_model_v2<M: MLModel>(
    config: TrainEmbeddingConfig,
    handle: TrainerHandle<TrainerMessage, M>,
) -> M {
    let (phrases, testing_phrases, mut state) = init_model_state(config, &handle);
    let batch_size = state.batch_size();

    loop {
        let training_error = state.train(&phrases, batch_size);
        let recv_timeout = state.training_paused().then_some(Duration::from_secs(1));

        while let Ok(msg) = handle.try_recv(recv_timeout) {
            let action = handle.run(&state.model, msg);
            match action.apply(&mut state, &handle) {
                std::ops::ControlFlow::Continue(()) => continue,
                std::ops::ControlFlow::Break(()) => break,
            }
        }

        if state.should_report_round() {
            report_round(&mut state, &testing_phrases, training_error, None);
        }

        if state.should_report_test_set_round() {
            report_round(&mut state, &phrases, training_error, Some("train"));
        }

        if state.should_report_round_number_only() {
            let training_error = training_error
                .map(|loss| format!("(train_loss {loss:<12.10})"))
                .unwrap_or_default();

            info!("Round {} completed {}", state.round + 1, training_error);
        }

        if state.halt {
            info!("Stopping rounds run...");
            break;
        }

        state.complete_round();
    }

    state.model
}

fn init_model_state<M>(
    config: TrainEmbeddingConfig,
    handle: &TrainerHandle<TrainerMessage, M>,
) -> (Vec<Vec<String>>, Vec<Vec<String>>, TrainerState<M>)
where
    M: MLModel,
{
    info!("Initialising vocab and train/test sets");
    let phrase_split_seed = config.phrase_split_seed.unwrap_or_default();
    let phrase_rng: RngStrategy = if phrase_split_seed >= 0 {
        RngStrategy::testable(phrase_split_seed as u32)
    } else {
        RngStrategy::default()
    };
    let (mut phrases, mut vocab) = init_phrases_and_vocab(&config, phrase_rng.to_arc());
    let mut testing_phrases = match config.phrase_test_set_split_pct.filter(|x| *x > 0.0) {
        Some(pct) => split_training_and_testing(
            &mut phrases,
            pct,
            config.phrase_test_set_max_tokens,
            config.sample_from_pattern.as_deref(),
        ),
        None => phrases.clone(),
    };
    let rng = RngStrategy::default();

    let (char_vocab, str_vocab, gdt_vocab) = if config.use_character_tokens && !config.use_gdt {
        info!("Transforming vocab and train/test sets to character-level tokens");
        phrases = characterize_phrases(phrases);
        testing_phrases = characterize_phrases(testing_phrases);
        vocab = compute_vocab(&phrases, Some(&testing_phrases));
        let char_vocab = vocab
            .into_iter()
            .map(|c| c.chars().next().unwrap())
            .chain(['\n'])
            .collect::<Vec<_>>();

        (Some(char_vocab), None, None)
    } else if config.use_gdt {
        let token_type = if config.gdt_bpe_enable {
            gdt::token::VocabTokenType::BPE
        } else if config.gdt_word_mode {
            gdt::token::VocabTokenType::Word
        } else {
            gdt::token::VocabTokenType::Char
        };
        info!("Building GDT vocab (type = {token_type:?})...");
        let vocab_builder = gdt::token::Vocab::new_builder(token_type)
            .with_max_vocab_size(config.gdt_bpe_vocab_size);

        let corpus = phrases.iter().chain(testing_phrases.iter()).fold(
            String::with_capacity(1 >> 24),
            |mut str, x| {
                x.iter().for_each(|x| {
                    str.push_str(&x);
                    str.push(' ')
                });
                str.push('\n');
                str
            },
        );

        let vocab_builder = vocab_builder.from_corpus(&corpus);

        let gdt_vocab = vocab_builder.with_char_fallback().build().unwrap();

        (None, None, Some(gdt_vocab))
    } else {
        let str_vocab = vocab.into_iter().collect::<Vec<_>>();

        (None, Some(str_vocab), None)
    };

    let token_count = |x: &Vec<Vec<String>>| x.iter().map(|phrase| phrase.len()).sum::<usize>();
    let char_token_len = char_vocab.as_ref().map(|x| x.len());
    let str_token_len = str_vocab.as_ref().map(|x| x.len());
    let token_len = char_token_len.or(str_token_len).unwrap_or_default();
    info!(
        "Completed token initialisation: (vocab size = {} tokens)",
        token_len
    );
    info!("Loaded: [ tokens(train_set) = {}, sequences(train_set) = {}, tokens(test_set) = {}, sequences(test_set) = {} ]",
        token_count(&phrases), phrases.len(),
        token_count(&testing_phrases), testing_phrases.len()
    );

    let model: M = if let (true, Some(vocab)) = (
        config.use_character_tokens && config.use_transformer && !config.use_gdt,
        &char_vocab,
    ) {
        info!("Using CharacterTransformer (S2S)...");
        let builder = DecoderBuilder::new(
            config.input_stride_width,
            config.embedding_size,
            vocab.len(),
        );
        let builder = builder
            .with_dropout_rate(0.0)
            .with_feed_forward_hidden_dimension(config.hidden_layer_nodes);

        let hidden_deep_layer_nodes = config
            .hidden_deep_layer_nodes
            .as_ref()
            .map(|x| x.split(',').collect_vec());

        let builder = if let Some(block_count) = hidden_deep_layer_nodes
            .as_ref()
            .and_then(|x| x.get(0))
            .and_then(|x| x.parse::<usize>().ok())
        {
            builder.with_block_count(block_count)
        } else {
            builder
        };

        let builder = if let Some(head_count) = hidden_deep_layer_nodes
            .as_ref()
            .and_then(|x| x.get(1))
            .and_then(|x| x.parse::<usize>().ok())
        {
            builder.with_head_count(head_count)
        } else {
            builder
        };

        let s2s = CharacterTransformer::from_builder_ordered(builder, &vocab, rng).unwrap();
        M::from_s2s(s2s, config.train_rate)
    } else if let (true, Some(vocab)) = (config.use_gdt, gdt_vocab) {
        info!("Using GenerativeDecoderTransformer (GDT)...");
        let builder = DecoderBuilder::new(
            config.input_stride_width,
            config.embedding_size,
            vocab.len(),
        );
        let builder = builder
            // .with_rng(rng.clone()) // TODO: inject rng instance here?
            .with_dropout_rate(0.0)
            .with_feed_forward_hidden_dimension(config.hidden_layer_nodes);

        let hidden_deep_layer_nodes = config
            .hidden_deep_layer_nodes
            .as_ref()
            .map(|x| x.split(',').collect_vec());

        let builder = if let Some(block_count) = hidden_deep_layer_nodes
            .as_ref()
            .and_then(|x| x.get(0))
            .and_then(|x| x.parse::<usize>().ok())
        {
            builder.with_block_count(block_count)
        } else {
            builder
        };

        let builder = if let Some(head_count) = hidden_deep_layer_nodes
            .as_ref()
            .and_then(|x| x.get(1))
            .and_then(|x| x.parse::<usize>().ok())
        {
            builder.with_head_count(head_count)
        } else {
            builder
        };

        let gdt = GenerativeDecoderTransformer::from_builder(builder, vocab, rng).unwrap();

        M::from_gdt(gdt, config.train_rate, config.sample_from_pattern.clone())
    } else if let Some(vocab) =
        str_vocab.or(char_vocab.map(|x| x.into_iter().map(|c| c.to_string()).collect()))
    {
        info!("Using Embedding...");
        let hidden_layer_shape = TrainerState::<()>::build_hidden_layer_shape(&config);

        let embedding = Embedding::new_builder_ordered(&vocab)
            .with_embedding_dimensions(config.embedding_size)
            .with_hidden_layer_custom_shape(hidden_layer_shape)
            .with_input_stride_width(config.input_stride_width)
            .with_activation_mode(config.activation_mode.into())
            .with_rng(rng)
            .build()
            .unwrap();

        M::from_embedding(embedding, config.train_rate)
    } else {
        panic!("invalid config build state");
    };
    let state = TrainerState::new(model, &config, &handle.tx);

    (phrases, testing_phrases, state)
}

fn report_round<M: MLModel>(
    state: &mut TrainerState<M>,
    testing_phrases: &Vec<Vec<String>>,
    training_error: Option<NodeValue>,
    label: Option<&str>,
) {
    let validation_results = if let Some(embedding) = state.model.as_embedding() {
        validate::validate_embeddings(embedding, testing_phrases)
    } else if let Some(s2s) = state.model.as_s2s() {
        let testing_phrases = TrainerState::<M>::join_phrases(&testing_phrases, Some(true)); // TODO: remove from hot path
        Ok(validate::validate_s2s(s2s, &testing_phrases))
    } else if let Some(gdt) = state.model.as_gdt() {
        let char_mode = gdt.vocab().token_type().is_char();
        let testing_phrases = TrainerState::<M>::join_phrases(&testing_phrases, Some(char_mode)); // TODO: remove from hot path
        validate::validate_gdt(gdt, &testing_phrases)
    } else {
        panic!("unknown model type: unable to run validation")
    };

    match validation_results {
        Ok(validation_results) => {
            let report = validate::generate_training_report(
                state.round,
                training_error.unwrap_or(0.0),
                validation_results,
                state
                    .last_report
                    .as_ref()
                    .map(|(last_dt, report)| (last_dt.elapsed(), report.round)),
                label,
            );

            validate::log_training_round(&report);

            state.last_report = Some((std::time::Instant::now(), report));
        }
        Err(e) => {
            error!(
                "Failed to validate testing round {}: '{e}'.. pausing",
                state.round + 1
            );
            state.pause();
        }
    }
}

type DeterministicHashSet<T> =
    HashSet<T, std::hash::BuildHasherDefault<std::collections::hash_map::DefaultHasher>>;

pub fn init_phrases_and_vocab(
    config: &TrainEmbeddingConfig,
    rng: Arc<dyn RNG>,
) -> (Vec<Vec<String>>, DeterministicHashSet<String>) {
    let phrases = parse_phrases(&config);

    let (min_len, max_len) = config.phrase_word_length_bounds;

    let mut phrases: Vec<Vec<String>> = phrases
        .into_iter()
        .filter(|phrase| {
            min_len.map_or(true, |min_len| phrase.len() > min_len)
                && max_len.map_or(true, |max_len| phrase.len() < max_len)
        })
        .collect();

    let vocab_counts = phrases
        .iter()
        .flat_map(|phrase| phrase.iter())
        .cloned()
        .fold(HashMap::new(), |mut counts, word| {
            if &word != " " {
                counts.entry(word).and_modify(|x| *x += 1).or_insert(1_u64);
            }
            counts
        });

    if !config.phrase_disable_shuffle {
        phrases.sort_by_cached_key(|phrase| {
            phrase
                .iter()
                .map(|word| -(vocab_counts.get(word).unwrap_or(&1).pow(2) as i128))
                .sum::<i128>()
        });
    }

    if let Some(phrase_train_set_size) = config.phrase_train_set_size {
        phrases.truncate(phrase_train_set_size);
    }

    debug!(
        "First 3 phrases: {:#?}",
        phrases
            .iter()
            .take(3)
            .map(|phrase| (
                phrase.join(" "),
                phrase
                    .iter()
                    .map(|word| vocab_counts[word].pow(2))
                    .join("|")
            ))
            .collect::<Vec<_>>()
    );

    let vocab = compute_vocab(&phrases, None);

    if !config.phrase_disable_shuffle {
        plane::ml::ShuffleRng::shuffle_vec(&rng, &mut phrases);
    }

    // TODO: cleanup
    Some(&phrases)
        .filter(|x| x.is_empty())
        .ok_or(())
        .expect_err("phrases is empty");

    (phrases, vocab)
}

fn compute_vocab(
    phrases: &Vec<Vec<String>>,
    testing_phrases: Option<&Vec<Vec<String>>>,
) -> DeterministicHashSet<String> {
    phrases
        .iter()
        .chain(testing_phrases.into_iter().flatten())
        .flat_map(|phrase| {
            phrase
                .join(" ")
                .chars()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn characterize_phrases(phrases: Vec<Vec<String>>) -> Vec<Vec<String>> {
    let separator = " ".to_string();
    phrases
        .into_iter()
        .map(|phrase| {
            Itertools::intersperse(phrase.into_iter(), separator.clone())
                .flat_map(|word| word.chars().map(|c| c.to_string()).collect_vec())
                .collect_vec()
        })
        .collect_vec()
}

pub fn parse_phrases(config: &TrainEmbeddingConfig) -> Vec<Vec<String>> {
    match &config.input_txt_path {
        Some(path) => read_plaintext_phrases(&path),
        None => {
            use std::io::Read;

            // TODO: remove fallback when old config files archived
            let mut file =
                File::open("res/phrase_list.json").expect("Failed to open fallback train set file"); // TODO: error handling
            let mut phrase_json = String::new();
            file.read_to_string(&mut phrase_json).unwrap();
            let phrase_json: Value = serde_json::from_str(&phrase_json).unwrap();

            phrase_json
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
                .collect()
        }
    }
}

pub fn read_plaintext_phrases(file_path: &str) -> Vec<Vec<String>> {
    use std::io::Read;

    let mut file = File::open(file_path).unwrap(); // TODO: error handling
    let mut buffer = String::new();
    file.read_to_string(&mut buffer).unwrap();

    let phrases: Vec<Vec<String>> = buffer
        .lines()
        .map(|line| {
            line.split_ascii_whitespace()
                .into_iter()
                .map(|value| value.to_owned())
                .collect()
        })
        .collect();
    phrases
}

pub fn split_training_and_testing(
    phrases: &mut Vec<Vec<String>>,
    test_phrases_pct: f64,
    test_set_max_tokens: Option<usize>,
    _sample_from_pattern: Option<&str>, // TODO: use for sampling test phrases
) -> Vec<Vec<String>> {
    let testing_sample_count = {
        let testing_ratio = test_phrases_pct as NodeValue / 100.0;
        (phrases.len() as NodeValue * testing_ratio) as usize
    };
    let testing_sample_count = match test_set_max_tokens {
        Some(max_words) => {
            let mut total_word_count = 0;
            let phrases_within_limits = phrases
                .iter()
                .rev()
                .take_while(|phrase| {
                    let was_below_limit = total_word_count < max_words;
                    total_word_count += phrase.len();
                    was_below_limit
                })
                .count();

            testing_sample_count.min(phrases_within_limits)
        }
        None => testing_sample_count,
    };
    let offset = phrases.len() - testing_sample_count;
    let testing_phrases = phrases.split_off(offset.clamp(0, phrases.len() - 1));
    testing_phrases
}

mod validate {
    use std::time::Duration;

    use anyhow::{Context, Result};
    use tracing::{error, info};

    use plane::ml::{
        embeddings::Embedding, gdt::GenerativeDecoderTransformer,
        seq2seq::transformer::CharacterTransformer, NodeValue,
    };

    use crate::messages::TrainerReport;

    pub struct TestSetValidation {
        validation_errors: Vec<(NodeValue, NodeValue)>,
        predictions_pct: NodeValue,
    }

    pub fn generate_training_report(
        round: usize,
        training_error: NodeValue,
        validation_results: TestSetValidation,
        last_report_time: Option<(Duration, usize)>,
        label: Option<&str>,
    ) -> TrainerReport {
        let label = label.map(|x| x.to_string());
        let val_count = validation_results.validation_errors.len() as NodeValue;
        let (validation_error, nll) = validation_results.validation_errors.iter().fold(
            (0.0, 0.0),
            |sum, (validation_error, nll)| {
                (
                    sum.0 + (validation_error / val_count),
                    sum.1 + (nll / val_count),
                )
            },
        );

        let ms_per_round = last_report_time.map(|(duration, last_round)| {
            duration.as_millis() / (round - last_round).max(1) as u128
        });
        let generated_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis();
        let predictions_pct = validation_results.predictions_pct;

        TrainerReport {
            round,
            training_error,
            ms_per_round,
            predictions_pct,
            validation_error,
            nll,
            label,
            generated_time,
        }
    }

    pub fn log_training_round(report: &TrainerReport) {
        let ms_per_round = report
            .ms_per_round
            .map(|ms_per_round| format!("(ms/round={ms_per_round:>4.1})"))
            .unwrap_or_default();

        let label = report
            .label
            .as_ref()
            .map(|label| format!("[{label}] "))
            .unwrap_or_default();

        info!(
                "{label}round = {:<6} |  train_loss = {:<12.10}, val_pred_acc: {:0>4.1}%, val_loss = {:<2.6e}, val_nll = {:<6.3} {ms_per_round}",
                report.round + 1, report.training_error, report.predictions_pct, report.validation_error, report.nll
        );
    }

    pub fn validate_embeddings(
        embedding: &Embedding,
        testing_phrases: &Vec<Vec<String>>,
    ) -> Result<TestSetValidation> {
        let mut validation_errors = vec![];
        let mut correct_first_word_predictions = 0;
        let mut total_first_word_predictions = 0;

        // TODO: take batch_size num of phrases or of training pairs randomly to validate..
        // once perf hit is ngligable, run on each train iter and save/plot data
        for testing_phrase in testing_phrases.iter() {
            for testing_phrase_window in testing_phrase.windows(embedding.input_stride_width() + 1)
            {
                let (actual, context_word_vectors) = testing_phrase_window
                    .split_last()
                    .context("should have last element")?;

                let context_word_vectors = context_word_vectors
                    .into_iter()
                    .map(|x| x.as_str())
                    .collect::<Vec<_>>();

                let predicted = embedding.predict_next(&context_word_vectors[..])?;
                if &predicted == actual {
                    correct_first_word_predictions += 1;
                }

                total_first_word_predictions += 1;
            }

            let error = embedding.compute_error(testing_phrase)?;
            let nll = embedding.nll_batch(&vec![testing_phrase.iter().cloned().collect()])?;
            validation_errors.push((error, nll));
        }

        let predictions_pct = correct_first_word_predictions as NodeValue * 100.0
            / total_first_word_predictions.max(1) as NodeValue;

        Ok(TestSetValidation {
            validation_errors,
            predictions_pct,
        })
    }

    pub fn validate_s2s(
        transformer: &CharacterTransformer,
        testing_phrases: &str,
    ) -> TestSetValidation {
        use itertools::Itertools;

        let mut validation_errors = vec![];
        let mut correct_first_word_predictions = 0;
        let mut total_first_word_predictions = 0;

        for testing_phrase_window in testing_phrases
            .chars()
            .chunks(transformer.input_stride_width() + 1)
            .into_iter()
        {
            let testing_phrase_window = testing_phrase_window.collect_vec();
            let (&last_token, context_tokens) = testing_phrase_window.split_last().unwrap();

            let predicted = transformer.predict_next(&context_tokens).unwrap();
            let actual = last_token;

            if predicted == actual {
                correct_first_word_predictions += 1;
            }

            let error = transformer
                .compute_error(&[context_tokens, &[actual]].concat())
                .unwrap();
            let nll = transformer.nll(&context_tokens, actual).unwrap();
            validation_errors.push((error, nll));
            total_first_word_predictions += 1;
        }

        let predictions_pct = correct_first_word_predictions as NodeValue * 100.0
            / total_first_word_predictions as NodeValue;

        TestSetValidation {
            validation_errors,
            predictions_pct,
        }
    }

    pub fn validate_gdt(
        transformer: &GenerativeDecoderTransformer,
        testing_phrases: &str,
    ) -> Result<TestSetValidation> {
        let mut validation_errors = vec![];
        let mut correct_first_word_predictions = 0;
        let mut total_first_word_predictions = 0;
        let testing_tokens = transformer.vocab().encode(testing_phrases)?;

        let input_stride_width = transformer.input_stride_width();
        for testing_phrase_window in testing_tokens.chunks(input_stride_width + 1).into_iter() {
            let (last_token, context_tokens) = testing_phrase_window.split_last().unwrap();

            let predicted = match transformer.predict_arg_max(&context_tokens) {
                Ok((predicted, _prob)) => predicted,
                Err(e) => {
                    error!("failed prediction: {e}");
                    continue;
                }
            };
            let actual = last_token;

            if &predicted == actual {
                correct_first_word_predictions += 1;
            }

            let error = transformer.compute_error(&[context_tokens, &[actual.clone()]].concat())?;
            let nll = transformer.nll(&context_tokens, actual.clone())?;
            validation_errors.push((error, nll));
            total_first_word_predictions += 1;
        }

        let predictions_pct = correct_first_word_predictions as NodeValue * 100.0
            / total_first_word_predictions as NodeValue;

        Ok(TestSetValidation {
            validation_errors,
            predictions_pct,
        })
    }

    pub fn should_report_round(round: usize, training_rounds: usize) -> bool {
        let round_1based = round + 1;

        round_1based <= 3
            || (round_1based <= 1000 && round_1based % 100 == 0)
            || (round_1based <= 10000 && round_1based % 1000 == 0)
            || (round_1based <= 100000 && round_1based % 10000 == 0)
            || (round_1based <= 1000000 && round_1based % 100000 == 0)
            || round_1based == training_rounds
    }
}

pub mod writer {
    use std::{
        collections::{HashMap, HashSet},
        fs::File,
        io::{BufReader, Write},
        path::PathBuf,
    };

    use anyhow::{Context, Result};
    use itertools::Itertools;
    use serde_json::Value;

    use plane::ml::embeddings::Embedding;
    use tracing::info;

    use crate::{config::TrainEmbeddingConfig, messages::TrainerStateMetadata, model::MLModel};

    #[cfg(not(feature = "plot"))]
    pub fn plot_training_loss(config: &&TrainEmbeddingConfig, metadata: &TrainerStateMetadata) {}

    #[cfg(feature = "plot")]
    pub fn plot_training_loss(config: &&TrainEmbeddingConfig, metadata: &TrainerStateMetadata) {
        use plotly::{Plot, Scatter};

        let x_points = metadata
            .training_error_history
            .iter()
            .map(|x| x.0 + 1)
            .collect();

        let y_points = metadata
            .training_error_history
            .iter()
            .map(|x| x.1)
            .collect();

        let mut plot = Plot::new();

        let trace = Scatter::new(x_points, y_points);
        plot.add_trace(trace.name("train_loss"));

        let title = "untitled";
        let title = format!(
            "Training Progess [{}]",
            config.output_label.as_ref().unwrap_or(&title.to_string())
        );
        let title = plotly::common::Title::new(&title);

        plot.set_layout(plot.layout().clone().title(title));

        #[cfg(feature = "cli")]
        plot.show();
    }

    pub fn read_model_from_disk(
        file_path: &str,
    ) -> Result<(String, TrainEmbeddingConfig, TrainerStateMetadata)> {
        info!("Loading model from path: {file_path}");
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);

        let mut snapshot: Value = serde_json::from_reader(reader)?;

        let config: TrainEmbeddingConfig =
            serde_json::from_value(snapshot["_trainer_config"].take())
                .context("unable to extract trainer config from saved model")?;

        let state: TrainerStateMetadata = serde_json::from_value(snapshot["_trainer_state"].take())
            .context("unable to extract trainer metatdata from saved model")?;

        let snapshot = serde_json::to_string(&snapshot)?;

        info!("Extracted model configuration and training state from file");
        Ok((snapshot, config, state))
    }

    pub fn write_model_to_disk<M: MLModel>(
        embedding: &M,
        config: &TrainEmbeddingConfig,
        state: &TrainerStateMetadata,
        label: &str,
        output_dir: Option<&str>,
        output_label: Option<&str>,
        output_ext: &str,
    ) -> PathBuf {
        let snapshot = embedding.snapshot().unwrap();
        let mut snapshot: Value = serde_json::from_str(&snapshot).unwrap();

        let output_label = state.output_label.as_deref().or(output_label);
        let mut config = config.clone();
        config.output_label = output_label.map(|x| x.to_string());

        snapshot["_trainer_config"] = serde_json::to_value(&config).unwrap();
        snapshot["_trainer_state"] = serde_json::to_value(&state).unwrap();

        let snapshot_pretty = serde_json::to_string_pretty(&snapshot).unwrap();
        let fpath = path::create_output_fpath(
            ("model", output_ext),
            output_dir,
            output_label,
            Some(state),
            label,
        );

        File::create(&fpath)
            .unwrap()
            .write_fmt(format_args!("{snapshot_pretty}"))
            .unwrap();

        fpath
    }

    pub fn write_embedding_tsv_to_disk(
        embedding: &Embedding,
        label: &str,
        output_dir: &Option<String>,
        output_label: &Option<String>,
    ) {
        let vocab = embedding.vocab().keys().cloned().collect::<Vec<_>>();

        let fpath_vectors = path::create_output_fpath(
            ("embedding", "_vectors.tsv"),
            output_dir.as_deref(),
            output_label.as_deref(),
            None,
            label,
        );
        let fpath_labels = path::create_output_fpath(
            ("embedding", "_labels.tsv"),
            output_dir.as_deref(),
            output_label.as_deref(),
            None,
            label,
        );

        File::create(fpath_vectors)
            .unwrap()
            .write_fmt(format_args!(
                "{}",
                vocab
                    .iter()
                    .map(|word| embedding.embeddings(&word).unwrap().iter().join("\t"))
                    .join("\n")
            ))
            .unwrap();

        File::create(fpath_labels)
            .unwrap()
            .write_fmt(format_args!("{}", vocab.iter().join("\n")))
            .unwrap();
    }

    pub fn write_results_to_disk(embedding: &Embedding, label: &str) {
        let vocab = embedding.vocab().keys().cloned().collect::<HashSet<_>>();

        let fpath_nearest =
            path::create_output_fpath(("embedding", "_nearest.json"), None, None, None, label);
        let fpath_predictions =
            path::create_output_fpath(("embedding", "_predictions.json"), None, None, None, label);
        let fpath_embeddings =
            path::create_output_fpath(("embedding", "_embeddings.csv"), None, None, None, label);

        File::create(fpath_nearest)
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

        File::create(fpath_predictions)
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

        File::create(fpath_embeddings)
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

    mod path {
        use std::{
            fs::DirBuilder,
            path::{Path, PathBuf},
            time::SystemTime,
        };

        use crate::messages::TrainerStateMetadata;

        pub struct CreateExportPathOptions<'a> {
            fname_prefix_ext: (&'a str, &'a str),
            output_dir: Option<&'a str>,
            output_label: Option<&'a str>,
            label: &'a str,
            is_unique_path: bool,
        }

        pub fn create_output_fpath(
            fname_prefix_ext: (&str, &str),
            output_dir: Option<&str>,
            output_label: Option<&str>,
            state: Option<&TrainerStateMetadata>,
            label: &str,
        ) -> PathBuf {
            let options = CreateExportPathOptions {
                fname_prefix_ext,
                output_dir: output_dir,
                output_label: output_label,
                label,
                is_unique_path: true,
            };
            create_output_fpath_advanced(options, state)
        }

        pub fn create_output_fpath_advanced(
            options: CreateExportPathOptions,
            state: Option<&TrainerStateMetadata>,
        ) -> PathBuf {
            let dir_path = create_output_dir(options.output_dir, options.output_label.clone());
            let fname_description = match (options.output_label, state) {
                (Some(_), Some(state)) => metadata_fname_description(state),
                _ => default_fname_description(options.label),
            };
            let (fname_prefix, fname_ext) = options.fname_prefix_ext;
            let fpath = dir_path.join(format!("{fname_prefix}-{fname_description}{fname_ext}"));

            if let (Ok(true), true) = (fpath.try_exists(), options.is_unique_path) {
                let fpath_prefix = dir_path.join(format!("{fname_prefix}-{fname_description}"));
                if let Some(value) = get_next_path(fpath_prefix, fname_ext.to_string()) {
                    return value;
                }
            }
            fpath
        }

        fn metadata_fname_description(state: &TrainerStateMetadata) -> String {
            let trainer_report = state.training_report.as_ref();
            let predictions_pct = trainer_report.map(|report| report.predictions_pct.round());
            let predictions_pct = predictions_pct.unwrap_or_default();
            let round = state.current_round + 1;
            format!("r{round}-{predictions_pct}pct")
        }

        fn default_fname_description(label: &str) -> String {
            format!("{label}-{}.json", systime())
        }

        fn systime() -> u128 {
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        }

        fn get_next_path(fpath_prefix: PathBuf, fpath_postfix: String) -> Option<PathBuf> {
            let fpath_prefix = fpath_prefix.to_str()?;
            let fpath_postfix = fpath_postfix.trim_start_matches('.');
            for i in 2..50 {
                let fpath: PathBuf = format!("{fpath_prefix}.{i}.{fpath_postfix}").into();
                if let Ok(false) = fpath.try_exists() {
                    return Some(fpath);
                }
            }
            None
        }

        fn create_output_dir(output_dir: Option<&str>, output_label: Option<&str>) -> PathBuf {
            let output_dir = output_dir.unwrap_or("out");
            let dir_path = Path::new(&output_dir);
            let dir_path = match output_label {
                Some(label) => dir_path.join(label),
                None => dir_path.to_path_buf(),
            };
            DirBuilder::new().recursive(true).create(&dir_path).unwrap();
            dir_path
        }
    }
}
