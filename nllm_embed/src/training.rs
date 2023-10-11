use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    sync::Arc,
    time::{Duration, Instant},
};

use plane::ml::{
    embeddings::TrainBatchConfig,
    gdt::{self, GenerativeDecoderTransformer},
    transformer::decoder::builder::DecoderBuilder,
    NodeValue, RngStrategy, RNG,
};

use anyhow::Result;
use itertools::Itertools;
use serde_json::Value;
use tracing::{error, info, warn};

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
    pub last_report: Option<(Instant, TrainerReport)>,
    paused: bool,
    batches_trained: usize,
    pub round: usize,
    learn_rate: NodeValue,
    training_loss_logger: BoundedValueLogger<(usize, NodeValue)>,
    rounds_until_pause_enabled: usize,
    halt: bool,
    use_detailed_nll: bool,
    pending_autosave: bool,
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
            training_loss_logger: BoundedValueLogger::new(1000),
            round_reported: false,
            print_round_number: false,
            supress_auto_report: false,
            use_detailed_nll: false,
            pending_autosave: false,
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
                self.training_loss_logger.push((self.round, error));
                Some(error)
            }
            Ok(non_finite_error) => {
                error!(
                    "Failed perform embedding model training iteration: training loss = {}",
                    non_finite_error
                );
                self.pause();
                None
            }
            Err(e) => {
                error!("Failed to train embedding model {e}");
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

        if !self.paused && !self.is_pending_round_report() && self.pending_autosave {
            self.pending_autosave = false;
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

        let pending_autosave = self.last_autosave.elapsed() > self.autosave_interval;
        if !self.paused && pending_autosave && !self.pending_autosave {
            self.trigger_round_report_test_set();
            self.pending_autosave = true;
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

    pub fn sample_from_pattern(&self) -> Option<&str> {
        self.sample_from_pattern.as_deref()
    }

    pub fn batch_size(&self) -> TrainBatchConfig {
        if self.inital_config.single_batch_iterations {
            TrainBatchConfig::SingleBatch(self.inital_config.batch_size)
        } else {
            TrainBatchConfig::Batches(self.inital_config.batch_size)
        }
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

    pub fn is_pending_round_report(&self) -> bool {
        self.force_report_test_set || self.force_report_train_set
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

    pub fn use_detailed_nll(&self) -> bool {
        self.use_detailed_nll
    }

    pub fn enable_detailed_nll(&mut self, enable: bool) {
        self.use_detailed_nll = enable;
    }
}

impl<M> TrainerState<M> {
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
    defer_init: bool,
) -> M {
    let (phrases, testing_phrases, mut state) = init_model_state(config, &handle, defer_init);
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
    defer_init: bool,
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
    let mut phrases = init_phrases_and_vocab(&config, phrase_rng.to_arc());
    let testing_phrases = match config.phrase_test_set_split_pct.filter(|x| *x > 0.0) {
        Some(pct) => split_training_and_testing(
            &mut phrases,
            pct,
            config.phrase_test_set_max_tokens,
            config.sample_from_pattern.as_deref(),
        ),
        None => phrases.clone(),
    };
    let rng = RngStrategy::default();

    let gdt_vocab = {
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

        let gdt_vocab = if defer_init {
            vocab_builder.build().unwrap()
        } else {
            let vocab_builder = match config.sample_from_pattern.as_deref() {
                Some("###human") | Some("###human:") => vocab_builder
                    .add_word_token_literal("###human:")
                    .add_word_token_literal("###ctx__:")
                    .add_word_token_literal("###chat_:"),
                _ => vocab_builder,
            };

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
            let vocab_builder = vocab_builder.with_char_fallback();
            let vocab = vocab_builder.build().unwrap();
            if config.gdt_bpe_vocab_size.is_power_of_two() && !vocab.len().is_power_of_two() {
                warn!(
                    "Potential vocab size misalignment: [actual]/[requested] vocab size: {}/{}",
                    vocab.len(),
                    config.gdt_bpe_vocab_size
                );
            }
            vocab
        };

        gdt_vocab
    };

    let token_count = |x: &Vec<Vec<String>>| x.iter().map(|phrase| phrase.len()).sum::<usize>();
    let token_len = gdt_vocab.len();
    info!(
        "Completed token initialisation: (vocab size = {} tokens)",
        token_len
    );
    info!("Loaded: [ tokens(train_set) = {}, sequences(train_set) = {}, tokens(test_set) = {}, sequences(test_set) = {} ]",
        token_count(&phrases), phrases.len(),
        token_count(&testing_phrases), testing_phrases.len()
    );

    info!("Using GenerativeDecoderTransformer (GDT)...");
    let model = build_model(&config, gdt_vocab, rng);
    let state = TrainerState::new(model, &config, &handle.tx);
    info!("Initialized GDT model.");

    (phrases, testing_phrases, state)
}

fn build_model<M>(config: &TrainEmbeddingConfig, vocab: gdt::token::Vocab, rng: RngStrategy) -> M
where
    M: MLModel,
{
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

    let csv_block_count = hidden_deep_layer_nodes
        .as_ref()
        .and_then(|x| x.get(0))
        .and_then(|x| x.parse::<usize>().ok());

    let csv_head_count = hidden_deep_layer_nodes
        .as_ref()
        .and_then(|x| x.get(1))
        .and_then(|x| x.parse::<usize>().ok());

    let builder = if let Some(block_count) = config.decoder_blocks.or(csv_block_count) {
        builder.with_block_count(block_count)
    } else {
        builder
    };
    let builder = if let Some(head_count) = config.decoder_heads.or(csv_head_count) {
        builder.with_head_count(head_count)
    } else {
        builder
    };

    let gdt = GenerativeDecoderTransformer::from_builder(builder, vocab, rng).unwrap();
    M::from_gdt(gdt, config.train_rate, config.sample_from_pattern.clone())
}

fn report_round<M: MLModel>(
    state: &mut TrainerState<M>,
    testing_phrases: &Vec<Vec<String>>,
    training_error: Option<NodeValue>,
    label: Option<&str>,
) {
    let validation_results = if let Some(gdt) = state.model.as_gdt() {
        let char_mode = gdt.vocab().token_type().is_char();
        let testing_phrases = TrainerState::<M>::join_phrases(&testing_phrases, Some(char_mode)); // TODO: remove from hot path
        validate::validate_gdt(
            gdt,
            &testing_phrases,
            state.sample_from_pattern(),
            state.use_detailed_nll(),
        )
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

pub fn init_phrases_and_vocab(
    config: &TrainEmbeddingConfig,
    rng: Arc<dyn RNG>,
) -> Vec<Vec<String>> {
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
    if !config.phrase_disable_shuffle {
        plane::ml::ShuffleRng::shuffle_vec(&rng, &mut phrases);
    }

    // TODO: cleanup
    Some(&phrases)
        .filter(|x| x.is_empty())
        .ok_or(())
        .expect_err("phrases is empty");

    phrases
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
    let file = File::open(file_path).unwrap(); // TODO: error handling
    let reader = BufReader::new(file);

    let phrases: Vec<Vec<String>> = reader
        .lines()
        .map(|line| {
            line.unwrap_or_default()
                .split_ascii_whitespace()
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
    sample_from_pattern: Option<&str>,
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
    let offset = (phrases.len() - testing_sample_count).clamp(0, phrases.len() - 1);
    let testing_phrases = match sample_from_pattern.and_then(|prefix| {
        phrases[..offset]
            .iter()
            .enumerate()
            .rev()
            .find(|(_, x)| x.join(" ").starts_with(prefix))
    }) {
        Some((offset, _)) => phrases.split_off(offset),
        None => phrases.split_off(offset),
    };
    testing_phrases
}

mod validate {
    use std::time::Duration;

    use anyhow::{Context, Result};
    use tracing::info;

    use plane::ml::{
        gdt::{token::Token, GenerativeDecoderTransformer},
        LayerValues, NodeValue,
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

    pub fn validate_gdt(
        transformer: &GenerativeDecoderTransformer,
        testing_phrases: &str,
        sample_from_pattern: Option<&str>,
        use_detailed_nll: bool,
    ) -> Result<TestSetValidation> {
        let testing_tokens = transformer.vocab().encode(testing_phrases)?;
        let input_stride_width = transformer.input_stride_width();
        let sample_from_token = sample_from_pattern
            .filter(|x| x.starts_with("###human"))
            .map(|_| Token::word("###human:"));

        let testing_slices: Vec<&[Token]> =
            chunk_by_inclusive(&testing_tokens, sample_from_token, input_stride_width + 1)
                .collect();

        #[cfg(feature = "multi_threaded")]
        use rayon::prelude::*;

        let testing_slices = {
            #[cfg(feature = "multi_threaded")]
            {
                testing_slices.par_iter()
            }
            #[cfg(not(feature = "multi_threaded"))]
            {
                testing_slices.iter()
            }
        };
        let validation_results = testing_slices
            .map(|testing_phrase_window| {
                let (last_token, context_tokens) = testing_phrase_window.split_last().unwrap();

                let error =
                    transformer.compute_error(&[context_tokens, &[last_token.clone()]].concat())?;

                let (nll, correct_predictions) = if use_detailed_nll {
                    let (nll_each, correct_predictions): (Vec<_>, Vec<_>) = transformer
                        .nll_each(&testing_phrase_window)?
                        .into_iter()
                        .unzip();
                    (LayerValues::from(nll_each).ave(), correct_predictions)
                } else {
                    let (predicted, _) = transformer
                        .predict_arg_max(&context_tokens)
                        .context("failed prediction")?;
                    let correct_predictions = vec![&predicted == last_token];
                    let nll = transformer.nll(&context_tokens, last_token.clone())?;
                    (nll, correct_predictions)
                };
                Ok(((error, nll), correct_predictions))
            })
            .collect::<Result<Vec<_>>>()?;

        let (validation_errors, predictions): (Vec<_>, Vec<_>) =
            validation_results.into_iter().unzip();

        let correct_first_word_predictions = predictions.iter().flatten().filter(|x| **x).count();
        let total_first_word_predictions = predictions.iter().flatten().count();
        let predictions_pct = 100.0 * correct_first_word_predictions as NodeValue
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

    fn chunk_by_inclusive<T: std::cmp::PartialEq>(
        mut slice: &[T],
        chunk_prefix_pattern: Option<T>,
        max_chunk_size: usize,
    ) -> impl Iterator<Item = &[T]> {
        std::iter::from_fn(move || match slice.len() {
            0 => None,
            len => {
                let pos = match chunk_prefix_pattern
                    .as_ref()
                    .and_then(|needle| slice.iter().skip(1).position(|x| x == needle))
                {
                    Some(pos) => (pos + 1).min(max_chunk_size),
                    None => max_chunk_size,
                };

                let mid = pos.min(len);
                let (first, second) = slice.split_at(mid);

                slice = second;
                Some(first)
            }
        })
    }
}

pub mod writer {
    use std::{
        fs::File,
        io::{BufReader, BufWriter},
        path::PathBuf,
    };

    use anyhow::{Context, Result};
    use serde_json::Value;

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

        let fpath = path::create_output_fpath(
            ("model", output_ext),
            output_dir,
            output_label,
            Some(state),
            label,
        );
        let file = &mut File::create(&fpath).unwrap();
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &snapshot).unwrap();

        fpath
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
