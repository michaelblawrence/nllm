use std::{
    collections::{HashMap, HashSet},
    fs::File,
    rc::Rc,
    time::{Duration, Instant},
};

use plane::ml::{
    embeddings::{Embedding, TrainBatchConfig},
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
        TrainerHandle, TrainerHandleActions, TrainerMessage, TrainerReport, TrainerStateMetadata, TrainerHandleSender,
    },
};

pub struct TrainerState {
    pub embedding: Embedding,
    pub handle_tx: TrainerHandleSender<TrainerMessage>,
    pub output_label: Option<String>,
    pub learn_rate: f64,
    pub training_rounds: usize,
    pub print_round_number: bool,
    pub supress_auto_report: bool,
    pub force_report_all: bool,
    pub snapshot: (Option<String>, Instant),
    pub last_report: Option<(Instant, TrainerReport)>,
    paused: bool,
    pub round: usize,
    training_loss_logger: BoundedValueLogger<(usize, f64)>,
    rounds_until_pause_enabled: usize,
    halt: bool,
    force_report_test_set: bool,
    force_report_train_set: bool,
    round_reported: bool,
    total_train_time: Duration,
    inital_config: TrainEmbeddingConfig,
}

impl TrainerState {
    fn new(
        embedding: Embedding,
        config: &TrainEmbeddingConfig,
        handle_tx: &TrainerHandleSender<TrainerMessage>,
    ) -> Self {
        Self {
            embedding,
            handle_tx: handle_tx.clone(),
            inital_config: config.clone(),
            output_label: config.output_label.clone(),
            learn_rate: config.train_rate,
            training_rounds: config.training_rounds,
            paused: config.pause_on_start,
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
            total_train_time: Duration::ZERO,
            round: 0,
        }
    }

    fn train(&mut self, phrases: &Vec<Vec<String>>, batch_size: TrainBatchConfig) -> Option<f64> {
        if self.training_paused() {
            return None;
        }
        let started = Instant::now();

        let train_error = self.embedding.train(phrases, self.learn_rate, batch_size);

        let train_duration = started.elapsed();
        self.total_train_time += train_duration;

        self.try_record_train_iteration(train_error)
    }

    fn try_record_train_iteration(&mut self, train_error: Result<f64>) -> Option<f64> {
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
        } else if !self.training_paused() {
            self.round += 1;
        }
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
        let json = self.embedding.snapshot().unwrap();
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
        match self.inital_config.single_batch_iterations {
            true => TrainBatchConfig::SingleBatch(self.inital_config.batch_size),
            false => self.inital_config.batch_size.into(),
        }
    }

    pub fn hidden_layer_shape(&self) -> Vec<usize> {
        Self::build_hidden_layer_shape(&self.inital_config)
    }

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

    pub fn set_embedding(&mut self, embedding: Embedding, metadata: TrainerStateMetadata) {
        self.embedding = embedding;
        self.round = metadata.current_round;
        self.training_rounds = metadata.training_rounds;
        self.learn_rate = metadata.learn_rate;
        self.output_label = metadata.output_label;
        self.total_train_time = Duration::from_secs(metadata.total_train_seconds);

        self.training_loss_logger = {
            let logger_capacity = self.training_loss_logger.capacity();
            let mut logger: BoundedValueLogger<(usize, f64)> = metadata
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
        let run_completed = self.round >= self.training_rounds;

        if quit_on_complete && run_completed && !self.halt {
            let metadata = self.metadata();
            let msg = TrainerMessage::WriteModelAndMetadataToDisk(metadata);

            self.handle_tx
                .send(msg)
                .expect("failed to invoke model write to disk");

            info!("Triggered action: save model to file");
        }

        self.halt = true;
    }
}

pub fn setup_and_train_embeddings_v2<F>(
    config: TrainEmbeddingConfig,
    handle: TrainerHandle<TrainerMessage, F>,
) -> Embedding
where
    F: Fn(&Embedding, TrainerMessage) -> TrainerHandleActions,
{
    let (phrases, testing_phrases, mut state) = init_embedding_state(config, &handle);
    let batch_size = state.batch_size();

    loop {
        let training_error = state.train(&phrases, batch_size);
        let recv_timeout = state.training_paused().then_some(Duration::from_secs(1));

        while let Ok(msg) = handle.try_recv(recv_timeout) {
            let action = handle.run(&state.embedding, msg);
            action.apply(&mut state, &handle);
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

    state.embedding
}

fn init_embedding_state<F>(
    config: TrainEmbeddingConfig,
    handle: &TrainerHandle<TrainerMessage, F>,
) -> (Vec<Vec<String>>, Vec<Vec<String>>, TrainerState)
where
    F: Fn(&Embedding, TrainerMessage) -> TrainerHandleActions,
{
    info!("Initialising vocab and train/test sets");
    let rng: RngStrategy = Default::default();
    let (mut phrases, mut vocab) = init_phrases_and_vocab(&config, rng.to_rc());
    let mut testing_phrases = match config.phrase_test_set_split_pct.filter(|x| *x > 0.0) {
        Some(pct) => {
            split_training_and_testing(&mut phrases, pct, config.phrase_test_set_max_tokens)
        }
        None => phrases.clone(),
    };

    if config.use_character_tokens {
        info!("Tranforming vocab and train/test sets to character-level tokens");
        let word_mode = false; // TODO config?
        phrases = characterize_phrases(phrases, Some(config.input_stride_width), word_mode);
        testing_phrases =
            characterize_phrases(testing_phrases, Some(config.input_stride_width), word_mode);
        vocab = compute_vocab(&phrases);
    }

    let token_count = |x: &Vec<Vec<String>>| x.iter().map(|phrase| phrase.len()).sum::<usize>();
    info!(
        "Completed token initialisation: (vocab size = {} tokens)",
        vocab.len()
    );
    info!("Loaded: [ tokens(train_set) = {}, sequences(train_set) = {}, tokens(test_set) = {}, sequences(test_set) = {} ]",
        token_count(&phrases), phrases.len(),
        token_count(&testing_phrases), testing_phrases.len()
    );

    let hidden_layer_shape = TrainerState::build_hidden_layer_shape(&config);

    let embedding = Embedding::new_builder(vocab)
        .with_embedding_dimensions(config.embedding_size)
        .with_hidden_layer_custom_shape(hidden_layer_shape)
        .with_input_stride_width(config.input_stride_width)
        .with_activation_mode(config.activation_mode)
        .with_rng(rng)
        .build()
        .unwrap();

    let state = TrainerState::new(embedding, &config, &handle.tx);

    (phrases, testing_phrases, state)
}

fn report_round(
    state: &mut TrainerState,
    testing_phrases: &Vec<Vec<String>>,
    training_error: Option<f64>,
    label: Option<&str>,
) {
    match validate::validate_embeddings(&state.embedding, testing_phrases) {
        Ok((validation_errors, predictions_pct)) => {
            let report = validate::generate_training_report(
                state.round,
                training_error.unwrap_or(0.0),
                validation_errors,
                predictions_pct,
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
    rng: Rc<dyn RNG>,
) -> (Vec<Vec<String>>, HashSet<String>) {
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

    phrases.sort_by_cached_key(|phrase| {
        phrase
            .iter()
            .map(|word| -(vocab_counts.get(word).unwrap_or(&1).pow(2) as i128))
            .sum::<i128>()
    });

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

    let vocab = compute_vocab(&phrases);

    plane::ml::ShuffleRng::shuffle_vec(&rng, &mut phrases);

    // TODO: cleanup
    Some(&phrases)
        .filter(|x| x.is_empty())
        .ok_or(())
        .expect_err("phrases is empty");

    (phrases, vocab)
}

fn compute_vocab(phrases: &Vec<Vec<String>>) -> HashSet<String> {
    phrases
        .iter()
        .flat_map(|phrase| {
            phrase
                .join(" ")
                .chars()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn characterize_phrases(
    phrases: Vec<Vec<String>>,
    min_word_len: Option<usize>,
    word_mode: bool,
) -> Vec<Vec<String>> {
    let separator = " ".to_string();
    phrases
        .into_iter()
        .filter_map(|phrase| match min_word_len {
            Some(min_word_len)
                if word_mode && phrase.iter().any(|word| word.len() < min_word_len) =>
            {
                None
            }
            _ => Some(
                Itertools::intersperse(phrase.into_iter(), separator.clone())
                    .flat_map(|word| word.chars().map(|c| c.to_string()).collect_vec())
                    .collect_vec(),
            ),
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
    use tracing::info;

    use plane::ml::{embeddings::Embedding, NodeValue};

    use crate::messages::TrainerReport;

    pub fn generate_training_report(
        round: usize,
        training_error: f64,
        validation_errors: Vec<(f64, f64)>,
        predictions_pct: f64,
        last_report_time: Option<(Duration, usize)>,
        label: Option<&str>,
    ) -> TrainerReport {
        let label = label.map(|x| x.to_string());
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

        let ms_per_round = last_report_time.map(|(duration, last_round)| {
            duration.as_millis() / (round - last_round).max(1) as u128
        });
        let generated_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis();

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
            .map(|ms_per_round| format!("(ms/round={ms_per_round:<4.1})"))
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
    ) -> Result<(Vec<(f64, f64)>, f64)> {
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

        Ok((validation_errors, predictions_pct))
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

    use crate::{config::TrainEmbeddingConfig, messages::TrainerStateMetadata};

    pub fn plot_training_loss(
        _embedding: &Embedding,
        config: &&TrainEmbeddingConfig,
        metadata: &TrainerStateMetadata,
    ) {
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
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);

        let mut snapshot: Value = serde_json::from_reader(reader)?;

        let config: TrainEmbeddingConfig =
            serde_json::from_value(snapshot["_trainer_config"].take())
                .context("unable to extract trainer config from saved model")?;

        let state: TrainerStateMetadata = serde_json::from_value(snapshot["_trainer_state"].take())
            .context("unable to extract trainer metatdata from saved model")?;

        let snapshot = serde_json::to_string(&snapshot)?;

        Ok((snapshot, config, state))
    }

    pub fn write_model_to_disk(
        embedding: &Embedding,
        config: &TrainEmbeddingConfig,
        state: &TrainerStateMetadata,
        label: &str,
        output_dir: &Option<String>,
        output_label: &Option<String>,
    ) -> PathBuf {
        let snapshot = embedding.snapshot().unwrap();
        let mut snapshot: Value = serde_json::from_str(&snapshot).unwrap();

        let output_label = state.output_label.as_ref().or(output_label.as_ref());
        let mut config = config.clone();
        config.output_label = output_label.cloned();

        snapshot["_trainer_config"] = serde_json::to_value(&config).unwrap();
        snapshot["_trainer_state"] = serde_json::to_value(&state).unwrap();

        let snapshot_pretty = serde_json::to_string_pretty(&snapshot).unwrap();
        let fpath = path::create_output_fpath(
            ("model", ".json"),
            output_dir.as_ref(),
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
            output_dir.as_ref(),
            output_label.as_ref(),
            None,
            label,
        );
        let fpath_labels = path::create_output_fpath(
            ("embedding", "_labels.tsv"),
            output_dir.as_ref(),
            output_label.as_ref(),
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

        pub fn create_output_fpath(
            fname_prefix_ext: (&str, &str),
            output_dir: Option<&String>,
            output_label: Option<&String>,
            state: Option<&TrainerStateMetadata>,
            label: &str,
        ) -> PathBuf {
            let dir_path = create_output_dir(output_dir, output_label.clone());
            let fname_description = match (output_label, state) {
                (Some(_), Some(state)) => metadata_fname_description(state),
                _ => default_fname_description(label),
            };
            let (fname_prefix, fname_ext) = fname_prefix_ext;
            let fpath = dir_path.join(format!("{fname_prefix}-{fname_description}{fname_ext}"));

            if let Ok(true) = fpath.try_exists() {
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

        fn create_output_dir(
            output_dir: Option<&String>,
            output_label: Option<&String>,
        ) -> PathBuf {
            let output_dir = output_dir.cloned().unwrap_or("out".to_string());
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
