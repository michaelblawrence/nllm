use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufReader, Write},
    rc::Rc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use itertools::Itertools;
use plane::ml::{
    embeddings::{Embedding, TrainBatchConfig},
    JsRng, NodeValue, RNG,
};

use serde_json::Value;
use tracing::{error, info};

use crate::{
    config::TrainEmbeddingConfig,
    messages::{
        TrainerHandle, TrainerHandleActions, TrainerMessage, TrainerReport, TrainerStateMetadata,
    },
};

pub struct TrainerState {
    pub embedding: Embedding,
    pub batch_size: TrainBatchConfig,
    pub learn_rate: f64,
    pub training_rounds: usize,
    pub supress_auto_report: bool,
    pub force_report: bool,
    pub force_report_all: bool,
    pub snapshot: (Option<String>, Instant),
    pub last_report: Option<(Instant, TrainerReport)>,
    pub paused: bool,
    pub halt: bool,
    pub round: usize,
    pub rng: Rc<dyn RNG>,
    inital_config: TrainEmbeddingConfig,
}

impl TrainerState {
    fn new(embedding: Embedding, config: &TrainEmbeddingConfig, rng: Rc<dyn RNG>) -> Self {
        let batch_size = match config.single_batch_iterations {
            true => TrainBatchConfig::SingleBatch(config.batch_size),
            false => config.batch_size.into(),
        };
        Self {
            inital_config: config.clone(),
            embedding,
            batch_size,
            learn_rate: config.train_rate,
            training_rounds: config.training_rounds,
            paused: config.pause_on_start,
            snapshot: (None, Instant::now()),
            supress_auto_report: false,
            force_report: false,
            force_report_all: false,
            last_report: None,
            halt: false,
            rng,
            round: 0,
        }
    }

    fn complete_round(&mut self) {
        if !self.paused {
            self.round += 1;
        }
        self.force_report = false;
    }

    fn pause(&mut self) {
        self.paused = true;
    }

    fn perform_snapshot(&mut self) {
        let json = self.embedding.snapshot().unwrap();
        self.snapshot = (Some(json), Instant::now());
    }

    fn should_perform_autosave(&self) -> bool {
        !self.paused
            && self.snapshot.1.elapsed().as_secs() > self.inital_config.snapshot_interval_secs
    }

    fn should_report_round(&self) -> bool {
        if self.force_report_all || self.force_report {
            return true;
        }

        !self.paused
            && !self.supress_auto_report
            && should_report_round(self.round, self.training_rounds)
    }

    pub fn metadata(&self) -> TrainerStateMetadata {
        TrainerStateMetadata {
            learn_rate: self.learn_rate,
            training_rounds: self.training_rounds,
            current_round: self.round,
            training_report: self.last_report.as_ref().map(|(_, report)| report.clone()),
        }
    }

    pub fn input_stride_width(&self) -> usize {
        self.inital_config.input_stride_width
    }

    pub fn hidden_layer_shape(&self) -> Vec<usize> {
        Self::build_hidden_layer_shape(&self.inital_config)
    }

    pub fn build_hidden_layer_shape(config: &TrainEmbeddingConfig) -> Vec<usize> {
        let layer_node_counts = [
            Some(config.hidden_layer_nodes),
            config.hidden_deep_layer_nodes.filter(|x| *x > 0),
        ];

        layer_node_counts.iter().cloned().flatten().collect()
    }
}

pub fn setup_and_train_embeddings_v2<F>(
    config: TrainEmbeddingConfig,
    handle: TrainerHandle<TrainerMessage, F>,
) -> Embedding
where
    F: Fn(&Embedding, TrainerMessage) -> TrainerHandleActions,
{
    let rng: Rc<dyn RNG> = Rc::new(JsRng::default());
    let (mut phrases, mut vocab) = init_phrases_and_vocab(&config, rng.clone());
    let mut testing_phrases = match config.phrase_test_set_split_pct.filter(|x| *x > 0.0) {
        Some(pct) => split_training_and_testing(&mut phrases, pct),
        None => phrases.clone(),
    };

    if config.use_character_tokens {
        phrases = characterize_phrases(phrases);
        testing_phrases = characterize_phrases(testing_phrases);
        vocab = compute_vocab(&phrases);
    }

    dbg!((phrases.len(), vocab.len())); // TODO: replace with tracing logging
    let hidden_layer_shape = TrainerState::build_hidden_layer_shape(&config);

    let embedding = Embedding::new_builder(vocab, rng.clone())
        .with_embedding_dimensions(config.embedding_size)
        .with_hidden_layer_custom_shape(hidden_layer_shape)
        .with_input_stride_width(config.input_stride_width)
        .with_activation_mode(config.activation_mode)
        .build();

    let mut state = TrainerState::new(embedding, &config, rng);

    loop {
        let training_error = if !state.paused {
            match state
                .embedding
                .train(&phrases, state.learn_rate, state.batch_size)
            {
                Ok(error) => {
                    if state.should_perform_autosave() {
                        state.perform_snapshot();
                    }
                    Some(error)
                }
                Err(e) => {
                    error!("Failed to train embedding model {e}");
                    if let (Some(_), when) = state.snapshot {
                        info!("Snapshot available from {when:?}. See help menu ('h') to restore... pausing")
                    }
                    state.pause();
                    None
                }
            }
        } else {
            None
        };

        let recv_timeout = state.paused.then_some(Duration::from_secs(5));

        while let Ok(msg) = handle.try_recv(recv_timeout) {
            let handler = &handle.handler;
            let action = handler(&state.embedding, msg);
            action.apply(&mut state, &handle);
        }

        if state.should_report_round() {
            match validate_embeddings(&state.embedding, &testing_phrases) {
                Ok((validation_errors, predictions_pct)) => {
                    let report = generate_training_report(
                        state.round,
                        training_error.unwrap_or(0.0),
                        validation_errors,
                        predictions_pct,
                        state
                            .last_report
                            .as_ref()
                            .map(|(last_dt, report)| (last_dt.elapsed(), report.round)),
                    );

                    log_training_round(&report);

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

        if state.halt {
            info!("Stopping rounds run...");
            break;
        }

        state.complete_round();

        if state.round == state.training_rounds && !state.paused {
            info!("Completed rounds run, pausing...");
            state.pause();
        }
    }

    state.embedding
}

pub fn init_phrases_and_vocab(
    config: &TrainEmbeddingConfig,
    rng: Rc<dyn RNG>,
) -> (Vec<Vec<String>>, HashSet<String>) {
    let phrases = parse_phrases();

    let (min_len, max_len) = config.phrase_word_length_bounds;

    let mut phrases: Vec<Vec<String>> = phrases
        .into_iter()
        .filter(|phrase| phrase.len() > min_len && phrase.len() < max_len)
        .collect();

    let vocab_counts = phrases
        .iter()
        .flat_map(|phrase| phrase.iter())
        .cloned()
        .fold(HashMap::new(), |mut counts, word| {
            counts.entry(word).and_modify(|x| *x += 1).or_insert(1_i32);
            counts
        });

    phrases.sort_by_cached_key(|phrase| {
        -phrase
            .iter()
            .map(|word| vocab_counts[word].pow(2))
            .sum::<i32>()
    });

    phrases.truncate(config.phrase_train_set_size);
    info!(
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
    (phrases, vocab)
}

fn compute_vocab(phrases: &Vec<Vec<String>>) -> HashSet<String> {
    phrases
        .iter()
        .flat_map(|phrase| phrase.iter())
        .cloned()
        .collect()
}

fn characterize_phrases(phrases: Vec<Vec<String>>) -> Vec<Vec<String>> {
    phrases
        .into_iter()
        .map(|phrase| {
            phrase
                .into_iter()
                .flat_map(|word| word.chars().map(|c| c.to_string()).collect_vec())
                .collect_vec()
        })
        .collect_vec()
}

pub fn parse_phrases() -> Vec<Vec<String>> {
    let phrase_json = include_str!("../../../res/phrase_list.json");
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

fn generate_training_report(
    round: usize,
    training_error: f64,
    validation_errors: Vec<(f64, f64)>,
    predictions_pct: f64,
    last_report_time: Option<(Duration, usize)>,
) -> TrainerReport {
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
        .map(|(duration, last_round)| duration.as_millis() / (round - last_round).max(1) as u128);

    TrainerReport {
        round,
        training_error,
        ms_per_round,
        predictions_pct,
        validation_error,
        nll,
    }
}

fn log_training_round(report: &TrainerReport) {
    let ms_per_round = report
        .ms_per_round
        .map(|ms_per_round| format!("(ms/round={ms_per_round:<4.1})"))
        .unwrap_or_default();

    info!(
            "round = {:<6} |  train_loss = {:<12.10}, val_pred_acc: {:0>4.1}%, val_loss = {:<2.6e}, val_nll = {:<6.3} {ms_per_round}",
            report.round + 1, report.training_error, report.predictions_pct, report.validation_error, report.nll
    );
}

fn validate_embeddings(
    embedding: &Embedding,
    testing_phrases: &Vec<Vec<String>>,
) -> Result<(Vec<(f64, f64)>, f64)> {
    let mut validation_errors = vec![];
    let mut correct_first_word_predictions = 0;
    let mut total_first_word_predictions = 0;

    for testing_phrase in testing_phrases.iter() {
        for testing_phrase_window in testing_phrase.windows(embedding.input_stride_width() + 1) {
            let (last_word_vector, context_word_vectors) = testing_phrase_window
                .split_last()
                .context("should have last element")?;

            let last_words = context_word_vectors
                .into_iter()
                .map(|x| x.as_str())
                .collect::<Vec<_>>();

            let predicted = embedding.predict_next(&last_words[..])?;
            let actual = last_word_vector;

            if &predicted == actual {
                correct_first_word_predictions += 1;
            }

            let error = embedding.compute_error(testing_phrase)?;
            let nll = embedding.nll(&last_words, &actual)?;
            validation_errors.push((error, nll));
            total_first_word_predictions += 1;
        }
    }

    let predictions_pct = correct_first_word_predictions as NodeValue * 100.0
        / total_first_word_predictions.max(1) as NodeValue;

    Ok((validation_errors, predictions_pct))
}

pub fn split_training_and_testing(
    phrases: &mut Vec<Vec<String>>,
    test_phrases_pct: f64,
) -> Vec<Vec<String>> {
    let testing_ratio = test_phrases_pct as NodeValue / 100.0;
    let testing_sample_count = phrases.len() as NodeValue * testing_ratio;
    let offset = phrases.len() - testing_sample_count as usize;
    let testing_phrases = phrases.split_off(offset.clamp(0, phrases.len() - 1));
    testing_phrases
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
) {
    let snapshot = embedding.snapshot().unwrap();
    let mut snapshot: Value = serde_json::from_str(&snapshot).unwrap();

    snapshot["_trainer_config"] = serde_json::to_value(&config).unwrap();
    snapshot["_trainer_state"] = serde_json::to_value(&state).unwrap();

    let snapshot_pretty = serde_json::to_string_pretty(&snapshot).unwrap();
    let fpath = output::create_output_fpath(
        ("model", ".json"),
        output_dir.as_ref(),
        output_label.as_ref(),
        Some(state),
        label,
    );

    File::create(fpath)
        .unwrap()
        .write_fmt(format_args!("{snapshot_pretty}"))
        .unwrap();
}

pub fn write_embedding_tsv_to_disk(
    embedding: &Embedding,
    label: &str,
    output_dir: &Option<String>,
    output_label: &Option<String>,
) {
    let vocab = embedding.vocab().keys().cloned().collect::<Vec<_>>();

    let fpath_vectors = output::create_output_fpath(
        ("embedding", "_vectors.tsv"),
        output_dir.as_ref(),
        output_label.as_ref(),
        None,
        label,
    );
    let fpath_labels = output::create_output_fpath(
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
        output::create_output_fpath(("embedding", "_nearest.json"), None, None, None, label);
    let fpath_predictions =
        output::create_output_fpath(("embedding", "_predictions.json"), None, None, None, label);
    let fpath_embeddings =
        output::create_output_fpath(("embedding", "_embeddings.csv"), None, None, None, label);

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

mod output {
    use std::{
        fs::DirBuilder,
        path::{Path, PathBuf},
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
        let fpath = dir_path.join(format!("{fname_prefix}-{}{fname_ext}", fname_description));
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
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    }

    fn create_output_dir(output_dir: Option<&String>, output_label: Option<&String>) -> PathBuf {
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
