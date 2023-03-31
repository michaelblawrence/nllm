use std::{
    collections::{HashMap, HashSet},
    io::stdin,
    ops::ControlFlow,
    rc::Rc,
    sync::{mpsc, Arc},
    thread,
};

use anyhow::{anyhow, Context, Result};
use clap::{Args, Parser};
use config::TrainEmbeddingConfig;
use crossterm::event::{self, Event, KeyCode, KeyEvent};
use itertools::Itertools;
use messages::{TrainerHandleActions, TrainerMessage};
use plane::ml::{embeddings::Embedding, JsRng, NetworkActivationMode, NodeValue, RNG};
use serde::{Deserialize, Serialize};
use tracing::{info, metadata::LevelFilter};

mod config {
    use clap::{command, Args, Parser, Subcommand};
    use serde::{Deserialize, Serialize};

    use plane::ml::{NetworkActivationMode, NodeValue};

    #[derive(Parser, Debug, Clone)]
    #[command(args_conflicts_with_subcommands = true)]
    pub struct Cli {
        #[command(subcommand)]
        command: Option<Command>,

        #[command(flatten)]
        train_command: TrainEmbeddingConfig,
    }

    impl Cli {
        pub fn command(&self) -> Command {
            self.command
                .clone()
                .unwrap_or(Command::TrainEmbedding(self.train_command.clone()))
        }
    }

    #[derive(Subcommand, Debug, Clone)]
    pub enum Command {
        #[command(name = "train", author, version, about, long_about = None)]
        TrainEmbedding(TrainEmbeddingConfig),

        #[command(name = "load", arg_required_else_help = true)]
        LoadEmbedding(LoadEmbeddingConfig),
    }

    #[derive(Args, Debug, Clone)]
    pub struct LoadEmbeddingConfig {
        pub file_path: String,
    }

    #[derive(Parser, Debug, Clone, Serialize, Deserialize)]
    pub struct TrainEmbeddingConfig {
        #[arg(short = 'n', long, default_value_t = 4)]
        pub embedding_size: usize,

        #[arg(short = 'h', long, default_value_t = 75)]
        pub hidden_layer_nodes: usize,

        #[arg(short = 'H', long, default_value_t = 0)]
        pub hidden_deep_layer_nodes: usize,

        #[arg(short = 'c', long, default_value_t = 1000)]
        pub training_rounds: usize,

        #[arg(short = 'w', long, default_value_t = 3)]
        pub input_stride_width: usize,

        #[arg(short = 'b', long, default_value_t = 16)]
        pub batch_size: usize,

        #[arg(short = 'r', long, default_value_t = 1e-3)]
        pub train_rate: NodeValue,

        #[arg(short = 'p', long, default_value_t = false)]
        #[serde(default)]
        pub pause_on_start: bool,

        #[arg(short = 'o', long, default_value = None)]
        #[serde(default)]
        pub output_dir: Option<String>,

        #[arg(short = 'O', long, default_value = None)]
        #[serde(default)]
        pub output_label: Option<String>,

        #[arg(short = 'S', long, default_value_t = 120)]
        pub snapshot_interval_secs: u64,

        #[arg(short = 'T', long, default_value_t = 150)]
        pub phrase_train_set_size: usize,

        #[arg(short = 'B', long, value_parser = parse_range::<usize>, default_value = "5..10")]
        pub phrase_word_length_bounds: (usize, usize),

        #[arg(short = 'X', long, default_value = "20")]
        pub phrase_test_set_split_pct: Option<NodeValue>,

        #[arg(short = 'm', long, default_value_t = NetworkActivationMode::Tanh)]
        pub activation_mode: NetworkActivationMode,
    }

    fn parse_range<T>(s: &str) -> Result<(T, T), Box<dyn std::error::Error + Send + Sync + 'static>>
    where
        T: std::str::FromStr,
        T::Err: std::error::Error + Send + Sync + 'static,
    {
        let pos = s
            .find("..")
            .ok_or_else(|| format!("invalid KEY=value: no `..=` found in `{s}`"))?;
        let range = (s[..pos].parse()?, s[pos + 2..].parse()?);
        Ok(range)
    }
}

fn main() -> Result<()> {
    configure_logging();
    let (config, resumed_state) = parse_cli_args()?;

    let config_clone = config.clone();
    let (tx, handle) = messages::TrainerHandle::new(move |embedding, msg| {
        handle_trainer_message(msg, embedding, &config_clone)
    });

    let config_clone = config.clone();
    let thread = thread::spawn(move || {
        let embedding = training::setup_and_train_embeddings_v2(config_clone, handle);

        training::write_results_to_disk(&embedding, "bin-convo-midlen");
    });

    if let Some((snapshot, state)) = resumed_state {
        tx.send(TrainerMessage::ReplaceEmbeddingState(snapshot, state));
    }

    let config_clone = config.clone();
    let ui_thread = thread::spawn(move || {
        use KeyCode::Char;

        loop {
            let event = event::read().unwrap();
            match event {
                Event::Key(KeyEvent { code: Char(c), .. }) => {
                    match parse_repl_char(c, &tx, &config_clone) {
                        Ok(_) => (),
                        Err(_) => return,
                    }
                }
                _ => (),
            }
        }
    });

    thread.join().unwrap();
    ui_thread.join().unwrap();

    Ok(())
}

fn parse_repl_char(
    c: char,
    tx: &mpsc::Sender<TrainerMessage>,
    config: &TrainEmbeddingConfig,
) -> Result<(), ()> {
    match c {
        'r' => {
            tx.send(TrainerMessage::PrintStatus);
        }
        'a' => {
            tx.send(TrainerMessage::TogglePrintAllStatus);
        }
        'x' => {
            tx.send(TrainerMessage::ReloadFromSnapshot);
        }
        'z' => {
            tx.send(TrainerMessage::ForceSnapshot);
        }
        'n' => {
            tx.send(TrainerMessage::PredictRandomPhrase);
        }
        'w' => {
            tx.send(TrainerMessage::WriteStatsToDisk);
        }
        'W' => {
            tx.send(TrainerMessage::WriteEmbeddingTsvToDisk);
        }
        's' => {
            tx.send(TrainerMessage::WriteModelToDisk);
        }
        ',' => {
            tx.send(TrainerMessage::MultiplyLearnRateBy(0.5));
        }
        '.' => {
            tx.send(TrainerMessage::MultiplyLearnRateBy(2.0));
        }
        'e' => {
            tx.send(TrainerMessage::IncreaseMaxRounds(config.training_rounds));
        }
        'p' => {
            tx.send(TrainerMessage::TogglePause);
        }
        'q' => {
            tx.send(TrainerMessage::Halt);
            return Err(());
        }
        'h' => {
            println!(
                r"Trainer help: 
                        'r' => report status
                        'n' => print new random phrase
                        'x' => reload from auto-save snapshot
                        'z' => force snapshot
                        'w' => write stats to disk
                        'W' => write embedding tsv to disk
                        's' => save model to disk
                        ',' => divide learn rate by 2
                        '.' => multiply learn rate by 2
                        'e' => extend training rounds
                        'h' => display help
                        'p' => toggle pause
                        'q' => quit
                        "
            )
        }
        _ => (),
    }
    Ok(())
}

fn handle_trainer_message(
    msg: TrainerMessage,
    embedding: &Embedding,
    config: &TrainEmbeddingConfig,
) -> TrainerHandleActions {
    match msg {
        TrainerMessage::PrintStatus => TrainerHandleActions::PrintStatus,
        TrainerMessage::Halt => TrainerHandleActions::Halt,
        TrainerMessage::TogglePause => TrainerHandleActions::TogglePause,
        TrainerMessage::TogglePrintAllStatus => TrainerHandleActions::TogglePrintAllStatus,
        TrainerMessage::ReloadFromSnapshot => TrainerHandleActions::ReloadFromSnapshot,
        TrainerMessage::ForceSnapshot => TrainerHandleActions::ForceSnapshot,
        TrainerMessage::WriteStatsToDisk => {
            training::write_results_to_disk(&embedding, "embed");
            TrainerHandleActions::Nothing
        }
        TrainerMessage::WriteEmbeddingTsvToDisk => {
            training::write_embedding_tsv_to_disk(
                &embedding,
                "embed",
                &config.output_dir,
                &config.output_label,
            );
            TrainerHandleActions::Nothing
        }
        TrainerMessage::WriteModelToDisk => {
            TrainerHandleActions::DispatchWithMetadata(Arc::new(|x| {
                TrainerMessage::WriteModelAndMetadataToDisk(x)
            }))
        }
        TrainerMessage::WriteModelAndMetadataToDisk(metadata) => {
            training::write_model_to_disk(
                &embedding,
                &config,
                &metadata,
                "embed",
                &config.output_dir,
                &config.output_label,
            );
            TrainerHandleActions::Nothing
        }
        TrainerMessage::PredictRandomPhrase => {
            let vocab_idx = JsRng::default().rand_range(0, embedding.vocab().len());
            let seed_word = embedding.vocab().keys().nth(vocab_idx).unwrap();
            let predicted_phrase = embedding.predict_iter(seed_word).join(" ");
            info!("Predicted a new random phrase:  {}", predicted_phrase);
            TrainerHandleActions::Nothing
        }
        TrainerMessage::IncreaseMaxRounds(x) => TrainerHandleActions::IncreaseMaxRounds(x),
        TrainerMessage::MultiplyLearnRateBy(x) => TrainerHandleActions::LearnRateMulMut(x),
        TrainerMessage::ReplaceEmbeddingState(embedding, state) => {
            TrainerHandleActions::ReplaceEmbeddingState(embedding, state)
        }
    }
}

fn parse_cli_args() -> Result<(
    TrainEmbeddingConfig,
    Option<(String, messages::TrainerStateMetadata)>,
)> {
    let cli = config::Cli::parse();
    let (config, resumed_state) = match cli.command() {
        config::Command::TrainEmbedding(config) => (config, None),
        config::Command::LoadEmbedding(config) => {
            let rng = Rc::new(JsRng::default());
            let (snapshot, mut config, state) =
                training::read_model_from_disk(&config.file_path, rng)?;

            config.pause_on_start = true;
            (config, Some((snapshot, state)))
        }
    };
    Ok((config, resumed_state))
}

fn configure_logging() {
    let subscriber = tracing_subscriber::fmt()
        .compact()
        .with_target(false)
        .with_max_level(LevelFilter::INFO)
        .finish();

    tracing::subscriber::set_global_default(subscriber).unwrap();
}

mod messages {
    use anyhow::Result;
    use serde::{Deserialize, Serialize};

    use std::sync::{
        mpsc::{self, Receiver, Sender},
        Arc,
    };

    use plane::ml::{embeddings::Embedding, NodeValue};

    use crate::training::{self, TrainerReport};

    pub enum TrainerMessage {
        PrintStatus,
        TogglePrintAllStatus,
        ReloadFromSnapshot,
        ForceSnapshot,
        WriteStatsToDisk,
        WriteEmbeddingTsvToDisk,
        WriteModelToDisk,
        WriteModelAndMetadataToDisk(TrainerStateMetadata),
        MultiplyLearnRateBy(NodeValue),
        IncreaseMaxRounds(usize),
        PredictRandomPhrase,
        TogglePause,
        Halt,
        ReplaceEmbeddingState(String, TrainerStateMetadata),
    }

    pub enum TrainerHandleActions {
        Nothing,
        LearnRateMulMut(NodeValue),
        IncreaseMaxRounds(usize),
        DispatchWithMetadata(Arc<dyn Fn(TrainerStateMetadata) -> TrainerMessage>),
        TogglePause,
        TogglePrintAllStatus,
        ReloadFromSnapshot,
        ForceSnapshot,
        PrintStatus,
        Halt,
        ReplaceEmbeddingState(String, TrainerStateMetadata),
    }

    pub struct TrainerHandle<Message: Send, F> {
        pub tx: Sender<Message>,
        pub rx: Receiver<Message>,
        pub handler: F,
    }

    impl<Message, F> TrainerHandle<Message, F>
    where
        Message: Send,
        F: Fn(&Embedding, Message) -> TrainerHandleActions,
    {
        pub fn new(handler: F) -> (Sender<Message>, Self) {
            let (tx, rx) = mpsc::channel();

            (tx.clone(), Self { tx, rx, handler })
        }

        pub fn send(&self, t: Message) -> Result<(), mpsc::SendError<Message>> {
            self.tx.send(t)
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct TrainerStateMetadata {
        pub learn_rate: f64,
        pub training_rounds: usize,
        pub current_round: usize,
        pub training_report: Option<TrainerReport>,
    }
}

mod training {
    use std::{
        collections::HashMap,
        fs::{DirBuilder, File},
        io::{BufReader, Read, Write},
        path,
        rc::Rc,
        time::{Duration, Instant},
    };

    use anyhow::Context;
    use plane::ml::{embeddings::Embedding, NetworkActivationMode, NodeValue};
    use serde_json::Value;
    use tracing::{debug, error, info};

    use crate::{
        config::TrainEmbeddingConfig,
        messages::{TrainerHandleActions, TrainerStateMetadata},
    };

    use super::*;

    pub fn setup_and_train_embeddings_v2<
        F: Fn(&Embedding, TrainerMessage) -> TrainerHandleActions,
    >(
        config: TrainEmbeddingConfig,
        handle: messages::TrainerHandle<TrainerMessage, F>,
    ) -> Embedding {
        let rng: Rc<dyn RNG> = Rc::new(JsRng::default());

        let (mut phrases, vocab) = init_phrases_and_vocab(&config, rng.clone());
        let testing_phrases = match config.phrase_test_set_split_pct.filter(|x| *x > 0.0) {
            Some(pct) => split_training_and_testing(&mut phrases, pct),
            None => phrases.clone(),
        };

        let layer_node_counts = [
            Some(config.hidden_layer_nodes),
            Some(config.hidden_deep_layer_nodes).filter(|x| *x > 0),
        ];

        let mut embedding = Embedding::new_builder(vocab, rng.clone())
            .with_embedding_dimensions(config.embedding_size)
            .with_hidden_layer_custom_shape(layer_node_counts.into_iter().flatten().collect())
            .with_input_stride_width(config.input_stride_width)
            .with_activation_mode(config.activation_mode)
            .build();

        let mut last_report: Option<(Instant, TrainerReport)> = None;
        let mut learn_rate = config.train_rate;
        let mut training_rounds = config.training_rounds;
        let mut force_report_all = false;
        let mut snapshot: (Option<String>, Instant) = (None, Instant::now());
        let mut paused = config.pause_on_start;
        let mut round = 0;

        loop {
            let training_error = if !paused {
                match embedding.train(&phrases, learn_rate, config.batch_size) {
                    Ok(error) => Some(error),
                    Err(e) => {
                        error!("Failed to train embedding model {e}");
                        if let (Some(_), when) = snapshot {
                            info!("Snapshot available from {when:?}. See help menu ('h') to restore... pausing")
                        }
                        paused = true;
                        None
                    }
                }
            } else {
                None
            };

            if !paused
                && training_error.is_some()
                && snapshot.1.elapsed().as_secs() > config.snapshot_interval_secs
            {
                let json = embedding.snapshot().unwrap();
                snapshot = (Some(json), Instant::now());
            }

            let mut force_report = false;
            let mut halt = false;

            while let Ok(msg) = if !paused {
                handle.rx.try_recv().map_err(|_| ())
            } else {
                handle
                    .rx
                    .recv_timeout(Duration::from_secs(5))
                    .map_err(|_| ())
            } {
                let action = (handle.handler)(&embedding, msg);
                match action {
                    TrainerHandleActions::Nothing => (),
                    TrainerHandleActions::LearnRateMulMut(factor) => {
                        let old_learn_rate = learn_rate;
                        learn_rate *= factor;
                        info!("Changed learn rate from {old_learn_rate} to {learn_rate}");
                    }
                    TrainerHandleActions::TogglePrintAllStatus => {
                        force_report_all = !force_report_all
                    }
                    TrainerHandleActions::TogglePause => {
                        paused = !paused;
                        if paused {
                            info!("Pausing rounds run, pausing...");
                        } else {
                            info!("Resuming rounds run, starting...");
                        }
                    }
                    TrainerHandleActions::IncreaseMaxRounds(rounds) => {
                        let old_training_rounds = training_rounds;
                        training_rounds += rounds;
                        info!("Changed max training rounds from {old_training_rounds} to {training_rounds}");
                        if paused {
                            info!("Resuming rounds run, starting...");
                            paused = false;
                        }
                    }
                    TrainerHandleActions::ReloadFromSnapshot => {
                        if let Some(snapshot) = &snapshot.0 {
                            info!("Recovering state from snapshot, pausing...");
                            embedding = Embedding::from_snapshot(&snapshot, rng.clone()).unwrap();
                            paused = true;
                        } else {
                            info!("No snapshot found to recover state from, continuing...");
                        }
                    }
                    TrainerHandleActions::ForceSnapshot => {
                        let json = embedding.snapshot().unwrap();
                        snapshot = (Some(json), Instant::now());
                        info!("Forced snapshot save to memory");
                    }
                    TrainerHandleActions::PrintStatus => force_report = true,
                    TrainerHandleActions::Halt => {
                        force_report = true;
                        halt = true;
                    }
                    TrainerHandleActions::DispatchWithMetadata(factory_fn) => {
                        let factory_fn = factory_fn.as_ref();
                        let state = TrainerStateMetadata {
                            learn_rate,
                            training_rounds,
                            current_round: round,
                            training_report: last_report.as_ref().map(|(_, report)| report.clone()),
                        };
                        let message = factory_fn(state);
                        handle.send(message).unwrap();
                    }
                    TrainerHandleActions::ReplaceEmbeddingState(snapshot, new_state) => {
                        embedding = Embedding::from_snapshot(&snapshot, rng.clone()).unwrap();
                        round = new_state.current_round;
                        training_rounds = new_state.training_rounds;
                        learn_rate = new_state.learn_rate;

                        last_report = None;
                        paused = true;
                        info!("Restored loaded model and state, pausing...");
                    }
                }
            }

            if force_report_all
                || force_report
                || should_report_round(round, config.training_rounds)
            {
                match validate_embeddings(&embedding, &testing_phrases, round) {
                    Ok((validation_errors, predictions_pct)) => {
                        let report = generate_training_report(
                            round,
                            training_error.unwrap_or(0.0),
                            validation_errors,
                            predictions_pct,
                            last_report
                                .as_ref()
                                .map(|(last_dt, report)| (last_dt.elapsed(), report.round)),
                        );

                        log_training_round(&report);

                        last_report = Some((std::time::Instant::now(), report));
                    }
                    Err(e) => {
                        error!("Failed to validate testing round {e}.. pausing");
                        paused = true;
                    }
                }
            }

            if halt {
                info!("Stopping rounds run...");
                break;
            }

            if !paused {
                round += 1;
            }

            if round == training_rounds && !paused {
                info!("Completed rounds run, pausing...");
                paused = true;
            }
        }

        embedding
    }

    pub fn init_phrases_and_vocab(
        config: &TrainEmbeddingConfig,
        rng: Rc<dyn RNG>,
    ) -> (Vec<Vec<String>>, HashSet<String>) {
        let phrases = training::parse_phrases();

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
        use itertools::Itertools;

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

        let vocab: HashSet<String> = phrases
            .iter()
            .flat_map(|phrase| phrase.iter())
            .cloned()
            .collect();

        dbg!((phrases.len(), vocab.len()));

        plane::ml::ShuffleRng::shuffle_vec(&rng, &mut phrases);
        (phrases, vocab)
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

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TrainerReport {
        round: usize,
        training_error: NodeValue,
        ms_per_round: Option<u128>,
        predictions_pct: NodeValue,
        validation_error: NodeValue,
        nll: NodeValue,
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

        let ms_per_round = last_report_time.map(|(duration, last_round)| {
            duration.as_millis() / (round - last_round).max(1) as u128
        });

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
        round: usize,
    ) -> Result<(Vec<(f64, f64)>, f64)> {
        let mut validation_errors = vec![];
        let mut correct_first_word_predictions = 0;
        let mut total_first_word_predictions = 0;

        for testing_phrase in testing_phrases.iter() {
            for testing_phrase_window in testing_phrase.windows(embedding.input_stride_width() + 1)
            {
                let (last_word_vector, context_word_vectors) =
                    testing_phrase_window
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
            / total_first_word_predictions as NodeValue;
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
        rng: Rc<dyn RNG>,
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
        use itertools::Itertools;

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
        use itertools::Itertools;

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
        use itertools::Itertools;

        let vocab = embedding.vocab().keys().cloned().collect::<HashSet<_>>();

        let fpath_nearest =
            output::create_output_fpath(("embedding", "_nearest.json"), None, None, None, label);
        let fpath_predictions = output::create_output_fpath(
            ("embedding", "_predictions.json"),
            None,
            None,
            None,
            label,
        );
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
