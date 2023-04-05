use std::{sync::mpsc, thread};

use anyhow::{anyhow, Result};
use clap::{Args, FromArgMatches};
use crossterm::event::{self, Event, KeyCode, KeyEvent};
use tracing::metadata::LevelFilter;

use config::TrainEmbeddingConfig;
use messages::TrainerMessage;

mod config;
mod messages;
mod training;

fn main() -> Result<()> {
    configure_logging();
    let (config, resumed_state) = parse_cli_args()?;

    let config_clone = config.clone();
    let (tx, handle) = messages::TrainerHandle::new(move |embedding, msg: TrainerMessage| {
        let handle_action = msg.apply(embedding, &config_clone);
        handle_action
    });

    let config_clone = config.clone();
    let thread = thread::spawn(move || {
        let _embedding = training::setup_and_train_embeddings_v2(config_clone, handle);
    });

    if let Some((snapshot, state)) = resumed_state {
        tx.send(TrainerMessage::ReplaceEmbeddingState(snapshot, state))?;
    }

    let config_clone = config.clone();
    let ui_thread = thread::spawn(move || {
        use KeyCode::Char;

        loop {
            match event::read().unwrap() {
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
) -> Result<()> {
    match c {
        'r' => tx.send(TrainerMessage::PrintStatus)?,
        'o' => tx.send(TrainerMessage::PrintTrainingStatus)?,
        'R' => tx.send(TrainerMessage::SuppressAutoPrintStatus)?,
        'a' => tx.send(TrainerMessage::TogglePrintAllStatus)?,
        'x' => tx.send(TrainerMessage::ReloadFromSnapshot)?,
        'z' => tx.send(TrainerMessage::ForceSnapshot)?,
        'n' => tx.send(TrainerMessage::PredictRandomPhrase)?,
        'w' => tx.send(TrainerMessage::WriteStatsToDisk)?,
        'W' => tx.send(TrainerMessage::WriteEmbeddingTsvToDisk)?,
        's' => tx.send(TrainerMessage::WriteModelToDisk)?,
        ',' => tx.send(TrainerMessage::MultiplyLearnRateBy(0.5))?,
        '.' => tx.send(TrainerMessage::MultiplyLearnRateBy(2.0))?,
        'e' => tx.send(TrainerMessage::IncreaseMaxRounds(config.training_rounds))?,
        'p' => tx.send(TrainerMessage::TogglePause)?,
        'q' => {
            tx.send(TrainerMessage::Halt)?;
            Err(anyhow!("Application Halted"))?
        }
        'h' => {
            println!(
                r"Trainer help: 
                        'r' => report status (testing set)
                        'o' => report status (training set)
                        'R' => supress auto-report status
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
            );
        }
        _ => (),
    }
    Ok(())
}

fn configure_logging() {
    let subscriber = tracing_subscriber::fmt()
        .compact()
        .with_target(false)
        .with_max_level(LevelFilter::INFO)
        .finish();

    tracing::subscriber::set_global_default(subscriber).unwrap();
}

fn parse_cli_args() -> Result<(
    TrainEmbeddingConfig,
    Option<(String, messages::TrainerStateMetadata)>,
)> {
    let cli = clap::Command::new("embed").disable_help_flag(true).arg(
        clap::Arg::new("help")
            .long("help")
            .action(clap::ArgAction::Help),
    );
    let cli = config::Cli::augment_args(cli);
    let matches = cli.get_matches();
    let cli = config::Cli::from_arg_matches(&matches)
        .map_err(|err| err.exit())
        .unwrap();

    // let cli = config::Cli::parse();

    let (config, resumed_state) = match cli.command() {
        config::Command::TrainEmbedding(config) => (config, None),
        config::Command::LoadEmbedding(config) => load_embedding(config)?,
    };
    Ok((config, resumed_state))
}

fn load_embedding(
    load_config: config::LoadEmbeddingConfig,
) -> Result<(
    TrainEmbeddingConfig,
    Option<(String, messages::TrainerStateMetadata)>,
)> {
    let file_path = &load_config.file_path;
    let (snapshot, mut config, state) = training::writer::read_model_from_disk(file_path)?;

    config.pause_on_start = true;
    if let Some(hidden_layer_nodes) = load_config.hidden_layer_nodes {
        config.hidden_layer_nodes = hidden_layer_nodes;
    }
    if let Some(hidden_deep_layer_nodes) = load_config.hidden_deep_layer_nodes {
        config.hidden_deep_layer_nodes = Some(hidden_deep_layer_nodes);
    }
    if let Some(input_stride_width) = load_config.input_stride_width {
        config.input_stride_width = input_stride_width;
    }
    Ok((config, Some((snapshot, state))))
}
