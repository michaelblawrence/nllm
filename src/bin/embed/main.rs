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
use crossterm::event::{self, Event, KeyCode, KeyEvent};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use tracing::{info, metadata::LevelFilter};

use config::TrainEmbeddingConfig;
use messages::{TrainerHandleActions, TrainerMessage};

use plane::ml::{embeddings::Embedding, JsRng, NetworkActivationMode, NodeValue, RNG};

mod config;
mod messages;
mod training;

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
        TrainerMessage::IncreaseMaxRounds(x) => TrainerHandleActions::IncreaseMaxRounds(x),
        TrainerMessage::MultiplyLearnRateBy(x) => TrainerHandleActions::LearnRateMulMut(x),
        TrainerMessage::ReplaceEmbeddingState(embedding, state) => {
            TrainerHandleActions::ReplaceEmbeddingState(embedding, state)
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
        TrainerMessage::PredictRandomPhrase => {
            let vocab_idx = JsRng::default().rand_range(0, embedding.vocab().len());
            let seed_word = embedding.vocab().keys().nth(vocab_idx).unwrap();
            let predicted_phrase = embedding.predict_iter(seed_word).join(" ");
            info!("Predicted a new random phrase:  {}", predicted_phrase);
            TrainerHandleActions::Nothing
        }
    }
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
    let cli = config::Cli::parse();
    let (config, resumed_state) = match cli.command() {
        config::Command::TrainEmbedding(config) => (config, None),
        config::Command::LoadEmbedding(config) => load_embedding(config)?,
    };
    Ok((config, resumed_state))
}

fn load_embedding(
    config: config::LoadEmbeddingConfig,
) -> Result<(
    TrainEmbeddingConfig,
    Option<(String, messages::TrainerStateMetadata)>,
)> {
    let rng = Rc::new(JsRng::default());
    let file_path = &config.file_path;
    let (snapshot, mut config, state) = training::read_model_from_disk(file_path, rng)?;

    config.pause_on_start = true;
    Ok((config, Some((snapshot, state))))
}
