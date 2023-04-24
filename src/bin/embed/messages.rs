use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use tracing::info;

use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

use plane::ml::{
    embeddings::{builder::EmbeddingBuilder, Embedding},
    NodeValue,
};

use crate::training;

#[derive(Debug, Serialize, Deserialize)]
pub enum TrainerMessage {
    PrintTrainingStatus,
    PrintStatus,
    TogglePrintAllStatus,
    ReloadFromSnapshot,
    ForceSnapshot,
    WriteStatsToDisk,
    WriteEmbeddingTsvToDisk,
    RenameOutputLabel(String),
    WriteModelToDisk,
    WriteModelAndMetadataToDisk(TrainerStateMetadata),
    MultiplyLearnRateBy(NodeValue),
    IncreaseMaxRounds(usize),
    PredictRandomPhrase,
    TogglePause,
    Halt,
    ReplaceEmbeddingState(String, TrainerStateMetadata),
    SuppressAutoPrintStatus,
    PrintEachRoundNumber,
    PlotTrainingLossGraph,
    PlotTrainingLossGraphDispatch(TrainerStateMetadata),
    NoOp,
    UnpauseForSingleIteration,
}

impl TrainerMessage {
    pub fn apply(
        self,
        embedding: &Embedding,
        config: &crate::config::TrainEmbeddingConfig,
    ) -> TrainerHandleActions {
        match self {
            TrainerMessage::PrintTrainingStatus => TrainerHandleActions::PrintTrainingStatus,
            TrainerMessage::PrintStatus => TrainerHandleActions::PrintStatus,
            TrainerMessage::Halt => TrainerHandleActions::Halt,
            TrainerMessage::UnpauseForSingleIteration => {
                TrainerHandleActions::UnpauseForSingleIteration
            }
            TrainerMessage::TogglePause => TrainerHandleActions::TogglePause,
            TrainerMessage::TogglePrintAllStatus => TrainerHandleActions::TogglePrintAllStatus,
            TrainerMessage::ReloadFromSnapshot => TrainerHandleActions::ReloadFromSnapshot,
            TrainerMessage::ForceSnapshot => TrainerHandleActions::ForceSnapshot,
            TrainerMessage::PrintEachRoundNumber => TrainerHandleActions::PrintEachRoundNumber,
            TrainerMessage::SuppressAutoPrintStatus => {
                TrainerHandleActions::SuppressAutoPrintStatus
            }
            TrainerMessage::IncreaseMaxRounds(x) => TrainerHandleActions::IncreaseMaxRounds(x),
            TrainerMessage::MultiplyLearnRateBy(x) => TrainerHandleActions::LearnRateMulMut(x),
            TrainerMessage::ReplaceEmbeddingState(embedding, state) => {
                TrainerHandleActions::ReplaceEmbeddingState(embedding, state)
            }
            TrainerMessage::PlotTrainingLossGraph => {
                TrainerHandleActions::DispatchWithMetadata(TrainerMessage::PlotTrainingLossGraph)
            }
            TrainerMessage::PlotTrainingLossGraphDispatch(metadata) => {
                training::writer::plot_training_loss(&embedding, &config, &metadata);
                TrainerHandleActions::Nothing
            }
            TrainerMessage::RenameOutputLabel(output_label) => {
                TrainerHandleActions::RenameOutputLabel(output_label)
            }
            TrainerMessage::WriteModelToDisk => {
                TrainerHandleActions::DispatchWithMetadata(TrainerMessage::WriteModelToDisk)
            }
            TrainerMessage::WriteModelAndMetadataToDisk(metadata) => {
                let path = training::writer::write_model_to_disk(
                    &embedding,
                    &config,
                    &metadata,
                    "embed",
                    &config.output_dir,
                    &config.output_label,
                );
                info!("Model written to disk: {}", path.to_string_lossy());
                TrainerHandleActions::Nothing
            }
            TrainerMessage::WriteStatsToDisk => {
                training::writer::write_results_to_disk(&embedding, "embed");
                info!("Results written to disk");
                TrainerHandleActions::Nothing
            }
            // TODO: dispatch back metadata for consistent output label handling
            TrainerMessage::WriteEmbeddingTsvToDisk => {
                training::writer::write_embedding_tsv_to_disk(
                    &embedding,
                    "embed",
                    &config.output_dir,
                    &config.output_label,
                );
                info!("Embedding TSV written to disk");
                TrainerHandleActions::Nothing
            }
            TrainerMessage::PredictRandomPhrase => {
                let sep = if config.use_character_tokens { "" } else { " " };
                let generated_phrase = embedding.generate_sequence_string(sep);
                info!("Generated a new phrase:  {}", generated_phrase);
                TrainerHandleActions::Nothing
            }
            TrainerMessage::NoOp => TrainerHandleActions::Nothing,
        }
    }
    pub fn create_response(&self, metadata: TrainerStateMetadata) -> Option<Self> {
        match self {
            TrainerMessage::WriteModelToDisk => {
                Some(TrainerMessage::WriteModelAndMetadataToDisk(metadata))
            }
            TrainerMessage::PlotTrainingLossGraph => {
                Some(TrainerMessage::PlotTrainingLossGraphDispatch(metadata))
            }
            _ => None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum TrainerHandleActions {
    Nothing,
    LearnRateMulMut(NodeValue),
    IncreaseMaxRounds(usize),
    DispatchWithMetadata(TrainerMessage),
    RenameOutputLabel(String),
    TogglePause,
    TogglePrintAllStatus,
    ReloadFromSnapshot,
    ForceSnapshot,
    PrintTrainingStatus,
    PrintStatus,
    Halt,
    ReplaceEmbeddingState(String, TrainerStateMetadata),
    SuppressAutoPrintStatus,
    PrintEachRoundNumber,
    UnpauseForSingleIteration,
}

impl TrainerHandleActions {
    pub fn apply<F>(
        self,
        state: &mut training::TrainerState,
        handle: &TrainerHandle<TrainerMessage, F>,
    ) where
        F: Fn(&Embedding, TrainerMessage) -> TrainerHandleActions,
    {
        match self {
            TrainerHandleActions::Nothing => (),
            TrainerHandleActions::LearnRateMulMut(factor) => {
                let old_learn_rate = state.learn_rate;
                state.learn_rate *= factor;
                let learn_rate = state.learn_rate;
                info!("Changed learn rate from {old_learn_rate} to {learn_rate}");
            }
            TrainerHandleActions::TogglePrintAllStatus => {
                state.force_report_all = !state.force_report_all
            }
            TrainerHandleActions::UnpauseForSingleIteration => {
                info!("Pausing after next round...");
                state.defer_pause(1);
            }
            TrainerHandleActions::TogglePause => {
                if state.training_paused() {
                    state.unpause();
                    info!("Resuming rounds run, starting...");
                } else {
                    state.pause();
                    info!("Pausing rounds run, pausing...");
                }
            }
            TrainerHandleActions::IncreaseMaxRounds(rounds) => {
                let old_training_rounds = state.training_rounds;
                state.training_rounds += rounds;
                let training_rounds = state.training_rounds;
                info!(
                    "Changed max training rounds from {old_training_rounds} to {training_rounds}"
                );
                if state.unpause() {
                    info!("Resuming rounds run, starting...");
                }
            }
            TrainerHandleActions::ReloadFromSnapshot => {
                let state_recovered = if let Some(snapshot) = &state.snapshot.0 {
                    info!("Recovering state from snapshot..");
                    match EmbeddingBuilder::from_snapshot(&snapshot).and_then(|x| x.build()) {
                        Ok(embedding) => {
                            state.embedding = embedding;
                            Ok(())
                        }
                        Err(e) => Err(e),
                    }
                } else {
                    Err(anyhow!(
                        "No snapshot found to recover state from, continuing..."
                    ))
                };

                match state_recovered {
                    Ok(_) => {
                        info!("Recovered state from snapshot, pausing..");
                        state.pause();
                    }
                    Err(e) => info!("Failed to restore snapshot: '{}', continuing...", e),
                };
            }
            TrainerHandleActions::ForceSnapshot => {
                let json = state.embedding.snapshot().unwrap();
                state.snapshot = (Some(json), Instant::now());
                info!("Forced snapshot save to memory");
            }
            TrainerHandleActions::PrintTrainingStatus => state.trigger_round_report_train_set(),
            TrainerHandleActions::PrintStatus => state.trigger_round_report_test_set(),
            TrainerHandleActions::SuppressAutoPrintStatus => {
                state.supress_auto_report = !state.supress_auto_report
            }
            TrainerHandleActions::PrintEachRoundNumber => {
                state.print_round_number = !state.print_round_number
            }
            TrainerHandleActions::Halt => {
                state.trigger_round_report_test_set();
                state.halt();
            }
            TrainerHandleActions::DispatchWithMetadata(request) => {
                let metadata = state.metadata();
                let message = request.create_response(metadata).unwrap();
                handle.send(message).unwrap();
            }
            TrainerHandleActions::RenameOutputLabel(output_label) => {
                state.set_output_label(output_label);
            }
            TrainerHandleActions::ReplaceEmbeddingState(snapshot, new_state) => {
                let (new_embedding, build_ctx) = EmbeddingBuilder::from_snapshot(&snapshot)
                    .unwrap()
                    .with_hidden_layer_custom_shape(state.hidden_layer_shape())
                    .with_input_stride_width(state.input_stride_width())
                    .build_advanced()
                    .unwrap();

                if build_ctx.rebuilt_network {
                    let old_shape = state.embedding.shape().desc_pretty();
                    let new_shape = new_embedding.shape().desc_pretty();
                    info!("Built new hidden model from restored snapshot embedding data with shape = [{}] (previously [{}])",
                        new_shape, old_shape
                    );
                }

                state.set_embedding(new_embedding, new_state);
                state.pause();
                info!("Restored loaded model and state, pausing...");
            }
        }
    }
}

pub type TrainerHandleSender<T> = mpsc::Sender<T>;

pub struct TrainerHandle<Message: Send, F> {
    pub tx: TrainerHandleSender<Message>,
    pub rx: mpsc::Receiver<Message>,
    pub handler: F,
}

impl<Message, F> TrainerHandle<Message, F>
where
    Message: Send,
    F: Fn(&Embedding, Message) -> TrainerHandleActions,
{
    pub fn new(handler: F) -> (TrainerHandleSender<Message>, Self) {
        let (tx, rx) = mpsc::channel();

        (tx.clone(), Self { tx, rx, handler })
    }

    pub fn send(&self, t: Message) -> Result<(), mpsc::SendError<Message>> {
        self.tx.send(t)
    }

    #[cfg(feature = "thread")]
    pub fn try_recv(&self, timeout: Option<Duration>) -> Result<Message> {
        match timeout {
            Some(timeout) => self
                .rx
                .recv_timeout(timeout)
                .context("failed to recv message"),
            None => self.rx.try_recv().context("failed to recv message"),
        }
    }

    #[cfg(not(feature = "thread"))]
    pub fn try_recv(&self, timeout: Option<Duration>) -> Result<Message> {
        match timeout {
            Some(timeout) => {
                std::thread::sleep(timeout);
                self.rx.try_recv().context("failed to recv message")
            }
            None => self.rx.try_recv().context("failed to recv message"),
        }
    }

    pub fn run(&self, embedding: &Embedding, msg: Message) -> TrainerHandleActions {
        let handler = &self.handler;
        handler(&embedding, msg)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainerStateMetadata {
    pub learn_rate: f64,
    pub training_rounds: usize,
    pub current_round: usize,
    #[serde(default)]
    pub output_label: Option<String>,
    #[serde(default)]
    pub total_train_seconds: u64,
    #[serde(default)]
    pub training_report: Option<TrainerReport>,
    #[serde(default)]
    pub training_error_history: Vec<(usize, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerReport {
    pub round: usize,
    pub training_error: NodeValue,
    pub ms_per_round: Option<u128>,
    pub predictions_pct: NodeValue,
    pub validation_error: NodeValue,
    pub nll: NodeValue,
    pub label: Option<String>,
    #[serde(default)]
    pub generated_time: u128,
}
