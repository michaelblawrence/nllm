use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use tracing::info;

use std::{
    ops::ControlFlow,
    sync::{mpsc, Arc},
    time::{Duration, Instant},
};

use plane::ml::{
    embeddings::builder::EmbeddingBuilder, seq2seq::transformer::CharacterTransformer, NodeValue,
};

use crate::{model::MLModel, training};

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
    Yield,
    NoOp,
    UnpauseForSingleIteration,
    PlotHeatMapGraphs,
    PrintConfig,
}

impl TrainerMessage {
    pub fn apply<M: MLModel>(
        self,
        model: &M,
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
                training::writer::plot_training_loss(&config, &metadata);
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
                    model,
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
                match model.as_embedding() {
                    Some(embedding) => training::writer::write_results_to_disk(&embedding, "embed"),
                    None => todo!("not yet implement for s2s"),
                };
                info!("Results written to disk");
                TrainerHandleActions::Nothing
            }
            // TODO: dispatch back metadata for consistent output label handling
            TrainerMessage::WriteEmbeddingTsvToDisk => {
                match model.as_embedding() {
                    Some(embedding) => training::writer::write_embedding_tsv_to_disk(
                        &embedding,
                        "embed",
                        &config.output_dir,
                        &config.output_label,
                    ),
                    None => todo!("not yet implement for s2s"),
                };
                info!("Embedding TSV written to disk");
                TrainerHandleActions::Nothing
            }
            TrainerMessage::PredictRandomPhrase => {
                let sep = if config.use_character_tokens { "" } else { " " };
                let generated_phrase = model.generate_sequence_string(sep);
                info!("Generated a new phrase:  {}", generated_phrase);
                TrainerHandleActions::Nothing
            }
            TrainerMessage::PlotHeatMapGraphs => {
                // plane::ml::transformer::linear::Linear::show_all_heatmap_plots();
                TrainerHandleActions::Nothing
            }
            TrainerMessage::PrintConfig => {
                let config_json = serde_json::to_string_pretty(&config).unwrap();
                info!("Model initialization config: {}", config_json);
                TrainerHandleActions::Nothing
            }
            TrainerMessage::Yield => TrainerHandleActions::Nothing,
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
    pub fn apply<M: MLModel>(
        self,
        state: &mut training::TrainerState<M>,
        handle: &TrainerHandle<TrainerMessage, M>,
    ) -> ControlFlow<()> {
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
                return ControlFlow::Break(());
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
                    if let Some(_) = state.model.as_embedding() {
                        match EmbeddingBuilder::from_snapshot(&snapshot).and_then(|x| x.build()) {
                            Ok(model) => {
                                state.model = model.into();
                                Ok(())
                            }
                            Err(e) => Err(e),
                        }
                    } else {
                        panic!("can not reload s2s yet")
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
                let json = state.model.snapshot().unwrap();
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
                if state.batches_trained_since_process_started() > 0 {
                    state.trigger_round_report_test_set();
                }
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
                let new_model = if let Some(embedding) = state.model.as_embedding() {
                    let (new_embedding, build_ctx) = EmbeddingBuilder::from_snapshot(&snapshot)
                        .unwrap()
                        .with_hidden_layer_custom_shape(state.hidden_layer_shape())
                        .with_input_stride_width(state.input_stride_width())
                        .build_advanced()
                        .unwrap();

                    if build_ctx.rebuilt_network {
                        let old_shape = embedding.shape().desc_pretty();
                        let new_shape = new_embedding.shape().desc_pretty();
                        info!("Built new hidden model from restored snapshot embedding data with shape = [{}] (previously [{}])",
                        new_shape, old_shape
                    );
                    }
                    new_embedding.into()
                } else if let Some(_) = state.model.as_s2s() {
                    let new_s2s = serde_json::from_str::<CharacterTransformer>(&snapshot).unwrap();
                    M::from_s2s(new_s2s, new_state.learn_rate)
                } else {
                    panic!("can not restore unknown model");
                };

                state.set_model(new_model, new_state);

                state.pause();
                info!("Restored loaded model and state, pausing...");
            }
        }
        ControlFlow::Continue(())
    }
}

pub type TrainerHandleSender<T> = mpsc::Sender<T>;

pub struct TrainerHandle<Message: Send, M> {
    pub tx: TrainerHandleSender<Message>,
    pub rx: mpsc::Receiver<Message>,
    pub handler: Arc<dyn Fn(&M, Message) -> TrainerHandleActions>,
}

unsafe impl<Message: Send, M> Send for TrainerHandle<Message, M> {}

impl<Message, M> TrainerHandle<Message, M>
where
    Message: Send,
{
    pub fn new<F>(handler: F) -> (TrainerHandleSender<Message>, Self)
    where
        F: Fn(&M, Message) -> TrainerHandleActions + 'static,
    {
        let (tx, rx) = mpsc::channel();

        (
            tx.clone(),
            Self {
                tx,
                rx,
                handler: Arc::new(handler),
            },
        )
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

    pub fn run(&self, embedding: &M, msg: Message) -> TrainerHandleActions {
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
