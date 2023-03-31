use anyhow::Result;
use serde::{Deserialize, Serialize};

use std::sync::{
    mpsc::{self, Receiver, Sender},
    Arc,
};

use plane::ml::{embeddings::Embedding, NodeValue};

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerReport {
    pub round: usize,
    pub training_error: NodeValue,
    pub ms_per_round: Option<u128>,
    pub predictions_pct: NodeValue,
    pub validation_error: NodeValue,
    pub nll: NodeValue,
}
