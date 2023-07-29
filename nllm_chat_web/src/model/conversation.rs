use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum ChatMessage {
    FromModel(String),
    FromUser(String),
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ChatConvesation {
    messages: Vec<ChatMessage>,
}
