use clap::{command, Args, Parser, Subcommand};
use serde::{Deserialize, Serialize};

use plane::ml::{NetworkActivationMode, NodeValue};

#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
pub struct TrainEmbeddingConfig {
    #[arg(short = 'n', long, default_value_t = 4)]
    pub embedding_size: usize,

    #[arg(short = 'h', long, default_value_t = 75)]
    pub hidden_layer_nodes: usize,

    #[arg(short = 'H', long, default_value = None)]
    pub hidden_deep_layer_nodes: Option<usize>,

    #[arg(short = 'c', long, default_value_t = 1000)]
    pub training_rounds: usize,

    #[arg(short = 'w', long, default_value_t = 3)]
    pub input_stride_width: usize,

    #[arg(short = 'b', long, default_value_t = 16)]
    pub batch_size: usize,

    #[arg(short = 's', long, default_value_t = false)]
    #[serde(default)]
    pub single_batch_iterations: bool,

    #[arg(short = 'C', long = "char", default_value_t = false)]
    #[serde(default)]
    pub use_character_tokens: bool,

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

    #[arg(short = 'i', long, default_value = None)]
    #[serde(default)]
    pub input_txt_path: Option<String>,

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

#[derive(Args, Debug, Clone)]
pub struct LoadEmbeddingConfig {
    pub file_path: String,

    #[arg(short = 'h', long, default_value = None)]
    pub hidden_layer_nodes: Option<usize>,

    #[arg(short = 'H', long, default_value = None)]
    pub hidden_deep_layer_nodes: Option<usize>,

    #[arg(short = 'w', long, default_value = None)]
    pub input_stride_width: Option<usize>,
}

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

fn parse_range<T>(s: &str) -> Result<(T, T), Box<dyn std::error::Error + Send + Sync + 'static>>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let pos = s.find("..");
    let pos = pos.ok_or_else(|| format!("invalid KEY=value: no `..` found in `{s}`"))?;

    let range = (s[..pos].parse()?, s[pos + 2..].parse()?);
    Ok(range)
}
