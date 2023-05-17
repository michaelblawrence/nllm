use clap::{command, Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

use plane::ml::{NetworkActivationMode, NodeValue};

#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
pub struct TrainEmbeddingConfig {
    #[arg(short = 'n', long, default_value_t = 4)]
    pub embedding_size: usize,

    #[arg(short = 'h', long, default_value_t = 75)]
    pub hidden_layer_nodes: usize,

    #[arg(short = 'H', long, default_value = None)]
    pub hidden_deep_layer_nodes: Option<String>,

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

    #[arg(short = 'N', long, default_value_t = false)]
    #[serde(default)]
    pub sample_from_newline: bool,

    #[arg(short = 'L', long, default_value_t = false)]
    #[serde(default)]
    pub use_transformer: bool,

    #[arg(short = 'r', long, default_value_t = 1e-3)]
    pub train_rate: NodeValue,

    #[arg(short = 'q', long, default_value_t = false)]
    #[serde(default)]
    pub quit_on_complete: bool,

    #[arg(short = 'p', long, default_value_t = false)]
    #[serde(default)]
    pub pause_on_start: bool,

    #[arg(short = 'o', long, default_value = None)]
    #[serde(default)]
    pub output_dir: Option<String>,

    #[arg(short = 'O', long, default_value = None)]
    #[serde(default)]
    pub output_label: Option<String>,

    #[arg(short = 'Z', long, default_value = None)]
    #[serde(default)]
    pub output_label_append_details: bool,

    #[arg(short = 'i', long, default_value = None)]
    #[serde(default)]
    pub input_txt_path: Option<String>,

    #[arg(long, default_value = None)]
    #[serde(default)]
    pub repl: Option<String>,

    #[arg(short = 'S', long, default_value_t = 120)]
    pub snapshot_interval_secs: u64,

    #[arg(short = 'T', long, default_value = None)]
    pub phrase_train_set_size: Option<usize>,

    #[arg(short = 'B', long, value_parser = parse_range::<usize>, default_value = "5..10")]
    pub phrase_word_length_bounds: (Option<usize>, Option<usize>),

    #[arg(short = 'X', long, default_value = "20")]
    pub phrase_test_set_split_pct: Option<NodeValue>,

    #[arg(short = 'W', long, default_value = None)]
    #[serde(default)]
    pub phrase_test_set_max_tokens: Option<usize>,

    #[arg(short = 'w', long, default_value = None)]
    #[serde(default)]
    pub phrase_split_seed: Option<isize>,

    #[arg(short = 'm', long, value_enum, default_value_t = LayerActivationConfig::Tanh)]
    pub activation_mode: LayerActivationConfig,
}

#[derive(Args, Debug, Clone)]
pub struct LoadEmbeddingConfig {
    pub file_path: String,

    #[arg(short = 'h', long, default_value = None)]
    pub hidden_layer_nodes: Option<usize>,

    #[arg(short = 'H', long, default_value = None)]
    pub hidden_deep_layer_nodes: Option<String>,

    #[arg(short = 'w', long, default_value = None)]
    pub input_stride_width: Option<usize>,

    #[arg(short = 'b', long, default_value = None)]
    pub batch_size: Option<usize>,

    #[arg(short = 'i', long, default_value = None)]
    pub input_txt_path: Option<String>,

    #[arg(short = 'W', long, default_value = None)]
    pub phrase_test_set_max_tokens: Option<usize>,

    #[arg(long, default_value_t = false)]
    pub force_continue: bool,

    #[arg(long, default_value = None)]
    pub repl: Option<String>,
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

#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize)]
pub enum LayerActivationConfig {
    Linear,
    SoftMax,
    Sigmoid,
    Tanh,
    RelU,
}

impl Into<NetworkActivationMode> for LayerActivationConfig {
    fn into(self) -> NetworkActivationMode {
        use NetworkActivationMode::*;
        match self {
            Self::Linear => Linear,
            Self::SoftMax => SoftMaxCrossEntropy,
            Self::Sigmoid => Sigmoid,
            Self::Tanh => Tanh,
            Self::RelU => RelU,
        }
    }
}

fn parse_range<T>(
    s: &str,
) -> Result<(Option<T>, Option<T>), Box<dyn std::error::Error + Send + Sync + 'static>>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let s = s.trim();
    let pos = s.find("..");
    let pos = pos.ok_or_else(|| format!("invalid KEY=value: no `..` found in `{s}`"))?;

    let range_start = if pos > 0 {
        Some(s[..pos].parse()?)
    } else {
        None
    };
    let range_end = if pos + 2 < s.len() {
        Some(s[pos + 2..].parse()?)
    } else {
        None
    };

    Ok((range_start, range_end))
}
