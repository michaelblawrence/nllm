use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::Read,
    path::Path,
};

use anyhow::{bail, Context, Result};

use plane::ml::{
    embeddings::{builder::EmbeddingBuilder, Embedding},
    gdt::GenerativeDecoderTransformer,
    seq2seq::transformer::CharacterTransformer,
    LayerValues, RngStrategy,
};

pub enum RespondModel {
    Embedding(Embedding),
    S2S(CharacterTransformer),
    GDT(GenerativeDecoderTransformer),
}

impl RespondModel {
    pub fn from_snapshot(use_transformer: bool, use_gdt: bool, json: &str) -> Result<Self> {
        if use_gdt {
            Ok(Self::GDT(serde_json::from_str(json)?))
        } else if use_transformer {
            Ok(Self::S2S(serde_json::from_str(json)?))
        } else {
            let rng = RngStrategy::default();
            Ok(Self::Embedding(
                EmbeddingBuilder::from_snapshot(&json)
                    .context("failed to rebuild state from snapshot")?
                    .with_rng(rng.clone())
                    .build()?,
            ))
        }
    }
    pub fn description(&self) -> &str {
        match self {
            RespondModel::Embedding(_) => "Embedding",
            RespondModel::S2S(_) => "CharacterTransformer",
            RespondModel::GDT(_) => "GenerativeDecoderTransformer",
        }
    }
    pub fn vocab(&self) -> HashMap<String, usize> {
        match self {
            RespondModel::Embedding(embedding) => embedding.vocab().clone(),
            RespondModel::S2S(s2s) => s2s
                .vocab()
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect(),
            RespondModel::GDT(gdt) => gdt
                .vocab()
                .dictionary()
                .into_iter()
                .map(|(x, y)| (x.to_string(), y))
                .collect(),
        }
    }
    pub fn predict_from_iter<'a>(
        &'a self,
        input_txt: &str,
    ) -> Box<dyn Iterator<Item = String> + 'a> {
        match self {
            RespondModel::Embedding(embedding) => {
                let append_space = embedding.vocab().contains_key(" ");
                let token_sequence: Vec<String> =
                    to_context_tokens(input_txt, true, append_space).collect();
                Box::new(embedding.predict_from_iter(&token_sequence))
            }
            RespondModel::S2S(s2s) => {
                let mut token_sequence: Vec<char> = input_txt.chars().collect();
                if s2s.vocab().contains_key(&' ') {
                    token_sequence.push(' ');
                }
                Box::new(
                    s2s.predict_from_iter(&token_sequence)
                        .map(|c| c.to_string()),
                )
            }
            RespondModel::GDT(gdt) => {
                let token_sequence = match gdt.vocab().encode(input_txt) {
                    Ok(token_sequence) => token_sequence,
                    Err(e) => {
                        println!("Failed to encode prompt sequence: {e}");
                        return Box::new(std::iter::empty());
                    }
                };
                Box::new(
                    gdt.predict_from_iter(&token_sequence)
                        .map(|c| c.appendable()),
                )
            }
        }
    }

    pub fn predict_next(&self, input_txt: &str) -> Result<String> {
        match self {
            RespondModel::Embedding(embedding) => {
                let append_space = embedding.vocab().contains_key(" ");
                let token_sequence: Vec<String> =
                    to_context_tokens(input_txt, true, append_space).collect();
                embedding.predict_next(&token_sequence)
            }
            RespondModel::S2S(s2s) => {
                let mut token_sequence: Vec<char> = input_txt.chars().collect();
                if s2s.vocab().contains_key(&' ') {
                    token_sequence.push(' ');
                }
                Ok(s2s.predict_next(&token_sequence)?.to_string())
            }
            RespondModel::GDT(gdt) => {
                let token_sequence = gdt.vocab().encode(input_txt)?;
                Ok(gdt.predict_next(&token_sequence)?.to_string())
            }
        }
    }

    pub fn control_vocab(&self) -> String {
        match self {
            RespondModel::Embedding(embedding) => embedding.control_vocab().to_string(),
            RespondModel::S2S(s2s) => s2s.control_vocab().to_string(),
            RespondModel::GDT(gdt) => gdt.control_vocab().to_string(),
        }
    }

    pub fn sample_probabilities(&self, input_txt: &str) -> Result<LayerValues> {
        match self {
            RespondModel::Embedding(embedding) => {
                let append_space = embedding.vocab().contains_key(" ");
                let token_sequence: Vec<String> =
                    to_context_tokens(input_txt, true, append_space).collect();
                embedding.sample_probabilities(&token_sequence)
            }
            RespondModel::S2S(s2s) => {
                let mut token_sequence: Vec<char> = input_txt.chars().collect();
                if s2s.vocab().contains_key(&' ') {
                    token_sequence.push(' ');
                }
                s2s.sample_probabilities(&token_sequence)
            }
            RespondModel::GDT(gdt) => {
                let token_sequence = gdt.vocab().encode(input_txt)?;
                gdt.sample_probabilities(&token_sequence)
            }
        }
    }
}

pub fn load(model_fpath: Option<&str>) -> Result<(RespondModel, ExtractedModelConfig)> {
    let json = match read_model_file(model_fpath) {
        Some(value) => value,
        None => bail!("failed to read model file"),
    };

    let (model, state) = from_json(&json)?;

    println!("..");
    println!(
        "Loaded {} model from '{}'",
        model.description(),
        model_fpath.clone().unwrap_or_default()
    );

    Ok((model, state))
}

pub fn from_json(json: &str) -> Result<(RespondModel, ExtractedModelConfig)> {
    let (_json, mut state) = extract_config(&json).expect("failed to parse config");

    let model = RespondModel::from_snapshot(state.use_transformer, state.use_gdt, &json)
        .expect("failed to rebuild instance from snapshot state");

    let char_mode = model
        .vocab()
        .iter()
        .all(|(token, id)| *id == 0 || token.chars().count() == 1);

    state.char_mode = Some(char_mode);

    // let embedding = EmbeddingBuilder::from_snapshot(&json)
    //     .expect("failed to rebuild state from snapshot")
    //     .with_rng(rng.clone())
    //     .build()
    //     .expect("failed to rebuild instance from snapshot state");

    Ok((model, state))
}

pub struct PromptConfig<'a> {
    pub use_gdt: bool,
    pub char_mode: bool,
    pub vocab_supervised_predictions_enabled: bool, // TODO: remove
    pub chat_mode: PromptChatMode,
    pub vocab: Option<&'a HashSet<String>>, // TODO: remove
}

#[derive(Debug, Clone, Copy)]
pub enum PromptChatMode {
    DirectPrompt,
    TripleHashHumanPrompt,
}

pub fn process_prompt<'a>(
    model: &'a RespondModel,
    rng: &RngStrategy,
    input_txt: &'a str,
    PromptConfig {
        use_gdt,
        char_mode,
        vocab_supervised_predictions_enabled: _,
        chat_mode,
        vocab: _,
    }: &PromptConfig,
) -> Result<impl Iterator<Item = (String, &'a str)> + 'a> {
    let (char_mode, use_gdt) = (*char_mode, *use_gdt);
    let append_space = char_mode && model.vocab().contains_key(" ");

    let prompt_txt = match chat_mode {
        PromptChatMode::DirectPrompt => input_txt.to_string(),
        PromptChatMode::TripleHashHumanPrompt => {
            format!(
                r#"###human: {} ###ctx__: "" ###chat_: "#,
                input_txt.to_ascii_lowercase()
            )
        }
    };

    match model.predict_next(&prompt_txt) {
        Err(e) => {
            println!("Error retrieving response: ({e})");
            return Err(e);
        }
        _ => (),
    };

    let response = model.predict_from_iter(&prompt_txt);

    let response: Box<dyn Iterator<Item = (String, &str)>> = if use_gdt {
        Box::new(response.into_iter().map(|x| (x, "")))
    } else if char_mode {
        if append_space {
            Box::new(response.into_iter().map(|x| (x, "")))
        } else {
            Box::new(
                [(prompt_txt.to_string(), "")]
                    .into_iter()
                    .chain(response.into_iter().skip(1).map(|x| (x, ""))),
            )
        }
    } else {
        Box::new(response.into_iter().map(|x| (x, " ")))
    };

    match chat_mode {
        PromptChatMode::DirectPrompt => Ok(response),
        PromptChatMode::TripleHashHumanPrompt => {
            Ok(Box::new(response.take_while(|(token, _separator)| {
                !token.trim_start().starts_with("###")
            })))
        }
    }
}

fn read_model_file(fpath_arg: Option<&str>) -> Option<String> {
    let path = match fpath_arg {
        Some(path) => {
            let path = Path::new(path);
            if !path.exists() {
                println!("File not found");
                return None;
            }
            path
        }
        None => {
            println!("usage: [respond] <MODEL_FILE_PATH>");
            return None;
        }
    };

    let mut file = File::open(path).expect("failed to open file");
    let mut buf = String::new();

    file.read_to_string(&mut buf)
        .expect("failed to read file contents");

    Some(buf)
}

fn to_context_tokens<'a>(
    input_txt: &'a str,
    char_mode: bool,
    append_space: bool,
) -> impl Iterator<Item = String> + 'a {
    let context_tokens: Vec<String> = if char_mode {
        let mut vec: Vec<String> = input_txt.chars().map(|c| c.to_string()).collect();
        if append_space {
            vec.push(" ".to_string())
        }
        vec
    } else {
        input_txt
            .split_whitespace()
            .map(|c| c.to_string())
            .collect()
    };
    context_tokens.into_iter()
}

fn is_token_alphabetic(token: &String) -> bool {
    token.chars().next().unwrap_or(' ').is_alphabetic()
}

pub struct ExtractedModelConfig {
    pub input_txt_path: Option<String>,
    pub use_transformer: bool,
    pub use_gdt: bool,
    pub char_mode: Option<bool>,
}

fn extract_config(json: &str) -> Result<(String, ExtractedModelConfig)> {
    use serde_json::Value;

    let mut snapshot: Value = serde_json::from_str(&json)?;

    let config: Value = serde_json::from_value(snapshot["_trainer_config"].take())
        .context("unable to extract trainer config from saved model")?;

    let _state: Value = serde_json::from_value(snapshot["_trainer_state"].take())
        .context("unable to extract trainer metatdata from saved model")?;

    let snapshot = serde_json::to_string(&snapshot)?;

    let input_txt_path = config["input_txt_path"].as_str().map(|x| x.to_string());
    let use_transformer = config["use_transformer"].as_bool().unwrap_or_default();
    let use_gdt = config["use_gdt"].as_bool().unwrap_or_default();

    Ok((
        snapshot,
        ExtractedModelConfig {
            input_txt_path,
            use_transformer,
            use_gdt,
            char_mode: None,
        },
    ))
}
