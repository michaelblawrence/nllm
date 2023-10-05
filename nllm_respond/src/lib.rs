use std::{collections::HashMap, fs::File, io::Read, path::Path};

use anyhow::{bail, Context, Result};

use plane::ml::{
    gdt::{GenerativeDecoderTransformer, InferContext, InferSampling},
    LayerValues, NodeValue,
};

pub enum RespondModel {
    GDT(GenerativeDecoderTransformer),
}

impl RespondModel {
    pub fn from_snapshot(json: &str) -> Result<Self> {
        Ok(Self::GDT(serde_json::from_str(json)?))
    }
    pub fn description(&self) -> &str {
        match self {
            RespondModel::GDT(_) => "GenerativeDecoderTransformer",
        }
    }
    pub fn vocab(&self) -> HashMap<String, usize> {
        match self {
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
            RespondModel::GDT(gdt) => {
                let token_sequence = match gdt.vocab().encode(input_txt) {
                    Ok(token_sequence) => token_sequence,
                    Err(e) => {
                        println!("Failed to encode prompt sequence: {e}");
                        return Box::new(std::iter::empty());
                    }
                };
                let temp = std::env::var("NLLM_TEMP")
                    .ok()
                    .and_then(|x| x.parse::<NodeValue>().ok())
                    .unwrap_or_else(|| {
                        println!("PLEASE SET 'NLLM_TEMP' env var");
                        1.0
                    });
                Box::new(
                    gdt.generate_sequence_iter(
                        &token_sequence,
                        InferContext {
                            sampling: InferSampling::Temperature(temp),
                            max_len: 40,
                        },
                    )
                    .map(|c| c.appendable()),
                )
            }
        }
    }

    pub fn predict_next(&self, input_txt: &str) -> Result<String> {
        match self {
            RespondModel::GDT(gdt) => {
                let token_sequence = gdt.vocab().encode(input_txt)?;
                Ok(gdt.predict_next(&token_sequence)?.to_string())
            }
        }
    }

    pub fn control_vocab(&self) -> String {
        match self {
            RespondModel::GDT(gdt) => gdt.control_vocab().to_string(),
        }
    }

    pub fn sample_probabilities(&self, input_txt: &str) -> Result<LayerValues> {
        match self {
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
    let (_json, state) = extract_config(&json).expect("failed to parse config");

    let model =
        RespondModel::from_snapshot(&json).expect("failed to rebuild instance from snapshot state");

    Ok((model, state))
}

pub struct PromptConfig {
    pub chat_mode: PromptChatMode,
}

#[derive(Debug, Clone, Copy)]
pub enum PromptChatMode {
    DirectPrompt,
    TripleHashHumanPrompt,
}

pub fn process_prompt<'a>(
    model: &'a RespondModel,
    input_txt: &'a str,
    PromptConfig { chat_mode }: &PromptConfig,
) -> Result<impl Iterator<Item = (String, &'a str)> + 'a> {
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

    let response_iter = model.predict_from_iter(&prompt_txt);

    let token_separator_pair_iter: Box<dyn Iterator<Item = (String, &str)>> =
        Box::new(response_iter.map(|x| (x, "")));

    match chat_mode {
        PromptChatMode::DirectPrompt => Ok(token_separator_pair_iter),
        PromptChatMode::TripleHashHumanPrompt => Ok(Box::new(
            token_separator_pair_iter
                .take_while(|(token, _separator)| !token.trim_start().starts_with("###")),
        )),
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

pub struct ExtractedModelConfig {
    pub input_txt_path: Option<String>,
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
    let use_gdt = config["use_gdt"].as_bool().unwrap_or(true);

    assert!(use_gdt, "use_gdt must be true, if set");

    Ok((snapshot, ExtractedModelConfig { input_txt_path }))
}
