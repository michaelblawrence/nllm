use std::{
    collections::{HashSet, VecDeque},
    env,
    fs::File,
    io::{self, Read, Write},
    path::Path,
    thread,
    time::Duration,
};

use anyhow::{Context, Result};
use itertools::Itertools;
use plane::ml::{embeddings::builder::EmbeddingBuilder, RngStrategy, SamplingRng};

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let fpath_arg = args.first();
    let mut fpath_arg = fpath_arg.cloned();

    while run(&mut fpath_arg) {}
}

fn run(model_fpath: &mut Option<String>) -> bool {
    let json = match read_model_file(model_fpath.as_ref()) {
        Some(value) => value,
        None => return false,
    };

    let (json, state) = extract_config(json).expect("failed to parse config");
    let rng = RngStrategy::default();

    let embedding = EmbeddingBuilder::from_snapshot(&json)
        .expect("failed to rebuild state from snapshot")
        .with_rng(rng.clone())
        .build()
        .expect("failed to rebuild instance from snapshot state");

    println!("..");
    println!(
        "Loaded model from '{}'",
        model_fpath.clone().unwrap_or_default()
    );

    let char_mode = embedding
        .vocab()
        .iter()
        .all(|(token, id)| *id == 0 || token.len() == 1);

    let append_space = char_mode && embedding.vocab().contains_key(" ");

    println!("Starting... (character input mode = {char_mode})");
    println!("   - Commands available: [ '.load <path>' | '.reload' | '.supervise' | '.reset' | '.quit' ]");
    println!("");

    let stdin = io::stdin();

    let mut vocab_supervised_predictions_enabled = false;
    let mut vocab: Option<HashSet<String>> = None;

    let default_min_word_len = 3;

    loop {
        println!("");
        print!("Enter prompt for model: ");
        io::stdout().flush().unwrap();

        let input_txt = read_line(&stdin);
        let input_txt = match process_repl_commands(&input_txt, model_fpath.as_ref()) {
            CliReplActions::ProcessInput => input_txt,
            CliReplActions::Reload(input_path) => {
                model_fpath.replace(input_path);
                return true;
            }
            CliReplActions::Quit => {
                return false;
            }
            CliReplActions::SetVocabSupervisionEnabled(enabled) => {
                vocab_supervised_predictions_enabled = enabled;
                if vocab_supervised_predictions_enabled {
                    configure_vocab(&mut vocab, &state, default_min_word_len);
                }
                continue;
            }
            CliReplActions::Reprompt => {
                continue;
            }
        };

        let context_tokens: Vec<String> =
            to_context_tokens(&input_txt, char_mode, append_space).collect();
        let context_tokens: Vec<&str> = context_tokens.iter().map(|x| x.as_str()).collect();

        match embedding.predict_next(&context_tokens) {
            Err(e) => {
                println!("Error retrieving response: ({e})");
                continue;
            }
            _ => (),
        };

        let response: Vec<_> = if let (Some(vocab), true) = (
            vocab.as_ref(),
            char_mode && vocab_supervised_predictions_enabled,
        ) {
            predict_supervised(
                &embedding,
                &context_tokens,
                vocab,
                default_min_word_len,
                &rng,
            )
        } else {
            embedding.predict_from_iter(&context_tokens).collect()
        };

        let response = if char_mode {
            if append_space {
                response.join("")
            } else {
                format!("{input_txt}{}", &response.join("")[1..])
            }
        } else {
            response.join(" ")
        };

        print_prompt_response(&response);
    }
}

fn read_model_file(fpath_arg: Option<&String>) -> Option<String> {
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

    let mut file = File::open(path).unwrap();
    let mut buf = String::new();

    file.read_to_string(&mut buf).unwrap();

    Some(buf)
}

fn read_line(stdin: &io::Stdin) -> String {
    let mut input_txt = String::new();
    while let Err(_) = stdin.read_line(&mut input_txt) {}
    input_txt.trim().to_string()
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

fn predict_supervised(
    embedding: &plane::ml::embeddings::Embedding,
    context_tokens: &Vec<&str>,
    vocab: &HashSet<String>,
    min_word_len: usize,
    rng: &RngStrategy,
) -> Vec<String> {
    let ctrl_token = embedding.control_vocab();
    let word_separator_token = " ".to_string();
    let max_vocab_len: usize = vocab.iter().map(|x| x.len()).max().unwrap();

    let mut context_queue = VecDeque::from_iter(context_tokens.iter().map(|x| x.to_string()));
    let mut word_queue = vec![];
    let mut generated_queue = vec![];
    let mut last_token = context_tokens.last().cloned().unwrap_or(ctrl_token);

    while last_token != ctrl_token {
        let context_tokens_str: Vec<&str> = context_queue.iter().map(|x| x.as_str()).collect();
        let generated_token = match embedding.predict_next(&context_tokens_str) {
            Ok(token) => token,
            Err(e) => {
                println!("Error retrieving response: ({e})");
                continue;
            }
        };

        if !is_token_alphabetic(&generated_token) {
            word_queue.clear();
        } else if &generated_token == ctrl_token {
            break;
        } else {
            word_queue.push(generated_token.to_string());
        }

        generated_queue.push(generated_token.to_string());
        context_queue.push_back(generated_token);

        let possible_vocab_word = word_queue.join("");
        let insert_separator = if vocab.contains(&possible_vocab_word) {
            true
        } else if (min_word_len..max_vocab_len)
            .into_iter()
            .rev()
            .any(|distance| {
                let start_idx = possible_vocab_word.len().saturating_sub(distance + 1);
                let trucated_vocab_word = &possible_vocab_word[start_idx..];

                let contains = vocab.contains(trucated_vocab_word);
                if contains {
                    generated_queue.truncate(generated_queue.len() - word_queue.len());
                    context_queue.truncate(context_queue.len() - word_queue.len());
                    trucated_vocab_word.chars().for_each(|c| {
                        generated_queue.push(c.to_string());
                        context_queue.push_back(c.to_string());
                    });
                }

                contains
            })
        {
            true
        } else {
            false
        };

        if insert_separator {
            word_queue.clear();

            let context_tokens_str: Vec<&str> = context_queue.iter().map(|x| x.as_str()).collect();
            let word_separator_token =
                match predict_word_separator_token(embedding, rng, &context_tokens_str) {
                    Ok(token) => token,
                    Err(e) => {
                        println!("Error sampling separator token: ({e})");
                        word_separator_token.to_string()
                    }
                };

            generated_queue.push(word_separator_token.clone());
            context_queue.push_back(word_separator_token.clone());
        }

        last_token = &context_queue.back().unwrap();
    }

    while generated_queue.last().unwrap() != ctrl_token {
        generated_queue.pop();
    }

    generated_queue
}

fn predict_word_separator_token(
    embedding: &plane::ml::embeddings::Embedding,
    rng: &RngStrategy,
    context_tokens_str: &[&str],
) -> Result<String> {
    let token_probabilites = embedding.sample_probabilities(&context_tokens_str)?;
    let ordered_tokens: Vec<_> = embedding
        .vocab()
        .iter()
        .sorted_by_key(|(_, &id)| id)
        .map(|(token, _)| token)
        .collect();

    let word_separator_probabilites: Vec<_> = ordered_tokens
        .iter()
        .zip(token_probabilites.iter())
        .filter(|(token, _)| !is_token_alphabetic(&token))
        .collect();

    let probabilities = word_separator_probabilites.iter().map(|(_, &x)| x);
    let sum_probabilities: f64 = probabilities.clone().sum();
    let probabilities = probabilities.map(|x| x / sum_probabilities).collect();

    let idx = rng.sample_uniform(&probabilities)?;
    let token = *word_separator_probabilites[idx].0;

    Ok(token.clone())
}

fn is_token_alphabetic(token: &String) -> bool {
    token.chars().next().unwrap_or(' ').is_alphabetic()
}

fn parse_word_level_vocab(input_txt_path: &str, min_word_len: Option<usize>) -> HashSet<String> {
    use io::{BufRead, BufReader};

    let file = File::open(Path::new(&input_txt_path)).unwrap();
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let mut words = HashSet::new();

    if let Some(min_word_len) = min_word_len {
        while let Some(Ok(line)) = lines.next() {
            for word in line.split_whitespace().filter(|w| w.len() >= min_word_len) {
                words.insert(word.to_string());
            }
        }
    } else {
        while let Some(Ok(line)) = lines.next() {
            for word in line.split_whitespace() {
                let word = word.trim_matches(|c| !char::is_alphanumeric(c));
                words.insert(word.to_string());
            }
        }
    }

    words
}

fn configure_vocab(
    vocab: &mut Option<HashSet<String>>,
    state: &ExtractedModelConfig,
    default_min_word_len: usize,
) {
    let min_word_len = Some(default_min_word_len);
    match &state.input_txt_path {
        Some(input_txt_path) => {
            vocab.get_or_insert_with(|| {
                println!("Loading input_txt vocab...");
                parse_word_level_vocab(&input_txt_path, min_word_len)
            });
            println!("Enabled word-level response generation supervison!");
        }
        None => {
            println!("Failed to enable word-level response generation supervison: input_txt vocab file path is missing");
        }
    }
}

fn print_prompt_response(response: &str) {
    print!("Model Response:        ");
    io::stdout().flush().unwrap();

    for word in response.split_whitespace() {
        print!(" {word}");

        io::stdout().flush().unwrap();
        thread::sleep(Duration::from_millis(25))
    }

    println!("");
}

struct ExtractedModelConfig {
    input_txt_path: Option<String>,
}

fn extract_config(json: String) -> Result<(String, ExtractedModelConfig)> {
    use serde_json::Value;

    let mut snapshot: Value = serde_json::from_str(&json)?;

    let config: Value = serde_json::from_value(snapshot["_trainer_config"].take())
        .context("unable to extract trainer config from saved model")?;

    let _state: Value = serde_json::from_value(snapshot["_trainer_state"].take())
        .context("unable to extract trainer metatdata from saved model")?;

    let snapshot = serde_json::to_string(&snapshot)?;

    let input_txt_path = config["input_txt_path"].as_str().map(|x| x.to_string());

    Ok((snapshot, ExtractedModelConfig { input_txt_path }))
}

enum CliReplActions {
    ProcessInput,
    Reload(String),
    Quit,
    SetVocabSupervisionEnabled(bool),
    Reprompt,
}

fn process_repl_commands(input_txt: &str, model_fpath: Option<&String>) -> CliReplActions {
    if input_txt == ".quit" {
        return CliReplActions::Quit;
    }

    if input_txt == ".supervise" {
        return CliReplActions::SetVocabSupervisionEnabled(true);
    }

    if input_txt == ".reset" {
        return CliReplActions::SetVocabSupervisionEnabled(false);
    }

    if input_txt.starts_with(".load") {
        let input_tokens = input_txt.split_ascii_whitespace().collect::<Vec<_>>();
        if let [".load", input_path] = input_tokens[..] {
            let input_path = input_path.to_string();
            return CliReplActions::Reload(input_path);
        } else {
            println!("Command usage: .load <MODEL_FILE_PATH>");
            return CliReplActions::Reprompt;
        }
    }

    if input_txt == ".reload" {
        if let Some(curr_fpath) = model_fpath {
            let curr_fpath = Path::new(&curr_fpath);
            let round_models: Vec<_> = curr_fpath
                .parent()
                .expect("failed to traverse path to model dir")
                .read_dir()
                .expect("failed to open model dir")
                .flatten()
                .filter_map(|f| {
                    Some((
                        f.file_name()
                            .to_string_lossy()
                            .split_once("model-r")
                            .and_then(|split| split.1.split('-').next())
                            .and_then(|round_str| round_str.parse::<usize>().ok())?,
                        f.path(),
                    ))
                })
                .collect();

            let latest_round_model = round_models.iter().max_by_key(|(round, _)| round);
            if let Some((_, latest_model_fpath)) = latest_round_model {
                let latest_model_fpath = latest_model_fpath.as_path();
                if latest_model_fpath != curr_fpath {
                    let input_path = latest_model_fpath.to_string_lossy().to_string();
                    return CliReplActions::Reload(input_path);
                }
            }
        }

        println!("No latest model found in model directory");
        return CliReplActions::Reprompt;
    }

    CliReplActions::ProcessInput
}
