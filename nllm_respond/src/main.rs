use std::{
    collections::HashSet,
    env,
    fs::File,
    io::{self, Write},
    path::Path,
};

use plane::ml::RngStrategy;
use respond::{ExtractedModelConfig, PromptChatMode, PromptConfig};

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let fpath_arg = args.first();
    let mut fpath_arg = fpath_arg.cloned();

    while run(&mut fpath_arg) {}
}

fn run(model_fpath: &mut Option<String>) -> bool {
    let (model, state) = match respond::load(model_fpath.as_deref()) {
        Ok((model, state)) => (model, state),
        Err(_) => return false,
    };

    let char_mode = state.char_mode.expect("missing char_mode");

    println!("Starting... (character input mode = {char_mode})");
    println!("   - Commands available: [ '.load <path>' | '.reload' | '.supervise' | '.reset' | '.quit' ]");

    // TODO: add inference mode hint to model json?
    let detected_human_ctx_chat_format = model.description() == "GenerativeDecoderTransformer"
        && model.vocab().keys().any(|token| token.starts_with("###"));

    let chat_mode = if detected_human_ctx_chat_format {
        println!("Detected triple hash prompt encoding");
        PromptChatMode::TripleHashHumanPrompt
    } else {
        PromptChatMode::DirectPrompt
    };
    
    println!("");

    let stdin = io::stdin();

    let mut vocab_supervised_predictions_enabled = false;
    let mut vocab: Option<HashSet<String>> = None;

    let default_min_word_len = 3;
    let rng = RngStrategy::default();

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

        let config = PromptConfig {
            use_gdt: state.use_gdt,
            char_mode,
            vocab_supervised_predictions_enabled,
            chat_mode,
            vocab: vocab.as_ref(),
        };
        let response = match respond::process_prompt(&model, &rng, &input_txt, &config) {
            Ok(x) => x,
            Err(_) => return false,
        };

        print_prompt_response(response, &state);
    }
}

fn read_line(stdin: &io::Stdin) -> String {
    let mut input_txt = String::new();
    while let Err(_) = stdin.read_line(&mut input_txt) {}
    input_txt.trim().to_string()
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

fn print_prompt_response<'a>(
    mut response_iter: impl Iterator<Item = (String, &'a str)>,
    state: &ExtractedModelConfig,
) {
    print!("Model Response:        ");
    io::stdout().flush().unwrap();

    let mut response = String::new();
    while let Some((token, separator)) = response_iter.next() {
        response += &format!("{token}{separator}");
        let split_whitespace = response.split_whitespace();
        let whole_word_count = split_whitespace.clone().count().saturating_sub(1);
        let mut final_token = None;
        for word in split_whitespace.take(whole_word_count) {
            print!(" {word}");

            io::stdout().flush().unwrap();
            // thread::sleep(Duration::from_millis(25))
            final_token = Some(word);
        }
        if let Some(token) = final_token {
            let token_start_idx = response.find(token).expect("token should be in sequence");
            let token_end_idx = token_start_idx + token.len();
            response = response[token_end_idx + 1..].trim_start().to_string();
        }
    }

    println!("");
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
