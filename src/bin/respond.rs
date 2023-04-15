use std::{
    env,
    fs::File,
    io::{self, Read, Write},
    path::Path,
    rc::Rc,
};

use plane::ml::{embeddings::builder::EmbeddingBuilder, JsRng};

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

    let rng = Rc::new(JsRng::default());
    let embedding = EmbeddingBuilder::from_snapshot(&json, rng)
        .expect("failed to rebuild state from snapshot")
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
    println!("   - Commands available: [ '.load <path>' | '.reload' | '.quit' ]");
    println!("");

    let stdin = io::stdin();

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

        let response: Vec<_> = embedding.predict_from_iter(&context_tokens).collect();
        let response = if char_mode {
            if append_space {
                response.join("")
            } else {
                format!("{input_txt}{}", &response.join("")[1..])
            }
        } else {
            response.join(" ")
        };

        println!("Model Response:        {response}");
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



enum CliReplActions {
    ProcessInput,
    Reload(String),
    Quit,
    Reprompt,
}

fn process_repl_commands(input_txt: &str, model_fpath: Option<&String>) -> CliReplActions {
    if input_txt == ".quit" {
        return CliReplActions::Quit;
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