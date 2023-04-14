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
    let json = match read_model_file(args.first()) {
        Some(value) => value,
        None => return,
    };

    let rng = Rc::new(JsRng::default());
    let embedding = EmbeddingBuilder::from_snapshot(&json, rng)
        .expect("failed to rebuild state from snapshot")
        .build()
        .expect("failed to rebuild instance from snapshot state");

    let char_mode = embedding
        .vocab()
        .iter()
        .all(|(token, id)| *id == 0 || token.len() == 1);

    let append_space = char_mode && embedding.vocab().contains_key(" ");

    println!("Starting... (character input mode = {char_mode})");

    let stdin = io::stdin();

    loop {
        println!("..");
        println!("..");
        print!("Enter prompt for model (or type 'quit'): ");
        io::stdout().flush().unwrap();
        let input_txt = read_line(&stdin);

        if &input_txt == "quit" {
            break;
        }

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

        println!("..");
        println!("Model Response: {response}");
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