use std::collections::{HashMap, HashSet};

use console::ConsoleCounter;
use itertools::Itertools;

use crate::config::{parse_args, BpeConfig};

fn main() {
    let (
        corpus,
        BpeConfig {
            num_merges,
            iters,
            lower_transform,
            max_token_length,
        },
    ) = match parse_args() {
        Some((corpus, config)) => {
            println!("Starting with config: {config:?}");
            (corpus, config)
        }
        None => return,
    };

    let tokenize_input = if lower_transform {
        corpus.to_lowercase()
    } else {
        corpus
    };

    let tokens = tokenize(&tokenize_input);
    let source_vocab: HashSet<_> = tokens.iter().collect();

    let encoded_tokens = {
        let mut iter_counter = ConsoleCounter::show("iter");
        let mut encoded_tokens: Vec<String> = tokens.clone();

        for iter in 0..iters {
            iter_counter.set_value(iter + 1);
            let mut frequencies = build_byte_pair_frequencies(&encoded_tokens, max_token_length);
            merge_byte_pairs(&mut encoded_tokens, &mut frequencies.1, num_merges);
        }

        encoded_tokens
            .into_iter()
            .filter(|x| x.len() > 0)
            .collect_vec()
    };

    let vocab: HashSet<_> = encoded_tokens
        .iter()
        .chain(source_vocab.iter().copied())
        .collect();

    println!("");
    println!("vocab: {:?}", vocab);

    println!(
        "Encoded tokens: {:?}",
        encoded_tokens
            .iter()
            .take(100)
            .cloned()
            .collect_vec()
            .join("|")
    );

    println!("Enter input for tokenization: ");
    let encode_input = {
        let mut buf = String::new();
        std::io::stdin().read_line(&mut buf).unwrap();
        if lower_transform {
            buf.to_lowercase()
        } else {
            buf
        }
    };
    println!("encoding: {:?}", encode_input);

    let (by_char_count, vocab_decoder) = process_vocab(vocab);
    println!("vocab size: {}", vocab_decoder.len());

    let encoded_tokens = encode_bpe(&encode_input, &by_char_count);
    println!("encoded: {:?}", encoded_tokens);

    let decoded_tokens = decode_bpe(&encoded_tokens, vocab_decoder);
    println!("decoded: {:?}", decoded_tokens.join("|"));

    let ratio = (encoded_tokens.len() as f64) / (encode_input.chars().count() as f64);
    println!("compression ratio: {:.1}%", ratio * 100.0);
}

fn process_vocab(
    vocab: HashSet<&String>,
) -> (
    Vec<(usize, HashMap<&String, usize>)>,
    HashMap<usize, &String>,
) {
    let indexed_vocab = vocab
        .iter()
        .copied()
        .enumerate()
        .map(|(c, x)| (c + 1, x))
        .collect_vec();

    let by_char_count = {
        let mut counts: Vec<(usize, HashMap<&String, usize>)> = indexed_vocab
            .iter()
            .group_by(|(_, x)| x.chars().count())
            .into_iter()
            .map(|(k, v)| (k, v.copied().map(|(c, x)| (x, c)).collect()))
            .collect_vec();
        counts.sort_by_key(|x| x.0);
        counts
    };

    let vocab_decoder: HashMap<usize, _> = indexed_vocab.iter().copied().collect();
    (by_char_count, vocab_decoder)
}

fn encode_bpe(
    encode_input: &String,
    by_char_count: &[(usize, HashMap<&String, usize>)],
) -> Vec<usize> {
    let mut encode_str = encode_input.clone();
    let mut encoded_tokens = vec![];

    while encode_str.len() > 0 {
        match try_consume_token(&mut encode_str, &by_char_count) {
            Some(token) => {
                encoded_tokens.push(token);
            }
            None => {
                let discarded = encode_str.remove(0);
                println!("unknown token: {:?}", discarded);
                encoded_tokens.push(0);
            }
        }
    }

    fn try_consume_token(
        encode_input: &mut String,
        by_char_count: &[(usize, HashMap<&String, usize>)],
    ) -> Option<usize> {
        for (token_len, tokens) in by_char_count.iter().rev() {
            let truncated: String = encode_input.chars().take(*token_len).collect();
            if truncated.chars().count() != *token_len {
                continue;
            }
            if let Some(token) = tokens.get(&truncated) {
                *encode_input = encode_input.chars().skip(*token_len).collect();
                return Some(*token);
            }
        }
        None
    }

    encoded_tokens
}

fn tokenize(text: &str) -> Vec<String> {
    text.chars().map(|c| c.to_string()).collect()
}

fn build_byte_pair_frequencies(
    tokens: &[String],
    max_token_length: usize,
) -> (HashMap<(String, String), usize>, Vec<(String, String)>) {
    let mut frequencies: HashMap<(String, String), usize> = HashMap::new();
    for chunk in tokens.windows(2) {
        let pair = (chunk[0].to_string(), chunk[1].to_string());
        *frequencies.entry(pair).or_insert(0) += 1;
    }

    let sorted_pairing_counts = {
        let mut pairings = frequencies
            .iter()
            .filter(|((lhs, rhs), _)| {
                let len = (lhs.len(), rhs.len());
                len.0 + len.1 <= max_token_length
                    && len.0 != 0
                    && len.1 != 0
                    && (len.0 == 1 || !lhs.chars().last().unwrap().is_whitespace())
                    && (len.1 == 1 || !rhs.chars().next().unwrap().is_whitespace())
            })
            .collect_vec();

        pairings.sort_by_key(|(_, count)| *count);
        pairings
    };

    let sorted_pairings = sorted_pairing_counts
        .into_iter()
        .map(|(pair, _)| pair)
        .cloned()
        .collect_vec();

    (frequencies, sorted_pairings)
}

fn merge_byte_pairs(
    tokens: &mut Vec<String>,
    sorted_pairs: &mut Vec<(String, String)>,
    num_merges: usize,
) {
    let mut counter = ConsoleCounter::show("  merge");
    for merge_idx in 0..num_merges {
        counter.set_value(merge_idx + 1);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        if let Some(pair) = &sorted_pairs.pop() {
            let mut idx = vec![];
            for (i, chunk_pair) in tokens.windows(2).enumerate() {
                if chunk_pair[0] == pair.0 && chunk_pair[1] == pair.1 {
                    idx.push(i);
                }
            }

            let replacement = format!("{}{}", pair.0, pair.1);
            for &i in idx.iter() {
                tokens[i] = replacement.clone();
                tokens[i + 1] = String::new();
            }
        } else {
            break;
        }
    }
}

fn decode_bpe<'a>(
    encoded_tokens: &'a Vec<usize>,
    vocab_decoder: HashMap<usize, &'a String>,
) -> Vec<&'a str> {
    encoded_tokens
        .iter()
        .map(|x| vocab_decoder.get(&x).map(|a| a.as_str()).unwrap_or("<?>"))
        .collect_vec()
}

mod config {
    use std::{env, fs::File, io::Read};

    use anyhow::Context;

    #[derive(Debug)]
    pub struct BpeConfig {
        pub num_merges: usize,
        pub iters: i32,
        pub lower_transform: bool,
        pub max_token_length: usize,
    }

    pub fn parse_args() -> Option<(String, BpeConfig)> {
        let usage_help_msg = "Usage: bpe [corpus_fname] [merges] [iters] [to_lower_case]";
        match env::args().nth(1).as_ref().map(|x| x.as_str()) {
            Some("-h" | "--help") | None => {
                println!("{usage_help_msg}");
                return None;
            }
            _ => (),
        }

        let corpus = env::args()
            .nth(1)
            .context("missing fpath arg")
            .and_then(|f| Ok(File::open(f)?))
            .and_then(|mut f| {
                let mut buffer = String::new();
                f.read_to_string(&mut buffer)?;
                Ok(buffer)
            })
            .map_err(|e| println!("ERROR: {e}\n\n{usage_help_msg}"))
            .ok()?;

        let num_merges = 1;
        let num_merges = env::args()
            .nth(2)
            .and_then(|f| f.parse().ok())
            .unwrap_or(num_merges);

        let iters = 2;
        let iters = env::args()
            .nth(3)
            .and_then(|f| f.parse().ok())
            .unwrap_or(iters);

        let lower_transform = false;
        let lower_transform = env::args()
            .nth(4)
            .and_then(|f| f.parse().ok())
            .unwrap_or(lower_transform);

        let max_token_length = 10;

        Some((
            corpus,
            BpeConfig {
                num_merges,
                iters,
                lower_transform,
                max_token_length,
            },
        ))
    }
}

mod console {
    use std::fmt::Display;

    pub struct ConsoleCounter(&'static str);

    impl ConsoleCounter {
        pub fn show(label: &'static str) -> Self {
            println!("");
            Self(label)
        }

        pub fn set_value<T: Display>(&mut self, value: T) {
            print!("{}: {}\r", self.0, value);
        }
    }

    impl Drop for ConsoleCounter {
        fn drop(&mut self) {
            println!("{}: <done>", self.0);
        }
    }
}
