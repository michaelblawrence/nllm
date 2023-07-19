use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use tracing::error;

pub struct BytePairEncoder;

impl BytePairEncoder {
    pub fn build(
        corpus: &str,
        num_merges: usize,
        iters: usize,
        max_token_length: usize,
        lower_transform: bool,
    ) -> HashSet<String> {
        let tokenize_input = if lower_transform {
            corpus.to_lowercase()
        } else {
            corpus.to_string()
        };

        let tokens = Self::tokenize(&tokenize_input);
        let source_vocab: HashSet<_> = tokens.iter().collect();

        let encoded_tokens = {
            let mut encoded_tokens: Vec<String> = tokens.clone();

            for _ in 0..iters {
                let mut frequencies =
                    Self::build_byte_pair_frequencies(&encoded_tokens, max_token_length);
                Self::merge_byte_pairs(&mut encoded_tokens, &mut frequencies.1, num_merges);
            }

            encoded_tokens
                .into_iter()
                .filter(|x| x.len() > 0)
                .collect_vec()
        };

        let vocab: HashSet<_> = encoded_tokens
            .into_iter()
            .chain(source_vocab.into_iter().cloned())
            .collect();

        vocab
    }

    pub fn encode_bpe(
        encode_input: &str,
        by_char_count: &[(usize, HashMap<&str, usize>)],
    ) -> Vec<usize> {
        let mut encode_str = encode_input.to_string();
        let mut encoded_tokens = vec![];

        while encode_str.len() > 0 {
            match try_consume_token(&mut encode_str, &by_char_count) {
                Some(token) => {
                    encoded_tokens.push(token);
                }
                None => {
                    let discarded = encode_str.remove(0);
                    error!("unknown token: {:?}", discarded);
                    encoded_tokens.push(0);
                }
            }
        }

        fn try_consume_token(
            encode_input: &mut String,
            by_char_count: &[(usize, HashMap<&str, usize>)],
        ) -> Option<usize> {
            for (token_len, tokens) in by_char_count.iter().rev() {
                let truncated: String = encode_input.chars().take(*token_len).collect();
                if truncated.chars().count() != *token_len {
                    continue;
                }
                if let Some(token) = tokens.get(&truncated.as_str()) {
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
        for _ in 0..num_merges {
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
}
