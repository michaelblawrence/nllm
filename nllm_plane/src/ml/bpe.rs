use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

use itertools::Itertools;
use tracing::{error, info};

type MaybeStr<'a> = Cow<'a, str>;

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
            let mut encoded_tokens: Vec<MaybeStr<'_>> = tokens.clone();

            for i in 0..iters {
                info!("Building BPE: iter {i}/{iters}");
                let vec = encoded_tokens.clone();
                let mut frequencies = Self::build_byte_pair_frequencies(&vec, max_token_length);
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
            .map(|x| x.into_owned())
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

    fn tokenize<'a>(text: &'a str) -> Vec<Cow<'a, str>> {
        let slices = text
            .char_indices()
            .tuple_windows()
            .map(|((x0, _), (y0, _))| Cow::from(&text[x0..y0]))
            .collect();
        info!("Consumed {} corpus bytes", text.len());
        slices
    }

    fn build_byte_pair_frequencies<'a>(
        tokens: &'a [MaybeStr<'a>],
        max_token_length: usize,
    ) -> (
        HashMap<(MaybeStr<'a>, MaybeStr<'a>), usize>,
        Vec<(MaybeStr<'a>, MaybeStr<'a>)>,
    ) {
        let mut frequencies: HashMap<(&MaybeStr<'_>, &MaybeStr<'_>), usize> = HashMap::new();

        for chunk in tokens.windows(2) {
            let pair = (&chunk[0], &chunk[1]);
            *frequencies.entry(pair).or_insert(0) += 1;
        }
        info!("Counted BPE token frequencies: N = {}", tokens.len());

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
            .map(|(&(lhs, rhs), _)| (lhs.clone(), rhs.clone()))
            .collect_vec();

        let frequencies = frequencies
            .into_iter()
            .map(|((lhs, rhs), x)| ((lhs.clone(), rhs.clone()), x))
            .collect();

        info!(
            "Sorted BPE token pair frequencies: N(pairs) = {}",
            sorted_pairings.len()
        );

        (frequencies, sorted_pairings)
    }

    fn merge_byte_pairs(
        tokens: &mut [MaybeStr<'_>],
        sorted_pairs: &mut Vec<(MaybeStr<'_>, MaybeStr<'_>)>,
        num_merges: usize,
    ) {
        for merge_idx in 0..num_merges {
            if let Some(pair) = &sorted_pairs.pop() {
                let mut idx = Vec::with_capacity(tokens.len());
                for (i, chunk_pair) in tokens.windows(2).enumerate() {
                    if &*chunk_pair[0] == &pair.0 && &*chunk_pair[1] == &pair.1 {
                        idx.push(i);
                    }
                }
                info!(
                    "Matched BPE token pairs: N = {} (iter: {merge_idx}/{num_merges})",
                    tokens.len()
                );

                let replacement = format!("{}{}", pair.0, pair.1);
                if let [i] = idx[..] {
                    tokens[i] = Cow::from(replacement);
                    tokens[i + 1] = Cow::from("");
                } else {
                    for &i in idx.iter() {
                        tokens[i] = Cow::from(replacement.clone());
                        tokens[i + 1] = Cow::from("");
                    }
                }
            } else {
                break;
            }
        }
    }
}
