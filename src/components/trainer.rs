mod handle;
mod nearest;

use std::rc::Rc;

use yew::prelude::*;

use crate::components::PlotComponent;

use hook::use_embeddings;
use nearest::EmbeddingNearest;
pub use parser::TrainEmbeddingConfig;

#[derive(Properties, Clone, PartialEq)]
pub struct EmbeddingTrainerProps {
    pub config: Rc<TrainEmbeddingConfig>,
}

#[function_component]
pub fn EmbeddingTrainer(props: &EmbeddingTrainerProps) -> Html {
    let embedding_training_config = (*props.config).clone();

    let chart_points = use_state(|| vec![]);
    let embedding_handle = use_state(|| handle::EmbeddingHandle::default());
    let vocab_and_phrases =
        use_state(|| Rc::new((Default::default(), Default::default(), Default::default())));
    let train_iter_count = use_state(|| 0);

    let setup_state = {
        let embedding_handle = embedding_handle.clone();
        let vocab_and_phrases = vocab_and_phrases.clone();

        move |(e_handle, v_and_p)| {
            embedding_handle.set(e_handle);
            vocab_and_phrases.set(v_and_p);
        }
    };

    let train_remaining_iters = use_embeddings(props.config.clone(), setup_state, {
        let chart_points = chart_points.clone();
        let train_iter_count = train_iter_count.clone();

        move |embedding, error| {
            let mut errors = (*chart_points).clone();
            errors.push((errors.len() as f64, error));

            chart_points.set(errors);
            train_iter_count.set(*train_iter_count + 1);
            embedding
        }
    });

    let onclick_train_iter = {
        let train_remaining_iters = train_remaining_iters.clone();
        let embedding_training_config = embedding_training_config.clone();

        move |_| {
            let iters = embedding_training_config.training_rounds.max(1);
            train_remaining_iters.set(*train_remaining_iters + iters);
        }
    };

    let onclick_train_stop = {
        let train_remaining_iters = train_remaining_iters.clone();

        move |_| {
            train_remaining_iters.set(0);
        }
    };

    html! {
        <div class="trainer">
            <div class="trainer_header">
                <h2>{"WASM Word Embeddings"}</h2>
            </div>
            <div class="trainer_controls">
                <p>{
                    format!(
                        "Extracted {} vocab words from {} phrases (mean length = {} words)",
                        vocab_and_phrases.0.len(),
                        vocab_and_phrases.1.len(),
                        vocab_and_phrases.1.iter().map(|x| x.len()).sum::<usize>() as f32
                            / vocab_and_phrases.1.len() as f32
                    )
                }</p>
                <button onclick={onclick_train_iter}>{ format!("Run Training Iteration") }</button>
                <button onclick={onclick_train_stop} disabled={*train_remaining_iters == 0}>{ format!("Stop Training Iterations") }</button>
                <p>{format!("Queued iterations = {}",*train_remaining_iters)}</p>
            </div>
            <PlotComponent points={(*chart_points).clone()} />
            <EmbeddingNearest
                vocab_and_phrases={(*vocab_and_phrases).clone()}
                embedding={(*embedding_handle).clone()}
                iter_hint={*train_iter_count}
            />
        </div>
    }
}

mod hook {
    use std::{collections::HashSet, rc::Rc};

    use yew::prelude::*;
    use yew_hooks::use_timeout;

    use crate::ml::{embeddings::Embedding, JsRng};

    use super::{
        handle::{self, EmbeddingHandle},
        parser, TrainEmbeddingConfig,
    };

    #[hook]
    pub fn use_embeddings<F, V>(
        config: Rc<TrainEmbeddingConfig>,
        set_initial_state_fn: V,
        with_emedding_fn: F,
    ) -> UseStateHandle<usize>
    where
        F: FnOnce(Embedding, f64) -> Embedding + 'static,
        V: FnOnce(
                (
                    EmbeddingHandle,
                    Rc<(HashSet<String>, Vec<Vec<String>>, Vec<Vec<String>>)>,
                ),
            ) -> ()
            + 'static,
    {
        let embedding_training_config = (*config).clone();

        let vocab_and_phrases =
            use_state(|| Rc::new((Default::default(), Default::default(), Default::default())));
        let embedding_handle = use_state(|| handle::EmbeddingHandle::default());

        let train_remaining_iters = use_state(|| 0_usize);

        let train_timeout = use_timeout(
            {
                let vocab_and_phrases = vocab_and_phrases.clone();
                let embedding_handle = embedding_handle.clone();
                let embedding_training_config = embedding_training_config.clone();
                let train_remaining_iters = train_remaining_iters.clone();

                move || {
                    if *train_remaining_iters <= 0 {
                        return;
                    }

                    embedding_handle.set(embedding_handle.replace_with(
                        |mut embedding_instance| {
                            let (_, phrases, testing_phrases) = &**vocab_and_phrases;

                            let TrainEmbeddingConfig {
                                word_locality_factor,
                                train_rate,
                                ..
                            } = embedding_training_config;

                            let error = parser::train_embedding(
                                &mut embedding_instance,
                                &phrases,
                                train_rate,
                                word_locality_factor,
                                testing_phrases,
                            );

                            let embedding_instance = with_emedding_fn(embedding_instance, error);

                            embedding_instance
                        },
                    ));

                    train_remaining_iters.set(*train_remaining_iters - 1);
                }
            },
            5,
        );

        use_effect_with_deps(
            {
                let train_timeout = train_timeout.clone();
                move |train_remaining_iters: &usize| {
                    if *train_remaining_iters > 0 {
                        train_timeout.reset();
                    }
                }
            },
            *train_remaining_iters,
        );

        use_effect_with_deps(
            {
                let vocab_and_phrases = vocab_and_phrases.clone();
                let embedding_training_config = embedding_training_config.clone();
                let train_remaining_iters = train_remaining_iters.clone();
                move |config: &TrainEmbeddingConfig| {
                    let (vocab, mut phrases) =
                        parser::parse_vocab_and_phrases(Some(config.max_vocab_words_count));
                    let TrainEmbeddingConfig {
                        max_phrases_count,
                        test_phrases_pct,
                        ..
                    } = embedding_training_config;

                    let testing_phrases = parser::split_training_and_testing(
                        &mut phrases,
                        max_phrases_count,
                        test_phrases_pct,
                    );

                    let v_and_p = Rc::new((vocab, phrases, testing_phrases));
                    vocab_and_phrases.set(v_and_p);
                    train_remaining_iters.set(0);
                    move || {}
                }
            },
            embedding_training_config.clone(),
        );

        use_effect_with_deps(
            {
                let embedding_handle = embedding_handle.clone();
                let embedding_training_config = embedding_training_config.clone();
                let train_remaining_iters = train_remaining_iters.clone();

                move |vocab_and_phrases: &Rc<(
                    HashSet<String>,
                    Vec<Vec<String>>,
                    Vec<Vec<String>>,
                )>| {
                    let (vocab, ..) = &**vocab_and_phrases;
                    let TrainEmbeddingConfig {
                        embedding_size,
                        training_rounds,
                        ..
                    } = embedding_training_config;

                    let embedding_instance =
                        Embedding::new(vocab.clone(), embedding_size, Rc::new(JsRng::default()));

                    let next_handle = embedding_handle.replace(embedding_instance);
                    set_initial_state_fn((next_handle.tick(), vocab_and_phrases.clone()));
                    embedding_handle.set(next_handle);
                    train_remaining_iters.set(training_rounds);

                    move || {}
                }
            },
            (*vocab_and_phrases).clone(),
        );

        train_remaining_iters
    }
}

mod parser {
    use std::collections::{HashMap, HashSet};

    use serde::{Deserialize, Serialize};

    use crate::ml::{embeddings::Embedding, JsRng, NodeValue, ShuffleRng};

    #[derive(Clone, PartialEq, Serialize, Deserialize)]
    pub struct TrainEmbeddingConfig {
        pub embedding_size: usize,
        pub training_rounds: usize,
        pub max_phrases_count: usize,
        pub max_vocab_words_count: usize,
        pub word_locality_factor: usize,
        pub train_rate: NodeValue,
        pub test_phrases_pct: NodeValue,
    }

    pub(crate) fn train_embedding(
        embedding: &mut Embedding,
        phrases: &Vec<Vec<String>>,
        train_rate: f64,
        word_locality_factor: usize,
        testing_phrases: &Vec<Vec<String>>,
    ) -> f64 {
        let train_timer_label = &"embedding train";
        web_sys::console::time_with_label(train_timer_label);

        embedding
            .train(&phrases, train_rate, word_locality_factor)
            .unwrap();

        web_sys::console::time_end_with_label(train_timer_label);

        embedding.compute_error_batch(&testing_phrases).unwrap()
    }

    pub(crate) fn split_training_and_testing(
        phrases: &mut Vec<Vec<String>>,
        max_phrases_count: usize,
        test_phrases_pct: f64,
    ) -> Vec<Vec<String>> {
        _ = phrases.split_off(max_phrases_count.min(phrases.len()));

        let testing_count = phrases.len() as NodeValue * (test_phrases_pct as NodeValue / 100.0);
        let testing_count = testing_count as usize;

        phrases.split_off(phrases.len() - testing_count)
    }

    pub(crate) fn parse_vocab_and_phrases(
        max_vocab: Option<usize>,
    ) -> (HashSet<String>, Vec<Vec<String>>) {
        let phrase_json = include_str!("../../res/phrase_list.json");
        let phrase_json: serde_json::Value = serde_json::from_str(phrase_json).unwrap();
        let mut phrases: Vec<Vec<String>> = phrase_json
            .as_array()
            .unwrap()
            .into_iter()
            .map(|value| {
                value
                    .as_array()
                    .unwrap()
                    .into_iter()
                    .map(|value| value.as_str().unwrap().to_owned())
                    .collect()
            })
            .collect();

        JsRng::default().shuffle_vec(&mut phrases);

        let vocab = phrases
            .iter()
            .map(|phrase| phrase.iter().cloned())
            .flatten()
            .fold(HashMap::new(), |mut map, c| {
                *map.entry(c).or_insert(0_usize) += 1;
                map
            })
            .into_iter();
        let vocab: HashSet<(String, usize)> = vocab.collect();
        let mut vocab: Vec<_> = vocab.into_iter().collect();

        JsRng::default().shuffle_vec(&mut vocab);
        vocab.sort_by_key(|(_, count)| -(*count as i64));

        let vocab = vocab.into_iter().map(|(v, _)| v);
        let vocab: HashSet<String> = match max_vocab {
            Some(max_vocab) => vocab.take(max_vocab).collect(),
            None => vocab.collect(),
        };

        if max_vocab.is_some() {
            phrases = phrases
                .into_iter()
                .map(|phrase| {
                    phrase
                        .into_iter()
                        .take_while(|word| vocab.contains(word))
                        .collect::<Vec<_>>()
                })
                .filter(|x| !x.is_empty())
                .collect();
            phrases.sort_by_key(|phrase| -(phrase.len() as i64));
        }

        (vocab, phrases)
    }
}
