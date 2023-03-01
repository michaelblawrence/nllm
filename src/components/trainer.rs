mod handle;

use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

use serde::{Deserialize, Serialize};
use web_sys::console::log_1;
use yew::prelude::*;

use crate::{
    components::PlotComponent,
    ml::{embeddings::Embedding, JsRng, NodeValue, ShuffleRng, RNG},
};

#[derive(Properties, Clone, PartialEq)]
pub struct EmbeddingTrainerProps {
    pub config: Rc<TrainEmbeddingConfig>,
}

#[function_component]
pub fn EmbeddingTrainer(props: &EmbeddingTrainerProps) -> Html {
    let embedding_training_config = (*props.config).clone();

    let vocab_and_phrases =
        use_state(|| Rc::new((Default::default(), Default::default(), Default::default())));
    let chart_points = use_state(|| vec![]);
    let train_iter_count = use_state(|| 0);
    let embedding_handle = use_state(|| handle::EmbeddingHandle::default());

    use_effect_with_deps(
        {
            let vocab_and_phrases = vocab_and_phrases.clone();
            let embedding_training_config = embedding_training_config.clone();
            move |config: &TrainEmbeddingConfig| {
                let (vocab, mut phrases) =
                    parse_vocab_and_phrases(Some(config.max_vocab_words_count));
                let TrainEmbeddingConfig {
                    max_phrases_count,
                    test_phrases_pct,
                    ..
                } = embedding_training_config;

                let testing_phrases =
                    split_training_and_testing(&mut phrases, max_phrases_count, test_phrases_pct);

                vocab_and_phrases.set(Rc::new((vocab, phrases, testing_phrases)));
                move || {}
            }
        },
        (*props.config).clone(),
    );

    use_effect_with_deps(
        {
            let chart_points = chart_points.clone();
            let embedding_handle = embedding_handle.clone();
            let embedding_training_config = embedding_training_config.clone();

            move |vocab_and_phrases: &Rc<(HashSet<String>, Vec<Vec<String>>, Vec<Vec<String>>)>| {
                let (vocab, phrases, testing_phrases) = &**vocab_and_phrases;
                let TrainEmbeddingConfig {
                    embedding_size,
                    training_rounds,
                    word_locality_factor,
                    train_rate,
                    ..
                } = embedding_training_config;

                let mut embedding_instance =
                    Embedding::new(vocab.clone(), embedding_size, Rc::new(JsRng::default()));

                let mut testing_phrases_errors = vec![];

                for _ in 0..training_rounds {
                    let error = train_embedding(
                        &mut embedding_instance,
                        &phrases,
                        train_rate,
                        word_locality_factor,
                        testing_phrases,
                    );
                    testing_phrases_errors.push(error);
                }

                let error_chart_points = testing_phrases_errors
                    .iter()
                    .enumerate()
                    .map(|(i, x)| (i as f64, *x))
                    .collect::<Vec<(f64, f64)>>();

                chart_points.set(error_chart_points);
                embedding_handle.set(embedding_handle.replace(embedding_instance));

                move || {}
            }
        },
        (*vocab_and_phrases).clone(),
    );

    let onclick_train_iter = {
        let chart_points = chart_points.clone();
        let vocab_and_phrases = vocab_and_phrases.clone();
        let train_iter_count = train_iter_count.clone();
        let embedding_handle = embedding_handle.clone();
        let embedding_training_config = embedding_training_config.clone();

        move |_| {
            let next_handle = embedding_handle.replace_with(|mut embedding_instance| {
                let (_, phrases, testing_phrases) = &**vocab_and_phrases;
                let mut errors = (*chart_points).clone();

                let TrainEmbeddingConfig {
                    word_locality_factor,
                    train_rate,
                    ..
                } = embedding_training_config;

                let error = train_embedding(
                    &mut embedding_instance,
                    &phrases,
                    train_rate,
                    word_locality_factor,
                    testing_phrases,
                );

                errors.push((errors.len() as f64, error));

                chart_points.set(errors);
                train_iter_count.set(*train_iter_count + 1);
                embedding_instance
            });
            embedding_handle.set(next_handle);
        }
    };

    html! {
        <div class="trainer">
            <h2>{"WASM Word Embeddings"}</h2>
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
            <EmbeddingNearest
                vocab_and_phrases={(*vocab_and_phrases).clone()}
                embedding={(*embedding_handle).clone()}
                iter_hint={*train_iter_count}
            />
            <PlotComponent points={(*chart_points).clone()} />
        </div>
    }
}

#[derive(Properties, Clone, PartialEq)]
pub struct EmbeddingNearestProps {
    pub vocab_and_phrases: Rc<(HashSet<String>, Vec<Vec<String>>, Vec<Vec<String>>)>,
    pub embedding: handle::EmbeddingHandle,
    #[prop_or_default]
    pub iter_hint: usize,
}

#[function_component]
pub fn EmbeddingNearest(props: &EmbeddingNearestProps) -> Html {
    let vocab_and_phrases = props.vocab_and_phrases.clone();
    let embedding = props.embedding.clone();
    let iter_hint = props.iter_hint.clone();

    let random_vocab_seed = use_state(|| 0);
    let nearest_target = use_state(|| Some("anyone".to_string()));
    let nearest_message = use_state(|| "".to_string());

    use_effect_with_deps(
        {
            let nearest_message = nearest_message.clone();

            move |(nearest_target, embedding, i): &(Option<String>, handle::EmbeddingHandle, _)| {
                log_1(&format!("iter hint = {i}").into());
                nearest_message.set(format!(
                    "Nearest vocab word to '{}' is ...",
                    nearest_target.clone().unwrap_or_else(|| "...".to_string())
                ));

                if let Some(word) = nearest_target {
                    let nearest = embedding.borrow().nearest(&word);
                    log_1(&format!("nearest = {nearest:?}").into());

                    match nearest {
                        Ok(next_nearest) => nearest_message.set(format!(
                            "Nearest vocab word to '{}' is {} (with a distance of {})",
                            word, next_nearest.0, next_nearest.1
                        )),
                        Err(_) => (),
                    }
                }

                move || {}
            }
        },
        ((*nearest_target).clone(), embedding.clone(), iter_hint),
    );

    use_effect_with_deps(
        {
            let nearest_target = nearest_target.clone();

            move |(vocab_and_phrases, _): &(
                Rc<(HashSet<String>, Vec<Vec<String>>, Vec<Vec<String>>)>,
                _,
            )| {
                let (vocab, _, _) = &**vocab_and_phrases;
                let rand_range = JsRng::default().rand_range(0, vocab.len());
                if let Some(random_vocab) = vocab.iter().nth(rand_range) {
                    nearest_target.set(Some(random_vocab.clone()));
                }

                move || {}
            }
        },
        (vocab_and_phrases.clone(), *random_vocab_seed),
    );

    let onclick_new_vocab_word = {
        let random_vocab_seed = random_vocab_seed.clone();

        move |_| random_vocab_seed.set(*random_vocab_seed + 1)
    };

    html! {
        <div class="trainer-nearest">
            <h3>{"Test random 'nearest' word"}</h3>
            <button onclick={onclick_new_vocab_word}>{ format!("New word") }</button>
            <p>{nearest_message.to_string()}</p>
        </div>
    }
}

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

fn train_embedding(
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

fn split_training_and_testing(
    phrases: &mut Vec<Vec<String>>,
    max_phrases_count: usize,
    test_phrases_pct: f64,
) -> Vec<Vec<String>> {
    _ = phrases.split_off(max_phrases_count.min(phrases.len()));

    let testing_count = phrases.len() as NodeValue * (test_phrases_pct as NodeValue / 100.0);
    let testing_count = testing_count as usize;

    phrases.split_off(phrases.len() - testing_count)
}

fn parse_vocab_and_phrases(max_vocab: Option<usize>) -> (HashSet<String>, Vec<Vec<String>>) {
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
            .map(|phrase| phrase.into_iter().take_while(|word| vocab.contains(word)).collect::<Vec<_>>())
            .filter(|x| !x.is_empty())
            .collect();
        phrases.sort_by_key(|phrase| -(phrase.len() as i64));
    }

    (vocab, phrases)
}
