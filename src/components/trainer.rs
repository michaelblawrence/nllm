use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

use serde::{Deserialize, Serialize};
use wasm_bindgen::__rt::WasmRefCell;
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

    let vocab_and_phrases = use_state(|| {
        parse_vocab_and_phrases(Some(embedding_training_config.max_vocab_words_count))
    });
    let chart_points = use_state(|| vec![]);
    let random_vocab_seed = use_state(|| 0);
    let train_fn = use_state(|| None);

    let nearest_target = use_state(|| Some("anyone".to_string()));
    let nearest_message = use_state(|| "".to_string());

    use_effect_with_deps(
        {
            let vocab_and_phrases = vocab_and_phrases.clone();
            move |config: &TrainEmbeddingConfig| {
                vocab_and_phrases.set(parse_vocab_and_phrases(Some(config.max_vocab_words_count)));
                move || {}
            }
        },
        (*props.config).clone(),
    );
    use_effect_with_deps(
        {
            let nearest_message = nearest_message.clone();
            let train_fn = train_fn.clone();
            move |nearest_target: &Option<String>| {
                nearest_message.set(format!(
                    "Nearest vocab word to '{}' is ...",
                    nearest_target.clone().unwrap_or_else(|| "...".to_string())
                ));
                // if let Some(mut train_fn) = train_fn.as_ref().map(|train_fn: | train_fn.borrow_mut())
                // {
                //     if let Some(nearest_target) = nearest_target {
                //         let tasks = vec![
                //             EmbeddingTasks::SkipTrain(),
                //             EmbeddingTasks::GetNearest(nearest_target.clone()),
                //         ];
                //         let (error, task_outputs) = train_fn(tasks);
                //         let task_outputs = train_fn();
                //         if let Some(EmbeddingTasksOutput::GetNearest(
                //             EmbeddingTasks::GetNearest(word),
                //             Ok(next_nearest),
                //         )) = task_outputs.into_iter().find(|x| {
                //             if let EmbeddingTasksOutput::GetNearest(..) = &x {
                //                 true
                //             } else {
                //                 false
                //             }
                //         }) {
                //             nearest_message.set(format!(
                //                 "Nearest vocab word to '{}' is {} (with a distance of {})",
                //                 word, next_nearest.0, next_nearest.1
                //             ));
                //         }
                //     }
                // }
                move || {}
            }
        },
        (*nearest_target).clone(),
    );

    use_effect_with_deps(
        {
            let nearest_target = nearest_target.clone();

            move |(vocab_and_phrases, _): &(
                UseStateHandle<(HashSet<String>, Vec<Vec<String>>)>,
                _,
            )| {
                let (vocab, _) = &**vocab_and_phrases;
                let rand_range = JsRng::default().rand_range(0, vocab.len());
                if let Some(random_vocab) = vocab.iter().nth(rand_range) {
                    nearest_target.set(Some(random_vocab.clone()));
                }

                move || {}
            }
        },
        (vocab_and_phrases.clone(), *random_vocab_seed),
    );

    use_effect_with_deps(
        {
            let chart_points = chart_points.clone();
            let train_fn = train_fn.clone();
            let nearest_target = nearest_target.clone();

            move |vocab_and_phrases: &UseStateHandle<(HashSet<String>, Vec<Vec<String>>)>| {
                let (vocab, phrases) = &**vocab_and_phrases;

                let (errors, train) =
                    train_embeddings((vocab.clone(), phrases.clone()), embedding_training_config);
                let error_chart_points = errors
                    .iter()
                    .enumerate()
                    .map(|(i, x)| (i as f64, *x))
                    .collect::<Vec<(f64, f64)>>();

                chart_points.set(error_chart_points);
                train_fn.set(Some(WasmRefCell::new(train)));
                move || {}
            }
        },
        vocab_and_phrases.clone(),
    );

    let onclick_train_iter = {
        let chart_points = chart_points.clone();
        let train_fn = train_fn.clone();
        let nearest_target = nearest_target.clone();
        let nearest_message = nearest_message.clone();

        move |_| {
            if let Some(mut train_fn) = train_fn.as_ref().map(|train_fn| train_fn.borrow_mut()) {
                let mut errors = (*chart_points).clone();

                let mut tasks = vec![];
                if let Some(nearest_target) = &*nearest_target {
                    tasks.push(EmbeddingTasks::GetNearest(nearest_target.to_string()));
                }

                let (error, task_outputs) = train_fn(tasks);
                errors.push((errors.len() as f64, error));

                chart_points.set(errors);
                if let Some(EmbeddingTasksOutput::GetNearest(
                    EmbeddingTasks::GetNearest(word),
                    Ok(next_nearest),
                )) = task_outputs.into_iter().find(|x| {
                    if let EmbeddingTasksOutput::GetNearest(..) = &x {
                        true
                    } else {
                        false
                    }
                }) {
                    nearest_message.set(format!(
                        "Nearest vocab word to '{}' is {} (with a distance of {})",
                        word, next_nearest.0, next_nearest.1
                    ));
                }
            }
        }
    };

    let onclick_new_vocab_word = {
        let random_vocab_seed = random_vocab_seed.clone();

        move |_| random_vocab_seed.set(*random_vocab_seed + 1)
    };

    html! {
        <div class="trainer">
            <h2>{"WASM Word Embeddings"}</h2>
            <p>{format!("Extracted {} vocab words from {} phrases", vocab_and_phrases.0.len(), vocab_and_phrases.1.len())}</p>
            <button onclick={onclick_train_iter}>{ format!("Run Training Iteration") }</button>
            <div class="trainer-nearest">
                <h3>{"Test random 'nearest' word"}</h3>
                <button onclick={onclick_new_vocab_word}>{ format!("New word") }</button>
                <p>{nearest_message.to_string()}</p>
            </div>
            <PlotComponent points={(*chart_points).clone()} />
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

pub(crate) type EmbeddingError = NodeValue;

#[derive(Clone)]
pub enum EmbeddingTasks {
    SkipTrain(),
    GetNearest(String),
}

pub enum EmbeddingTasksOutput {
    GetNearest(EmbeddingTasks, Result<(String, NodeValue), ()>),
    None,
}

pub(crate) fn train_embeddings(
    (vocab, mut phrases): (HashSet<String>, Vec<Vec<String>>),
    TrainEmbeddingConfig {
        embedding_size,
        training_rounds,
        word_locality_factor,
        max_phrases_count,
        max_vocab_words_count: _,
        train_rate,
        test_phrases_pct,
    }: TrainEmbeddingConfig,
) -> (
    Vec<EmbeddingError>,
    impl FnMut(Vec<EmbeddingTasks>) -> (EmbeddingError, Vec<EmbeddingTasksOutput>),
) {
    let rng = Rc::new(JsRng::default());
    let mut embedding = Embedding::new(vocab, embedding_size, rng.clone());

    let testing_phrases =
        split_training_and_testing(&mut phrases, max_phrases_count, test_phrases_pct);
    let mut testing_phrases_errors = vec![];

    let mut train_fn = move |tasks: Vec<EmbeddingTasks>| {
        let train_timer_label = &"embedding train";
        web_sys::console::time_with_label(train_timer_label);
        let skip = tasks
            .iter()
            .find(|x| {
                if let EmbeddingTasks::SkipTrain() = &x {
                    true
                } else {
                    false
                }
            })
            .is_some();

        if !skip {
            embedding
                .train(&phrases, train_rate, word_locality_factor)
                .unwrap();
        }
        web_sys::console::time_end_with_label(train_timer_label);

        let error = embedding.compute_error_batch(&testing_phrases).unwrap();

        let task_outputs = tasks
            .iter()
            .map(|task| match &task {
                EmbeddingTasks::GetNearest(word) => {
                    EmbeddingTasksOutput::GetNearest(task.clone(), embedding.nearest(word))
                }
                EmbeddingTasks::SkipTrain() => EmbeddingTasksOutput::None,
            })
            .collect();

        (error, task_outputs)
    };

    for _ in 0..training_rounds {
        let (error, _) = train_fn(vec![]);
        testing_phrases_errors.push(error);
    }

    (testing_phrases_errors, train_fn)
}

pub(crate) fn split_training_and_testing(
    phrases: &mut Vec<Vec<String>>,
    max_phrases_count: usize,
    test_phrases_pct: f64,
) -> Vec<Vec<String>> {
    _ = phrases.split_off(max_phrases_count.min(phrases.len()));
    let testing_phrases = phrases.split_off(
        phrases.len()
            - (phrases.len() as NodeValue * (test_phrases_pct as NodeValue / 100.0)) as usize,
    );
    // log_1(
    //     &format!(
    //         "{:#?}",
    //         phrases.iter().map(|x| x.join(" ")).collect::<Vec<_>>()
    //     )
    //     .into(),
    // );
    testing_phrases
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
            .filter(|phrase| phrase.iter().any(|word| vocab.contains(word)))
            .collect();
        phrases.sort_by_key(|phrase| -(phrase.len() as i64));
    }

    (vocab, phrases)
}
