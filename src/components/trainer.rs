use std::{collections::HashSet, rc::Rc};

use wasm_bindgen::__rt::WasmRefCell;
use yew::prelude::*;

use crate::{
    components::PlotComponent,
    ml::{embeddings::Embedding, JsRng, NodeValue, RNG},
};

#[function_component]
pub fn EmbeddingTrainer() -> Html {
    let vocab_and_phrases = use_state(|| parse_vocab_and_phrases());
    let chart_points = use_state(|| vec![]);
    let train_fn = use_state(|| None);

    let nearest_target = use_state(|| Some("anyone".to_string()));
    let nearest_message = use_state(|| "".to_string());

    use_effect_with_deps(
        {
            let nearest_message = nearest_message.clone();
            move |nearest_target: &Option<String>| {
                nearest_message.set(format!(
                    "Nearest vocab word to '{}' is ...",
                    nearest_target.clone().unwrap_or_else(|| "...".to_string())
                ));
                move || {}
            }
        },
        (*nearest_target).clone(),
    );

    use_effect_with_deps(
        {
            let chart_points = chart_points.clone();
            let train_fn = train_fn.clone();
            let nearest_target = nearest_target.clone();

            move |vocab_and_phrases: &UseStateHandle<(HashSet<String>, Vec<Vec<String>>)>| {
                let (vocab, phrases) = &**vocab_and_phrases;
                let rand_range = JsRng::default().rand_range(0, vocab.len());
                let random_vocab = vocab.iter().nth(rand_range).unwrap();
                nearest_target.set(Some(random_vocab.clone()));

                let (errors, train) = train_embeddings(
                    (vocab.clone(), phrases.clone()),
                    TrainEmbeddingConfig {
                        embedding_size: 2,
                        training_rounds: 0,
                        max_vocab_count: 50,
                        train_rate: 1e-5,
                        test_phrases_pct: 20.0,
                    },
                );
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
                )) = task_outputs.first()
                {
                    nearest_message.set(format!(
                        "Nearest vocab word to '{}' is {}",
                        word, next_nearest
                    ));
                }
            }
        }
    };

    html! {
        <div>
            <button onclick={onclick_train_iter}>{ format!("Run Training Iteration") }</button>
            <p>{nearest_message.to_string()}</p>
            <PlotComponent points={(*chart_points).clone()} />
        </div>
    }
}

pub(crate) struct TrainEmbeddingConfig {
    pub(crate) embedding_size: usize,
    pub(crate) training_rounds: usize,
    pub(crate) max_vocab_count: usize,
    pub(crate) train_rate: NodeValue,
    pub(crate) test_phrases_pct: NodeValue,
}

pub(crate) type EmbeddingError = NodeValue;

#[derive(Clone)]
pub enum EmbeddingTasks {
    GetNearest(String),
}

pub enum EmbeddingTasksOutput {
    GetNearest(EmbeddingTasks, Result<String, ()>),
}

pub(crate) fn train_embeddings(
    (vocab, mut phrases): (HashSet<String>, Vec<Vec<String>>),
    TrainEmbeddingConfig {
        embedding_size,
        training_rounds,
        max_vocab_count,
        train_rate,
        test_phrases_pct,
    }: TrainEmbeddingConfig,
) -> (
    Vec<EmbeddingError>,
    impl FnMut(Vec<EmbeddingTasks>) -> (EmbeddingError, Vec<EmbeddingTasksOutput>),
) {
    let mut embedding = Embedding::new(vocab, embedding_size, Rc::new(JsRng::default()));
    embedding.shuffle_vec(&mut phrases);

    let testing_phrases =
        split_training_and_testing(&mut phrases, max_vocab_count, test_phrases_pct);
    let mut testing_phrases_errors = vec![];

    let mut train_fn = move |tasks: Vec<EmbeddingTasks>| {
        embedding.train(&phrases, train_rate).unwrap();
        let error = embedding.compute_error_batch(&testing_phrases).unwrap();

        let task_outputs = tasks
            .iter()
            .map(|task| match &task {
                EmbeddingTasks::GetNearest(word) => {
                    EmbeddingTasksOutput::GetNearest(task.clone(), embedding.nearest(word))
                }
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
    max_vocab_count: usize,
    test_phrases_pct: f64,
) -> Vec<Vec<String>> {
    _ = phrases.split_off(max_vocab_count);
    let testing_phrases = phrases.split_off(
        phrases.len()
            - (phrases.len() as NodeValue * (test_phrases_pct as NodeValue / 100.0)) as usize,
    );
    testing_phrases
}

pub(crate) fn parse_vocab_and_phrases() -> (HashSet<String>, Vec<Vec<String>>) {
    let vocab_json = include_str!("../../res/vocab_set.json");
    let vocab_json: serde_json::Value = serde_json::from_str(vocab_json).unwrap();
    let vocab: HashSet<String> = vocab_json
        .as_array()
        .unwrap()
        .into_iter()
        .map(|value| value.as_str().unwrap().to_string())
        .collect();

    let phrase_json = include_str!("../../res/phrase_list.json");
    let phrase_json: serde_json::Value = serde_json::from_str(phrase_json).unwrap();
    let phrases: Vec<Vec<String>> = phrase_json
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
    (vocab, phrases)
}
