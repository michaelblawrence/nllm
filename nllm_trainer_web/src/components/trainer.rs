mod handle;
mod hook;
mod nearest;
mod parser;

use std::rc::Rc;

use itertools::Itertools;
use yew::prelude::*;

use crate::components::PlotComponent;

use hook::use_embeddings;
use nearest::EmbeddingNearest;
pub use parser::TrainEmbeddingConfig;
pub use handle::RefCellHandle;

#[derive(Properties, Clone, PartialEq)]
pub struct EmbeddingTrainerProps {
    pub config: Rc<TrainEmbeddingConfig>,
}

#[function_component]
pub fn EmbeddingTrainer(props: &EmbeddingTrainerProps) -> Html {
    let chart_points = use_state(|| vec![]);
    let nll_chart_points = use_state(|| vec![]);
    let train_iter_count = use_state(|| 0);
    let generated_phrase = use_state(|| String::new());

    let (embedding_handle, vocab_and_phrases, train_remaining_iters) = use_embeddings(
        props.config.clone(),
        {
            let chart_points = chart_points.clone();
            let nll_chart_points = nll_chart_points.clone();
            let train_iter_count = train_iter_count.clone();
            let generated_phrase = generated_phrase.clone();

            move |embedding, error, pvt| {
                let mut errors = (*chart_points).clone();
                errors.push((errors.len() as f64, error));

                chart_points.set(errors);
                train_iter_count.set(*train_iter_count + 1);
                generated_phrase.set(embedding.predict_iter("money").join(" "));

                let (_, _, training_set) = &*pvt;
                let nll = embedding.nll_batch(&training_set).unwrap();
                let mut nlls = (*nll_chart_points).clone();
                nlls.push((nlls.len() as f64, nll));

                nll_chart_points.set(nlls);

                embedding
            }
        },
        {
            let chart_points = chart_points.clone();
            let nll_chart_points = nll_chart_points.clone();
            let generated_phrase = generated_phrase.clone();

            move || {
                chart_points.set(vec![]);
                nll_chart_points.set(vec![]);
                generated_phrase.set(String::new());
            }
        },
    );

    let onclick_train_iter = {
        let train_remaining_iters = train_remaining_iters.clone();
        let embedding_training_config = props.config.clone();

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
                <p>{&*generated_phrase}</p>
            </div>
            <div class="trainer_graph">
                <p>{"Training Errors"}</p>
                <PlotComponent points={(*chart_points).clone()} />
                <p>{"Testing Loss (NLL)"}</p>
                <PlotComponent points={(*nll_chart_points).clone()} />
            </div>
            <EmbeddingNearest
                vocab_and_phrases={(*vocab_and_phrases).clone()}
                embedding={(*embedding_handle).clone()}
                iter_hint={*train_iter_count}
            />
        </div>
    }
}
