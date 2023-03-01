mod components;
mod ml;

use std::rc::Rc;

use yew::prelude::*;
use yew_hooks::use_local_storage;

#[function_component]
fn App() -> Html {
    use crate::components::{trainer::TrainEmbeddingConfig, EmbeddingTrainer, JsonEditor};

    let storage = use_local_storage::<TrainEmbeddingConfig>("embeddings_config".to_string());

    let embedding_training_config = use_state({
        let storage = storage.clone();
        move || {
            if let Some(input_config) = &*storage {
                Rc::new(input_config.clone())
            } else {
                Rc::new(TrainEmbeddingConfig {
                    embedding_size: 2,
                    training_rounds: 0,
                    max_phrases_count: 300,
                    max_vocab_words_count: 125,
                    word_locality_factor: 2,
                    train_rate: 1e-2,
                    test_phrases_pct: 20.0,
                })
            }
        }
    });

    let on_json_change = {
        let embedding_training_config = embedding_training_config.clone();
        let storage = storage.clone();
        move |input_json: String| {
            if let Ok(input_config) = serde_json::from_str::<TrainEmbeddingConfig>(&input_json) {
                storage.set(input_config.clone());
                embedding_training_config.set(Rc::new(input_config));
            }
        }
    };

    let json = serde_json::to_string_pretty(embedding_training_config.clone().as_ref()).unwrap();

    html! {
        <div>
        <EmbeddingTrainer config={&*embedding_training_config}/>
        <JsonEditor json_input={json} on_json_change={on_json_change} />
        </div>
    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}
