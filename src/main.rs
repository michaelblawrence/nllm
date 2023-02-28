mod components;
mod ml;

use std::rc::Rc;

use yew::prelude::*;

#[function_component]
fn App() -> Html {
    use crate::components::{EmbeddingTrainer, JsonEditor};

    let embedding_training_config = use_state(move || {
        Rc::new(components::trainer::TrainEmbeddingConfig {
            embedding_size: 2,
            training_rounds: 0,
            max_phrases_count: 300,
            max_vocab_words_count: 125,
            word_locality_factor: 2,
            train_rate: 1e-2,
            test_phrases_pct: 20.0,
        })
    });

    let on_json_change = {
        let embedding_training_config = embedding_training_config.clone();
        move |input_json: String| {
            if let Ok(input_config) = serde_json::from_str(&input_json) {
                embedding_training_config.set(Rc::new(input_config));
            }
        }
    };

    let json = serde_json::to_string_pretty(embedding_training_config.clone().as_ref()).unwrap();

    html! {
        <div>
            <JsonEditor json_input={json} on_json_change={on_json_change} />
            <EmbeddingTrainer config={&*embedding_training_config}/>
        </div>
    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}
