use std::rc::Rc;

use tracing::{metadata::LevelFilter, Level};
use tracing_subscriber::prelude::*;
use yew::prelude::*;
use yew_hooks::use_local_storage;

#[function_component]
fn App() -> Html {
    use plane::components::{trainer::TrainEmbeddingConfig, EmbeddingTrainer, JsonEditor};

    let storage = use_local_storage::<TrainEmbeddingConfig>("embeddings_config".to_string());
    let json_is_valid = use_state(|| true);

    let embedding_training_config = use_state({
        let storage = storage.clone();
        move || {
            if let Some(input_config) = &*storage {
                Rc::new(input_config.clone())
            } else {
                Rc::new(TrainEmbeddingConfig {
                    embedding_size: 10,
                    hidden_layer_nodes: 100,
                    training_rounds: 0,
                    max_phrases_count: 800,
                    max_vocab_words_count: 300,
                    input_stride_width: 3,
                    batch_size: 10,
                    train_rate: 1e-3,
                    process_all_batches: false,
                    test_phrases_pct: 20.0,
                })
            }
        }
    });

    let on_json_change = {
        let embedding_training_config = embedding_training_config.clone();
        let json_is_valid = json_is_valid.clone();
        let storage = storage.clone();
        move |input_json: String| {
            if let Ok(input_config) = serde_json::from_str::<TrainEmbeddingConfig>(&input_json) {
                storage.set(input_config.clone());
                json_is_valid.set(true);
                embedding_training_config.set(Rc::new(input_config));
            } else {
                json_is_valid.set(false);
            }
        }
    };

    let json = serde_json::to_string_pretty(embedding_training_config.clone().as_ref()).unwrap();

    html! {
        <div>
        <EmbeddingTrainer config={&*embedding_training_config}/>
        <JsonEditor json_input={json} on_json_change={on_json_change} input_invalid={!*json_is_valid}/>
        </div>
    }
}

fn main() {
    use tracing_subscriber::fmt::{self, format, time};

    let fmt_layer = fmt::layer()
        .with_ansi(true)
        .with_timer(time::UtcTime::rfc_3339())
        .with_writer(tracing_web::MakeConsoleWriter)
        .with_span_events(format::FmtSpan::ACTIVE)
        .with_filter(LevelFilter::from_level(Level::INFO));

    let perf_layer =
        tracing_web::performance_layer().with_details_from_fields(format::Pretty::default());

    tracing_subscriber::registry()
        .with(fmt_layer)
        .with(perf_layer)
        .init();

    yew::Renderer::<App>::new().render();
}
