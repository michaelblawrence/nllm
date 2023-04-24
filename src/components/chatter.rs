use tracing::debug;
use web_sys::HtmlTextAreaElement;
use yew::prelude::*;

use crate::ml::embeddings::builder::EmbeddingBuilder;

#[cfg(feature = "include_res")]
const MODEL_JSON: &str = include_str!(
    "../../out/labelled/train/imdb-reviews-neg-v2-e32-L150x150x75x50/model-r1273568-42pct.json"
);
#[cfg(not(feature = "include_res"))]
const MODEL_JSON: &str = "";

fn is_valid_prompt(value: &str) -> bool {
    value.trim().ends_with("?")
}

#[function_component(EmbeddingChat)]
pub fn embedding_chat() -> Html {
    let embedding_state = use_state(|| {
        let json = &MODEL_JSON;
        EmbeddingBuilder::from_snapshot(json)
            .unwrap()
            .build()
            .unwrap()
    });

    let prompt_text = use_state(|| None);
    let prompt_response = use_state(|| None);
    let invalid_json = use_state(|| false);

    use_effect_with_deps(
        {
            let embedding_state = embedding_state.clone();
            let prompt_response = prompt_response.clone();
            move |prompt_input: &Option<String>| {
                if let Some(prompt_input) = prompt_input {
                    let prompt_input_chars: Vec<_> =
                        prompt_input.chars().map(|c| c.to_string()).collect();
                    let context_tokens: Vec<_> =
                        prompt_input_chars.iter().map(|c| c.as_str()).collect();
                    let response_tokens = embedding_state
                        .predict_from_iter(&context_tokens[..])
                        .collect::<Vec<_>>();
                    prompt_response.set(Some(response_tokens.join("")));
                } else {
                    prompt_response.set(None);
                }
                || {}
            }
        },
        (*prompt_text).clone(),
    );

    let on_prompt_input = {
        let invalid_json = invalid_json.clone();
        let prompt_text = prompt_text.clone();
        let prompt_response = prompt_response.clone();
        move |event: InputEvent| {
            if let Some(value) = event
                .target_dyn_into::<HtmlTextAreaElement>()
                .map(|x| x.value())
            {
                let is_valid_prompt = is_valid_prompt(&value);
                debug!("PROMPT INPUT (valid = {}): {} ", is_valid_prompt, &value);

                if is_valid_prompt {
                    prompt_text.set(Some(value));
                    invalid_json.set(true);
                } else {
                    prompt_text.set(None);
                    prompt_response.set(None);
                    invalid_json.set(false);
                }
            }
        }
    };

    html! {
        <div class="json-editor">
            <h3>{"Experimental: prompt response interface"}</h3>
            <p>
                {format!("you said: {}", (&*prompt_text).clone().unwrap_or_else(|| "<waiting... (end prompts with '?')>".to_string()))}
            </p>
            <p>
                {format!("i responded: {}", &*(&*prompt_response).clone().unwrap_or_else(|| "<probably waiting for prompt>".to_string()))}
            </p>
            <textarea
                oninput={on_prompt_input}
                class={classes!(
                    (*invalid_json).then_some("json-invalid"),
                )}
            />
        </div>
    }
}
