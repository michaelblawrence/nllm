use serde_json::Value;
use tracing::debug;
use web_sys::HtmlTextAreaElement;
use yew::prelude::*;

#[derive(Properties, PartialEq, Clone)]
pub struct JsonEditorProps {
    #[prop_or_default]
    pub on_json_change: Callback<String>,
    #[prop_or_default]
    pub json_input: String,
    #[prop_or_default]
    pub input_invalid: bool,
}

fn is_valid_json(value: &str) -> bool {
    serde_json::from_str::<Value>(value).is_err()
}

#[function_component(JsonEditor)]
pub fn json_editor(props: &JsonEditorProps) -> Html {
    let on_json_change = props.on_json_change.clone();
    let input_invalid = props.input_invalid.clone();

    let invalid_json = use_state(|| false);
    use_effect_with_deps(
        {
            let invalid_json = invalid_json.clone();
            move |json_input: &String| {
                invalid_json.set(is_valid_json(&json_input));
                || {}
            }
        },
        props.json_input.clone(),
    );
    let on_json_input = {
        let invalid_json = invalid_json.clone();
        move |event: InputEvent| {
            if let Some(value) = event
                .target_dyn_into::<HtmlTextAreaElement>()
                .map(|x| x.value())
            {
                debug!("JSON: {} ", &value);
                invalid_json.set(is_valid_json(&value));
                on_json_change.emit(value);
            }
        }
    };

    html! {
        <div class="json-editor">
            // <Prism
            //     code=""
            //     language="json"
            //     theme="okaidia"
            // />
            <textarea
                oninput={on_json_input}
                value={props.json_input.clone()}
                class={classes!(
                    (*invalid_json || input_invalid).then_some("json-invalid"),
                )}
            />
        </div>
    }
}
