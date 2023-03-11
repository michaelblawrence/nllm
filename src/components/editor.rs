use tracing::info;
use web_sys::HtmlTextAreaElement;
use yew::prelude::*;

#[derive(Properties, PartialEq, Clone)]
pub struct JsonEditorProps {
    #[prop_or_default]
    pub on_json_change: Callback<String>,
    #[prop_or_default]
    pub json_input: String,
}

#[function_component(JsonEditor)]
pub fn json_editor(props: &JsonEditorProps) -> Html {
    let on_json_change = props.on_json_change.clone();
    let on_json_input = move |event: InputEvent| {
        if let Some(value) = event
            .target_dyn_into::<HtmlTextAreaElement>()
            .map(|x| x.value())
        {
            info!("{} ", &value);
            on_json_change.emit(value);
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
            />
        </div>
    }
}
