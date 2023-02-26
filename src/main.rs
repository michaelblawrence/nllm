mod ml;
mod components;

use yew::prelude::*;

#[function_component]
fn App() -> Html {
    use crate::components::EmbeddingTrainer;

    html! {
        <EmbeddingTrainer/>
    }
}

mod tester;

fn main() {
    yew::Renderer::<App>::new().render();
}
