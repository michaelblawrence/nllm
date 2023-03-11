use std::{collections::HashSet, rc::Rc};

use tracing::info;
use yew::prelude::*;

use crate::ml::{JsRng, RNG};
use super::handle;

#[derive(Properties, Clone, PartialEq)]
pub struct EmbeddingNearestProps {
    pub vocab_and_phrases: Rc<(HashSet<String>, Vec<Vec<String>>, Vec<Vec<String>>)>,
    pub embedding: handle::EmbeddingHandle,
    #[prop_or_default]
    pub iter_hint: usize,
}

#[function_component]
pub fn EmbeddingNearest(props: &EmbeddingNearestProps) -> Html {
    let vocab_and_phrases = props.vocab_and_phrases.clone();
    let embedding = props.embedding.clone();
    let iter_hint = props.iter_hint.clone();

    let random_vocab_seed = use_state(|| 0);
    let nearest_target = use_state(|| Some("anyone".to_string()));
    let nearest_message = use_state(|| "".to_string());

    use_effect_with_deps(
        {
            let nearest_message = nearest_message.clone();

            move |(nearest_target, embedding, i): &(
                Option<String>,
                handle::EmbeddingHandle,
                _,
            )| {
                info!("iter hint = {i}");
                nearest_message.set(format!(
                    "Nearest vocab word to '{}' is ...",
                    nearest_target.clone().unwrap_or_else(|| "...".to_string())
                ));

                if let Some(word) = nearest_target {
                    let nearest = embedding.borrow().nearest(&word);
                    info!("nearest = {nearest:?}");

                    match nearest {
                        Ok(next_nearest) => nearest_message.set(format!(
                            "Nearest vocab word to '{}' is {} (with a distance of {})",
                            word, next_nearest.0, next_nearest.1
                        )),
                        Err(_) => (),
                    }
                }

                move || {}
            }
        },
        ((*nearest_target).clone(), embedding.clone(), iter_hint),
    );

    use_effect_with_deps(
        {
            let nearest_target = nearest_target.clone();

            move |(vocab_and_phrases, _): &(
                Rc<(HashSet<String>, Vec<Vec<String>>, Vec<Vec<String>>)>,
                _,
            )| {
                let (vocab, _, _) = &**vocab_and_phrases;
                let rand_range = JsRng::default().rand_range(0, vocab.len());
                if let Some(random_vocab) = vocab.iter().nth(rand_range) {
                    nearest_target.set(Some(random_vocab.clone()));
                }

                move || {}
            }
        },
        (vocab_and_phrases.clone(), *random_vocab_seed),
    );

    let onclick_new_vocab_word = {
        let random_vocab_seed = random_vocab_seed.clone();

        move |_| random_vocab_seed.set(*random_vocab_seed + 1)
    };

    html! {
        <div class="trainer-nearest">
            <h3>{"Test random 'nearest' word"}</h3>
            <button onclick={onclick_new_vocab_word}>{ format!("New word") }</button>
            <p>{nearest_message.to_string()}</p>
        </div>
    }
}
