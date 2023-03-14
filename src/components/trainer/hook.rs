use std::{collections::HashSet, rc::Rc};

use yew::prelude::*;
use yew_hooks::use_timeout;

use crate::ml::{embeddings::Embedding, JsRng};

use super::{
    handle::{self, EmbeddingHandle},
    parser, TrainEmbeddingConfig,
};

pub type VocabAndPhrases = (HashSet<String>, Vec<Vec<String>>, Vec<Vec<String>>);

#[hook]
pub fn use_embeddings<F, C>(
    config: Rc<TrainEmbeddingConfig>,
    with_emedding_fn: F,
    cleanup_fn: C
) -> (
    UseStateHandle<EmbeddingHandle>,
    UseStateHandle<Rc<VocabAndPhrases>>,
    UseStateHandle<usize>,
)
where
    F: FnOnce(Embedding, f64, Rc<VocabAndPhrases>) -> Embedding + 'static,
    C: FnOnce() -> () + 'static,
{
    let vocab_and_phrases =
        use_state(|| Rc::new((Default::default(), Default::default(), Default::default())));
    let embedding_handle = use_state(|| handle::EmbeddingHandle::default());

    let train_remaining_iters = use_state(|| 0_usize);

    let train_timeout = use_timeout(
        {
            let vocab_and_phrases = vocab_and_phrases.clone();
            let embedding_handle = embedding_handle.clone();
            let config = config.clone();
            let train_remaining_iters = train_remaining_iters.clone();

            move || {
                if *train_remaining_iters <= 0 {
                    return;
                }

                embedding_handle.set(embedding_handle.replace_with(
                    |mut embedding_instance| {
                        let (_, phrases, _) = &**vocab_and_phrases;

                        let error = parser::train_embedding(
                            &mut embedding_instance,
                            &phrases,
                            config.train_rate,
                            config.batch_size,
                            config.process_all_batches,
                        );

                        with_emedding_fn(embedding_instance, error, (*vocab_and_phrases).clone())
                    },
                ));

                train_remaining_iters.set(*train_remaining_iters - 1);
            }
        },
        5,
    );

    use_effect_with_deps(
        {
            let train_timeout = train_timeout.clone();
            move |train_remaining_iters: &usize| {
                if *train_remaining_iters > 0 {
                    train_timeout.reset();
                }
            }
        },
        *train_remaining_iters,
    );

    use_effect_with_deps(
        {
            let vocab_and_phrases = vocab_and_phrases.clone();
            let train_remaining_iters = train_remaining_iters.clone();

            move |config: &Rc<TrainEmbeddingConfig>| {
                let (vocab, mut phrases) =
                    parser::parse_vocab_and_phrases(Some(config.max_vocab_words_count));

                let testing_phrases = parser::split_training_and_testing(
                    &mut phrases,
                    config.max_phrases_count,
                    config.test_phrases_pct,
                );

                let v_and_p = Rc::new((vocab, phrases, testing_phrases));
                vocab_and_phrases.set(v_and_p);
                train_remaining_iters.set(0);

                move || {cleanup_fn();}
            }
        },
        config.clone(),
    );

    use_effect_with_deps(
        {
            let embedding_handle = embedding_handle.clone();
            let config = config.clone();
            let train_remaining_iters = train_remaining_iters.clone();

            move |vocab_and_phrases: &Rc<VocabAndPhrases>| {
                let (vocab, ..) = &**vocab_and_phrases;

                embedding_handle.set(embedding_handle.replace(Embedding::new(
                    vocab.clone(),
                    config.embedding_size,
                    config.input_stride_width,
                    vec![config.hidden_layer_nodes],
                    Rc::new(JsRng::default()),
                )));

                // train_remaining_iters.set(config.training_rounds);

                move || {}
            }
        },
        (*vocab_and_phrases).clone(),
    );

    (embedding_handle, vocab_and_phrases, train_remaining_iters)
}
