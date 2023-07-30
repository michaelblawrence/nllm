use plane::ml::RngStrategy;
use std::{sync::Arc, time::Duration};
use tokio::sync::broadcast;
use tracing::info;

pub fn spawn_local_inference(tx: broadcast::Sender<String>) -> broadcast::Sender<String> {
    let model_fpath = std::env::var("NLLM_MODEL_PATH").ok();
    let (model, state) = respond::load(model_fpath.as_deref()).unwrap();

    let (model_tx, mut model_rx) = broadcast::channel::<String>(100);

    tokio::spawn({
        let tx = tx.clone();
        let rng = RngStrategy::default();
        let ctx = Arc::new((model, state, rng, tx.clone()));
        async move {
            while let Ok(prompt) = model_rx.recv().await {
                let ctx = ctx.clone();

                tokio::task::spawn_blocking(move || {
                    let (model, state, rng, tx) = &*ctx;
                    match infer(&prompt, &model, &state, &rng) {
                        Ok(messages) => {
                            for msg in messages {
                                tx.send(msg).unwrap();
                            }
                        }
                        Err(_) => return,
                    };
                })
                .await
                .unwrap();
            }
        }
    });

    model_tx
}

pub fn infer<'a>(
    prompt: &'a str,
    model: &'a respond::RespondModel,
    state: &'a respond::ExtractedModelConfig,
    rng: &'a RngStrategy,
) -> anyhow::Result<impl Iterator<Item = String> + 'a> {
    let char_mode = state.char_mode.expect("missing char_mode");
    let inference_started = std::time::Instant::now();
    let config = respond::PromptConfig {
        use_gdt: state.use_gdt,
        char_mode,
        vocab_supervised_predictions_enabled: false,
        vocab: None,
    };

    enum State {
        Infer(String),
        Complete,
    }
    let mut state: Option<State> = None;
    let mut response = respond::process_prompt(&model, &rng, &prompt, &config)?;

    Ok(std::iter::from_fn(move || match state.take() {
        None => {
            state = Some(State::Infer(String::new()));
            Some(format!("Chat: 　"))
        }
        Some(State::Infer(mut response_msg)) => {
            if let Some((token, separator)) = response.next() {
                response_msg += &format!("{token}{separator}");

                if inference_started.elapsed() > Duration::from_secs(15) {
                    info!("Timed out on prompt `{prompt}`");

                    state = Some(State::Complete);
                    Some(format!("Chat: {response_msg}..."))
                } else {
                    let msg = format!("[CHAT_PARTIAL]: {response_msg}");

                    state = Some(State::Infer(response_msg));
                    Some(msg)
                }
            } else {
                state = Some(State::Complete);
                Some(format!("Chat: {response_msg}"))
            }
        }
        Some(State::Complete) => {
            info!("Completed user prompt response");
            None
        }
    }))
}

pub fn run_inference(
    prompt: &str,
    model: &respond::RespondModel,
    state: &respond::ExtractedModelConfig,
    rng: &RngStrategy,
    tx: &tokio::sync::broadcast::Sender<String>,
) {
    let char_mode = state.char_mode.expect("missing char_mode");
    let inference_started = std::time::Instant::now();
    let config = respond::PromptConfig {
        use_gdt: state.use_gdt,
        char_mode,
        vocab_supervised_predictions_enabled: false,
        vocab: None,
    };
    let response = match respond::process_prompt(&model, &rng, &prompt, &config) {
        Ok(x) => x,
        Err(_) => return,
    };

    _ = tx.send(format!("Chat: 　"));
    let mut response_msg = String::new();
    for (token, separator) in response {
        response_msg += &format!("{token}{separator}");
        _ = tx.send(format!("[CHAT_PARTIAL]: {response_msg}"));

        if inference_started.elapsed() > Duration::from_secs(15) {
            info!("Timed out on prompt `{prompt}`");
            _ = tx.send(format!("Chat: {response_msg}..."));
            return;
        }
    }

    info!("Completed user prompt response");
    _ = tx.send(format!("Chat: {response_msg}"));
}
