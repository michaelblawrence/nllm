use anyhow::Context;
use aws_sdk_lambda::types::InvokeMode;
use plane::ml::RngStrategy;
use serde_json::json;
use std::{sync::Arc, time::Duration};
use tokio::sync::broadcast;
use tracing::{error, info};

pub async fn spawn_lambda_inference_client(
    lambda_arn: &str,
    tx: broadcast::Sender<String>,
) -> broadcast::Sender<String> {
    let (model_tx, mut model_rx) = broadcast::channel::<String>(100);

    use aws_sdk_lambda::{primitives::Blob, types::InvocationType, Client as LambdaClient};

    let shared_config = aws_config::load_from_env().await;
    let client = LambdaClient::new(&shared_config);
    let lambda_arn = lambda_arn.to_owned();
    let function_url_config = client
        .get_function_url_config()
        .function_name(&lambda_arn)
        .send()
        .await
        .expect("error extracting lambda url config");

    match function_url_config.invoke_mode() {
        Some(InvokeMode::ResponseStream) => {}
        Some(_) => panic!("lambda response streaming is not enabled on server"),
        None => panic!("lambda config missing function invoke mode"),
    }
    let function_url = function_url_config
        .function_url()
        .expect("lambda config missing function url")
        .to_owned();

    let function_name = client.invoke().function_name(&lambda_arn);
    let factory = move |prompt: Option<&str>| {
        function_name
            .clone()
            .set_payload(prompt.map(|prompt| Blob::new(json!({ "prompt": prompt }).to_string())))
    };

    let function = factory(None);
    function
        .invocation_type(InvocationType::DryRun)
        .send()
        .await
        .expect("failed to validate lambda connection");

    info!("Connected to lambda endpoint...");

    tokio::spawn(async move {
        let http_client = reqwest::Client::new();
        while let Ok(prompt) = model_rx.recv().await {
            info!("Requesting inference from remote");
            match http_client
                .post(&function_url)
                .json(&json!({ "prompt": &prompt }))
                .send()
                .await
            {
                Ok(response) => {
                    use futures::StreamExt;
                    use tokio::io::AsyncBufReadExt;

                    info!("Started streaming inference response from remote: prompt = `{prompt}`");
                    match response.error_for_status_ref() {
                        Ok(_) => (),
                        Err(e) => {
                            error!("Error streaming inference response from remote: {e}");
                            continue;
                        }
                    }
                    let stream = response.bytes_stream();
                    let stream = stream.map(|result| {
                        result.map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))
                    });

                    let read = tokio_util::io::StreamReader::new(stream);
                    let reader = tokio::io::BufReader::new(read);
                    let mut lines = reader.lines();
                    let mut last_line = String::new();

                    while let Ok(Some(msg)) = lines.next_line().await {
                        last_line.clear();
                        last_line.push_str(&msg);
                        tx.send(msg).unwrap();
                    }
                    info!("Completed streaming inference response from remote: {last_line}");
                }
                Err(e) => error!("Error starting inference streaming from remote: {e}"),
            }
        }
    });

    model_tx
}
pub fn spawn_local_inference(
    model_fpath: Option<&str>,
    tx: broadcast::Sender<String>,
) -> broadcast::Sender<String> {
    let (model, state) = respond::load(model_fpath).unwrap();
    let (model_tx, mut model_rx) = broadcast::channel::<String>(100);

    let rng = RngStrategy::default();
    let ctx = Arc::new((model, state, rng, tx.clone()));

    tokio::spawn(async move {
        while let Ok(prompt) = model_rx.recv().await {
            let ctx = ctx.clone();

            let prompt: Arc<str> = prompt.into();
            let infer_task = tokio::task::spawn_blocking({
                let prompt = prompt.clone();
                move || {
                    let (model, state, rng, tx) = &*ctx;
                    let messages = infer(&prompt, &model, &state, &rng)?;
                    for msg in messages {
                        tx.send(msg)?;
                    }
                    anyhow::Result::<()>::Ok(())
                }
            });

            match infer_task.await.context("failed to join task") {
                Ok(Err(e)) | Err(e) => {
                    error!("Error performing model inference for prompt = '{prompt}': {e}");
                }
                _ => (),
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
    let config = to_prompt_config(state);

    let env_timeout_secs = std::env::var("NLLM_INFER_TIMEOUT_MS").ok();
    let env_timeout_secs = env_timeout_secs.and_then(|x| x.parse::<u64>().ok());
    let default_timeout_ms = 15_000;
    let timeout_duration = Duration::from_millis(env_timeout_secs.unwrap_or(default_timeout_ms));
    let inference_started = std::time::Instant::now();

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

                if inference_started.elapsed() > timeout_duration {
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

pub fn to_prompt_config(state: &respond::ExtractedModelConfig) -> respond::PromptConfig<'_> {
    use respond::PromptChatMode::*;
    let char_mode = state.char_mode.expect("missing char_mode");
    let use_human_ctx_chat_format = std::env::var("NLLM_TRIPLE_HASH_PROMPT")
        .map(|x| x != "0")
        .unwrap_or(false);
    let chat_mode = if use_human_ctx_chat_format {
        TripleHashHumanPrompt
    } else {
        DirectPrompt
    };
    respond::PromptConfig {
        use_gdt: state.use_gdt,
        char_mode,
        vocab_supervised_predictions_enabled: false,
        chat_mode,
        vocab: None,
    }
}

pub fn run_inference(
    prompt: &str,
    model: &respond::RespondModel,
    state: &respond::ExtractedModelConfig,
    rng: &RngStrategy,
    tx: &tokio::sync::broadcast::Sender<String>,
) {
    let inference_started = std::time::Instant::now();
    let config = to_prompt_config(&state);

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
