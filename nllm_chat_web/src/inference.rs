use anyhow::{bail, Context};
use aws_sdk_lambda::types::InvokeMode;
use serde_json::json;
use tokio::sync::broadcast;
use tracing::{debug, error, info};

use std::{sync::Arc, time::Duration};

#[derive(Debug, Clone)]
pub struct ModelPromptRequest {
    pub prompt: String,
    pub username: String,
    pub use_beta_function: bool,
    pub index: usize,
}

pub async fn spawn_lambda_inference_client(
    lambda_arn: &str,
    tx: broadcast::Sender<String>,
) -> broadcast::Sender<ModelPromptRequest> {
    let (model_tx, mut model_rx) = broadcast::channel::<ModelPromptRequest>(100);

    use aws_sdk_lambda::{primitives::Blob, types::InvocationType, Client as LambdaClient};

    let shared_config = aws_config::load_from_env().await;
    let client = LambdaClient::new(&shared_config);
    let lambda_arn = lambda_arn.to_owned();
    _ = get_lambda_funtion_url(&client, &lambda_arn, None).await;

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
        while let Ok(ModelPromptRequest {
            prompt,
            username,
            use_beta_function,
            index,
        }) = model_rx.recv().await
        {
            // TODO: consider moving to env var
            let lambda_qualifier = if use_beta_function { "beta" } else { "prod" };
            info!("Resolving lambda function url (version = '{lambda_qualifier}')");
            let function_url =
                match get_lambda_funtion_url(&client, &lambda_arn, Some(lambda_qualifier)).await {
                    Ok(url) => url,
                    Err(e) => {
                        error!("failed to retrieve lambda url: {e}");
                        continue;
                    }
                };

            info!("Requesting inference from remote url = {function_url}");
            let body = json!({ "prompt": &prompt });
            let response = http_client.post(&function_url).json(&body).send().await;
            match response {
                Ok(response) => {
                    use futures::StreamExt;
                    use tokio::io::AsyncBufReadExt;

                    info!("Started streaming inference response from remote: prompt = `{prompt}`");
                    info!(target: "generation-started", index=index, prompt=prompt, username=username, remote=true);

                    match response.error_for_status_ref() {
                        Ok(_) => (),
                        Err(e) => {
                            error!("Error streaming inference response from remote: {e}");
                            info!(target: "generation-error", index=index, prompt=prompt, username=username, remote=true);
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
                    let mut response_msg = String::new();

                    while let Ok(Some(msg)) = lines.next_line().await {
                        response_msg.clear();
                        response_msg.push_str(&msg);
                        tx.send(msg).unwrap();
                    }
                    info!(target: "generation-completed", index=index, response=response_msg, username=username, remote=true);
                    info!("Completed streaming inference response from remote: response = `{response_msg}`");
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
) -> broadcast::Sender<ModelPromptRequest> {
    let (model, _) = respond::load(model_fpath).unwrap();
    let (model_tx, mut model_rx) = broadcast::channel::<ModelPromptRequest>(100);

    let ctx = Arc::new((model, tx.clone()));

    tokio::spawn(async move {
        while let Ok(prompt_request) = model_rx.recv().await {
            let ctx = ctx.clone();
            let prompt = prompt_request.prompt.clone();

            let infer_task = tokio::task::spawn_blocking({
                move || {
                    let (model, tx) = &*ctx;
                    let messages = infer(&prompt_request, &model)?;
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
    prompt_request: &'a ModelPromptRequest,
    model: &'a respond::RespondModel,
) -> anyhow::Result<impl Iterator<Item = String> + 'a> {
    let ModelPromptRequest {
        prompt,
        username,
        index,
        ..
    } = prompt_request;
    let config = to_prompt_config();

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
    let mut response = respond::process_prompt(&model, &prompt, &config)?;
    info!(target: "generation-started", index=index, prompt=prompt, username=username, remote=false);

    Ok(std::iter::from_fn(move || match state.take() {
        None => {
            state = Some(State::Infer(String::new()));
            Some(format!("Chat: 　"))
        }
        Some(State::Infer(mut response_msg)) => {
            if let Some((token, separator)) = response.next() {
                response_msg += &format!("{token}{separator}");

                if inference_started.elapsed() > timeout_duration {
                    info!(target: "generation-timeout", index=index, prompt=prompt, username=username, remote=false);

                    state = Some(State::Complete);
                    Some(format!("Chat: {response_msg}..."))
                } else {
                    let msg = format!("[CHAT_PARTIAL]: {response_msg}");

                    state = Some(State::Infer(response_msg));
                    Some(msg)
                }
            } else {
                state = Some(State::Complete);
                info!(target: "generation-completed", index=index, response=response_msg, username=username, remote=false);
                Some(format!("Chat: {response_msg}"))
            }
        }
        Some(State::Complete) => {
            debug!("Completed user prompt response");
            None
        }
    }))
}

async fn get_lambda_funtion_url(
    client: &aws_sdk_lambda::Client,
    lambda_arn: &str,
    lambda_qualifier: Option<&str>,
) -> anyhow::Result<String> {
    let function_url_config_builder = client.get_function_url_config().function_name(lambda_arn);
    let function_url_config_builder = if let Some(qualifier) = lambda_qualifier {
        function_url_config_builder.qualifier(qualifier)
    } else {
        function_url_config_builder
    };
    let function_url_config = function_url_config_builder
        .send()
        .await
        .context("error extracting lambda url config")?;

    match function_url_config.invoke_mode() {
        Some(InvokeMode::ResponseStream) => {}
        Some(_) => bail!("lambda response streaming is not enabled on server"),
        None => bail!("lambda config missing function invoke mode"),
    }
    let function_url = function_url_config
        .function_url()
        .context("lambda config missing function url")?
        .to_owned();

    Ok(function_url)
}

pub fn to_prompt_config() -> respond::PromptConfig {
    use respond::PromptChatMode::*;
    let use_human_ctx_chat_format = std::env::var("NLLM_TRIPLE_HASH_PROMPT")
        .map(|x| x != "0")
        .unwrap_or(false);
    let chat_mode = if use_human_ctx_chat_format {
        TripleHashHumanPrompt
    } else {
        DirectPrompt
    };
    respond::PromptConfig {
        chat_mode,
    }
}
