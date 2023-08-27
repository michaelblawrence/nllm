use anyhow::Context;
use aws_sdk_s3::Client as S3Client;
use hyper::{body::Body, Response};
use lambda_runtime::{service_fn, Error, LambdaEvent};
use serde_json::Value;
use tokio::io::{AsyncReadExt, BufReader};
use tracing::info;

use std::{sync::Arc, time::Duration};

mod env {
    pub const MODEL_S3_BUCKET: &str = "NLLM_MODEL_S3_BUCKET";
    pub const MODEL_S3_KEY: &str = "NLLM_MODEL_S3_KEY";
    pub const MODEL_PATH: &str = "NLLM_MODEL_PATH";
    pub const TRIPLE_HASH_PROMPT: &str = "NLLM_TRIPLE_HASH_PROMPT";
}

async fn func(
    event: LambdaEvent<Value>,
    ctx: Arc<(
        respond::RespondModel,
        respond::ExtractedModelConfig,
        respond::PromptChatMode
    )>,
) -> Result<Response<Body>, Error> {
    let prompt_txt = extract_prompt(event)?;

    // TODO: resolve this requirement another way
    let make_lowercase = true;
    let prompt_txt = if make_lowercase {
        prompt_txt.to_lowercase().to_owned()
    } else {
        prompt_txt
    };

    let (mut body_tx, body_rx) = Body::channel();
    let (tx, mut rx) = tokio::sync::broadcast::channel(100);

    tokio::task::spawn_blocking(move || {
        let (model, state, chat_mode) = &*ctx;
        for message in infer(&prompt_txt, &model, &state, *chat_mode)? {
            tx.send(message)?;
        }
        <Result<(), Error>>::Ok(())
    });

    tokio::spawn(async move {
        while let Ok(data) = rx.recv().await {
            body_tx.send_data((data + "\n").into()).await.unwrap();
        }
    });

    let resp = Response::builder()
        .header("content-type", "text/html")
        .body(body_rx)?;

    Ok(resp)
}

fn extract_prompt(event: LambdaEvent<Value>) -> anyhow::Result<String> {
    let prompt_txt = event
        .payload
        .get("prompt")
        .cloned()
        .or_else(|| {
            // fallback to html api request body
            serde_json::from_str::<Value>(event.payload.get("body")?.as_str()?)
                .ok()?
                .get("prompt")
                .cloned()
        })
        .context("request missing prompt field")?
        .as_str()
        .context("request prompt field invalid type")?
        .to_owned();
    Ok(prompt_txt)
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    // required to enable CloudWatch error logging by the runtime
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        // disable printing the name of the module in every log line.
        .with_target(false)
        // disabling time is handy because CloudWatch will add the ingestion time.
        .without_time()
        .init();

    let (model, state) = match (
        std::env::var(env::MODEL_S3_BUCKET),
        std::env::var(env::MODEL_S3_KEY),
    ) {
        (Ok(s3_bucket), Ok(s3_key)) => {
            info!("Loading model from S3...");
            let shared_config = aws_config::load_from_env().await;
            let client = S3Client::new(&shared_config);
            load_model_from_s3(client, &s3_bucket, &s3_key).await?
        }
        _ => {
            info!("Loading model from file...");
            let model_fpath = std::env::var(env::MODEL_PATH).ok();
            respond::load(model_fpath.as_deref())?
        }
    };

    let use_human_ctx_chat_format = std::env::var(env::TRIPLE_HASH_PROMPT)
        .map(|x| x != "0")
        .unwrap_or(false);
    let chat_mode = if use_human_ctx_chat_format {
        respond::PromptChatMode::TripleHashHumanPrompt
    } else {
        respond::PromptChatMode::DirectPrompt
    };

    let ctx = Arc::new((model, state, chat_mode));
    let handler = service_fn(|event| func(event, ctx.clone()));

    lambda_runtime::run_with_streaming_response(handler).await?;
    Ok(())
}

async fn load_model_from_s3(
    client: S3Client,
    s3_bucket: &str,
    s3_key: &str,
) -> anyhow::Result<(respond::RespondModel, respond::ExtractedModelConfig)> {
    let query = client.get_object().bucket(s3_bucket).key(s3_key);
    let response = query.send().await;

    let reader = response
        .map_err(|err| {
            let service_err = err.into_service_error();
            let meta = service_err.meta();
            tracing::info!("Error from aws when downloading: {}", meta.to_string());
            service_err
        })?
        .body
        .into_async_read();

    let mut reader = BufReader::new(reader);
    let mut json = String::new();

    reader.read_to_string(&mut json).await?;

    tracing::info!("Object is downloaded, length is {}", json.len());
    respond::from_json(&json)
}

pub fn infer<'a>(
    prompt: &'a str,
    model: &'a respond::RespondModel,
    state: &'a respond::ExtractedModelConfig,
    chat_mode: respond::PromptChatMode,
) -> anyhow::Result<impl Iterator<Item = String> + 'a> {
    let char_mode = state.char_mode.expect("missing char_mode");
    let inference_started = std::time::Instant::now();
    let config = respond::PromptConfig {
        use_gdt: state.use_gdt,
        char_mode,
        chat_mode,
    };

    enum State {
        Infer(String),
        Complete,
    }
    let mut state: Option<State> = None;
    let mut response = respond::process_prompt(&model, &prompt, &config)?;

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
