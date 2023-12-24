use anyhow::{bail, Context};
use axum::{
    extract::{
        ws::{Message, WebSocketUpgrade},
        State,
    },
    http::StatusCode,
    response::{AppendHeaders, Html, IntoResponse},
    routing::{get, post},
    Form, Router,
};
use futures::{sink::SinkExt, stream::StreamExt};
use reqwest::header;
use tokio::sync::broadcast;
use tower_http::services::ServeDir;

use std::{
    collections::{HashMap, HashSet},
    net::SocketAddr,
    sync::{Arc, Mutex},
};

use crate::tower_ext::ApiServiceExt;

mod chat;
mod inference;
mod logging;
mod model;
mod tower_ext;

mod env {
    pub const API_PORT: &str = "NLLM_API_PORT";
    pub const MODEL_PATH: &str = "NLLM_MODEL_PATH";
    pub const INFER_LAMBDA_ARN: &str = "NLLM_INFER_LAMBDA_ARN";
    pub const HOTRELOAD: &str = "NLLM_HOTRELOAD";
}

pub struct ApiAppState {
    handles: tokio::sync::Barrier,
    channel: broadcast::Sender<String>,
}

pub struct AppState {
    user_set: Mutex<HashSet<String>>,
    tx: broadcast::Sender<String>,
    model_tx: broadcast::Sender<inference::ModelPromptRequest>,
    api: ApiAppState,
}

#[tokio::main]
async fn main() {
    logging::configure_logging().await;

    let app_state = configure_app_state().await;
    if let Err(e) = configure_current_dir() {
        eprintln!("Failed to locate content dir: {e}");
        return;
    }

    let app = Router::new()
        .route("/", get(index))
        .route("/diagnostics/ws.js", get(diagnostics_ws_js))
        .route("/websocket", get(websocket_handler))
        .route("/keepalive", get(keepalive_websocket_handler))
        .nest(
            "/api",
            Router::new()
                .route("/", get(api_index))
                .route("/payload", post(api_payload)),
        )
        .nest_service("/scripts", ServeDir::new("public/scripts").no_cache())
        .nest_service("/icons", ServeDir::new("public/icons"))
        .nest_service("/images", ServeDir::new("public/images"))
        .with_state(app_state);

    let addr = socket_addr_from_env(env::API_PORT, 3000);
    tracing::info!("listening on {}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service_with_connect_info::<SocketAddr>())
        .await
        .unwrap();
}

async fn api_index() -> impl IntoResponse {
    match std::fs::read_to_string("admin.html") {
        Ok(html) => {
            let no_cache_header = vec![
                (header::CACHE_CONTROL, "no-cache, no-store"),
                (header::EXPIRES, "-1"),
            ];
            (StatusCode::OK, AppendHeaders(no_cache_header), Html(html))
        }
        Err(err) => (
            StatusCode::NOT_FOUND,
            AppendHeaders(vec![]),
            Html(err.to_string()),
        ),
    }
}

async fn api_payload(
    State(state): State<Arc<AppState>>,
    Form(map): Form<HashMap<String, String>>,
) -> Result<impl IntoResponse, String> {
    let mut rx = state.api.channel.subscribe();
    let barrier = state.api.handles.wait().await;
    let map_json = if barrier.is_leader() {
        // let map_json = serde_json::to_string_pretty(&map).map_err(|x| x.to_string())?;
        let map_json = map
            .get(&String::from("type"))
            .cloned()
            .unwrap_or_else(|| String::from("Unknown"));
        state.api.channel.send(map_json.clone()).unwrap();
        map_json
    } else {
        rx.recv().await.unwrap()
    };
    let no_cache_header = vec![
        (header::CACHE_CONTROL, "no-cache, no-store"),
        (header::EXPIRES, "-1"),
    ];
    let html = format!(r#"<p>{para}</p>"#, para = map_json);
    Ok((StatusCode::OK, AppendHeaders(no_cache_header), Html(html)))
}

async fn index() -> impl IntoResponse {
    match std::fs::read_to_string("index.html") {
        Ok(html) => {
            let no_cache_header = vec![
                (header::CACHE_CONTROL, "no-cache, no-store"),
                (header::EXPIRES, "-1"),
            ];
            (StatusCode::OK, AppendHeaders(no_cache_header), Html(html))
        }
        Err(err) => (
            StatusCode::NOT_FOUND,
            AppendHeaders(vec![]),
            Html(err.to_string()),
        ),
    }
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    connection: axum::extract::ConnectInfo<SocketAddr>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| chat::chat_websocket(socket, connection.ip(), state))
}

async fn keepalive_websocket_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(|socket| async {
        let (mut sender, mut receiver) = socket.split();
        loop {
            match sender.send(Message::Text(String::from("ping"))).await {
                Ok(_) => {}
                Err(_) => return,
            };
            match receiver.next().await {
                Some(Err(_)) => return,
                _ => {}
            };
        }
    })
}

// Include utf-8 file at **compile** time.
async fn diagnostics_ws_js() -> impl IntoResponse {
    let body: &'static str = if let Ok(_) = std::env::var(env::HOTRELOAD) {
        include_str!("../public/res/ws.js")
    } else {
        "// diagnostic js loaded"
    };
    let content_type_header = vec![(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/javascript"),
    )];
    (AppendHeaders(content_type_header), body)
}

async fn configure_app_state() -> Arc<AppState> {
    let (tx, rx) = broadcast::channel(100);
    std::mem::drop(rx);

    let model_tx = match (
        std::env::var(env::MODEL_PATH).ok(),
        std::env::var(env::INFER_LAMBDA_ARN).ok(),
    ) {
        (None, Some(lambda_arn)) => {
            inference::spawn_lambda_inference_client(&lambda_arn, tx.clone()).await
        }
        (model_fpath, None) => inference::spawn_local_inference(model_fpath.as_deref(), tx.clone()),
        (_, _) => panic!(
            "Either the NLLM_MODEL_PATH or NLLM_INFER_LAMBDA_ARN environment variable must be set"
        ),
    };

    let mut user_set = HashSet::new();
    user_set.insert("Chat".to_string());

    Arc::new(AppState {
        user_set: Mutex::new(user_set),
        tx,
        model_tx,
        api: ApiAppState {
            handles: tokio::sync::Barrier::new(2),
            channel: {
                let channel = broadcast::channel(1);
                std::mem::forget(channel.1);
                channel.0
            },
        },
    })
}

fn configure_current_dir() -> anyhow::Result<()> {
    let index_html_path = std::path::Path::new("./index.html");
    if !index_html_path.exists() {
        let current_exe = std::env::current_exe().context("failed to get cwd")?;
        let mut dir = Some(current_exe);
        while let Some(current) = dir {
            let index_test_path = current.join("nllm_chat_web/index.html");
            if index_test_path.exists() {
                std::env::set_current_dir(current.join("nllm_chat_web"))
                    .context("failed to set current dir to content root")?;

                return Ok(());
            }
            dir = current.parent().map(|x| x.to_path_buf());
        }
        bail!("failed to find index.html content root");
    }
    Ok(())
}

fn socket_addr_from_env(port_env_key: &str, default: u16) -> SocketAddr {
    let port = std::env::var(port_env_key).ok();
    let port = port.and_then(|x| x.parse().ok()).unwrap_or(default);
    SocketAddr::from(([0, 0, 0, 0], port))
}
