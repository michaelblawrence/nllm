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

use std::{collections::HashMap, sync::Arc};

pub mod env {
    pub const CTRL_PORT: &str = "NLLM_CTRL_PORT";
    pub const HOTRELOAD: &str = "NLLM_HOTRELOAD";
}

pub struct ApiAppState {
    pub handles: tokio::sync::Barrier,
    pub channel: broadcast::Sender<String>,
}

pub struct AppState {
    pub tx: broadcast::Sender<String>,
    pub api: ApiAppState,
}

pub async fn app(app_state: Arc<AppState>) -> Router<()> {
    Router::new()
        .route("/", get(api_index))
        .route("/payload", post(api_payload))
        .route("/diagnostics/ws.js", get(diagnostics_ws_js))
        .route("/keepalive", get(keepalive_websocket_handler))
        .with_state(app_state)
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
        include_str!("../../nllm_chat_web/public/res/ws.js")
    } else {
        "// diagnostic js loaded"
    };
    let content_type_header = vec![(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/javascript"),
    )];
    (AppendHeaders(content_type_header), body)
}
