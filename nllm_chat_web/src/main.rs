use axum::{
    body::Bytes,
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::{HeaderName, StatusCode},
    response::{AppendHeaders, Html, IntoResponse, Response},
    routing::get,
    Router,
};
use futures::{
    sink::SinkExt,
    stream::{SplitStream, StreamExt},
};
use reqwest::header;

use std::{
    collections::HashSet,
    net::SocketAddr,
    sync::{Arc, Mutex},
};
use tokio::sync::broadcast;
use tower_http::services::ServeDir;
use tracing::metadata::LevelFilter;

use crate::tower_ext::ApiServiceExt;

mod inference;
mod model;
mod tower_ext;

mod env {
    pub const API_PORT: &str = "NLLM_API_PORT";
    pub const MODEL_PATH: &str = "NLLM_MODEL_PATH";
    pub const INFER_LAMBDA_ARN: &str = "NLLM_INFER_LAMBDA_ARN";
    pub const HOTRELOAD: &str = "NLLM_HOTRELOAD";
}

struct AppState {
    user_set: Mutex<HashSet<String>>,
    tx: broadcast::Sender<String>,
    model_tx: broadcast::Sender<inference::ModelPromptRequest>,
}

#[tokio::main]
async fn main() {
    configure_logging();

    let app = Router::new()
        .route("/", get(index))
        .route("/diagnostics/ws.js", get(diagnostics_ws_js))
        .route("/websocket", get(websocket_handler))
        .route("/keepalive", get(keepalive_websocket_handler))
        .nest_service("/scripts", ServeDir::new("public/scripts").no_cache())
        .nest_service("/icons", ServeDir::new("public/icons"))
        .nest_service("/images", ServeDir::new("public/images"))
        .with_state(configure_app_state().await);

    let port = std::env::var(env::API_PORT).ok();
    let port = port.and_then(|x| x.parse().ok()).unwrap_or(3000_u16);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("listening on {}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

// async fn index() -> (StatusCode, Html<&'static str>) {
async fn index() -> (
    StatusCode,
    AppendHeaders<Vec<(HeaderName, &'static str)>>,
    Html<String>,
) {
    // (StatusCode::OK, Html(include_str!("../index.html")))

    match std::fs::read_to_string("index.html") {
        Ok(html) => (
            StatusCode::OK,
            AppendHeaders(vec![
                (header::CACHE_CONTROL, "no-cache, no-store"),
                (header::EXPIRES, "-1"),
            ]),
            Html(html),
        ),
        Err(err) => (
            StatusCode::NOT_FOUND,
            AppendHeaders(vec![]),
            Html(err.to_string()),
        ),
    }
}

// Include utf-8 file at **compile** time.
async fn diagnostics_ws_js() -> Response {
    let body: &'static str = if let Ok(_) = std::env::var(env::HOTRELOAD) {
        include_str!("../public/res/ws.js")
    } else {
        "// diagnostic js loaded"
    };
    let response: (_, axum::body::Full<Bytes>) = (
        [(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/javascript"),
        )],
        body.into(),
    );
    response.into_response()
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| websocket(socket, state))
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

fn configure_logging() {
    let subscriber = tracing_subscriber::fmt()
        .compact()
        .with_target(false)
        .with_timer(tracing_subscriber::fmt::time::time())
        .with_max_level(LevelFilter::INFO)
        .finish();

    tracing::subscriber::set_global_default(subscriber).unwrap();
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
    })
}

#[derive(Debug, Clone)]
enum ClientPayload {
    Message(String),
    PingNotification,
}

async fn get_username(
    receiver: &mut SplitStream<WebSocket>,
    state: &AppState,
) -> Result<String, &'static str> {
    // Loop until a text message is found.
    if let Some(Ok(Message::Text(name))) = receiver.next().await {
        // If username that is sent by client is not taken, fill username string.
        match check_username(&state, &name) {
            CheckUsernameAction::FoundNew => Ok(name),
            CheckUsernameAction::InvalidUsernameChars => {
                Err("Username contains invalid characters.")
            }
            CheckUsernameAction::InvalidUsernameLength => Err("Username is too long or short."),
            CheckUsernameAction::AlreadyExisted => Err("Username already taken."),
        }
    } else {
        Err("Username must be set first")
    }
}
// This function deals with a single websocket connection, i.e., a single
// connected client / user, for which we will spawn two independent tasks (for
// receiving / sending chat messages).
async fn websocket(stream: WebSocket, state: Arc<AppState>) {
    // By splitting, we can send and receive at the same time.
    let (mut sender, mut receiver) = stream.split();
    let (client_tx, client_rx) = broadcast::channel(10);

    // Username gets set in the receive loop, if it's valid.
    let name = match get_username(&mut receiver, &state).await {
        Ok(username) => username,
        Err(msg) => {
            let _ = sender.send(Message::Text(String::from(msg))).await;
            return;
        }
    };

    // We subscribe *before* sending the "joined" message, so that we will also
    // display it to our client.
    let rx = state.tx.subscribe();

    // Now send the "joined" message to all subscribers.
    let joined_msg = format!("{} joined.", name);
    let welcome_msg = format!("Chat: Hi {name}, if you ever need my help, just call me by starting your message with 'Hey Chat'");
    tracing::info!("{}", joined_msg);
    let _ = state.tx.send(joined_msg);
    let _ = sender.send(Message::Text(welcome_msg)).await;

    // Spawn the first task that will receive broadcast messages and send text
    // messages over the websocket to our client.
    let mut send_task = tokio::spawn(forward_payload_to_ws(client_rx, sender));

    tokio::spawn(broadcast_messages_to_payload(rx, client_tx.clone()));

    // Spawn a task that takes messages from the websocket, prepends the user
    // name, and sends them to all broadcast subscribers.
    let mut recv_task = tokio::spawn(forward_ws_to_broadcast(
        receiver,
        client_tx,
        state.clone(),
        name.clone(),
    ));

    // If any one of the tasks run to completion, we abort the other.
    tokio::select! {
        _ = (&mut send_task) => recv_task.abort(),
        _ = (&mut recv_task) => send_task.abort(),
    };

    // Send "user left" message (similar to "joined" above).
    let msg = format!("{} left.", name);
    tracing::info!("{}", msg);
    let _ = state.tx.send(msg);

    // Remove username from map so new clients can take it again.
    state.user_set.lock().unwrap().remove(&name);
}

async fn forward_ws_to_broadcast(
    mut ws_receiver: SplitStream<WebSocket>,
    client_tx: broadcast::Sender<ClientPayload>,
    state: Arc<AppState>,
    name: String,
) {
    let name = name.to_owned();
    let mut beta_mode = false;
    while let Some(Ok(Message::Text(text))) = ws_receiver.next().await {
        if text.starts_with('!') {
            let message = match process_cmd(&text[1..], &state, &client_tx) {
                PromptAction::ReportSuccess(cmd_response) => {
                    ClientPayload::Message(format!("🤖 {cmd_response}"))
                }
                PromptAction::ToggleBetaMode => {
                    beta_mode = !beta_mode;
                    ClientPayload::Message(format!(
                        "🤖 Experimental inference model enabled = {beta_mode}"
                    ))
                }
                PromptAction::ReportUnknownCommand(err_response) => {
                    ClientPayload::Message(format!("❓ {err_response}"))
                }
                PromptAction::DiscardMessage => continue,
            };
            client_tx.send(message).unwrap();
            continue;
        }

        let chatbot_prefixes = ["hey ai", "hey chat", "hi ai", "hi chat"];
        for prefix in chatbot_prefixes {
            if text.to_lowercase().starts_with(prefix) {
                let query = text[prefix.len()..].trim_start_matches(|c: char| !c.is_alphabetic());
                let prompt_request = inference::ModelPromptRequest {
                    prompt: query.to_string(),
                    use_beta_function: beta_mode,
                };
                let _ = state.model_tx.send(prompt_request);
                break;
            }
        }

        // Add username before message.
        let _ = state.tx.send(format!("{}: {}", name, text));
    }
}

async fn broadcast_messages_to_payload(
    mut rx: broadcast::Receiver<String>,
    client_tx1: broadcast::Sender<ClientPayload>,
) {
    while let Ok(msg) = rx.recv().await {
        if client_tx1.send(ClientPayload::Message(msg)).is_err() {
            break;
        }
    }
}

async fn forward_payload_to_ws(
    mut payload_rx: broadcast::Receiver<ClientPayload>,
    mut ws_sender: futures::stream::SplitSink<WebSocket, Message>,
) {
    use std::time::{Duration, SystemTime};
    let ping_recv_timeout = Duration::from_millis(20_000);
    let ping_dispatch_interval = Duration::from_millis(7_500);

    let mut last_seen = SystemTime::now();

    loop {
        tokio::select! {
            msg = payload_rx.recv() => {
                match msg {
                    Ok(ClientPayload::Message(msg)) => {
                        let message = Message::Text(msg);
                        if let Ok(Ok(_)) = tokio::time::timeout(ping_recv_timeout, ws_sender.send(message)).await {
                            last_seen = SystemTime::now();
                        } else {
                            // In any websocket error, break loop.
                            println!("failed to send message to client");
                            break;
                        }
                    }
                    Ok(ClientPayload::PingNotification) => { last_seen = SystemTime::now(); },
                    Err(_) => {
                        println!("closed connection");
                        break;
                    },
                }
            }
            _ = tokio::time::sleep(ping_dispatch_interval) => {
                let message = Message::Text("!PING".to_string());
                if let Ok(Ok(_)) = tokio::time::timeout(ping_recv_timeout, ws_sender.send(message)).await {
                } else {
                    println!("failed to ping client");
                    break;
                }
                if SystemTime::now().duration_since(last_seen).map(|since| since > ping_recv_timeout).unwrap_or(true) {
                    println!("timeout");
                    break;
                }
            }
        }
    }
}

enum PromptAction {
    ReportSuccess(String),
    ReportUnknownCommand(String),
    DiscardMessage,
    ToggleBetaMode,
}

fn process_cmd(
    text: &str,
    state: &AppState,
    client_tx: &broadcast::Sender<ClientPayload>,
) -> PromptAction {
    match text.trim().to_uppercase().as_str() {
        "USERS" => PromptAction::ReportSuccess(state.user_set.lock().unwrap().iter().fold(
            String::with_capacity(128),
            |mut acc, x| {
                acc.push_str(x);
                acc.push_str(", ");
                acc
            },
        )),
        "PONG" => {
            client_tx.send(ClientPayload::PingNotification).unwrap();
            PromptAction::DiscardMessage
        }
        "BETA" => PromptAction::ToggleBetaMode,
        _ => PromptAction::ReportUnknownCommand("Invalid Command".to_string()),
    }
}

fn check_username(state: &AppState, name: &str) -> CheckUsernameAction {
    let mut user_set = state.user_set.lock().unwrap();
    let name = name.trim();
    let name_len = name.len();
    let valid_chars = |x: char| x.is_alphanumeric() || x == '_' || x == '-';

    if user_set.contains(name) {
        return CheckUsernameAction::AlreadyExisted;
    }
    if name_len < 2 || name_len > 18 {
        return CheckUsernameAction::InvalidUsernameLength;
    }
    if !name.chars().all(valid_chars) {
        return CheckUsernameAction::InvalidUsernameChars;
    }

    user_set.insert(name.to_owned());
    CheckUsernameAction::FoundNew
}

enum CheckUsernameAction {
    FoundNew,
    InvalidUsernameChars,
    AlreadyExisted,
    InvalidUsernameLength,
}
