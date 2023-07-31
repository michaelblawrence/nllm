use axum::{
    body::Bytes,
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::get,
    Router,
};
use futures::{sink::SinkExt, stream::StreamExt};

use std::{
    collections::HashSet,
    net::SocketAddr,
    sync::{Arc, Mutex},
};
use tokio::sync::broadcast;
use tower_http::services::ServeDir;
use tracing::metadata::LevelFilter;

mod inference;
mod model;
struct AppState {
    // We require unique usernames. This tracks which usernames have been taken.
    user_set: Mutex<HashSet<String>>,
    // Channel used to send messages to all connected clients.
    tx: broadcast::Sender<String>,
    model_tx: broadcast::Sender<String>,
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

#[tokio::main]
async fn main() {
    configure_logging();

    // Set up application state for use with with_state().
    let mut user_set = HashSet::new();
    user_set.insert("Chat".to_string());
    let user_set = Mutex::new(user_set);
    let (tx, _rx) = broadcast::channel(100);
    let model_tx = match (
        std::env::var("NLLM_MODEL_PATH").ok(),
        std::env::var("NLLM_INFER_LAMBDA_ARN").ok(),
    ) {
        (None, Some(lambda_arn)) => {
            inference::spawn_lambda_inference_client(&lambda_arn, tx.clone()).await
        }
        (model_fpath, None) => inference::spawn_local_inference(model_fpath.as_deref(), tx.clone()),
        (_, _) => panic!(
            "Either the NLLM_MODEL_PATH or NLLM_INFER_LAMBDA_ARN environment variable must be set"
        ),
    };

    let app_state = Arc::new(AppState {
        user_set,
        tx,
        model_tx,
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/diagnostics/ws.js", get(diagnostics_ws_js))
        .route("/websocket", get(websocket_handler))
        .route("/keepalive", get(keepalive_websocket_handler))
        .nest_service("/scripts", ServeDir::new("public/scripts"))
        .with_state(app_state);

    let port = std::env::var("NLLM_API_PORT")
        .ok()
        .and_then(|port| port.parse().ok())
        .unwrap_or(3000_u16);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::debug!("listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
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

// This function deals with a single websocket connection, i.e., a single
// connected client / user, for which we will spawn two independent tasks (for
// receiving / sending chat messages).
async fn websocket(stream: WebSocket, state: Arc<AppState>) {
    // By splitting, we can send and receive at the same time.
    let (mut sender, mut receiver) = stream.split();

    // Username gets set in the receive loop, if it's valid.
    let mut username = String::new();
    // Loop until a text message is found.
    while let Some(Ok(message)) = receiver.next().await {
        if let Message::Text(name) = message {
            // If username that is sent by client is not taken, fill username string.
            let username_check = match check_username(&state, &name) {
                CheckUsernameAction::FoundNew => {
                    username.push_str(&name);
                    Ok(())
                }
                CheckUsernameAction::InvalidUsernameChars => {
                    Err("Username contains invalid characters.")
                }
                CheckUsernameAction::InvalidUsernameLength => Err("Username is too long or short."),
                CheckUsernameAction::AlreadyExisted => Err("Username already taken."),
            };

            // If not empty we want to quit the loop else we want to quit function.
            match username_check {
                Ok(()) => break,
                Err(msg) => {
                    // Only send our client that username is taken.
                    let _ = sender.send(Message::Text(String::from(msg))).await;

                    return;
                }
            }
        }
    }

    // We subscribe *before* sending the "joined" message, so that we will also
    // display it to our client.
    let mut rx = state.tx.subscribe();

    // Now send the "joined" message to all subscribers.
    let msg = format!("{} joined.", username);
    tracing::info!("{}", msg);
    let _ = state.tx.send(msg);
    let _ = sender.send(Message::Text(
        format!("Chat: Hi {username}, if you ever need my help, just call me by starting your message with 'hey ai'")
    )).await;

    // Spawn the first task that will receive broadcast messages and send text
    // messages over the websocket to our client.
    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            // In any websocket error, break loop.
            if sender.send(Message::Text(msg)).await.is_err() {
                break;
            }
        }
    });

    // Clone things we want to pass (move) to the receiving task.
    let tx = state.tx.clone();
    let model_tx = state.model_tx.clone();
    let name = username.clone();

    // Spawn a task that takes messages from the websocket, prepends the user
    // name, and sends them to all broadcast subscribers.
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(Message::Text(text))) = receiver.next().await {
            let chatbot_prefixes = ["hey ai", "hey chat", "hi ai", "hi chat"];
            for prefix in chatbot_prefixes {
                if text.to_lowercase().starts_with(prefix) {
                    let query = text[prefix.len()..]
                        .trim_start_matches(|c: char| !c.is_alphabetic());
                    
                    let _ = model_tx.send(query.to_string());
                    break;
                }
            }
            // Add username before message.
            let _ = tx.send(format!("{}: {}", name, text));
        }
    });

    // If any one of the tasks run to completion, we abort the other.
    tokio::select! {
        _ = (&mut send_task) => recv_task.abort(),
        _ = (&mut recv_task) => send_task.abort(),
    };

    // Send "user left" message (similar to "joined" above).
    let msg = format!("{} left.", username);
    tracing::info!("{}", msg);
    let _ = state.tx.send(msg);

    // Remove username from map so new clients can take it again.
    state.user_set.lock().unwrap().remove(&username);
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

// Include utf-8 file at **compile** time.
// async fn index() -> (StatusCode, Html<&'static str>) {
async fn index() -> (StatusCode, Html<String>) {
    // (StatusCode::OK, Html(include_str!("../index.html")))
    match std::fs::read_to_string("index.html") {
        Ok(html) => (StatusCode::OK, Html(html)),
        Err(err) => (StatusCode::NOT_FOUND, Html(err.to_string())),
    }
}

// Include utf-8 file at **compile** time.
async fn diagnostics_ws_js() -> Response {
    let body: &'static str = if let Some(_) = option_env!("NLLM_HOTRELOAD") {
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
