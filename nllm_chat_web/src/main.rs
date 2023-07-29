use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::get,
    Router,
};
use futures::{sink::SinkExt, stream::StreamExt};
use plane::ml::RngStrategy;
use std::{
    collections::HashSet,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::sync::broadcast;
use tracing::{info, metadata::LevelFilter};

mod model;

// Our shared state
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
        .with_max_level(LevelFilter::INFO)
        // .with_span_events(FmtSpan::CLOSE)
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
    let (model_tx, _model_rx) = broadcast::channel::<String>(100);
    let model_fpath = std::env::var("NLLM_MODEL_PATH").ok();
    let (model, state) = respond::load(model_fpath.as_deref()).unwrap();

    tokio::spawn({
        let tx = tx.clone();
        let model_tx = model_tx.clone();
        let rng = RngStrategy::default();
        let ctx = Arc::new((model, state, rng, tx.clone()));
        async move {
            let mut model_rx = model_tx.subscribe();
            while let Ok(prompt) = model_rx.recv().await {
                let ctx = ctx.clone();

                tokio::task::spawn_blocking(move || {
                    let (model, state, rng, tx) = &*ctx;
                    run_inference(&prompt, &model, &state, &rng, &tx)
                })
                .await
                .unwrap();
            }
        }
    });

    let app_state = Arc::new(AppState {
        user_set,
        tx,
        model_tx,
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/websocket", get(websocket_handler))
        .route("/keepalive", get(keepalive_websocket_handler))
        .with_state(app_state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    tracing::debug!("listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

fn run_inference(
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
            match check_username(&state, &name) {
                CheckUsernameAction::FoundNew => username.push_str(&name),
                CheckUsernameAction::AlreadyExisted => (),
            }

            // If not empty we want to quit the loop else we want to quit function.
            if !username.is_empty() {
                break;
            } else {
                // Only send our client that username is taken.
                let _ = sender
                    .send(Message::Text(String::from("Username already taken.")))
                    .await;

                return;
            }
        }
    }

    // We subscribe *before* sending the "joined" message, so that we will also
    // display it to our client.
    let mut rx = state.tx.subscribe();

    // Now send the "joined" message to all subscribers.
    let msg = format!("{} joined.", username);
    tracing::debug!("{}", msg);
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
            let chatbot_prefix = "hey ai";
            if text.to_lowercase().starts_with(chatbot_prefix) {
                let query =
                    text[chatbot_prefix.len()..].trim_start_matches(|c: char| !c.is_alphabetic());
                let _ = model_tx.send(query.to_string());
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
    tracing::debug!("{}", msg);
    let _ = state.tx.send(msg);

    // Remove username from map so new clients can take it again.
    state.user_set.lock().unwrap().remove(&username);
}

fn check_username(state: &AppState, name: &str) -> CheckUsernameAction {
    let mut user_set = state.user_set.lock().unwrap();

    if !user_set.contains(name) {
        user_set.insert(name.to_owned());
        CheckUsernameAction::FoundNew
    } else {
        CheckUsernameAction::AlreadyExisted
    }
}

enum CheckUsernameAction {
    FoundNew,
    AlreadyExisted,
}

// Include utf-8 file at **compile** time.
async fn index() -> (StatusCode, Html<String>) {
    match std::fs::read_to_string("index.html") {
        Ok(html) => (StatusCode::OK, Html(html)),
        Err(err) => (StatusCode::NOT_FOUND, Html(err.to_string())),
    }
}
