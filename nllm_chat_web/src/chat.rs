use axum::extract::ws::{Message, WebSocket};
use futures::{
    sink::SinkExt,
    stream::{SplitStream, StreamExt},
};
use tokio::sync::broadcast;

use std::{net::IpAddr, sync::Arc};

use crate::{inference::ModelPromptRequest, AppState};

#[derive(Debug, Clone)]
enum ClientPayload {
    Message(String),
    PingNotification,
}

// This function deals with a single websocket connection, i.e., a single
// connected client / user, for which we will spawn two independent tasks (for
// receiving / sending chat messages).
pub async fn chat_websocket(stream: WebSocket, ip: IpAddr, state: Arc<AppState>) {
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
    tracing::info!("{} IP = {}", joined_msg, ip);
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
    let name = name.into_boxed_str();
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
                let prompt_request = ModelPromptRequest {
                    prompt: query.to_string(),
                    use_beta_function: beta_mode,
                    username: name.to_string(),
                    index: next_prompt_index(),
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

fn next_prompt_index() -> usize {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static mut COUNTER: AtomicUsize = AtomicUsize::new(0);
    unsafe { COUNTER.fetch_add(1, Ordering::Relaxed) }
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
