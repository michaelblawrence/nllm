use std;
use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;

use ::axum;
use ::tokio::sync::broadcast::{self};
use ::tracing::Level;
use ::tracing_subscriber::prelude::*;
use ::tracing_subscriber::{self, Registry};
use anyhow::{bail, Context};

use nllm_embed_ctrl::{app, env, ApiAppState, AppState};

#[tokio::main]
async fn main() {
    configure_current_dir().unwrap();
    configure_logging();

    let app_state = configure_app_state();
    let app = app(app_state).await;

    let addr = socket_addr_from_env(env::CTRL_PORT, 3001);
    tracing::info!("listening on {}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

fn socket_addr_from_env(port_env_key: &str, default: u16) -> SocketAddr {
    let port = std::env::var(port_env_key).ok();
    let port = port.and_then(|x| x.parse().ok()).unwrap_or(default);
    SocketAddr::from(([0, 0, 0, 0], port))
}

pub fn configure_logging() {
    let stdout_log = tracing_subscriber::fmt::layer()
        .compact()
        .with_target(false)
        .with_timer(tracing_subscriber::fmt::time::time())
        .map_writer(|x| x.with_max_level(Level::INFO));
    let subscriber = Registry::default()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "with_axum_htmx_askama=debug".into()),
        )
        .with(stdout_log);
    tracing::subscriber::set_global_default(subscriber).unwrap();
}

pub fn configure_app_state() -> Arc<AppState> {
    let (tx, rx) = broadcast::channel(100);
    std::mem::drop(rx);

    let mut user_set = HashSet::new();
    user_set.insert("Chat".to_string());

    Arc::new(AppState {
        tx,
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
    let index_html_path = std::path::Path::new("./admin.html");
    if !index_html_path.exists() {
        let current_exe = std::env::current_exe().context("failed to get cwd")?;
        let mut dir = Some(current_exe);
        while let Some(current) = dir {
            let index_test_path = current.join("nllm_embed_ctrl/admin.html");
            if index_test_path.exists() {
                std::env::set_current_dir(current.join("nllm_embed_ctrl"))
                    .context("failed to set current dir to content root")?;

                return Ok(());
            }
            dir = current.parent().map(|x| x.to_path_buf());
        }
        bail!("failed to find admin.html content root");
    }
    Ok(())
}
