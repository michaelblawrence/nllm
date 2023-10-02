use tracing::Level;
use tracing_subscriber::{prelude::*, Registry};

pub async fn configure_logging() {
    let stdout_log = tracing_subscriber::fmt::layer()
        .compact()
        .with_target(false)
        .with_timer(tracing_subscriber::fmt::time::time())
        .map_writer(|x| x.with_max_level(Level::INFO));
    let subscriber = Registry::default().with(stdout_log);

    let subscriber: Box<dyn tracing::Subscriber + Send + Sync + 'static> = {
        #[cfg(feature = "cloudwatch")]
        {
            use tracing_subscriber::filter::LevelFilter;
            let sdk_config = aws_config::load_from_env().await;
            let client = aws_sdk_cloudwatchlogs::Client::new(&sdk_config);
            let log_group_name = "nllm-chat-web";
            match ensure_log_stream(&client, log_group_name).await {
                Ok(log_stream_name) => Box::new(
                    subscriber.with(
                        tracing_cloudwatch::layer()
                            .with_client(
                                client,
                                tracing_cloudwatch::ExportConfig::default()
                                    .with_batch_size(5)
                                    .with_interval(std::time::Duration::from_secs(1))
                                    .with_log_group_name(log_group_name)
                                    .with_log_stream_name(log_stream_name),
                            )
                            .with_code_location(true)
                            .with_target(false)
                            .with_filter(LevelFilter::from_level(Level::INFO)),
                    ),
                ),
                Err(e) => {
                    eprintln!("Failed to init cloudwatch log stream: {e}");
                    Box::new(subscriber)
                }
            }
        }
        #[cfg(not(feature = "cloudwatch"))]
        {
            Box::new(subscriber)
        }
    };

    tracing::subscriber::set_global_default(subscriber).unwrap();
}

#[cfg(feature = "cloudwatch")]
async fn ensure_log_stream(
    client: &aws_sdk_cloudwatchlogs::Client,
    log_group_name: &str,
) -> anyhow::Result<String> {
    use aws_sdk_cloudwatchlogs::error::SdkError;

    let ec2_instance_id = get_ec2_instance_id();
    let instance_id = ec2_instance_id.as_deref().unwrap_or("unknown");
    let log_stream_name = format!("tracing-stream--{instance_id}");
    println!("Creating log stream '{log_stream_name}'...");

    match client
        .create_log_stream()
        .log_group_name(log_group_name)
        .log_stream_name(&log_stream_name)
        .send()
        .await
    {
        Ok(_) => {
            println!("Created log stream '{log_stream_name}'");
            ()
        }
        Err(SdkError::ServiceError(e)) if e.err().is_resource_already_exists_exception() => (),
        Err(e) => Err(e.into_service_error())
            .context(format!("Error creating log stream '{log_stream_name}'"))?,
    }
    Ok(log_stream_name)
}

#[cfg(feature = "cloudwatch")]
fn get_ec2_instance_id() -> Option<String> {
    std::process::Command::new("ec2-metadata")
        .args(["-i"])
        .output()
        .ok()
        .and_then(|x| {
            use std::io::Read;
            let mut s = String::new();
            x.stdout.as_slice().read_to_string(&mut s).ok()?;
            s.split_once("id: ").map(|x| x.1.to_owned())
        })
}
