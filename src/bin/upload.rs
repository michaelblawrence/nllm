use std::{
    collections::HashSet,
    env,
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
};

use anyhow::Result;
use itertools::Itertools;
use mongodb::sync::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{info, metadata::LevelFilter};

#[derive(Serialize, Deserialize)]
struct ExportedModel {
    model: Value,
}

fn main() -> Result<()> {
    configure_logging();
    let args: Vec<String> = env::args().skip(1).collect();
    let fpath_arg = args.first();
    let file = File::open(fpath_arg.as_ref().unwrap()).unwrap();
    let reader = BufReader::new(file);
    let model = serde_json::from_reader(reader)?;

    let host = env::var("MONGODB_HOST").unwrap_or_else(|_| "mongodb://localhost:27017".to_string());

    let client = Client::with_uri_str(host)?;
    let database = client.database("mydb");
    let collection = database.collection::<ExportedModel>("models");

    // Insert some books into the "mydb.models" collection.
    collection.insert_one(ExportedModel { model }, None)?;

    info!("Uploaded model to db");
    Ok(())
}

fn configure_logging() {
    let subscriber = tracing_subscriber::fmt()
        .compact()
        .with_target(false)
        .with_max_level(LevelFilter::INFO)
        .finish();

    tracing::subscriber::set_global_default(subscriber).unwrap();
}
