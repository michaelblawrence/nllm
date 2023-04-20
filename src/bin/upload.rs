use std::{env, fs::File, io::BufReader, path::Path};

use anyhow::Result;

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
    let path = fpath_arg.cloned().unwrap();
    let path: &Path = Path::new(&path);
    let path = if path.is_dir() {
        info!("Attempting to find json in dir...");
        first_json_dir_entry(path)
    } else {
        path.to_path_buf()
    };

    info!("Opening model from disk: {}", path.to_str().unwrap());
    let file = File::open(path).unwrap();
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

fn first_json_dir_entry(path: &Path) -> std::path::PathBuf {
    path.read_dir()
        .unwrap()
        .find(|file| {
            let dir_entry = &file.as_ref().unwrap();
            let file_name = &dir_entry.file_name();
            let file_name = file_name.to_str().unwrap();
            let file_type = dir_entry.file_type().unwrap();
            file_type.is_file() && file_name.ends_with(".json")
        })
        .expect("could not match json files in dir")
        .unwrap()
        .path()
}

fn configure_logging() {
    let subscriber = tracing_subscriber::fmt()
        .compact()
        .with_target(false)
        .with_max_level(LevelFilter::INFO)
        .finish();

    tracing::subscriber::set_global_default(subscriber).unwrap();
}
