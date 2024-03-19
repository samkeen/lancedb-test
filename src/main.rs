mod db;

use db::EmbedStore;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use lancedb::connect;
use lancedb::connection::Connection;
use rand::distributions::Alphanumeric;
use rand::Rng;
use std::path::Path;
use std::{fs, io};

#[tokio::main]
async fn main() -> Result<(), lancedb::Error> {
    // RESET THE DB FOR TESTING
    reset_db().await.expect("Unable to reset db");

    let embedding_model = TextEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::AllMiniLML6V2,
        show_download_progress: true,
        ..Default::default()
    })
    .unwrap();

    let db = EmbedStore::new(embedding_model).await.unwrap();

    let text_lines = read_file_and_split_lines("tests/fixtures/mobi-dick.txt", 1000).unwrap();
    let alt_ids: Vec<String> = (1..=1000).map(|_| generate_id()).collect();
    let id = alt_ids[42].clone();
    db.add(alt_ids, text_lines).await.unwrap();

    db.create_index().await.unwrap();

    // FOUND example //////////
    let record = db.get(&id).await.unwrap();
    match record {
        None => {
            println!("The record with id: {id} was not found")
        }
        Some(record) => {
            println!("Found record[{}]: '{:?}'", id, record.id)
        }
    }
    // NOT FOUND example ////////
    let id = "abc";
    let record = db.get(id).await.unwrap();
    match record {
        None => {
            println!("The record with id: {id} was not found")
        }
        Some(record) => {
            println!("Found record[{}]: '{:?}'", id, record.id)
        }
    }

    // update record 42
    // db.update()

    let record_count = db.record_count().await.unwrap();
    println!("Number of items in Db: {record_count}");

    // Search example /////////
    println!("Starting search...");
    let search_result = db
        .search(
            "Call me Ishmael. Some years ago—never mind how long precisely—having",
            Some(3),
        )
        .await
        .unwrap();
    search_result.iter().for_each(|result| {
        println!(
            "results: Document[{}] <distance:{}>: '{}'",
            result.0.id, result.1, result.0.text
        );
    });

    Ok(())
}

fn generate_id() -> String {
    let mut rng = rand::thread_rng();
    let id: String = std::iter::repeat(())
        .map(|()| rng.sample(Alphanumeric))
        .map(char::from)
        .take(6)
        .collect();
    id
}

/// Initializes the database.
async fn reset_db() -> lancedb::Result<Connection> {
    if Path::new("data").exists() {
        fs::remove_dir_all("data").unwrap();
    }
    let db = connect("data/sample-lancedb").execute().await?;
    Ok(db)
}

/// Reads a file and splits its content into lines.
fn read_file_and_split_lines<P: AsRef<Path>>(path: P, limit: usize) -> io::Result<Vec<String>> {
    let content = fs::read_to_string(path)?;
    let lines: Vec<String> = content
        .lines()
        .filter(|line| !line.is_empty())
        .take(limit)
        .map(|line| line.to_string())
        .collect();
    Ok(lines)
}
