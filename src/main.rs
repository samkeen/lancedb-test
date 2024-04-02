mod db;

use crate::db::Document;
use db::EmbedStore;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use lancedb::connect;
use lancedb::connection::Connection;
use rand::Rng;
use std::path::Path;
use std::{fs, io};

#[tokio::main]
async fn main() -> Result<(), lancedb::Error> {
    // RESET THE DB FOR TESTING
    reset_db().await.unwrap();

    let embedding_model = TextEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::AllMiniLML6V2,
        show_download_progress: true,
        ..Default::default()
    })
    .unwrap();

    let db = EmbedStore::new(embedding_model).await.unwrap();

    // Read text lines from file
    let text_lines = read_file_lines("tests/fixtures/mobi-dick.txt", 1000).unwrap();

    // Create documents
    let documents: Vec<Document> = text_lines
        .into_iter()
        .map(|line| Document::new(&line))
        .collect();

    db.add(documents.clone()).await.unwrap();

    db.create_index().await.unwrap();

    let id = documents
        .first()
        .expect("The documents vector is empty")
        .id
        .clone();

    // FOUND example //////////
    let found_record = db.get(&id).await.unwrap();
    match found_record {
        None => {
            println!("The record with id: {id} was not found")
        }
        Some(record) => {
            println!("Found record[{}]: '{:?}'", id, record.id)
        }
    }

    // UPDATE example //////////
    let mut record_to_update = db.get(&id).await.unwrap();
    match record_to_update {
        None => {
            println!("The record with id: {id} was not found")
        }
        Some(orig_record) => {
            println!("Updating record[{}]: '{:?}'", id, orig_record.id);
            let updated_document = Document {
                id: orig_record.id,
                text: "New text".to_string(),
                created: orig_record.created,
                modified: orig_record.modified,
            };
            db.update(vec![updated_document]).await.unwrap();
            let updated_record = db.get(&id).await.unwrap();
            match updated_record {
                None => {}
                Some(updated_record) => {
                    assert_eq!("New text", updated_record.text);
                    assert_eq!(orig_record.created, updated_record.created);
                    assert!(
                        orig_record.modified < updated_record.modified,
                        "orig[{}], updated_record[{}]",
                        orig_record.modified,
                        updated_record.modified
                    )
                }
            }
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

    let record_count = db.record_count().await.unwrap();
    println!("Number of items in Db: {record_count}");

    // Search example /////////
    println!("Starting search...");
    let search_result = db
        .search(
            "Call me Ishmael. Some years ago—never mind how long precisely—having",
            None,
            Some(3),
        )
        .await
        .unwrap();
    search_result.iter().for_each(|result| {
        println!(
            "results: Document[{}][{}] <distance:{}>: '{}'",
            result.0.id, result.0.created, result.1, result.0.text
        );
    });

    Ok(())
}

/// Initializes the database.
async fn reset_db() -> lancedb::Result<Connection> {
    if Path::new("data").exists() {
        fs::remove_dir_all("data").unwrap();
    }
    let db = connect("data/sample-lancedb").execute().await?;
    Ok(db)
}

fn read_file_lines(path: &str, limit: usize) -> std::io::Result<Vec<String>> {
    let content = fs::read_to_string(path)?;
    let lines: Vec<String> = content
        .lines()
        .filter(|line| !line.is_empty())
        .take(limit)
        .map(|line| line.to_string())
        .collect();
    Ok(lines)
}
