mod db;

use db::EmbedStore;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use lancedb::connect;
use lancedb::connection::Connection;
use std::path::Path;
use std::{fs, io};

/// This is a simple exampled that demonstrates how to use LanceDB to store embeddings and then
/// search against those embeddings.
///
/// Using the `fastembed` crate, we generate embeddings for a list of documents.
///
/// Using some text lines from a sample document, we generate embeddings for each line. Then store
/// the embeddings in a LanceDB table along with the original text.
///
/// Table Schema:
/// - id: an integer field
/// - embeddings: a list of floats
/// - text: a string field
///
/// Additionally, an index on the `embeddings` field and search for a given text.
///
/// With the db populated, we then run a similarity search for a given of text and print the results.
/// ```
/// Searching for string: 'Call me Ishmael. Some years ago—never mind how long precisely—having'
/// Num records embedded: 1
/// Embedding dimension: 384
/// Number of 'records'> 4
/// Search results[count: 2]:
/// IDs> PrimitiveArray<Int32>
/// [
///   4,
///   216,
/// ]
/// Text> StringArray
/// [
///   "Call me Ishmael. Some years ago—never mind how long precisely—having",
///   "wherever you go, Ishmael, said I to myself, as I stood in the middle of",
/// ]
/// Similarity distances> PrimitiveArray<Float32>
/// [
///   0.09064292,
///   0.20795542,
/// ]
/// Embeddings> FixedSizeListArray<384>
/// [
///   PrimitiveArray<Float32>
/// [
///   -0.0030241862,
///   0.016034737,
///   0.052480992,
///   0.011406774,
///   -0.06568693,
///   -0.0030454374,
///      ...truncated...
/// ```
///
///
/// Thanks to the authors of this script for giving me a starting point
/// - https://github.com/lancedb/lancedb/blob/main/rust/lancedb/examples/simple.rs
///
///

#[tokio::main]
async fn main() -> Result<(), lancedb::Error> {
    // RESET THE DB FOR TESTING
    reset_db().await.expect("Unable to reset db");

    let embedding_model = TextEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::AllMiniLML6V2,
        show_download_progress: true,
        ..Default::default()
    })
    .expect("Unable to create embedding model");
    let db = EmbedStore::new(embedding_model).await;

    let text_lines = read_file_and_split_lines("tests/fixtures/mobi-dick.txt", 1000)
        .expect("Unable to read test file");

    db.add(text_lines).await;

    db.create_index(None).await.expect("Failed to create index");

    println!(
        "Number of items in Db: {}",
        db.record_count()
            .await
            .expect("Was unable to get record count")
    );

    if let Ok(search_result) = db
        .search("Call me Ishmael. Some years ago—never mind how long precisely—having")
        .await
    {
        for record_batch in &search_result {
            println!(
                "Number of 'records' returned from search> {}",
                record_batch.num_rows()
            );
            // let ids = record_batch.column_by_name("id").unwrap();
            // let embeddings = record_batch.column_by_name("embeddings").unwrap();
            // let text = record_batch.column_by_name("text").unwrap();
            // let distances = record_batch.column_by_name("_distance").unwrap();
            // println!("Search results[count: {}]:", ids.len());
            // println!("IDs> {:#?}", ids);
            // println!("Text> {:#?}", text);
            // println!("Similarity distances> {:#?}", distances);
            // println!("Embeddings> {:?}", embeddings);
            for field in record_batch.schema().fields() {
                println!("Field: {:#?}", field.name());
            }
            println!("BATCH SCHEMA FIELDS: {:#?}", record_batch.schema().fields);
            println!(
                "BATCH SCHEMA METADATA: {:#?}",
                record_batch.schema().metadata
            );
        }
    }
    Ok(())
}

/// Initializes the database.
async fn reset_db() -> lancedb::Result<Connection> {
    if Path::new("data").exists() {
        std::fs::remove_dir_all("data").unwrap();
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
