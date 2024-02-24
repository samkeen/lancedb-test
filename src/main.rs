use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{
    ArrayRef, FixedSizeListArray, Int32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use fastembed::{Embedding, EmbeddingModel, InitOptions, TextEmbedding};
use futures::TryStreamExt;

use vectordb::Connection;
use vectordb::{connect, Result, Table, TableRef};

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
/// Search results[count: 2]:
/// IDs> PrimitiveArray<Int32>
/// [
///   8,
///   239,
/// ]
/// Text> StringArray
/// [
///   "Call me Ishmael. Some years ago—never mind how long precisely—having",
///   "wherever you go, Ishmael, said I to myself, as I stood in the middle of",
/// ]
/// Embeddings> FixedSizeListArray<384>
/// [
///   PrimitiveArray<Float32>
/// [
///   -0.0030241862,
///   0.016034737,
///   0.052480992,
/// ```
///
///
/// Thanks to the authors of this script for giving me a starting point
/// - https://github.com/lancedb/lancedb/blob/main/rust/vectordb/examples/simple.rs
///
///

#[tokio::main]
async fn main() -> Result<()> {
    let db = init_db().await?;
    let documents = read_file_and_split_lines("tests/fixtures/mobi-dick.txt", 1000).unwrap();

    let embeddings = match create_embeddings(&documents) {
        Ok(embeddings) => {
            println!("Embeddings length: {}", embeddings.len());
            println!("Embedding dimension: {}", embeddings[0].len());
            embeddings
        }
        Err(e) => {
            panic!("Error: {:?}", e);
        }
    };

    println!("Initial table names> {:?}", db.table_names().await?);
    let tbl = create_table(db.clone(), embeddings, documents).await?;
    println!(
        "Number of records in the table> {}",
        tbl.count_rows(None).await?
    );
    create_index(tbl.as_ref(), "embeddings").await?;
    let search_result = search(
        tbl.as_ref(),
        "Call me Ishmael. Some years ago—never mind how long precisely—having",
    )
    .await?;
    for batch in &search_result {
        println!("Number of 'records'> {}", batch.num_columns());
        println!("Number of 'dimensions' in records> {}", batch.num_rows());
        let ids = batch.column_by_name("id").unwrap();
        let embeddings = batch.column_by_name("embeddings").unwrap();
        let text = batch.column_by_name("text").unwrap();
        // println!("Search results[count: {}]:", ids.len());
        // println!("IDs> {:#?}", ids);
        // println!("Text> {:#?}", text);
        // println!("Embeddings> {:?}", embeddings);

        println!("BATCH SCHEMA FIELDS: {:#?}", batch.schema().fields);
        println!("BATCH SCHEMA METADATA: {:#?}", batch.schema().metadata);

        // for column in batch.columns() {
        //     println!("COLUMN: {:#?}", column);
        // }
    }

    tbl.delete("id > 24").await.unwrap();
    db.drop_table("my_table").await.unwrap();
    Ok(())
}

/// Initializes the database.
async fn init_db() -> Result<Arc<dyn Connection>> {
    drop_data_dir();
    let db = connect("data/sample-lancedb").await?;
    Ok(db)
}

/// Drops the data directory if it exists.
fn drop_data_dir() {
    if Path::new("data").exists() {
        std::fs::remove_dir_all("data").unwrap();
    }
}

#[allow(dead_code)]
/// Opens a table with an existing table name.
async fn open_with_existing_tbl() -> Result<()> {
    let uri = "data/sample-lancedb";
    let db = connect(uri).await?;
    let _ = db
        .open_table_with_params("my_table", Default::default())
        .await
        .unwrap();
    Ok(())
}

/// Creates a table with embeddings and original text.
async fn create_table(
    db: Arc<dyn Connection>,
    embeddings: Vec<Vec<f32>>,
    text: Vec<String>,
) -> Result<TableRef> {
    assert_eq!(
        embeddings.len(),
        text.len(),
        "Embeddings and text must be the same length"
    );
    let dimensions_count = embeddings[0].len();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "embeddings",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimensions_count as i32,
            ),
            true,
        ),
        Field::new("text", DataType::Utf8, false),
    ]));

    let records_iter = create_record_batch(embeddings, text, schema.clone())
        .into_iter()
        .map(Ok);
    let batches = RecordBatchIterator::new(records_iter, schema.clone());

    let tbl = db
        .create_table("my_table", Box::new(batches), None)
        .await
        .unwrap();

    Ok(tbl)
}

/// Transforms a 2D vector into a 2D vector where each element is wrapped in an `Option`.
///
/// This function takes a 2D vector `source` as input and returns a new 2D vector where each element
/// is wrapped in an `Option`.
/// The outer vector is also wrapped in an `Option`. This is useful when you want to represent the
/// absence of data in your vector.
///
/// # Arguments
///
/// * `source` - A 2D vector that will be transformed.
///
/// # Returns
///
/// A 2D vector where each element is wrapped in an `Option`, and the outer vector is also wrapped in an `Option`.
///
/// # Example
///
/// ```
/// let source = vec![vec![1, 2, 3], vec![4, 5, 6]];
/// let result = wrap_in_option(source);
/// assert_eq!(result, vec![Some(vec![Some(1), Some(2), Some(3)]), Some(vec![Some(4), Some(5), Some(6)])]);
/// ```
fn wrap_in_option<T>(source: Vec<Vec<T>>) -> Vec<Option<Vec<Option<T>>>> {
    source
        .into_iter()
        .map(|inner_vec| Some(inner_vec.into_iter().map(|item| Some(item)).collect()))
        .collect()
}

/// Creates a record batch from a list of embeddings and a correlated list of original text.
fn create_record_batch(
    embeddings: Vec<Vec<f32>>,
    text: Vec<String>,
    schema: Arc<Schema>,
) -> Vec<RecordBatch> {
    let total_records_count = embeddings.len();
    let dimensions_count = embeddings[0].len();
    let wrapped_source = wrap_in_option(embeddings);
    vec![RecordBatch::try_new(
        schema.clone(),
        vec![
            // id field
            Arc::new(Int32Array::from_iter_values(0..total_records_count as i32)),
            // Embeddings field
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    wrapped_source,
                    dimensions_count as i32,
                ),
            ),
            // Text field
            Arc::new(Arc::new(StringArray::from(text)) as ArrayRef),
        ],
    )
    .unwrap()]
}

#[allow(dead_code)]
/// Creates an empty table with a schema.
async fn create_empty_table(db: Arc<dyn Connection>) -> Result<TableRef> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("item", DataType::Utf8, true),
    ]));
    let batches = RecordBatchIterator::new(vec![], schema.clone());
    db.create_table("empty_table", Box::new(batches), None)
        .await
}

/// Creates an index on a given field.
async fn create_index(table: &dyn Table, field_name: &str) -> Result<()> {
    table
        .create_index(&[field_name])
        .ivf_pq()
        .num_partitions(8)
        .build()
        .await
}

/// Searches for a given text in the table.
async fn search(table: &dyn Table, search_text: &str) -> Result<Vec<RecordBatch>> {
    let query = match create_embeddings(&[search_text.to_string()]) {
        Ok(embeddings) => {
            println!("Embeddings length: {}", embeddings.len());
            println!("Embedding dimension: {}", embeddings[0].len());

            embeddings
        }
        Err(e) => {
            panic!("Error: {:?}", e);
        }
    };
    // flattening a 2D vector into a 1D vector. This is necessary because the search
    // function of the Table trait expects a 1D vector as input. However, the
    // create_embeddings function returns a 2D vector (a vector of embeddings,
    // where each embedding is itself a vector)
    let query: Vec<f32> = query
        .into_iter()
        .flat_map(|embedding| embedding.to_vec())
        .collect();
    Ok(table
        .search(&query)
        .limit(2)
        .execute_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?)
}

/// Reads a file and splits its content into lines.
fn read_file_and_split_lines<P>(filename: P, limit: usize) -> io::Result<Vec<String>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    // return the first `limit` lines
    let lines = reader
        .lines()
        .take(limit)
        .collect::<std::result::Result<_, _>>()?;
    Ok(lines)
}

/// Generates embeddings for a list of documents.
fn create_embeddings(documents: &[String]) -> anyhow::Result<Vec<Embedding>> {
    let model = TextEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::AllMiniLML6V2,
        show_download_progress: true,
        ..Default::default()
    })?;
    // Generate embeddings with the default batch size, 256
    model.embed(documents.to_vec(), None)
}
