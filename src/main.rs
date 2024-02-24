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

#[tokio::main]
async fn main() -> Result<()> {
    let db = init_db().await?;
    let documents = read_file_and_split_lines("tests/fixtures/mobi-dick.txt").unwrap();

    let embeddings = match create_embeddings(documents[0..1000].to_vec()) {
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
    let tbl = create_table(db.clone(), embeddings, documents[0..1000].to_vec()).await?;
    println!(
        "Number of records in the table> {}",
        tbl.count_rows(None).await?
    );
    create_index(tbl.as_ref()).await?;
    let search_result = search(tbl.as_ref()).await?;
    // @TODO for each result, lookup the text for the returned embeddings
    for batch in &search_result {
        println!("Number of 'records'> {}", batch.num_columns());
        println!("Number of 'dimensions' in records> {}", batch.num_rows());
        for column in batch.columns() {
            println!("Record> {:?}", column);
        }
    }
    // println!("Batches> {:?}", search_result);

    create_empty_table(db.clone()).await.unwrap();
    tbl.delete("id > 24").await.unwrap();
    db.drop_table("my_table").await.unwrap();
    Ok(())
}

async fn init_db() -> Result<Arc<dyn Connection>> {
    drop_data_dir();
    let uri = "data/sample-lancedb";
    let db = connect(uri).await?;
    Ok(db)
}

fn drop_data_dir() {
    if Path::new("data").exists() {
        std::fs::remove_dir_all("data").unwrap();
    }
}

#[allow(dead_code)]
async fn open_with_existing_tbl() -> Result<()> {
    let uri = "data/sample-lancedb";
    let db = connect(uri).await?;
    let _ = db
        .open_table_with_params("my_table", Default::default())
        .await
        .unwrap();
    Ok(())
}

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

    // let more_records = vec![vec![1.0; dimensions_count]; 1000];
    // let more_records_iter = create_record_batch(more_records, schema.clone())
    //     .into_iter()
    //     .map(Ok);
    // let more_batches = RecordBatchIterator::new(more_records_iter, schema.clone());
    //
    // tbl.add(Box::new(more_batches), None).await.unwrap();

    Ok(tbl)
}

///
/// If TOTAL were 2, and DIM were 3, this creates the following:
// [
//     Some(
//         [
//             Some(1.0),
//             Some(1.0),
//             Some(1.0),
//         ],
//     ),
//     Some(
//         [
//             Some(1.0),
//             Some(1.0),
//             Some(1.0),
//         ],
//     ),
// ]
fn wrap_in_option<T>(source: Vec<Vec<T>>) -> Vec<Option<Vec<Option<T>>>> {
    source
        .into_iter()
        .map(|inner_vec| Some(inner_vec.into_iter().map(|item| Some(item)).collect()))
        .collect()
}

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
            // This is the id field
            Arc::new(Int32Array::from_iter_values(0..total_records_count as i32)),
            // This is the vector field
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    wrapped_source,
                    dimensions_count as i32,
                ),
            ),
            // This is the text field
            Arc::new(Arc::new(StringArray::from(text)) as ArrayRef),
        ],
    )
        .unwrap()]
}

async fn create_empty_table(db: Arc<dyn Connection>) -> Result<TableRef> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("item", DataType::Utf8, true),
    ]));
    let batches = RecordBatchIterator::new(vec![], schema.clone());
    db.create_table("empty_table", Box::new(batches), None)
        .await
}

async fn create_index(table: &dyn Table) -> Result<()> {
    table
        .create_index(&["embeddings"])
        .ivf_pq()
        .num_partitions(8)
        .build()
        .await
}

async fn search(table: &dyn Table) -> Result<Vec<RecordBatch>> {
    let query = match create_embeddings(vec![
        "Call me Ishmael. Some years ago—never mind how long precisely—having".to_string(),
    ]) {
        Ok(embeddings) => {
            println!("Embeddings length: {}", embeddings.len()); // -> Embeddings length: 4
            println!("Embedding dimension: {}", embeddings[0].len()); // -> Embedding dimension: 384
            // println!("Embedding: {:?}", embeddings[0]); // -> Embedding: [0.1, 0.2, 0.3, ...]
            embeddings
        }
        Err(e) => {
            panic!("Error: {:?}", e);
        }
    };
    // @TODO: I bet I can do something better than this
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

fn read_file_and_split_lines<P>(filename: P) -> io::Result<Vec<String>>
    where
        P: AsRef<Path>,
{
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let lines = reader.lines().collect::<std::result::Result<_, _>>()?;
    Ok(lines)
}

fn create_embeddings(documents: Vec<String>) -> anyhow::Result<Vec<Embedding>> {
    let model = TextEmbedding::try_new(InitOptions {
        // https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        model_name: EmbeddingModel::AllMiniLML6V2,
        show_download_progress: true,
        ..Default::default()
    });

    // Generate embeddings with the default batch size, 256
    let embeddings = model.expect("").embed(documents, None);
    embeddings
}
