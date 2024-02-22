use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, Int32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use fastembed::{Embedding, EmbeddingModel, InitOptions, TextEmbedding};
use futures::TryStreamExt;

use vectordb::Connection;
use vectordb::{connect, Result, Table, TableRef};

#[tokio::main]
async fn main() -> Result<()> {
    let db = init_db().await?;

    // into a Vec<&str>, remoing empty lines
    let documents = read_file_and_split_lines("tests/fixtures/mobi-dick.txt").unwrap();

    let embeddings = match create_embeddings(documents[0..1000].to_vec()) {
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

    println!("Initial table names> {:?}", db.table_names().await?);
    let _records = vec![vec![1.0f32; 128]; 1000];
    let tbl = create_table(db.clone(), embeddings).await?;
    println!(
        "Number of records in the table> {}",
        tbl.count_rows(None).await?
    );
    create_index(tbl.as_ref()).await?;
    let batches = search(tbl.as_ref()).await?;
    println!("Batches> {:?}", batches);

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

async fn create_table(db: Arc<dyn Connection>, records: Vec<Vec<f32>>) -> Result<TableRef> {
    let dimensions_count = records[0].len();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimensions_count as i32,
            ),
            true,
        ),
    ]));

    let records_iter = create_record_batch(records, schema.clone())
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

fn create_record_batch(source: Vec<Vec<f32>>, schema: Arc<Schema>) -> Vec<RecordBatch> {
    let total_records_count = source.len();
    let dimensions_count = source[0].len();
    let wrapped_source = wrap_in_option(source);
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
        .create_index(&["vector"])
        .ivf_pq()
        .num_partitions(8)
        .build()
        .await
}

// @TODO getting this error: Error: Store { message: "No vector column found to create index" }
// @TODO fix this: instead of [1.0; 128], use the embedding of one of the lines from the document
async fn search(table: &dyn Table) -> Result<Vec<RecordBatch>> {
    println!("Searching for vectors similar to [1.0; 128]...");
    Ok(table
        .search(&[1.0; 128])
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
