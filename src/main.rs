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



    /// Try some embeddings
    // let documents = vec![
    //     "Hello World!",
    //     "Another sentence",
    //     "And one last sentence",
    //     "This is a beer",
    //     "the beer is good",
    //     "weekends are too short",
    //     "The cat is warm",
    //     "Blankets are the best",
    //     "Rust is tuff, but that's okay",
    //     "Learn something new everyday",
    //     "I wan to to be a builder",
    // ];
    // read in tests/fixtures/mobi-dick.txt then split all the lines of the document
    // into a Vec<&str>, remoing empty lines
    let documents = read_file_and_split_lines("tests/fixtures/mobi-dick.txt").unwrap();


    let embeddings = match create_embeddings(documents) {
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

    // Convert embeddings to the required format
    // let embeddings_iter = embeddings.iter().map(|embedding| Some(embedding.iter().map(Some).collect::<Vec<_>>()));
    // Convert embeddings to the required format
    let embeddings_iter = embeddings.iter().map(|embedding| Some(embedding.iter().map(|&value| Some(value.clone())).collect::<Vec<_>>()));








    println!("Initial table names> {:?}", db.table_names().await?);
    let tbl = create_table(db.clone()).await?;
    // print the number of records in the table
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
    if std::path::Path::new("data").exists() {
        std::fs::remove_dir_all("data").unwrap();
    }
}

#[allow(dead_code)]
async fn open_with_existing_tbl() -> Result<()> {
    let uri = "data/sample-lancedb";
    let db = connect(uri).await?;
    // --8<-- [start:open_with_existing_file]
    let _ = db
        .open_table_with_params("my_table", Default::default())
        .await
        .unwrap();
    // --8<-- [end:open_with_existing_file]
    Ok(())
}

async fn create_table(db: Arc<dyn Connection>) -> Result<TableRef> {
    // --8<-- [start:create_table]
    const TOTAL: usize = 1000;
    const DIM: usize = 128;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            true,
        ),
    ]));

    // Create a RecordBatch stream needed by `create_table`.
    // this turns a vector into an iterator & .map(Ok) turns it into an iterator of Result<RecordBatch>
    let records_iter = create_record_batch(TOTAL, DIM, schema.clone())
        .into_iter()
        .map(Ok);
    let batches = RecordBatchIterator::new(records_iter, schema.clone());

    let tbl = db
        .create_table("my_table", Box::new(batches), None)
        .await
        .unwrap();

    // this turns a vector into an iterator & .map(Ok) turns it into an iterator of Result<RecordBatch>
    let more_records_iter = create_record_batch(TOTAL, DIM, schema.clone())
        .into_iter()
        .map(Ok);
    let more_batches = RecordBatchIterator::new(more_records_iter, schema.clone());

    tbl.add(Box::new(more_batches), None).await.unwrap();

    Ok(tbl)
}

fn create_record_batch(total: usize, dim: usize, schema: Arc<Schema>) -> Vec<RecordBatch> {
    vec![RecordBatch::try_new(
        schema.clone(),
        vec![
            // this creates the id's (0 -> <TOTAL-1>)
            Arc::new(Int32Array::from_iter_values(0..total as i32)),
            // If TOTAL were 2, and DIM were 3, this creates the following:
            // [
            //     Some(
            //         [
            //             1.0,
            //             1.0,
            //             1.0,
            //         ],
            //     ),
            //     Some(
            //         [
            //             1.0,
            //             1.0,
            //             1.0,
            //         ],
            //     ),
            // ]
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    (0..total).map(|_| Some(vec![Some(1.0); dim])),
                    dim as i32,
                ),
            ),
        ],
    )
    .unwrap()]
}

async fn create_empty_table(db: Arc<dyn Connection>) -> Result<TableRef> {
    // --8<-- [start:create_empty_table]
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("item", DataType::Utf8, true),
    ]));
    let batches = RecordBatchIterator::new(vec![], schema.clone());
    db.create_table("empty_table", Box::new(batches), None)
        .await
    // --8<-- [end:create_empty_table]
}

async fn create_index(table: &dyn Table) -> Result<()> {
    // --8<-- [start:create_index]
    table
        .create_index(&["vector"])
        .ivf_pq()
        .num_partitions(8)
        .build()
        .await
    // --8<-- [end:create_index]
}

async fn search(table: &dyn Table) -> Result<Vec<RecordBatch>> {
    // --8<-- [start:search]
    println!("Searching for vectors similar to [1.0; 128]...");
    Ok(table
        .search(&[1.0; 128])
        .limit(2)
        .execute_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?)
    // --8<-- [end:search]
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
