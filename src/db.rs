use crate::{drop_data_dir, generate_schema, wrap_in_option};
use arrow_array::types::Float32Type;
use arrow_array::{
    ArrayRef, FixedSizeListArray, Int32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use fastembed::{Embedding, EmbeddingModel, InitOptions, TextEmbedding};
use lancedb::connection::Connection;
use lancedb::table::AddDataOptions;
use lancedb::{connect, Table, TableRef};
use std::fmt::format;
use std::path::Path;
use std::sync::Arc;

const DB_DIR: &str = "data";
const DB_NAME: &str = "sample-lancedb";
const TABLE_NAME: &str = "documents";
pub struct EmbedStore {
    embedding_model: TextEmbedding,
    db_conn: Connection,
}

impl EmbedStore {
    pub async fn new(embedding_model: TextEmbedding) -> EmbedStore {
        EmbedStore {
            embedding_model,
            db_conn: Self::init_db_conn()
                .await
                .expect("Failed to initialize database"),
        }
    }

    pub async fn add(&self, text: &str) {
        let embeddings = self
            .create_embeddings(&[text.to_string()])
            .expect("Failed to create embeddings");
        let table = self
            .get_or_create_table(TABLE_NAME, embeddings[0].len())
            .await
            .expect("Failed to create table");
        let schema = table.schema().await.expect("Failed to get schema");
        let records_iter = self
            .create_record_batch(embeddings, vec![text.to_string()], schema.clone())
            .into_iter()
            .map(Ok);
        let batches = RecordBatchIterator::new(records_iter, schema.clone());
        let result = table
            .add(Box::new(batches), AddDataOptions::default())
            .await;
        !todo!("CONTINUE HERE");
    }

    pub fn get_all(&self) {
        println!("Getting embeddings");
    }

    pub fn search(&self) {
        println!("Searching embeddings");
    }

    pub fn delete(&self) {
        println!("Deleting embeddings");
    }

    pub fn update(&self) {
        println!("Updating embeddings");
    }

    async fn init_db_conn() -> lancedb::Result<Connection> {
        let db_path = format!("{}/{}", DB_DIR, DB_NAME);
        let db_conn = connect(db_path.as_str()).execute().await?;
        Ok(db_conn)
    }

    fn drop_data_dir() {
        if Path::new(DB_DIR).exists() {
            std::fs::remove_dir_all(DB_DIR).unwrap();
        }
    }

    fn create_embeddings(&self, documents: &[String]) -> anyhow::Result<Vec<Embedding>> {
        // let model = TextEmbedding::try_new(InitOptions {
        //     model_name: EmbeddingModel::AllMiniLML6V2,
        //     show_download_progress: true,
        //     ..Default::default()
        // })?;
        // Generate embeddings with the default batch size, 256
        self.embedding_model.embed(documents.to_vec(), None)
    }

    fn wrap_in_option<T>(&self, source: Vec<Vec<T>>) -> Vec<Option<Vec<Option<T>>>> {
        source
            .into_iter()
            .map(|inner_vec| Some(inner_vec.into_iter().map(|item| Some(item)).collect()))
            .collect()
    }

    /// Creates a record batch from a list of embeddings and a correlated list of original text.
    fn create_record_batch(
        &self,
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

    fn generate_schema(&self, dimensions_count: usize) -> Arc<Schema> {
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
        schema
    }

    async fn get_or_create_table(
        &self,
        table_name: &str,
        dimensions_count: usize,
    ) -> lancedb::Result<TableRef> {
        let db = self.db_conn.clone();
        let table_names = self.db_conn.table_names().await?;
        let table = table_names.iter().find(|&name| name == table_name);
        match table {
            Some(_) => {
                let table = db.open_table(table_name).execute().await?;
                Ok(table)
            }
            None => {
                let table = self
                    .create_empty_table(table_name, dimensions_count)
                    .await?;
                Ok(table)
            }
        }
    }

    /// Creates an empty table with a schema.
    async fn create_empty_table(
        &self,
        table_name: &str,
        dimensions_count: usize,
    ) -> lancedb::Result<TableRef> {
        let schema = generate_schema(dimensions_count);
        let batches = RecordBatchIterator::new(vec![], schema.clone());
        self.db_conn
            .create_table(table_name, Box::new(batches))
            .execute()
            .await
    }

    /// Creates an index on a given field.
    async fn create_index(&self, table: &dyn Table, field_name: &str) -> lancedb::Result<()> {
        table
            .create_index(&[field_name])
            .ivf_pq()
            .num_partitions(8)
            .build()
            .await
    }
}
