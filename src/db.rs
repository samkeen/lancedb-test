use crate::{create_embeddings, drop_data_dir, generate_schema, wrap_in_option};
use arrow_array::types::Float32Type;
use arrow_array::{
    ArrayRef, FixedSizeListArray, Int32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use fastembed::{Embedding, TextEmbedding};
use futures::TryStreamExt;
use lancedb::connection::Connection;
use lancedb::index::MetricType;
use lancedb::table::AddDataOptions;
use lancedb::{connect, Table, TableRef};
use std::path::Path;
use std::sync::Arc;

const DB_DIR: &str = "data";
const DB_NAME: &str = "sample-lancedb";
const TABLE_NAME: &str = "documents";
const EMBEDDING_DIMENSIONS: usize = 384;
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

    pub async fn add(&self, text: Vec<String>) {
        let embeddings = self
            .create_embeddings(&text)
            .expect("Failed to create embeddings");
        assert_eq!(
            embeddings[0].len(),
            EMBEDDING_DIMENSIONS,
            "Embedding dimensions mismatch"
        );
        let table = self
            .get_or_create_table(TABLE_NAME)
            .await
            .expect("Failed to create table");
        let schema = table.schema().await.expect("Failed to get schema");
        let records_iter = self
            .create_record_batch(embeddings, text, schema.clone())
            .into_iter()
            .map(Ok);
        let batches = RecordBatchIterator::new(records_iter, schema.clone());
        table
            .add(Box::new(batches), AddDataOptions::default())
            .await
            .expect("Failed adding data to table");
    }

    pub async fn get_all(&self) -> lancedb::Result<Vec<RecordBatch>> {
        let table = self
            .get_or_create_table(TABLE_NAME)
            .await
            .expect("Failed to create table");
        Ok(table
            .query()
            .execute_stream()
            .await?
            .try_collect::<Vec<_>>()
            .await?)
    }

    pub async fn record_count(&self) -> usize {
        let table = self
            .get_or_create_table(TABLE_NAME)
            .await
            .expect("Failed to create table");
        table.count_rows(None).await.expect("Failed to count rows")
    }

    pub async fn search(&self, search_text: &str) -> lancedb::Result<Vec<RecordBatch>> {
        let query = create_embeddings(&[search_text.to_string()])
            .expect("Failed to compute embeddings for query string");
        // flattening a 2D vector into a 1D vector. This is necessary because the search
        // function of the Table trait expects a 1D vector as input. However, the
        // create_embeddings function returns a 2D vector (a vector of embeddings,
        // where each embedding is itself a vector)
        let query: Vec<f32> = query
            .into_iter()
            .flat_map(|embedding| embedding.to_vec())
            .collect();
        let table = self
            .get_or_create_table(TABLE_NAME)
            .await
            .expect("Failed to create table");
        Ok(table
            .search(&query)
            .metric_type(MetricType::L2) // this is the default
            .limit(2)
            .execute_stream()
            .await?
            .try_collect::<Vec<_>>()
            .await?)
    }

    pub async fn delete<T: std::fmt::Display>(&self, id: T) {
        let table = self
            .get_or_create_table(TABLE_NAME)
            .await
            .expect("Failed to create table");
        table
            .delete(format!("id > {id}").as_str())
            .await
            .expect("Failed to delete table");
    }

    pub async fn update(&self, new_data: Vec<RecordBatch>) {
        todo!("Look at the docs for Table.merge_insert and implement this method.");
        // let table = self
        //     .get_or_create_table(TABLE_NAME)
        //     .await
        //     .expect("Failed to create table");
        // let mut merge_insert = table.merge_insert(&["id"]);
        // merge_insert
        //     .when_matched_update_all(None)
        //     .when_not_matched_insert_all();
        // merge_insert.execute(Box::new(new_data)).await.unwrap();
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

    async fn get_or_create_table(&self, table_name: &str) -> lancedb::Result<TableRef> {
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
                    .create_empty_table(table_name, EMBEDDING_DIMENSIONS)
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
