# Original file: simple.rs

source: [simple.rs](https://github.com/lancedb/lancedb/blob/main/rust/vectordb/examples/simple.rs)

```rust
// Copyright 2024 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, Int32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;

use vectordb::connection::Connection;
use vectordb::table::AddDataOptions;
use vectordb::{connect, Result, Table, TableRef};

#[tokio::main]
async fn main() -> Result<()> {
    if std::path::Path::new("data").exists() {
        std::fs::remove_dir_all("data").unwrap();
    }
    // --8<-- [start:connect]
    let uri = "data/sample-lancedb";
    let db = connect(uri).execute().await?;
    // --8<-- [end:connect]

    // --8<-- [start:list_names]
    println!("{:?}", db.table_names().await?);
    // --8<-- [end:list_names]
    let tbl = create_table(&db).await?;
    create_index(tbl.as_ref()).await?;
    let batches = search(tbl.as_ref()).await?;
    println!("{:?}", batches);

    create_empty_table(&db).await.unwrap();

    // --8<-- [start:delete]
    tbl.delete("id > 24").await.unwrap();
    // --8<-- [end:delete]

    // --8<-- [start:drop_table]
    db.drop_table("my_table").await.unwrap();
    // --8<-- [end:drop_table]
    Ok(())
}

#[allow(dead_code)]
async fn open_with_existing_tbl() -> Result<()> {
    let uri = "data/sample-lancedb";
    let db = connect(uri).execute().await?;
    // --8<-- [start:open_with_existing_file]
    let _ = db.open_table("my_table").execute().await.unwrap();
    // --8<-- [end:open_with_existing_file]
    Ok(())
}

async fn create_table(db: &Connection) -> Result<TableRef> {
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

    // Create a RecordBatch stream.
    let batches = RecordBatchIterator::new(
        vec![RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..TOTAL as i32)),
                Arc::new(
                    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                        (0..TOTAL).map(|_| Some(vec![Some(1.0); DIM])),
                        DIM as i32,
                    ),
                ),
            ],
        )
        .unwrap()]
        .into_iter()
        .map(Ok),
        schema.clone(),
    );
    let tbl = db
        .create_table("my_table", Box::new(batches))
        .execute()
        .await
        .unwrap();
    // --8<-- [end:create_table]

    let new_batches = RecordBatchIterator::new(
        vec![RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..TOTAL as i32)),
                Arc::new(
                    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                        (0..TOTAL).map(|_| Some(vec![Some(1.0); DIM])),
                        DIM as i32,
                    ),
                ),
            ],
        )
        .unwrap()]
        .into_iter()
        .map(Ok),
        schema.clone(),
    );
    // --8<-- [start:add]
    tbl.add(Box::new(new_batches), AddDataOptions::default())
        .await
        .unwrap();
    // --8<-- [end:add]

    Ok(tbl)
}

async fn create_empty_table(db: &Connection) -> Result<TableRef> {
    // --8<-- [start:create_empty_table]
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("item", DataType::Utf8, true),
    ]));
    db.create_empty_table("empty_table", schema).execute().await
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
    Ok(table
        .search(&[1.0; 128])
        .limit(2)
        .execute_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?)
    // --8<-- [end:search]
}
```