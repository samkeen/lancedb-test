# LanceDB test

Working with lanceDB and the vectordb crate to investigate storing embeddings on LanceDB

## Help for those new to LanceDB

(such as myself)

### Table search (query-building)

**Search**

`table.search()` is a shortcut to the
Table [QueryBuilder](https://docs.rs/vectordb/0.4.10/vectordb/table/trait.Table.html#tymethod.query).

```rust
// file: table.rs

fn search(&self, query: &[f32]) -> Query {
    self.query().nearest_to(query)
}
```

**Query building**

Options for a [Query](https://docs.rs/vectordb/0.4.10/vectordb/query/struct.Query.html)

```text
column: None,
limit: None,
nprobes: 20,
refine_factor: None,
metric_type: None,
use_index: true,
filter: None,
select: None,
prefilter: false,
```

I've also found at this point in time (Feb/2024),
the more mature [Python API docs](https://lancedb.github.io/lancedb/python/python/#lancedb.table.Table.compact_files)
to have slightly more helpful explanations to ML novices such as I.

- `nearest_to` - Find the nearest vectors to the given query vector. You are typically always going to use this, hence
  the shortcut `table.search()` method.
- `limit` - Set the maximum number of results to return.
- `columns` - which columns to return in the results
- `nprobes` - Set the number of probes to use*.
- `refine_factor` - Set the refine factor to use*.
- `metric_type` - Set the metric type to use. Can support `MetricType::L2` (default), `MetricType::Cosine`
  and `MetricType::Dot`.
- `use_index` - Whether to use an ANN index if available.
- `filter` - sql filter to refine the query with, optional.
- `select` - Return only the specified columns. Only select the specified columns. If not specified, all columns will be
  returned.
- `prefilter` - if True then apply the filter before vector search

*_For explanations of `nprobes` and `refine_factor`, see
the [Querying an ANN Index](https://lancedb.github.io/lancedb/ann_indexes/#querying-an-ann-index)_

Examples from the [Docs]()

```rust
// Run a vector search (ANN) query.

let stream = tbl.query().nearest_to(& [1.0, 2.0, 3.0])
.refine_factor(5)
.nprobes(10)
.execute_stream()
.await
.unwrap();
let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

// Run a SQL-style filter
let stream = tbl
.query()
.filter("id > 5")
.limit(1000)
.execute_stream()
.await
.unwrap();
let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

// Run a full scan query.
let stream = tbl
.query()
.execute_stream()
.await
.unwrap();
let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

// for any of these examples you could use the search() method on the table
let stream = tbl.search(& [1.0, 2.0, 3.0])
.refine_factor(5)
.nprobes(10)
.execute_stream()
.await
.unwrap();
let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
```