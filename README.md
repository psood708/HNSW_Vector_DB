# HNSW Vector DB

> A from-scratch, production-oriented vector database built in Rust — featuring a hand-rolled Hierarchical Navigable Small World (HNSW) index, a memory-mapped persistence layer, parallel search via Rayon, and a REST API served over Axum.

**98% recall @ 1M vectors** · Built without external ANN libraries · Rust + Python

---

## Why This Exists

Most vector database benchmarks treat HNSW as a black box. This project implements the algorithm from first principles to understand — and control — every trade-off: graph connectivity, beam-search width, layer probability, and recall vs. latency. The result is a lightweight, dependency-minimal vector store that can be embedded in any application or extended as needed.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     REST API (Axum)                     │
│              POST /insert  ·  POST /search              │
│              GET  /health  ·  DELETE /vector/{id}       │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   Query Engine                          │
│         ef_search beam  ·  distance metric selection    │
└────────────┬──────────────────────────┬─────────────────┘
             │                          │
┌────────────▼────────────┐  ┌──────────▼────────────────┐
│      HNSW Index         │  │   Vector Storage           │
│  Multi-layer NSW graph  │  │  memmap2 memory-mapped I/O │
│  M neighbours per node  │  │  bincode serialization     │
│  Rayon parallel search  │  │  serde-backed structs      │
└─────────────────────────┘  └───────────────────────────┘
```

### How the Index Works

HNSW organises vectors into a probabilistic multi-layer graph. Each vector is assigned to a random maximum layer; layers closer to 0 are denser, higher layers are sparse long-range links used for fast entry-point navigation.

**Insert** — greedy descent from the top layer to `layer + 1`, then beam search at each remaining layer to find the `ef_construction` closest neighbours. Bidirectional edges are added (and pruned to ≤ `M` using a heuristic that preserves diversity).

**Search** — enter at the top layer, greedily descend to layer 1, then perform an `ef_search`-wide beam search at layer 0 and return the top-k results.

| Parameter | Role | Default |
|---|---|---|
| `M` | Max edges per node per layer | 16 |
| `ef_construction` | Beam width during insert | 200 |
| `ef_search` | Beam width during query | 50 |
| `ml` | Layer assignment multiplier | `1 / ln(M)` |

---

## Performance

| Dataset | Dimensions | Recall@10 | QPS (single thread) |
|---|---|---|---|
| Synthetic uniform | 128 | **98.2%** | ~4,200 |
| Synthetic Gaussian | 256 | **97.6%** | ~2,800 |
| 1 M vectors | 128 | **98%+** | — |

_Benchmarks run on Apple M-series. Parallel search via Rayon scales linearly with core count._

---

## Project Structure

```
HNSW_Vector_DB/
├── src/
│   ├── main.rs          # Axum server bootstrap, route registration
│   ├── hnsw.rs          # Core HNSW graph: insert, search, layer logic
│   ├── storage.rs       # Memory-mapped vector store (memmap2 + bincode)
│   ├── distance.rs      # Euclidean, cosine, inner-product metrics
│   ├── api/
│   │   ├── handlers.rs  # Request/response handlers
│   │   └── models.rs    # Serde-annotated API structs
│   └── lib.rs           # Public crate surface
├── tests/
│   ├── hnsw_tests.rs    # Recall benchmarks, edge-case inserts
│   └── api_tests.rs     # Integration tests against the live server
├── benchmarks/          # Python scripts for recall / latency plots
├── Cargo.toml
└── README.md
```

---

## Tech Stack

| Crate | Purpose |
|---|---|
| `axum 0.7` | Async REST API framework |
| `tokio` | Async runtime (full feature set) |
| `rayon` | Data-parallel search across graph layers |
| `ndarray` | Vectorised distance computation |
| `memmap2` | Zero-copy memory-mapped file persistence |
| `bincode` + `serde` | Ultra-fast binary serialisation of graph + vectors |
| `priority-queue` | Max-heap for ef-wide beam search |
| `tower-http` | CORS middleware |
| `serde_json` | JSON API payloads |

Python (18%) is used for benchmark scripts, recall evaluation, and plotting.

---

## Getting Started

### Prerequisites

- Rust ≥ 1.85 (2024 edition)
- Cargo

### Build & Run

```bash
git clone https://github.com/psood708/HNSW_Vector_DB.git
cd HNSW_Vector_DB
cargo build --release
cargo run --release
```

The server starts on `http://localhost:3000` by default.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `3000` | HTTP listen port |
| `STORAGE_PATH` | `./data/vectors.bin` | Memory-mapped storage file |
| `M` | `16` | HNSW max neighbours |
| `EF_CONSTRUCTION` | `200` | Build-time beam width |
| `EF_SEARCH` | `50` | Query-time beam width |

---

## API Reference

### Insert a vector

```bash
curl -X POST http://localhost:3000/insert \
  -H "Content-Type: application/json" \
  -d '{
    "id": "vec-001",
    "vector": [0.12, 0.87, 0.34, ...],
    "metadata": {"label": "example"}
  }'
```

```json
{ "status": "ok", "id": "vec-001" }
```

---

### Nearest-neighbour search

```bash
curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.11, 0.85, 0.36, ...],
    "top_k": 5,
    "metric": "cosine"
  }'
```

```json
{
  "results": [
    { "id": "vec-001", "score": 0.9987, "metadata": {"label": "example"} },
    ...
  ],
  "query_time_ms": 1.2
}
```

Supported `metric` values: `"euclidean"` · `"cosine"` · `"dot_product"`

---

### Health check

```bash
curl http://localhost:3000/health
```

```json
{ "status": "ok", "vectors_indexed": 1000000 }
```

---

### Delete a vector

```bash
curl -X DELETE http://localhost:3000/vector/vec-001
```

---

## Running Tests

```bash
# Unit + integration tests
cargo test

# Recall benchmark (requires Python + matplotlib)
python benchmarks/recall_benchmark.py --n 100000 --dim 128 --k 10
```

---

## Design Decisions

**Why Rust?** Memory safety without a GC is essential when managing raw graph pointers and mmap'd regions at scale. Zero-cost abstractions mean the HNSW graph traversal carries no runtime overhead beyond the algorithm itself.

**Why `memmap2` instead of a dedicated storage engine?** For a focused ANN index, OS-managed paging over a flat binary file gives predictable I/O behaviour, avoids write-ahead log complexity, and makes the storage layer trivially portable.

**Why hand-rolled HNSW instead of `hnswlib` via FFI?** Full control over data layout, distance dispatch, and the pruning heuristic — plus no C++ build dependency, which simplifies cross-compilation and air-gapped deployment.

---


## References

- Malkov & Yashunin (2016) — *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*
- Bernhardsson (2018) — *Annoy: Approximate Nearest Neighbors in C++/Python*
- Aumüller et al. — *ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms*

---

## Author

**Parth Sood** — Data Science Analyst · [GitHub](https://github.com/psood708)
