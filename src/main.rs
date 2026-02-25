mod engine;
mod api;
mod storage;

use crate::engine::hnsw::HnswIndex;
use crate::api::{InsertRequest,SearchRequest,SearchResponse,SearchResult};

use axum::{routing::post, extract::State, Json, Router};
use std::sync::{Arc, RwLock};


type SharedState = Arc<RwLock<HnswIndex>>;

#[tokio::main]
async fn main(){
    // initalizing HNSW Index

    println!("🚀 Initializing Vector DB...");

    let index = HnswIndex::new(16);
    let shared_state= Arc::new(RwLock::new(index));

    // building the router
    let app = Router::new()
        .route("/insert",post(insert_handler))
        .route("/search",post(search_handler))
        .with_state(shared_state);

    // run server
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8000").await.unwrap();
    println!("📡 Mini-Vector DB Running on port 8000");
    axum::serve(listener,app).await.unwrap();
}


// async fn insert_vector(
//     State(state): State<SharedState>,
//     Json(payload): Json<InsertRequest>,
// ) -> &'static str{
//     let mut index = state.write().unwrap();
//     index.insert(payload.vector);
//     "Vector Inserted Successfully"
// }

// async fn search_vectors(
//     State(state): State<SharedState>,
//     Json(payload): Json<SearchRequest>,
// ) -> Json<SearchResponse> {
//     let index = state.read().unwrap();
//     let results = index.discover_nearest(&payload.query);

//     Json(SearchResponse{
//         matches: vec![SearchResult{ id: results.unwrap_or(0),score: 0.99}]
//     })
// }

pub async fn insert_handler(
    State(state): State<SharedState>,
    Json(payload): Json<InsertRequest>,
) -> &'static str {
    let mut index = state.write().expect("Failed to acquire write lock");
    index.insert(payload.vector);
    "Vector successfully indexed"
}

pub async fn search_handler(
    State(state): State<SharedState>,
    Json(payload): Json<SearchRequest>,
) -> Json<SearchResponse> {
    let index = state.read().expect("Failed to acquire read lock");
    
    // Using our HNSW search logic
    let result_id = index.discover_nearest(&payload.query);
    
    let matches = match result_id {
        Some(id) => vec![SearchResult { id, score: 1.0 }], // Score logic can be added later
        None => vec![],
    };

    Json(SearchResponse { matches })
}