use serde::{Deserialize, Serialize};

#[derive(serde::Deserialize)]
pub struct InsertRequest {
    pub vector: Vec<f32>,
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: Vec<f32>,
    pub k: usize,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub matches: Vec<SearchResult>,
}

#[derive(Serialize)]
pub struct SearchResult {
    pub id: usize,
    pub score: f32,
}