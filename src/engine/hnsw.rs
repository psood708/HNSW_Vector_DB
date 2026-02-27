use serde::{Serialize, Deserialize};
use rand::{rng, RngExt}; // Standard for rand 0.10.0
use crate::engine::distance::{Distance, CosineSimilarity};
use crate::storage::MmapStorage;


#[derive(PartialEq)]
struct SearchCandidate {
    id: usize,
    distance: f32,
}

impl Eq for SearchCandidate {}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Use the PartialOrd implementation we just wrote
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VectorNode {
    pub id: usize,
    pub vector: Vec<f32>,
    pub neighbors: Vec<Vec<usize>>,
}

impl VectorNode {
    pub fn new(id: usize, vector: Vec<f32>, max_layer: usize) -> Self {
        Self {
            id,
            vector,
            // Create a vector of vectors, one for each layer up to target_layer
            neighbors: vec![vec![]; max_layer + 1],
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct HnswIndex {
    pub nodes: Vec<VectorNode>,
    pub entry_point: Option<usize>,
    pub max_layer: usize,
    pub m: usize,
    pub m_l: f64,
}

impl HnswIndex {
    pub fn new(m: usize) -> Self {
        Self {
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
            m,
            // Standard multiplier to ensure log(N) layer distribution
            m_l: 1.0 / (m as f64).ln(),
        }
    }

pub fn insert(&mut self, vector: Vec<f32>) {
    let id = self.nodes.len();
    let target_layer = self.calculate_random_layer();
    let new_node = VectorNode::new(id, vector, target_layer);
    
    // 1. PUSH FIRST: This makes 'id' a valid index in self.nodes immediately
    self.nodes.push(new_node);

    // 2. Start connecting (if this wasn't the very first node)
    if let Some(mut curr_entry_id) = self.entry_point {
        // Navigation: Zoom down to target_layer
        for layer in (target_layer + 1..=self.max_layer).rev() {
            curr_entry_id = self.search_layer(&self.nodes[id].vector, curr_entry_id, layer);
        }

        // Connection: From target_layer down to 0
        let start_layer = std::cmp::min(target_layer, self.max_layer);
        for layer in (0..=start_layer).rev() {
            let neighbor_candidates = self.find_neighbors_for_layer(&self.nodes[id].vector, curr_entry_id, layer);
            
            for neighbor_id in neighbor_candidates {
                // Safety: Don't connect a node to itself
                if neighbor_id == id { continue; }

                // Add neighbor_id to the NEW node's list
                self.nodes[id].neighbors[layer].push(neighbor_id);
                
                // Add NEW node's ID to the NEIGHBOR'S list
                if let Some(neighbor_node) = self.nodes.get_mut(neighbor_id) {
                    if layer < neighbor_node.neighbors.len() {
                        neighbor_node.neighbors[layer].push(id);
                    }
                }
                
                // Prune the neighbor
                self.prune_neighbors(neighbor_id, layer);
            }
            // Move entry point down
            curr_entry_id = self.search_layer(&self.nodes[id].vector, curr_entry_id, layer);
        }
    }

    // 3. Update entry point if this node is the new "tallest"
    if self.entry_point.is_none() || target_layer > self.max_layer {
        self.entry_point = Some(id);
        self.max_layer = target_layer;
        println!("🆕 New Entry Point: Node {} at Layer {}", id, target_layer);
    }

    // 4. Prune the new node itself
    for layer in 0..=target_layer {
        self.prune_neighbors(id, layer);
    }

    println!("✅ Inserted node ID: {} at max layer: {}", id, target_layer);
}



 

   fn prune_neighbors(&mut self, node_id: usize, layer: usize) {
    if node_id >= self.nodes.len() { return; }
    
    // 1. Get the data we need and release the borrow on self immediately
    let node_vector = self.nodes[node_id].vector.clone();
    let mut neighbors = self.nodes[node_id].neighbors[layer].clone();

    if neighbors.len() <= self.m { return; }

    // 2. Now we can safely sort because 'neighbors' is a local clone, 
    // and we only read from 'self.nodes'
    neighbors.sort_by(|&a, &b| {
        let dist_a = CosineSimilarity::calculate(&node_vector, &self.nodes[a].vector);
        let dist_b = CosineSimilarity::calculate(&node_vector, &self.nodes[b].vector);
        // Higher similarity = closer (descending order)
        dist_b.partial_cmp(&dist_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    // 3. Put the sorted, truncated list back
    neighbors.truncate(self.m);
    self.nodes[node_id].neighbors[layer] = neighbors;
}

    pub fn calculate_random_layer(&self) -> usize {
        let mut my_rng = rng();
        let r: f64 = my_rng.random_range(0.0..1.0);
        let r_val = if r == 0.0 { 1e-9 } else { r };
        (-r_val.ln() * self.m_l).floor() as usize
    }

  pub fn search_layer(&self, query: &[f32], entry_point: usize, layer: usize) -> usize {
    let mut current_node = entry_point;
    
    // Get the REAL distance of the starting node first
    let mut best_dist = match self.nodes.get(current_node) {
        Some(node) => CosineSimilarity::calculate(query, &node.vector),
        None => return entry_point,
    };

    let mut changed = true;
    while changed {
        changed = false;
        if let Some(node) = self.nodes.get(current_node) {
            if let Some(neighbors) = node.neighbors.get(layer) {
                for &neighbor_id in neighbors {
                    if let Some(neighbor_node) = self.nodes.get(neighbor_id) {
                        let dist = CosineSimilarity::calculate(query, &neighbor_node.vector);
                        // SIMILARITY: Higher is closer to the target
                        if dist > best_dist {
                            best_dist = dist;
                            current_node = neighbor_id;
                            changed = true;
                        }
                    }
                }
            }
        }
    }
    current_node
}

    pub fn discover_nearest(&self, query: &[f32]) -> Option<usize> {
        let mut current_entry = self.entry_point?;
        
        // Start from max_layer and zoom down
        for layer in (1..=self.max_layer).rev() {
            current_entry = self.search_layer(query, current_entry, layer);
        }
        // Some(current_entry)
        // Final search on base layer
        Some(self.search_layer(query, current_entry, 0))
    }

    pub fn find_neighbors_for_layer(&self, query: &[f32], entry: usize, layer: usize) -> Vec<usize> {
    let mut candidates = std::collections::BinaryHeap::new();
    let mut visited = std::collections::HashSet::new();
    
    // Start with the entry point
    let dist = CosineSimilarity::calculate(query, &self.nodes[entry].vector);
    candidates.push(SearchCandidate { id: entry, distance: dist });
    visited.insert(entry);

    let mut top_neighbors = Vec::new();

    // Greedily explore neighbors
    while let Some(current) = candidates.pop() {
        top_neighbors.push(current.id);
        if top_neighbors.len() >= self.m { break; }

        if let Some(node) = self.nodes.get(current.id) {
            if let Some(neighbors) = node.neighbors.get(layer) {
                for &neighbor_id in neighbors {
                    if !visited.contains(&neighbor_id) {
                        visited.insert(neighbor_id);
                        if let Some(neighbor_node) = self.nodes.get(neighbor_id) {
                            let d = CosineSimilarity::calculate(query, &neighbor_node.vector);
                            candidates.push(SearchCandidate { id: neighbor_id, distance: d });
                        }
                    }
                }
            }
        }
    }
    top_neighbors
}

    pub fn save_to_mmap(&self, storage: &mut MmapStorage) {
        let encoded: Vec<u8> = bincode::serialize(self).expect("Failed to serialize index");
        
        if encoded.len() > storage.mmap.len() {
            panic!("Mmap buffer ({} bytes) too small for index ({} bytes)!", storage.mmap.len(), encoded.len());
        }
        
        storage.mmap[..encoded.len()].copy_from_slice(&encoded);
        storage.flush().expect("Failed to sync to disk");
    }
}