use serde::{Serialize,Deserialize};
use crate::engine::distance::{Distance,CosineSimilarity};
use crate::storage::MmapStorage;




#[derive(Debug)]
pub struct VectorNode{
    pub id: usize,
    pub vector: Vec<f32>,
    pub neighbors: Vec<Vec<usize>>,
}

impl VectorNode{
    pub fn new(id:usize,vector: Vec<f32>,max_layer: usize) -> Self{
        Self{
            id,
            vector,
            neighbors: vec![vec![]; max_layer+1],
        }
    }
}

#[derive(Serialize,Deserialize,Debug)]
pub struct HnswIndex{
    pub nodes: Vec<VectorNode>,
    pub entry_point: Option<usize>,
    pub max_layer: usize,
    pub m: usize,
    pub m_l: f64,
}

impl HnswIndex{
    pub fn new(m:usize) -> Self{
        Self{
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
            m,
            m_l: 1.0/(m as f64).ln(),
        }
    }

    pub fn insert(&mut self, vector: Vec<f32>){
        let id = self.nodes.len();
        let target_layer = self.calculate_random_layer();
        let mut new_node = VectorNode::new(id,vector,target_layer);

        if let Some(mut curr_entry_id) = self.entry_point{
            // zoom down to target_layer
            for layer in (target_layer + 1..=self.max_layer).rev(){
                curr_entry_id = self.search_layer(&new_node.vector,curr_entry_id,layer);
            }

            // insert and connct on all layers from target_layer  down to 0
            for layer in (0..=std::cmp::min(target_layer,self.max_layer)).rev(){
                let neighbors = self.find_neighbors_for_layer(&new_node.vector,curr_entry_id,layer);
                for neighbor_id in neighbors{
                    new_node.neighbors[layer].push(neighbor_id);
                    self.nodes[neighbor_id].neighbors[layer].push(id);
                    // TOODL implemnet pruning if neighbours > N
                    if self.nodes[neighbor_id].neighbors[layer].len() > self.m{
                        self.prune_neighbors(neighbor_id,layer);
                    }
                }
                curr_entry_id = new_node.id;
            }
        }

        // update global entry point if this node is highest
        if target_layer > self.max_layer || self.entry_point.is_none(){
            self.entry_point =  Some(id);
            self.max_layer = target_layer;
        }
        self.nodes.push(new_node);
    }


    fn prune_neighbors(&mut self, node_id: usize, layer: usize){
        let node_vector = self.nodes[node_id].vector.clone();
        let mut neighbors = self.nodes[node_id].neighbors[layer].clone();

        if neighbors.len() <= self.m{
            return;
        }

        // sort by distanc , for cosine highre  value is closer
        neighbors.sort_by(|&a,&b|{
            let dist_a = CosineSimilarity::calculate(&node_vector, &self.nodes[a].vector);
            let dist_b = CosineSimilarity::calculate(&node_vector, &self.nodes[b].vector);
            dist_b.partial_cmp(&dist_a).unwrap()
        });

        neighbors.truncate(self.m);
        self.nodes[node_id].neighbors[layer] = neighbors;

    }

    pub fn calculate_random_layer(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng::gen_range(&mut rng);
        (-r.ln() * self.m_l).floor() as usize
    }

    //  The core layer search
    pub fn search_layer(&self, query: &[f32], entry_point: usize, layer: usize) -> usize {
        let mut current_node = entry_point;
        let mut best_dist = CosineSimilarity::calculate(query, &self.nodes[current_node].vector);
        let mut changed = true;
        while changed {
            changed = false;
            for &neighbor_id in &self.nodes[current_node].neighbors[layer] {
                let dist = CosineSimilarity::calculate(query, &self.nodes[neighbor_id].vector);
                if dist > best_dist {
                    best_dist = dist;
                    current_node = neighbor_id;
                    changed = true;
                }
            }
        }
        current_node
    }

    // The top-level search method
    pub fn discover_nearest(&self, query: &[f32]) -> Option<usize> {
        let mut current_entry = self.entry_point?;
        for layer in (1..=self.max_layer).rev() {
            current_entry = self.search_layer(query, current_entry, layer);
        }
        Some(self.search_layer(query, current_entry, 0))
    }

    //  Helper for insertion  
    pub fn find_neighbors_for_layer(&self, query: &[f32], entry: usize, layer: usize) -> Vec<usize> {
        // For now, return the best one found; 
        // a real HNSW would return a set of top-M candidates
        vec![self.search_layer(query, entry, layer)]
    }

    pub fn save_to_mmap(&self, storage: &mut MmapStorage) {
        let encoded: Vec<u8> = bincode::serialize(self).unwrap();
        // Copy the encoded bytes into the memory-mapped region
        storage.mmap[..encoded.len()].copy_from_slice(&encoded);
        storage.flush().expect("Failed to sync to disk");
    }




}

