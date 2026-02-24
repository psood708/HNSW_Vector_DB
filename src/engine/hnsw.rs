#[derive(Debug)]
pub struct VectorNode{
    pub id: usize,
    pub vector: Vec<f32>,
    pub neighbors: Vec<Vec<usize>>,
}

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
            for layer in (0..=std::cmp::min(target_layer,self.target_layer)).rev(){
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
        let neighbors = &mut self.nodes[node_id].neighbors[layer];

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
    }
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