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