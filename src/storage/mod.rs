use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::path::Path;

pub struct MmapStorage {
    pub mmap: MmapMut,
}

impl MmapStorage {
    pub fn new(path: &Path, size: u64) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        // Ensure the file is large enough to hold our data
        file.set_len(size)?;

        // Map the file into memory 
        let mmap = unsafe { MmapMut::map_mut(&file)? };

        Ok(MmapStorage { mmap })
    }

    pub fn flush(&self) -> std::io::Result<()> {
        self.mmap.flush()
    }
}