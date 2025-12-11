use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TensorFormat {
    #[default]
    #[serde(rename = "f32-to-bf16")]
    F32ToBf16,
    #[serde(rename = "raw")]
    Raw,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TensorCacheEntry {
    pub name: String,
    pub shape: Vec<u32>,
    pub dtype: String,
    pub format: TensorFormat,
    pub byte_offset: usize,
    pub nbytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ShardFormat {
    #[default]
    #[serde(rename = "raw-shard")]
    RawShard,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TensorShardEntry {
    pub data_path: String,
    pub format: ShardFormat,
    pub nbytes: usize,
    pub records: Vec<TensorCacheEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "kebab-case")]
pub struct TensorCacheMetadata {
    pub param_size: f32,
    pub param_bytes: f32,
    pub bits_per_param: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TensorCache {
    pub metadata: TensorCacheMetadata,
    pub records: Vec<TensorShardEntry>,
}

impl TensorCache {
    pub fn from_str(s: &str) -> serde_json::Result<Self> {
        let cache: TensorCache = serde_json::from_str(s)?;
        Ok(cache)
    }

    pub fn to_string(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}

pub struct Parameters {
    tensor_cache: TensorCache,
}

impl Parameters {
    pub fn new(tensor_cache: TensorCache) -> Self {
        Self { tensor_cache }
    }

    pub fn from_str(s: &str) -> serde_json::Result<Self> {
        Ok(Self::new(TensorCache::from_str(s)?))
    }

    pub fn handle_shard(&mut self, filename: &str, contents: &[u8]) {
        if let Some(entry) = self
            .tensor_cache
            .records
            .iter()
            .find(|&v| &v.data_path == filename)
        {
            for record in &entry.records {
                let start = record.byte_offset;
                let end = start + record.nbytes;
                let slice = &contents[start..end];
                let shape = &record.shape;
                let dtype = record.dtype.as_str();
            }
            return;
        } else {
            return;
        };
    }
}
