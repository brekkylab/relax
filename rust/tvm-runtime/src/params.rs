use std::{collections::HashMap, path::PathBuf};

use serde::{Deserialize, Serialize};
use tvm_ffi::{DLDataType, DLDataTypeCode, DLDataTypeExt};

use crate::RTensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TensorFormat {
    #[default]
    #[serde(rename = "f32-to-bf16")]
    F32ToBf16,
    #[serde(rename = "raw")]
    Raw,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
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
#[serde(default, rename_all = "camelCase")]
pub struct TensorShardEntry {
    pub data_path: String,
    pub format: ShardFormat,
    pub nbytes: usize,
    pub records: Vec<TensorCacheEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "PascalCase")]
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

    #[serde(skip)]
    pool: HashMap<String, RTensor>,
}

impl TensorCache {
    pub fn from_str(s: &str) -> serde_json::Result<Self> {
        let cache: TensorCache = serde_json::from_str(s)?;
        Ok(cache)
    }

    pub fn from(
        path: &PathBuf,
        device_type: tvm_ffi::DLDeviceType,
        device_id: i32,
    ) -> anyhow::Result<Self> {
        let device = tvm_ffi::DLDevice::new(device_type, device_id);
        let tensor_cache_json_path = std::fs::read_to_string(path.join("tensor-cache.json"))
            .map_err(|e| anyhow::anyhow!("Failed to open tensor-cache.json: {}", e.to_string()))?;
        let mut tensor_cache = TensorCache::from_str(&tensor_cache_json_path).map_err(|e| {
            anyhow::anyhow!("Failed to deserialize tensor-cache.json: {}", e.to_string())
        })?;
        for file_record in tensor_cache.records.iter() {
            let record_bytes = std::fs::read(path.join(&file_record.data_path)).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to open the record {}: {}",
                    file_record.data_path,
                    e.to_string()
                )
            })?;
            for param_record in file_record.records.iter() {
                let dtype = DLDataType::try_from_str(&param_record.dtype).unwrap();
                let mut tensor = RTensor::new(device, param_record.shape.clone(), dtype);

                if dtype.code == DLDataTypeCode::kDLFloat as u8
                    && dtype.bits == 32
                    && param_record.format == TensorFormat::F32ToBf16
                {
                    // Decode bf16 to f32
                    let mut buffer: Vec<u16> = Vec::with_capacity(param_record.nbytes / 2);
                    let mut decoded: Vec<u32> = Vec::with_capacity(param_record.nbytes / 2);
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            record_bytes.as_ptr().wrapping_add(param_record.byte_offset),
                            buffer.as_mut_ptr() as *mut u8,
                            param_record.nbytes,
                        )
                    };
                    for (idx, item) in buffer.into_iter().enumerate() {
                        decoded[idx] = (item as u32) << 16;
                    }
                    tensor.copy_from(bytemuck::cast_slice(&decoded));
                } else {
                    // Copy sliced data
                    let sliced = unsafe {
                        std::slice::from_raw_parts(
                            record_bytes.as_ptr().wrapping_add(param_record.byte_offset),
                            param_record.nbytes,
                        )
                    };
                    tensor.copy_from(sliced);
                }

                tensor_cache.pool.insert(param_record.name.clone(), tensor);
            }
        }
        Ok(tensor_cache)
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
