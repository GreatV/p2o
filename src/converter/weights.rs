use anyhow::bail;
use prost::Message;
use std::fs::File;
use std::io::Read;

use crate::helper::{self, dt};
use crate::proto::onnx;
use crate::proto::{PaddleDataType, TensorDesc};

const MAX_PDIPARAMS_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MAX_TENSOR_BYTES_SOFT_LIMIT: usize = 8 * 1024 * 1024 * 1024;

impl super::Converter {
    pub fn load_paddle_weights(&mut self, weights_path: &str) -> anyhow::Result<()> {
        let mut file = File::open(weights_path)?;
        let file_len = file.metadata()?.len();
        if file_len > MAX_PDIPARAMS_BYTES {
            bail!(
                ".pdiparams file is too large: {} bytes exceeds soft limit {} bytes",
                file_len,
                MAX_PDIPARAMS_BYTES
            );
        }
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;

        let mut read_size = 0;
        let total_size = buf.len();
        let mut index = 0;

        while read_size < total_size {
            // Each Paddle parameter record starts with a 4-byte version marker.
            if read_size + 4 > total_size {
                bail!("Truncated .pdiparams header at byte {}", read_size);
            }
            read_size += 4;

            // Followed by an 8-byte LoD level count. Inference weights are plain tensors.
            if read_size + 8 > total_size {
                bail!("Truncated LoD header at byte {}", read_size);
            }
            let mut lod_bytes = [0u8; 8];
            lod_bytes.copy_from_slice(&buf[read_size..read_size + 8]);
            let lod_level = u64::from_le_bytes(lod_bytes);
            if lod_level != 0 {
                bail!("LoD tensors not supported");
            }
            read_size += 8;

            // Paddle then stores a 4-byte tensor metadata header before the TensorDesc size.
            if read_size + 4 > total_size {
                bail!("Truncated tensor metadata header at byte {}", read_size);
            }
            read_size += 4;

            if read_size + 4 > total_size {
                bail!("Truncated tensor desc size at byte {}", read_size);
            }
            let mut size_bytes = [0u8; 4];
            size_bytes.copy_from_slice(&buf[read_size..read_size + 4]);
            let tensor_desc_size = i32::from_le_bytes(size_bytes);
            if tensor_desc_size < 0 {
                bail!(
                    "Invalid tensor desc size {} at byte {}",
                    tensor_desc_size,
                    read_size
                );
            }
            let tensor_desc_size = tensor_desc_size as usize;
            read_size += 4;

            if read_size + tensor_desc_size > total_size {
                bail!(
                    "Truncated tensor desc: need {} bytes at byte {}",
                    tensor_desc_size,
                    read_size
                );
            }
            let tensor_desc_buf = &buf[read_size..read_size + tensor_desc_size];
            let tensor_desc = TensorDesc::decode(tensor_desc_buf)?;
            read_size += tensor_desc_size;

            let data_type = tensor_desc.data_type();
            let dims = tensor_desc.dims.clone();
            let numel = dims.iter().try_fold(1usize, |acc, &dim| {
                let dim = usize::try_from(dim)
                    .map_err(|_| anyhow::anyhow!("Invalid negative tensor dim {}", dim))?;
                acc.checked_mul(dim)
                    .ok_or_else(|| anyhow::anyhow!("Tensor element count overflow"))
            })?;

            let (onnx_dt, type_size) = match data_type {
                PaddleDataType::Fp32 => (dt::FLOAT, 4),
                PaddleDataType::Int32 => (dt::INT32, 4),
                PaddleDataType::Int64 => (dt::INT64, 8),
                PaddleDataType::Fp16 => (dt::FLOAT16, 2),
                PaddleDataType::Bf16 => (dt::BFLOAT16, 2),
                PaddleDataType::Fp64 => (dt::DOUBLE, 8),
                PaddleDataType::Int8 => (dt::INT8, 1),
                PaddleDataType::Uint8 => (dt::UINT8, 1),
                PaddleDataType::Bool => (dt::BOOL, 1),
                PaddleDataType::Int16 => (dt::INT16, 2),
                _ => bail!("Unsupported data_type: {:?}", data_type),
            };

            let data_size = numel
                .checked_mul(type_size)
                .ok_or_else(|| anyhow::anyhow!("Tensor byte size overflow"))?;
            if data_size > MAX_TENSOR_BYTES_SOFT_LIMIT {
                bail!(
                    "Tensor byte size {} exceeds soft limit {}",
                    data_size,
                    MAX_TENSOR_BYTES_SOFT_LIMIT
                );
            }
            if read_size + data_size > total_size {
                bail!(
                    "Truncated tensor data for parameter index {}: need {} bytes at byte {}",
                    index,
                    data_size,
                    read_size
                );
            }
            let tensor_data = &buf[read_size..read_size + data_size];
            read_size += data_size;

            if index >= self.state.param_names.len() {
                bail!(
                    "Weight file contains more tensors than discovered parameters (extra tensor at index {})",
                    index
                );
            }

            let name = &self.state.param_names[index];
            if let Some(expected) = self.state.param_meta.get(name) {
                if let Some(expected_dtype) = expected.onnx_dtype
                    && expected_dtype != onnx_dt
                {
                    bail!(
                        "Weight tensor '{}' dtype mismatch: expected {} from model metadata, got {}",
                        name,
                        helper::onnx_dtype_name(expected_dtype),
                        helper::onnx_dtype_name(onnx_dt)
                    );
                }
                if expected.dims != dims {
                    bail!(
                        "Weight tensor '{}' shape mismatch: expected {:?} from model metadata, got {:?}",
                        name,
                        expected.dims,
                        dims
                    );
                }
            }

            let onnx_tensor = onnx::TensorProto {
                name: name.clone(),
                dims: dims.clone(),
                data_type: onnx_dt,
                raw_data: tensor_data.to_vec(),
                ..Default::default()
            };

            self.onnx_graph.initializer.push(onnx_tensor);
            index += 1;
        }

        if index != self.state.param_names.len() {
            bail!(
                "Weight count mismatch: loaded {} tensors for {} parameters",
                index,
                self.state.param_names.len()
            );
        }
        Ok(())
    }
}
