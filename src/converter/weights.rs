use std::fs::File;
use std::io::Read;
use prost::Message;
use anyhow::bail;

use crate::proto::onnx;
use crate::proto::TensorDesc;
use crate::helper::dt;

impl super::Converter {
    pub fn load_paddle_weights(&mut self, weights_path: &str) -> anyhow::Result<()> {
        let mut file = File::open(weights_path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;

        let mut read_size = 0;
        let total_size = buf.len();
        let mut index = 0;

        while read_size < total_size {
            if read_size + 4 > total_size { break; }
            read_size += 4;

            if read_size + 8 > total_size { break; }
            let mut lod_bytes = [0u8; 8];
            lod_bytes.copy_from_slice(&buf[read_size..read_size + 8]);
            let lod_level = u64::from_le_bytes(lod_bytes);
            if lod_level != 0 {
                bail!("LoD tensors not supported");
            }
            read_size += 8;

            if read_size + 4 > total_size { break; }
            read_size += 4;

            if read_size + 4 > total_size { break; }
            let mut size_bytes = [0u8; 4];
            size_bytes.copy_from_slice(&buf[read_size..read_size + 4]);
            let tensor_desc_size = i32::from_le_bytes(size_bytes) as usize;
            read_size += 4;

            if read_size + tensor_desc_size > total_size { break; }
            let tensor_desc_buf = &buf[read_size..read_size + tensor_desc_size];
            let tensor_desc = TensorDesc::decode(tensor_desc_buf)?;
            read_size += tensor_desc_size;

            let data_type = tensor_desc.data_type();
            let dims = tensor_desc.dims.clone();
            let mut numel = 1;
            for &dim in &dims {
                numel *= dim;
            }

            use crate::proto::paddle::framework::proto::var_type::var_type::Type as PdType;
            let (onnx_dt, type_size) = match data_type {
                PdType::Fp32 => (dt::FLOAT, 4),
                PdType::Int32 => (dt::INT32, 4),
                PdType::Int64 => (dt::INT64, 8),
                PdType::Fp16 => (dt::FLOAT16, 2),
                PdType::Bf16 => (dt::BFLOAT16, 2),
                PdType::Fp64 => (dt::DOUBLE, 8),
                PdType::Int8 => (dt::INT8, 1),
                PdType::Uint8 => (dt::UINT8, 1),
                PdType::Bool => (dt::BOOL, 1),
                PdType::Int16 => (dt::INT16, 2),
                _ => bail!("Unsupported data_type: {:?}", data_type),
            };

            let data_size = (numel as usize) * type_size;
            if read_size + data_size > total_size { break; }
            let tensor_data = &buf[read_size..read_size + data_size];
            read_size += data_size;

            if index < self.param_names.len() {
                let name = &self.param_names[index];
                
                let onnx_tensor = onnx::TensorProto {
                    name: name.clone(),
                    dims: dims.clone(),
                    data_type: onnx_dt,
                    raw_data: tensor_data.to_vec(),
                    ..Default::default()
                };
                
                self.onnx_graph.initializer.push(onnx_tensor);
            }
            index += 1;
        }
        Ok(())
    }
}
