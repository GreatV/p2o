use serde_json::Value;
use crate::proto::onnx;
use crate::helper::{self, dt};

impl super::super::Converter {
    pub fn op_nearest_interp(&mut self, op: &Value) -> anyhow::Result<()> {
        let mut onnx_node = onnx::NodeProto {
            op_type: "Resize".to_string(),
            ..Default::default()
        };
        
        onnx_node.attribute.push(onnx::AttributeProto {
            name: "mode".to_string(),
            r#type: crate::helper::at::STRING,
            s: b"nearest".to_vec(),
            ..Default::default()
        });

        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if !inputs.is_empty() {
            onnx_node.input.push(self.get_tensor_name(inputs[0])?);
        }
        onnx_node.input.push("".to_string()); // roi
        
        let scale_name = format!("scales_{}", out_id);
        onnx_node.input.push(scale_name.clone());
        let mut scale_vals = vec![1.0f32, 1.0, 2.0, 2.0];
        if let Some(d_arr) = helper::attr(op, "scale").and_then(|d| d.as_array())
            && d_arr.len() >= 2 {
                scale_vals[2] = d_arr[0].get("D").and_then(|v| v.as_f64()).unwrap_or(2.0) as f32;
                scale_vals[3] = d_arr[1].get("D").and_then(|v| v.as_f64()).unwrap_or(2.0) as f32;
            }
        
        let mut raw_data = Vec::new();
        for &v in &scale_vals {
            raw_data.extend_from_slice(&v.to_le_bytes());
        }
        let scales_tensor = onnx::TensorProto {
            name: scale_name,
            dims: vec![4],
            data_type: dt::FLOAT,
            raw_data,
            ..Default::default()
        };
        self.onnx_graph.initializer.push(scales_tensor);
        
        onnx_node.output.push(self.get_tensor_name(out_id)?);
        self.onnx_graph.node.push(onnx_node);
        Ok(())
    }
}
