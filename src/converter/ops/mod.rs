
use serde_json::Value;
use crate::proto::onnx;
use crate::helper::{self, dt};

pub mod array;
pub mod math;
pub mod conv;
pub mod resize;

impl super::Converter {
    pub fn process_pass2_op(&mut self, op_type: &str, op: &Value) -> anyhow::Result<()> {
        match op_type {
            "p" | "0.combine" => return Ok(()),
            "1.full" | "1.full_int_array" => self.op_full(op_type, op)?,
            "1.data" => self.op_data(op)?,
            "1.fetch" => self.op_fetch(op)?,
            "1.slice" => self.op_slice(op)?,
            "1.flatten" => self.op_flatten(op)?,
            "1.scale" => self.op_scale(op)?,
            "1.swish" => self.op_swish(op)?,
            "1.stack" => self.op_stack(op)?,
            "1.concat" => self.op_concat(op)?,
            "1.nearest_interp" => self.op_nearest_interp(op)?,
            "1.pool2d" => self.op_pool2d(op)?,
            "1.dropout" => self.op_dropout(op)?,
            _ => {
                if op_type.starts_with("1.") {
                    let clean_op_type = op_type.trim_start_matches("1.").trim_end_matches("_");
                    self.convert_generic_op(clean_op_type, op)?;
                }
            }
        }
        Ok(())
    }

    pub fn op_data(&mut self, op: &Value) -> anyhow::Result<()> {
        let vinfo = self.build_value_info(op, true);
        self.onnx_graph.input.push(vinfo.clone());
        if let Ok(out_id) = helper::op_out_id(op) {
            self.id_to_name.insert(out_id, vinfo.name);
        }
        Ok(())
    }

    pub fn op_fetch(&mut self, op: &Value) -> anyhow::Result<()> {
        let vinfo = self.build_value_info(op, false);
        self.onnx_graph.output.push(vinfo);
        let inputs = helper::op_input_ids(op);
        if !inputs.is_empty() {
            let in_name = self.get_tensor_name(inputs[0])?;
            if let Some(out_name) = helper::attr(op, "name").and_then(|d| d.as_str()) {
                let node = onnx::NodeProto {
                    op_type: "Identity".to_string(),
                    input: vec![in_name],
                    output: vec![out_name.to_string()],
                    ..Default::default()
                };
                self.onnx_graph.node.push(node);
            }
        }
        Ok(())
    }

    pub fn op_full(&mut self, op_type: &str, op: &Value) -> anyhow::Result<()> {
        if let Ok(out_id) = helper::op_out_id(op)
            && let Some(vals) = self.constants.get(&out_id).cloned() {
                let mut is_float = false;
                let mut shape_dims = vec![];
                if let Some(d_arr) = helper::attr(op, "shape").and_then(|d| d.as_array()) {
                    shape_dims = d_arr.iter().filter_map(|x| x.get("D").and_then(|d| d.as_i64()).or_else(|| x.as_i64())).collect();
                }
                if op_type == "1.full" && helper::attr(op, "dtype").and_then(|d| d.as_str()) == Some("float32") {
                    is_float = true;
                }
                
                let mut tensor = onnx::TensorProto {
                    name: self.get_tensor_name(out_id)?,
                    dims: shape_dims.clone(),
                    ..Default::default()
                };
                
                let expected_volume = if tensor.dims.is_empty() { 1 } else { tensor.dims.iter().product::<i64>() };
                if expected_volume != vals.len() as i64 || (tensor.dims.is_empty() && op_type == "1.full_int_array") {
                    tensor.dims = vec![vals.len() as i64];
                }                
                let mut raw_data = Vec::new();
                if is_float {
                    tensor.data_type = dt::FLOAT;
                    for &v in &vals {
                        raw_data.extend_from_slice(&(v as f32).to_le_bytes());
                    }
                } else {
                    tensor.data_type = dt::INT64;
                    for &v in &vals {
                        raw_data.extend_from_slice(&(v as i64).to_le_bytes());
                    }
                }
                tensor.raw_data = raw_data;
                self.onnx_graph.initializer.push(tensor);
            }
        Ok(())
    }

    pub fn op_dropout(&mut self, op: &Value) -> anyhow::Result<()> {
        let mut onnx_node = onnx::NodeProto {
            op_type: "Identity".to_string(),
            ..Default::default()
        };
        let inputs = helper::op_input_ids(op);
        if !inputs.is_empty() {
            onnx_node.input.push(self.get_tensor_name(inputs[0])?);
        }
        if let Ok(out_id) = helper::op_out_id(op) {
            onnx_node.output.push(self.get_tensor_name(out_id)?);
        }
        self.onnx_graph.node.push(onnx_node);
        Ok(())
    }

    pub fn convert_generic_op(&mut self, op_type: &str, op: &Value) -> anyhow::Result<()> {
        let mut onnx_node = onnx::NodeProto {
            op_type: self.map_op_type(op_type)?,
            ..Default::default()
        };

        if let Some(attrs) = op.get("A") {
            onnx_node.attribute = self.extract_attributes(op_type, attrs);
        }

        let inputs = helper::op_input_ids(op);
        let mut num_inputs = inputs.len();
        if onnx_node.op_type == "ConvTranspose" {
            // Note: Paddle's 3rd input is output_size, not bias.
            // If a model uses ConvTranspose with an explicit bias input (e.g. as 4th input),
            // this logic would silently drop it and require an explicit Add node later.
            // PP-OCR usually uses a separate `1.add` for bias.
            num_inputs = 2; 
        }
        
        if onnx_node.op_type == "BatchNormalization" && inputs.len() >= 5 {
            onnx_node.input.push(self.get_tensor_name(inputs[0])?);
            onnx_node.input.push(self.get_tensor_name(inputs[3])?); // scale
            onnx_node.input.push(self.get_tensor_name(inputs[4])?); // bias
            onnx_node.input.push(self.get_tensor_name(inputs[1])?); // mean
            onnx_node.input.push(self.get_tensor_name(inputs[2])?); // var
        } else {
            for input_id in inputs.iter().take(num_inputs) {
                onnx_node.input.push(self.get_tensor_name(*input_id)?);
            }
        }

        if let Some(outputs) = op.get("O").and_then(|o| o.as_array()) {
            let mut num_outputs = outputs.len();
            if onnx_node.op_type == "BatchNormalization" {
                num_outputs = 1;
            }
            for output in outputs.iter().take(num_outputs) {
                if let Some(id) = output.get("%").and_then(|id| id.as_i64()) {
                    onnx_node.output.push(self.get_tensor_name(id)?);
                }
            }
        }

        self.onnx_graph.node.push(onnx_node);
        Ok(())
    }
}
