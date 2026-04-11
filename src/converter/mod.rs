use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use prost::Message;
use serde_json::Value;
use anyhow::bail;

use crate::proto::onnx;
use crate::helper::{self, dt};

mod attr;
mod ops;
mod weights;

#[cfg(test)]
mod tests;

pub struct Converter {
    pub onnx_graph: onnx::GraphProto,
    pub id_to_name: HashMap<i64, String>,
    pub param_names: Vec<String>,
    pub combines: HashMap<i64, Vec<i64>>,
    pub constants: HashMap<i64, Vec<f64>>,
}

impl Converter {
    pub fn new() -> Self {
        Self {
            onnx_graph: onnx::GraphProto {
                node: vec![],
                name: "Paddle2ONNX_Model".to_string(),
                initializer: vec![],
                input: vec![],
                output: vec![],
                value_info: vec![],
                ..Default::default()
            },
            id_to_name: HashMap::new(),
            param_names: Vec::new(),
            combines: HashMap::new(),
            constants: HashMap::new(),
        }
    }

    pub fn get_tensor_name(&self, id: i64) -> anyhow::Result<String> {
        if id == -1 {
            bail!("Undefined value id -1");
        }
        if id == 0 {
            return Ok("".to_string());
        }
        Ok(self.id_to_name.get(&id).cloned().unwrap_or_else(|| format!("tensor_{}", id)))
    }

    pub fn build_value_info(&mut self, op: &Value, is_input: bool) -> onnx::ValueInfoProto {
        let name = if is_input {
            helper::attr(op, "name").and_then(|d| d.as_str()).unwrap_or("input").to_string()
        } else {
            helper::attr(op, "name").and_then(|d| d.as_str()).unwrap_or("output").to_string()
        };

        let mut vi = onnx::ValueInfoProto {
            name,
            ..Default::default()
        };

        // Fallback: If O[0].TT is missing (e.g. 1.fetch), we should try to extract from input if it's fetch
        let mut tt_d = op.get("O")
            .and_then(|o| o.as_array())
            .and_then(|o| o.first())
            .and_then(|o| o.get("TT"))
            .and_then(|tt| tt.get("D"))
            .and_then(|d| d.as_array());

        // For fetch, TT might be missing on output, try input
        if tt_d.is_none() && !is_input {
            tt_d = op.get("I")
                .and_then(|i| i.as_array())
                .and_then(|i| i.first())
                .and_then(|i| i.get("TT"))
                .and_then(|tt| tt.get("D"))
                .and_then(|d| d.as_array());
        }

        if let Some(tt) = tt_d
            && tt.len() >= 2 {
                let elem_type_str = tt[0].get("#").and_then(|t| t.as_str()).unwrap_or("");
                let onnx_type = match elem_type_str {
                    "0.t_f32" => dt::FLOAT,
                    "0.t_i32" => dt::INT32,
                    "0.t_i64" => dt::INT64,
                    _ => dt::FLOAT,
                };
                
                let mut shape_proto = onnx::TensorShapeProto { dim: vec![] };
                if let Some(dims) = tt[1].as_array() {
                    for dim in dims {
                        if let Some(d) = dim.as_i64() {
                            let mut dim_proto = onnx::tensor_shape_proto::Dimension {
                                denotation: "".to_string(),
                                ..Default::default()
                            };
                            if d > 0 {
                                dim_proto.value = Some(onnx::tensor_shape_proto::dimension::Value::DimValue(d));
                            }
                            shape_proto.dim.push(dim_proto);
                        }
                    }
                }
                
                let tensor_type = onnx::type_proto::Tensor {
                    elem_type: onnx_type,
                    shape: Some(shape_proto),
                };
                
                vi.r#type = Some(onnx::TypeProto {
                    denotation: "".to_string(),
                    value: Some(onnx::type_proto::Value::TensorType(tensor_type)),
                });
            }
        
        vi
    }

    pub fn load_paddle_model(&mut self, json_path: &str) -> anyhow::Result<()> {
        println!("Loading PaddlePaddle JSON model: {}", json_path);
        let file = File::open(json_path)?;
        let data: Value = serde_json::from_reader(file)?;

        let ops = data.pointer("/program/regions/0/blocks/0/ops")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("Could not find ops array in JSON under /program/regions/0/blocks/0/ops"))?;

        // Pass 1: Extract parameters, constants, and combines
        for op in ops {
            if let Some(op_type) = helper::op_type(op) {
                if op_type == "p" {
                    let name = op.get("A")
                        .and_then(|a| a.as_array())
                        .and_then(|a| a.iter().find_map(|x| x.as_str()))
                        .unwrap_or("unknown_param");
                    
                    if let Ok(id) = helper::op_out_id(op) {
                        self.id_to_name.insert(id, name.to_string());
                        self.param_names.push(name.to_string());
                    }
                } else if op_type == "0.combine" {
                    let inputs = helper::op_input_ids(op);
                    if let Ok(out_id) = helper::op_out_id(op) {
                        self.combines.insert(out_id, inputs);
                    }
                } else if (op_type == "1.full" || op_type == "1.full_int_array")
                    && let Ok(out_id) = helper::op_out_id(op) {
                        let mut vals = Vec::new();
                        if let Some(d) = helper::attr(op, "value") {
                            if d.is_f64() {
                                vals.push(d.as_f64().unwrap());
                            } else if d.is_i64() {
                                vals.push(d.as_i64().unwrap() as f64);
                            } else if let Some(arr) = d.as_array() {
                                for item in arr {
                                    if let Some(f) = item.get("D").and_then(|d| d.as_f64()) {
                                        vals.push(f);
                                    } else if let Some(f) = item.as_f64() {
                                        vals.push(f);
                                    }
                                }
                            }
                        }
                        self.constants.insert(out_id, vals);
                    }
            }
        }
        
        // Paddle's save_combine writes vars in sorted-by-name order,
        // see python/paddle/static/io.py::_serialize_persistables.
        self.param_names.sort();
        println!("Found {} parameters.", self.param_names.len());

        // Pass 2: Convert computation operations
        for op in ops {
            if let Some(op_type) = helper::op_type(op) {
                self.process_pass2_op(op_type, op)?;
            }
        }

        Ok(())
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        let mut defined: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for init in &self.onnx_graph.initializer { defined.insert(&init.name); }
        for inp  in &self.onnx_graph.input       { defined.insert(&inp.name); }
        for node in &self.onnx_graph.node {
            for out in &node.output { defined.insert(out); }
        }
        for node in &self.onnx_graph.node {
            for inp in &node.input {
                if !inp.is_empty() && !defined.contains(inp.as_str()) {
                    bail!("node {} references undefined tensor {}", node.op_type, inp);
                }
            }
        }
        for out in &self.onnx_graph.output {
            if !defined.contains(out.name.as_str()) {
                bail!("graph output {} is never produced by any node", out.name);
            }
        }
        Ok(())
    }

    pub fn export_onnx(&mut self, output_path: &str, opset_version: i64) -> anyhow::Result<()> {
        self.validate()?;
        let model = onnx::ModelProto {
            ir_version: 8,
            opset_import: vec![onnx::OperatorSetIdProto {
                domain: "".to_string(),
                version: opset_version,
            }],
            producer_name: env!("CARGO_PKG_NAME").to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            graph: Some(self.onnx_graph.clone()),
            ..Default::default()
        };

        let mut buf = Vec::new();
        model.encode(&mut buf)?;

        let mut file = File::create(output_path)?;
        file.write_all(&buf)?;
        log::info!("Saved ONNX model to: {}", output_path);
        Ok(())
    }
}
