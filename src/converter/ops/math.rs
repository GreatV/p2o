use serde_json::Value;
use crate::proto::onnx;
use crate::helper::{self, dt};
use anyhow::bail;

impl super::super::Converter {
    pub fn op_scale(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("scale op missing inputs");
        }
        let in_name = self.get_tensor_name(inputs[0])?;
        
        let scale_name = if inputs.len() > 1 {
            self.get_tensor_name(inputs[1])?
        } else {
            let scale = helper::attr(op, "scale").and_then(|d| d.as_f64()).unwrap_or(1.0) as f32;
            let s_name = format!("scale_factor_{}", out_id);
            let mut scale_tensor = onnx::TensorProto {
                name: s_name.clone(),
                dims: vec![1],
                data_type: dt::FLOAT,
                ..Default::default()
            };
            scale_tensor.raw_data.extend_from_slice(&scale.to_le_bytes());
            self.onnx_graph.initializer.push(scale_tensor);
            s_name
        };

        let bias = helper::attr(op, "bias").and_then(|d| d.as_f64()).unwrap_or(0.0) as f32;
        let bias_after_scale = helper::attr(op, "bias_after_scale").and_then(|d| d.as_bool()).unwrap_or(true);

        let bias_name = format!("scale_bias_{}", out_id);
        let mut bias_tensor = onnx::TensorProto {
            name: bias_name.clone(),
            dims: vec![1],
            data_type: dt::FLOAT,
            ..Default::default()
        };
        bias_tensor.raw_data.extend_from_slice(&bias.to_le_bytes());
        self.onnx_graph.initializer.push(bias_tensor);

        if bias_after_scale {
            // Y = scale * X + bias
            let mul_out = format!("scale_mul_{}", out_id);
            let mul_node = onnx::NodeProto {
                op_type: "Mul".to_string(),
                input: vec![in_name, scale_name],
                output: vec![mul_out.clone()],
                ..Default::default()
            };
            self.onnx_graph.node.push(mul_node);
            
            let add_node = onnx::NodeProto {
                op_type: "Add".to_string(),
                input: vec![mul_out, bias_name],
                output: vec![self.get_tensor_name(out_id)?],
                ..Default::default()
            };
            self.onnx_graph.node.push(add_node);
        } else {
            // Y = scale * (X + bias)
            let add_out = format!("scale_add_{}", out_id);
            let add_node = onnx::NodeProto {
                op_type: "Add".to_string(),
                input: vec![in_name, bias_name],
                output: vec![add_out.clone()],
                ..Default::default()
            };
            self.onnx_graph.node.push(add_node);

            let mul_node = onnx::NodeProto {
                op_type: "Mul".to_string(),
                input: vec![add_out, scale_name],
                output: vec![self.get_tensor_name(out_id)?],
                ..Default::default()
            };
            self.onnx_graph.node.push(mul_node);
        }
        Ok(())
    }

    pub fn op_swish(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() { bail!("swish missing inputs"); }
        
        let in_name = self.get_tensor_name(inputs[0])?;
        let sigmoid_out = format!("swish_sigmoid_{}", out_id);
        
        let sigmoid_node = onnx::NodeProto {
            op_type: "Sigmoid".to_string(),
            input: vec![in_name.clone()],
            output: vec![sigmoid_out.clone()],
            ..Default::default()
        };
        self.onnx_graph.node.push(sigmoid_node);

        let mul_node = onnx::NodeProto {
            op_type: "Mul".to_string(),
            input: vec![in_name, sigmoid_out],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        };
        self.onnx_graph.node.push(mul_node);
        Ok(())
    }
}
