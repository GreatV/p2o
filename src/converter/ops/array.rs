use serde_json::Value;
use crate::proto::onnx;
use crate::helper::{self, dt};
use anyhow::bail;

impl super::super::Converter {
    pub fn op_slice(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let mut axes = vec![];
        let mut decrease_axis = vec![];

        if let Some(d_arr) = helper::attr(op, "axes").and_then(|d| d.as_array()) {
            axes = d_arr.iter().filter_map(|x| x.get("D").and_then(|d| d.as_i64())).collect();
        }
        if let Some(d_arr) = helper::attr(op, "decrease_axis").and_then(|d| d.as_array()) {
            decrease_axis = d_arr.iter().filter_map(|x| x.get("D").and_then(|d| d.as_i64())).collect();
        }

        let mut onnx_node = onnx::NodeProto {
            op_type: "Slice".to_string(),
            ..Default::default()
        };

        let inputs = helper::op_input_ids(op);
        for input_id in inputs.iter().take(3) {
            onnx_node.input.push(self.get_tensor_name(*input_id)?);
        }

        let axes_name = format!("slice_axes_{}", out_id);
        let mut axes_tensor = onnx::TensorProto {
            name: axes_name.clone(),
            dims: vec![axes.len() as i64],
            data_type: dt::INT64,
            raw_data: vec![],
            ..Default::default()
        };
        for &a in &axes {
            axes_tensor.raw_data.extend_from_slice(&a.to_le_bytes());
        }
        self.onnx_graph.initializer.push(axes_tensor);
        onnx_node.input.push(axes_name);

        let slice_out_name = if decrease_axis.is_empty() {
            self.get_tensor_name(out_id)?
        } else {
            format!("slice_out_{}", out_id)
        };
        onnx_node.output.push(slice_out_name.clone());
        self.onnx_graph.node.push(onnx_node);

        if !decrease_axis.is_empty() {
            let squeeze_axes_name = format!("squeeze_axes_{}", out_id);
            let mut squeeze_axes_tensor = onnx::TensorProto {
                name: squeeze_axes_name.clone(),
                dims: vec![decrease_axis.len() as i64],
                data_type: dt::INT64,
                raw_data: vec![],
                ..Default::default()
            };
            for &a in &decrease_axis {
                squeeze_axes_tensor.raw_data.extend_from_slice(&a.to_le_bytes());
            }
            self.onnx_graph.initializer.push(squeeze_axes_tensor);

            let squeeze_node = onnx::NodeProto {
                op_type: "Squeeze".to_string(),
                input: vec![slice_out_name, squeeze_axes_name],
                output: vec![self.get_tensor_name(out_id)?],
                ..Default::default()
            };
            self.onnx_graph.node.push(squeeze_node);
        }
        Ok(())
    }

    pub fn op_flatten(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let start_axis = helper::attr(op, "start_axis").and_then(|d| d.as_i64()).unwrap_or(1);
        let stop_axis = helper::attr(op, "stop_axis").and_then(|d| d.as_i64()).unwrap_or(-1);

        if stop_axis != -1 {
            bail!("flatten with explicit stop_axis={} not yet supported (only stop_axis=-1 is)", stop_axis);
        }
        
        let shape_name = format!("flatten_shape_{}", out_id);
        let mut shape_tensor = onnx::TensorProto {
            name: shape_name.clone(),
            dims: vec![start_axis + 1],
            data_type: dt::INT64,
            raw_data: vec![],
            ..Default::default()
        };
        for _ in 0..start_axis {
            shape_tensor.raw_data.extend_from_slice(&0i64.to_le_bytes());
        }
        shape_tensor.raw_data.extend_from_slice(&(-1i64).to_le_bytes());
        self.onnx_graph.initializer.push(shape_tensor);

        let mut reshape_node = onnx::NodeProto {
            op_type: "Reshape".to_string(),
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        };
        let inputs = helper::op_input_ids(op);
        if !inputs.is_empty() {
            reshape_node.input.push(self.get_tensor_name(inputs[0])?);
        }
        reshape_node.input.push(shape_name);
        self.onnx_graph.node.push(reshape_node);
        Ok(())
    }

    pub fn op_stack(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let axis = helper::attr(op, "axis").and_then(|d| d.as_i64()).unwrap_or(0);
        
        let inputs = helper::op_input_ids(op);
        let list_id = inputs.first().copied().unwrap_or(-1);
        let mut concat_inputs = Vec::new();
        
        let axes_name = format!("stack_axes_{}", out_id);
        let mut axes_tensor = onnx::TensorProto {
            name: axes_name.clone(),
            dims: vec![1],
            data_type: dt::INT64,
            raw_data: vec![],
            ..Default::default()
        };
        axes_tensor.raw_data.extend_from_slice(&axis.to_le_bytes());
        self.onnx_graph.initializer.push(axes_tensor);

        if let Some(tensors) = self.combines.get(&list_id) {
            for (idx, &t) in tensors.iter().enumerate() {
                let unsqueeze_out = format!("stack_unsqueezed_{}_{}", out_id, idx);
                let unsqueeze_node = onnx::NodeProto {
                    op_type: "Unsqueeze".to_string(),
                    input: vec![self.get_tensor_name(t)?, axes_name.clone()],
                    output: vec![unsqueeze_out.clone()],
                    ..Default::default()
                };
                self.onnx_graph.node.push(unsqueeze_node);
                concat_inputs.push(unsqueeze_out);
            }
        } else {
            bail!("Combine list_id {} not found for stack", list_id);
        }

        let mut concat_node = onnx::NodeProto {
            op_type: "Concat".to_string(),
            input: concat_inputs,
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        };
        concat_node.attribute.push(onnx::AttributeProto {
            name: "axis".to_string(),
            r#type: crate::helper::at::INT,
            i: axis,
            ..Default::default()
        });
        self.onnx_graph.node.push(concat_node);
        Ok(())
    }

    pub fn op_concat(&mut self, op: &Value) -> anyhow::Result<()> {
        let mut onnx_node = onnx::NodeProto {
            op_type: "Concat".to_string(),
            ..Default::default()
        };
        let inputs = helper::op_input_ids(op);
        if inputs.len() >= 2 {
            let list_id = inputs[0];
            let axis_id = inputs[1];
            if let Some(tensors) = self.combines.get(&list_id) {
                for &t in tensors {
                    onnx_node.input.push(self.get_tensor_name(t)?);
                }
            } else {
                bail!("Combine list_id {} not found for concat", list_id);
            }
            let axis = self.constants.get(&axis_id).and_then(|v| v.first()).copied().ok_or_else(|| anyhow::anyhow!("concat: missing axis constant for id {}", axis_id))? as i64;
            onnx_node.attribute.push(onnx::AttributeProto {
                name: "axis".to_string(),
                r#type: crate::helper::at::INT,
                i: axis,
                ..Default::default()
            });
        } else {
            bail!("Concat op missing enough inputs");
        }
        if let Ok(out_id) = helper::op_out_id(op) {
            onnx_node.output.push(self.get_tensor_name(out_id)?);
        }
        self.onnx_graph.node.push(onnx_node);
        Ok(())
    }
}
