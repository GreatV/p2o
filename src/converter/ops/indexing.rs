use crate::converter::Converter;
use crate::helper::{self, dt};
use crate::proto::onnx;
use anyhow::bail;
use serde_json::Value;

impl Converter {
    pub fn op_gather(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("gather missing inputs");
        }
        let axis = if let Some(&axis_id) = inputs.get(2) {
            self.state
                .constants
                .get(&axis_id)
                .and_then(|vals| vals.first())
                .copied()
                .ok_or_else(|| anyhow::anyhow!("gather requires constant axis input"))?
                as i64
        } else {
            0
        };
        let mut node = onnx::NodeProto {
            op_type: "Gather".to_string(),
            input: vec![
                self.get_tensor_name(inputs[0])?,
                self.get_tensor_name(inputs[1])?,
            ],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        };
        node.attribute.push(helper::attr_int("axis", axis));
        self.onnx_graph.node.push(node);
        Ok(())
    }

    pub fn op_gather_nd(&mut self, op: &Value) -> anyhow::Result<()> {
        self.require_opset(11, "gather_nd")?;
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("gather_nd missing inputs");
        }
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "GatherND".to_string(),
            input: vec![
                self.get_tensor_name(inputs[0])?,
                self.get_tensor_name(inputs[1])?,
            ],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        });
        Ok(())
    }

    pub fn op_index_select(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("index_select missing inputs");
        }
        let axis = helper::attr(op, "axis")
            .and_then(|d| d.as_i64())
            .unwrap_or(0);
        let mut node = onnx::NodeProto {
            op_type: "Gather".to_string(),
            input: vec![
                self.get_tensor_name(inputs[0])?,
                self.get_tensor_name(inputs[1])?,
            ],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        };
        node.attribute.push(helper::attr_int("axis", axis));
        self.onnx_graph.node.push(node);
        Ok(())
    }

    pub fn op_embedding(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("embedding missing inputs");
        }
        let mut node = onnx::NodeProto {
            op_type: "Gather".to_string(),
            input: vec![
                self.get_tensor_name(inputs[1])?,
                self.get_tensor_name(inputs[0])?,
            ],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        };
        node.attribute.push(helper::attr_int("axis", 0));
        self.onnx_graph.node.push(node);
        Ok(())
    }

    pub fn op_eye(&mut self, op: &Value) -> anyhow::Result<()> {
        self.require_opset(11, "eye")?;
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("eye missing inputs");
        }

        let rows_name = self.ensure_scalar_i64_input(inputs[0], out_id, "eye_rows")?;
        let cols_name = self.ensure_scalar_i64_input(inputs[1], out_id, "eye_cols")?;

        let zero_name = format!("eye_zero_{}", out_id);
        let one_name = format!("eye_one_{}", out_id);
        for (name, value) in [(zero_name.clone(), 0_i64), (one_name.clone(), 1_i64)] {
            let mut tensor = onnx::TensorProto {
                name,
                dims: vec![],
                data_type: dt::INT64,
                ..Default::default()
            };
            tensor.raw_data.extend_from_slice(&value.to_le_bytes());
            self.onnx_graph.initializer.push(tensor);
        }

        let rows_range = format!("eye_rows_range_{}", out_id);
        let cols_range = format!("eye_cols_range_{}", out_id);
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Range".to_string(),
            input: vec![zero_name.clone(), rows_name, one_name.clone()],
            output: vec![rows_range.clone()],
            ..Default::default()
        });
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Range".to_string(),
            input: vec![zero_name, cols_name, one_name],
            output: vec![cols_range.clone()],
            ..Default::default()
        });

        let rows_unsqueezed = format!("eye_rows_unsqueezed_{}", out_id);
        let cols_unsqueezed = format!("eye_cols_unsqueezed_{}", out_id);
        self.add_unsqueeze_node(
            rows_range,
            rows_unsqueezed.clone(),
            &[1],
            format!("eye_rows_axes_{}", out_id),
        );
        self.add_unsqueeze_node(
            cols_range,
            cols_unsqueezed.clone(),
            &[0],
            format!("eye_cols_axes_{}", out_id),
        );

        let equal_name = format!("eye_equal_{}", out_id);
        self.add_binary_node(
            "Equal",
            rows_unsqueezed,
            cols_unsqueezed,
            equal_name.clone(),
        );

        let to = helper::attr(op, "dtype")
            .and_then(|d| d.as_str())
            .and_then(helper::paddle_dtype_to_onnx)
            .unwrap_or(dt::FLOAT);
        self.add_cast_node(equal_name, self.get_tensor_name(out_id)?, to);
        Ok(())
    }

    pub fn op_take_along_axis(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("take_along_axis missing inputs");
        }
        let mut node = onnx::NodeProto {
            op_type: "GatherElements".to_string(),
            input: vec![
                self.get_tensor_name(inputs[0])?,
                self.get_tensor_name(inputs[1])?,
            ],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        };
        node.attribute.push(helper::attr_int(
            "axis",
            helper::attr(op, "axis")
                .and_then(|d| d.as_i64())
                .unwrap_or(0),
        ));
        self.onnx_graph.node.push(node);
        Ok(())
    }

    pub fn op_not_equal(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("not_equal missing inputs");
        }
        let equal_out = format!("not_equal_eq_{}", out_id);
        self.add_binary_node(
            "Equal",
            self.get_tensor_name(inputs[0])?,
            self.get_tensor_name(inputs[1])?,
            equal_out.clone(),
        );
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Not".to_string(),
            input: vec![equal_out],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        });
        Ok(())
    }
}
