use crate::converter::Converter;
use crate::helper::{self, dt};
use crate::proto::onnx;
use anyhow::bail;
use serde_json::Value;

impl Converter {
    pub fn op_flip(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let axes = helper::attr(op, "axis")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|value| {
                        value
                            .get("D")
                            .and_then(|v| v.as_i64())
                            .or_else(|| value.as_i64())
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let input_id = *helper::op_input_ids(op)
            .first()
            .ok_or_else(|| anyhow::anyhow!("flip missing input"))?;
        let input_shape = self
            .state
            .tensor_shapes
            .get(&input_id)
            .cloned()
            .or_else(|| {
                op.get("O")
                    .and_then(|o| o.as_array())
                    .and_then(|o| o.first())
                    .and_then(|o| o.get("TT"))
                    .and_then(|tt| tt.get("D"))
                    .and_then(|d| d.as_array())
                    .and_then(|tt| tt.get(1))
                    .and_then(|shape| shape.as_array())
                    .map(|dims| {
                        dims.iter()
                            .map(|dim| dim.as_i64().unwrap_or(-1))
                            .collect::<Vec<_>>()
                    })
            })
            .unwrap_or_default();
        let rank = input_shape.len() as i64;
        let normalized_axes = if axes.is_empty() {
            vec![0]
        } else {
            axes.into_iter()
                .map(|axis| if axis < 0 { axis + rank } else { axis })
                .collect::<Vec<_>>()
        };
        let axis_count = normalized_axes.len();
        let mut current_input = self.get_tensor_name(input_id)?;
        for (axis_index, axis) in normalized_axes.into_iter().enumerate() {
            let axis_len = input_shape.get(axis as usize).copied().unwrap_or(-1);
            if axis_len <= 0 {
                bail!("flip requires a static positive axis length");
            }

            let indices_name = format!("flip_indices_{}_{}", out_id, axis_index);
            let mut indices_tensor = onnx::TensorProto {
                name: indices_name.clone(),
                dims: vec![axis_len],
                data_type: dt::INT64,
                ..Default::default()
            };
            for idx in (0..axis_len).rev() {
                indices_tensor
                    .raw_data
                    .extend_from_slice(&idx.to_le_bytes());
            }
            self.onnx_graph.initializer.push(indices_tensor);

            let output_name = if axis_index + 1 == axis_count {
                self.get_tensor_name(out_id)?
            } else {
                format!("flip_axis_{}_{}", out_id, axis_index)
            };
            let mut node = onnx::NodeProto {
                op_type: "Gather".to_string(),
                input: vec![current_input, indices_name],
                output: vec![output_name.clone()],
                ..Default::default()
            };
            node.attribute.push(helper::attr_int("axis", axis));
            self.onnx_graph.node.push(node);
            current_input = output_name;
        }
        Ok(())
    }

    pub fn op_cumsum(&mut self, op: &Value) -> anyhow::Result<()> {
        self.require_opset(11, "cumsum")?;
        if helper::attr(op, "flatten")
            .and_then(|d| d.as_bool())
            .unwrap_or(false)
        {
            bail!("cumsum with flatten=true is not supported");
        }

        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("cumsum missing inputs");
        }
        let mut node = onnx::NodeProto {
            op_type: "CumSum".to_string(),
            input: vec![
                self.get_tensor_name(inputs[0])?,
                self.get_tensor_name(inputs[1])?,
            ],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        };
        for (name, enabled) in [
            (
                "exclusive",
                helper::attr(op, "exclusive")
                    .and_then(|d| d.as_bool())
                    .unwrap_or(false),
            ),
            (
                "reverse",
                helper::attr(op, "reverse")
                    .and_then(|d| d.as_bool())
                    .unwrap_or(false),
            ),
        ] {
            node.attribute
                .push(helper::attr_int(name, i64::from(enabled)));
        }
        self.onnx_graph.node.push(node);
        Ok(())
    }

    pub fn op_einsum(&mut self, op: &Value) -> anyhow::Result<()> {
        self.require_opset(12, "einsum")?;
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("einsum missing inputs");
        }

        let mut node_inputs = Vec::new();
        if let Some(expanded) = self.state.combines.get(&inputs[0]) {
            for &input_id in expanded {
                node_inputs.push(self.get_tensor_name(input_id)?);
            }
        } else {
            node_inputs.push(self.get_tensor_name(inputs[0])?);
        }

        let equation = helper::attr(op, "equation")
            .and_then(|d| d.as_str())
            .ok_or_else(|| anyhow::anyhow!("einsum missing equation"))?;

        let mut node = onnx::NodeProto {
            op_type: "Einsum".to_string(),
            input: node_inputs,
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        };
        node.attribute.push(helper::attr_str("equation", equation));
        self.onnx_graph.node.push(node);
        Ok(())
    }

    pub fn op_meshgrid(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("meshgrid missing inputs");
        }
        let input_ids = self
            .state
            .combines
            .get(&inputs[0])
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!("meshgrid missing 0.combine metadata for {}", inputs[0])
            })?;
        let output_ids =
            self.state.splits.get(&out_id).cloned().ok_or_else(|| {
                anyhow::anyhow!("meshgrid missing 0.split metadata for {}", out_id)
            })?;
        if input_ids.len() != output_ids.len() {
            bail!(
                "meshgrid expects matching input/output counts, got {} inputs and {} outputs",
                input_ids.len(),
                output_ids.len()
            );
        }
        if input_ids.is_empty() {
            bail!("meshgrid requires at least one input");
        }

        let grid_shape = format!("meshgrid_shape_{}", out_id);
        let mut shape_inputs = Vec::with_capacity(input_ids.len());
        for (index, input_id) in input_ids.iter().enumerate() {
            let shape_name = format!("meshgrid_shape_part_{}_{}", out_id, index);
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Shape".to_string(),
                input: vec![self.get_tensor_name(*input_id)?],
                output: vec![shape_name.clone()],
                ..Default::default()
            });
            shape_inputs.push(shape_name);
        }
        let mut concat = onnx::NodeProto {
            op_type: "Concat".to_string(),
            input: shape_inputs,
            output: vec![grid_shape.clone()],
            ..Default::default()
        };
        concat.attribute.push(helper::attr_int("axis", 0));
        self.onnx_graph.node.push(concat);

        let rank = input_ids.len() as i64;
        for (index, (&input_id, &output_id)) in input_ids.iter().zip(output_ids.iter()).enumerate()
        {
            let unsqueezed = format!("meshgrid_unsqueezed_{}_{}", out_id, index);
            let axes = (0..rank)
                .filter(|&axis| axis != index as i64)
                .collect::<Vec<_>>();
            self.add_unsqueeze_node(
                self.get_tensor_name(input_id)?,
                unsqueezed.clone(),
                &axes,
                format!("meshgrid_axes_{}_{}", out_id, index),
            );
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Expand".to_string(),
                input: vec![unsqueezed, grid_shape.clone()],
                output: vec![self.get_tensor_name(output_id)?],
                ..Default::default()
            });
        }
        Ok(())
    }
}
