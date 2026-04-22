use crate::converter::Converter;
use crate::helper::{self, dt};
use crate::proto::onnx;
use anyhow::bail;
use serde_json::Value;

impl Converter {
    pub fn op_tile(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("tile missing inputs");
        }

        let mut data_name = self.get_tensor_name(inputs[0])?;
        if matches!(self.state.tensor_shapes.get(&inputs[0]), Some(shape) if shape.is_empty()) {
            let unsqueezed_name = format!("tile_unsqueezed_{}", out_id);
            self.add_unsqueeze_node(
                data_name,
                unsqueezed_name.clone(),
                &[0],
                format!("tile_unsqueeze_axes_{}", out_id),
            );
            data_name = unsqueezed_name;
        }

        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Tile".to_string(),
            input: vec![data_name, self.get_tensor_name(inputs[1])?],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        });
        Ok(())
    }

    pub fn op_pad(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("pad missing inputs");
        }

        let paddings = helper::attr(op, "paddings")
            .and_then(|d| d.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| {
                        item.get("D")
                            .and_then(|v| v.as_i64())
                            .or_else(|| item.as_i64())
                    })
                    .collect::<Vec<_>>()
            })
            .ok_or_else(|| anyhow::anyhow!("pad: missing paddings"))?;
        if paddings.len() % 2 != 0 {
            bail!("pad: paddings must have even length");
        }

        let rank = paddings.len() / 2;
        let mut onnx_pads = Vec::with_capacity(paddings.len());
        for idx in 0..rank {
            onnx_pads.push(paddings[idx * 2]);
        }
        for idx in 0..rank {
            onnx_pads.push(paddings[idx * 2 + 1]);
        }

        let pads_name = format!("pad_pads_{}", out_id);
        let mut pads_tensor = onnx::TensorProto {
            name: pads_name.clone(),
            dims: vec![onnx_pads.len() as i64],
            data_type: dt::INT64,
            ..Default::default()
        };
        for pad in onnx_pads {
            pads_tensor.raw_data.extend_from_slice(&pad.to_le_bytes());
        }
        self.onnx_graph.initializer.push(pads_tensor);

        let mut node = onnx::NodeProto {
            op_type: "Pad".to_string(),
            input: vec![self.get_tensor_name(inputs[0])?, pads_name],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        };

        let mode = helper::attr(op, "mode")
            .or_else(|| helper::attr(op, "padding_mode"))
            .and_then(|d| d.as_str())
            .unwrap_or("constant");
        let onnx_mode = match mode {
            "reflect" => "reflect",
            "replicate" => "edge",
            _ => "constant",
        };
        if onnx_mode != "constant" {
            node.attribute.push(helper::attr_str("mode", onnx_mode));
        }

        if inputs.len() > 1 && inputs[1] != 0 {
            let mut value_name = self.get_tensor_name(inputs[1])?;
            if matches!(self.state.tensor_shapes.get(&inputs[1]), Some(shape) if shape == &vec![1])
            {
                let squeezed = format!("pad_value_{}", out_id);
                self.add_squeeze_node(
                    value_name,
                    squeezed.clone(),
                    Some(&[0]),
                    Some(format!("pad_value_axes_{}", out_id)),
                );
                value_name = squeezed;
            }
            node.input.push(value_name);
        }

        self.onnx_graph.node.push(node);
        Ok(())
    }

    pub fn op_pad3d(&mut self, op: &Value) -> anyhow::Result<()> {
        self.require_opset(11, "pad3d")?;

        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("pad3d missing paddings input");
        }

        let paddings_name = self.get_tensor_name(inputs[1])?;
        let data_format = helper::attr(op, "data_format")
            .and_then(|d| d.as_str())
            .unwrap_or("NCDHW");
        if !matches!(data_format, "NCDHW" | "NDHWC") {
            bail!("pad3d only supports NCDHW or NDHWC");
        }
        let zero_prefix_name = format!("pad3d_zero_prefix_{}", out_id);
        let mut zero_prefix = onnx::TensorProto {
            name: zero_prefix_name.clone(),
            dims: vec![2],
            data_type: dt::INT64,
            ..Default::default()
        };
        zero_prefix.raw_data.extend_from_slice(&0_i64.to_le_bytes());
        zero_prefix.raw_data.extend_from_slice(&0_i64.to_le_bytes());
        self.onnx_graph.initializer.push(zero_prefix);

        let mut slice_pad = |start: i64, end: i64, name: &str| -> anyhow::Result<String> {
            let output = format!("{}_{}", name, out_id);
            self.add_slice_node(
                paddings_name.clone(),
                output.clone(),
                &[start],
                &[end],
                Some(&[0]),
                None,
                &format!("{}_slice", output),
            )?;
            if matches!(
                self.state.tensor_types.get(&inputs[1]).map(String::as_str),
                Some(helper::paddle_tt::I64)
            ) {
                return Ok(output);
            }
            let cast_output = format!("{}_i64", output);
            self.add_cast_node(output, cast_output.clone(), dt::INT64);
            Ok(cast_output)
        };

        let w_begin = slice_pad(0, 1, "pad3d_w_begin")?;
        let w_end = slice_pad(1, 2, "pad3d_w_end")?;
        let h_begin = slice_pad(2, 3, "pad3d_h_begin")?;
        let h_end = slice_pad(3, 4, "pad3d_h_end")?;
        let d_begin = slice_pad(4, 5, "pad3d_d_begin")?;
        let d_end = slice_pad(5, 6, "pad3d_d_end")?;

        let starts_name = format!("pad3d_starts_{}", out_id);
        let ends_name = format!("pad3d_ends_{}", out_id);
        let onnx_pads_name = format!("pad3d_pads_{}", out_id);

        for (output, inputs) in [
            (starts_name.clone(), {
                if data_format == "NDHWC" {
                    vec![zero_prefix_name.clone(), h_begin, w_begin, d_begin]
                } else {
                    vec![zero_prefix_name.clone(), d_begin, h_begin, w_begin]
                }
            }),
            (ends_name.clone(), {
                if data_format == "NDHWC" {
                    vec![zero_prefix_name.clone(), h_end, w_end, d_end]
                } else {
                    vec![zero_prefix_name.clone(), d_end, h_end, w_end]
                }
            }),
            (
                onnx_pads_name.clone(),
                vec![starts_name.clone(), ends_name.clone()],
            ),
        ] {
            let mut node = onnx::NodeProto {
                op_type: "Concat".to_string(),
                input: inputs,
                output: vec![output],
                ..Default::default()
            };
            node.attribute.push(helper::attr_int("axis", 0));
            self.onnx_graph.node.push(node);
        }

        let mut node = onnx::NodeProto {
            op_type: "Pad".to_string(),
            input: vec![self.get_tensor_name(inputs[0])?, onnx_pads_name],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        };

        let mode = helper::attr(op, "mode")
            .and_then(|d| d.as_str())
            .unwrap_or("constant");
        let onnx_mode = match mode {
            "reflect" => "reflect",
            "replicate" => "edge",
            _ => "constant",
        };
        if onnx_mode != "constant" {
            node.attribute.push(helper::attr_str("mode", onnx_mode));
        } else if let Some(pad_value) = helper::attr(op, "pad_value").and_then(|d| d.as_f64())
            && pad_value != 0.0
        {
            let value_name = format!("pad3d_value_{}", out_id);
            let mut value = onnx::TensorProto {
                name: value_name.clone(),
                dims: vec![],
                data_type: dt::FLOAT,
                ..Default::default()
            };
            value
                .raw_data
                .extend_from_slice(&(pad_value as f32).to_le_bytes());
            self.onnx_graph.initializer.push(value);
            node.input.push(value_name);
        }

        self.onnx_graph.node.push(node);
        Ok(())
    }

    pub fn op_roll(&mut self, op: &Value) -> anyhow::Result<()> {
        self.require_opset(11, "roll")?;

        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("roll missing shifts input");
        }
        let axes = helper::attr(op, "axis")
            .and_then(|d| d.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| {
                        item.get("D")
                            .and_then(|v| v.as_i64())
                            .or_else(|| item.as_i64())
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let shifts = self
            .state
            .constants
            .get(&inputs[1])
            .map(|values| values.iter().map(|&value| value as i64).collect::<Vec<_>>())
            .ok_or_else(|| anyhow::anyhow!("roll currently requires constant shifts"))?;
        if !axes.is_empty() && shifts.len() != axes.len() {
            bail!("roll axes/shifts length mismatch");
        }

        let mut current_name = self.get_tensor_name(inputs[0])?;
        let rank = self
            .state
            .tensor_shapes
            .get(&inputs[0])
            .map(|shape| shape.len())
            .ok_or_else(|| anyhow::anyhow!("roll: missing rank metadata"))?;
        if axes.is_empty() {
            bail!("roll currently requires explicit axes");
        }

        for (index, (&axis_raw, &shift_raw)) in axes.iter().zip(shifts.iter()).enumerate() {
            let axis = if axis_raw < 0 {
                axis_raw + rank as i64
            } else {
                axis_raw
            };
            if axis < 0 || axis >= rank as i64 {
                bail!("roll axis {} out of range for rank {}", axis_raw, rank);
            }

            let dim = *self
                .state
                .tensor_shapes
                .get(&inputs[0])
                .and_then(|shape| shape.get(axis as usize))
                .ok_or_else(|| anyhow::anyhow!("roll: missing axis dim metadata"))?;
            if dim <= 0 {
                bail!("roll requires static positive axis dims");
            }

            let mut shift = shift_raw % dim;
            if shift < 0 {
                shift += dim;
            }
            if shift == 0 {
                continue;
            }

            let split = dim - shift;
            let tail_name = format!("roll_tail_{}_{}", out_id, index);
            let head_name = format!("roll_head_{}_{}", out_id, index);
            let concat_name = if index + 1 == axes.len() {
                self.get_tensor_name(out_id)?
            } else {
                format!("roll_axis_{}_{}", out_id, index)
            };

            self.add_slice_node(
                current_name.clone(),
                tail_name.clone(),
                &[split],
                &[dim],
                Some(&[axis]),
                None,
                &format!("roll_tail_{}_{}", out_id, index),
            )?;
            self.add_slice_node(
                current_name,
                head_name.clone(),
                &[0],
                &[split],
                Some(&[axis]),
                None,
                &format!("roll_head_{}_{}", out_id, index),
            )?;

            let mut concat = onnx::NodeProto {
                op_type: "Concat".to_string(),
                input: vec![tail_name, head_name],
                output: vec![concat_name.clone()],
                ..Default::default()
            };
            concat.attribute.push(helper::attr_int("axis", axis));
            self.onnx_graph.node.push(concat);
            current_name = concat_name;
        }

        if current_name != self.get_tensor_name(out_id)? {
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Identity".to_string(),
                input: vec![current_name],
                output: vec![self.get_tensor_name(out_id)?],
                ..Default::default()
            });
        }
        Ok(())
    }
}
