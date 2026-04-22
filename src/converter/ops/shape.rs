use crate::converter::Converter;
use crate::helper::{self, dt};
use crate::proto::onnx;
use anyhow::bail;
use serde_json::Value;

impl Converter {
    pub fn op_squeeze(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("squeeze missing inputs");
        }

        let input_id = inputs[0];
        let input_name = self.get_tensor_name(input_id)?;
        let output_name = self.get_tensor_name(out_id)?;
        let input_shape = self
            .state
            .tensor_shapes
            .get(&input_id)
            .cloned()
            .unwrap_or_default();
        let output_shape = self
            .state
            .tensor_shapes
            .get(&out_id)
            .cloned()
            .unwrap_or_default();
        let input_rank = input_shape.len() as i64;
        let output_rank = output_shape.len() as i64;

        if !input_shape.is_empty() && !output_shape.is_empty() && input_rank == output_rank {
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Identity".to_string(),
                input: vec![input_name],
                output: vec![output_name],
                ..Default::default()
            });
            return Ok(());
        }

        let axes_from_attr = helper::attr(op, "axis")
            .or_else(|| helper::attr(op, "axes"))
            .and_then(|value| {
                if let Some(axis) = value.as_i64() {
                    Some(vec![axis])
                } else {
                    value.as_array().map(|items| {
                        items
                            .iter()
                            .filter_map(|item| {
                                item.get("D")
                                    .and_then(|v| v.as_i64())
                                    .or_else(|| item.as_i64())
                            })
                            .collect::<Vec<_>>()
                    })
                }
            });

        let axes_from_input = inputs.get(1).and_then(|axis_id| {
            self.state
                .constants
                .get(axis_id)
                .map(|values| values.iter().map(|value| *value as i64).collect::<Vec<_>>())
        });

        let mut effective_axes = axes_from_input.or(axes_from_attr);
        if let Some(axes) = effective_axes.as_mut()
            && input_rank > 0
        {
            for axis in axes.iter_mut() {
                if *axis < 0 {
                    *axis += input_rank;
                }
            }
            axes.retain(|axis| *axis >= 0 && *axis < input_rank);
            axes.sort_unstable();
            axes.dedup();

            let expected_removed = if !output_shape.is_empty() && output_rank <= input_rank {
                (input_rank - output_rank) as usize
            } else {
                0
            };

            let mut filtered = Vec::with_capacity(axes.len());
            for &axis in axes.iter() {
                let dim = input_shape[axis as usize];
                if dim == 1 || dim < 0 {
                    filtered.push(axis);
                }
            }

            if expected_removed == 0 && !input_shape.is_empty() && !output_shape.is_empty() {
                filtered.clear();
            }
            *axes = filtered;
        }

        if matches!(effective_axes.as_ref(), Some(axes) if axes.is_empty()) {
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Identity".to_string(),
                input: vec![input_name],
                output: vec![output_name],
                ..Default::default()
            });
            return Ok(());
        }

        if let Some(axes) = effective_axes {
            self.add_squeeze_node(
                input_name,
                output_name,
                Some(&axes),
                Some(format!("squeeze_axes_{}", out_id)),
            );
        } else if let Some(axis_input_id) = inputs.get(1) {
            if self.target_opset < 13 {
                bail!("squeeze with dynamic axes requires opset >= 13");
            }
            let mut node = onnx::NodeProto {
                op_type: "Squeeze".to_string(),
                input: vec![input_name],
                output: vec![output_name],
                ..Default::default()
            };
            if self.target_opset >= 13 {
                node.input.push(self.get_tensor_name(*axis_input_id)?);
            }
            self.onnx_graph.node.push(node);
        } else {
            self.add_squeeze_node(input_name, output_name, None, None);
        }
        Ok(())
    }

    pub fn op_unsqueeze(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("unsqueeze missing inputs");
        }

        let input_id = inputs[0];
        let input_name = self.get_tensor_name(input_id)?;
        let output_name = self.get_tensor_name(out_id)?;
        let input_rank = self
            .state
            .tensor_shapes
            .get(&input_id)
            .map(|shape| shape.len() as i64)
            .unwrap_or(0);
        let output_rank = self
            .state
            .tensor_shapes
            .get(&out_id)
            .map(|shape| shape.len() as i64)
            .unwrap_or(input_rank + 1);

        let axes_from_attr = helper::attr(op, "axis")
            .or_else(|| helper::attr(op, "axes"))
            .and_then(|value| {
                if let Some(axis) = value.as_i64() {
                    Some(vec![axis])
                } else {
                    value.as_array().map(|items| {
                        items
                            .iter()
                            .filter_map(|item| {
                                item.get("D")
                                    .and_then(|v| v.as_i64())
                                    .or_else(|| item.as_i64())
                            })
                            .collect::<Vec<_>>()
                    })
                }
            });
        let axes_from_input = inputs.get(1).and_then(|axis_id| {
            self.state
                .constants
                .get(axis_id)
                .map(|values| values.iter().map(|value| *value as i64).collect::<Vec<_>>())
        });

        let mut effective_axes = axes_from_input.or(axes_from_attr);
        if let Some(axes) = effective_axes.as_mut() {
            for axis in axes.iter_mut() {
                if *axis < 0 {
                    *axis += output_rank;
                }
            }
            axes.sort_unstable();
            axes.dedup();
        }

        if let Some(axes) = effective_axes {
            self.add_unsqueeze_node(
                input_name,
                output_name,
                &axes,
                format!("unsqueeze_axes_{}", out_id),
            );
        } else if let Some(axis_input_id) = inputs.get(1) {
            if self.target_opset < 13 {
                bail!("unsqueeze with dynamic axes requires opset >= 13");
            }
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Unsqueeze".to_string(),
                input: vec![input_name, self.get_tensor_name(*axis_input_id)?],
                output: vec![output_name],
                ..Default::default()
            });
        } else {
            bail!("unsqueeze missing axes");
        }
        Ok(())
    }

    pub fn op_flatten(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("flatten missing inputs");
        }
        let input_id = inputs[0];
        let input_name = self.get_tensor_name(input_id)?;
        let input_rank = self
            .state
            .tensor_shapes
            .get(&input_id)
            .map(|dims| dims.len() as i64);
        let raw_start_axis = helper::attr(op, "start_axis")
            .and_then(|d| d.as_i64())
            .unwrap_or(1);
        let stop_axis = helper::attr(op, "stop_axis")
            .and_then(|d| d.as_i64())
            .unwrap_or(-1);
        let start_axis = match input_rank {
            Some(rank) if raw_start_axis < 0 => raw_start_axis + rank,
            _ => raw_start_axis,
        };
        let target_shape: Vec<i64> = op
            .get("O")
            .and_then(|o| o.as_array())
            .and_then(|o| o.first())
            .and_then(|o| o.get("TT"))
            .and_then(|tt| tt.get("D"))
            .and_then(|d| d.as_array())
            .and_then(|tt| tt.get(1))
            .and_then(|shape| shape.as_array())
            .map(|dims| dims.iter().filter_map(|dim| dim.as_i64()).collect())
            .unwrap_or_default();

        let multiple_unknown_dims = target_shape.iter().filter(|&&dim| dim == -1).count() > 1;
        if stop_axis != -1 && multiple_unknown_dims {
            let rank = input_rank
                .ok_or_else(|| anyhow::anyhow!("flatten: missing input rank metadata"))?;
            let normalized_stop_axis = if stop_axis < 0 {
                stop_axis + rank
            } else {
                stop_axis
            };
            if start_axis < 0 || normalized_stop_axis < start_axis || normalized_stop_axis >= rank {
                bail!(
                    "flatten: invalid axes start_axis={} stop_axis={} for rank {}",
                    start_axis,
                    normalized_stop_axis,
                    rank
                );
            }

            let shape_out = format!("flatten_input_shape_{}", out_id);
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Shape".to_string(),
                input: vec![input_name.clone()],
                output: vec![shape_out.clone()],
                ..Default::default()
            });

            let mut shape_parts = Vec::new();
            if start_axis > 0 {
                let prefix_out = format!("flatten_prefix_{}", out_id);
                self.add_slice_node(
                    shape_out.clone(),
                    prefix_out.clone(),
                    &[0],
                    &[start_axis],
                    Some(&[0]),
                    None,
                    &format!("flatten_prefix_{}", out_id),
                )?;
                shape_parts.push(prefix_out);
            }

            let middle_out = format!("flatten_middle_{}", out_id);
            let prod_out = format!("flatten_prod_{}", out_id);
            self.add_slice_node(
                shape_out.clone(),
                middle_out.clone(),
                &[start_axis],
                &[normalized_stop_axis + 1],
                Some(&[0]),
                None,
                &format!("flatten_middle_{}", out_id),
            )?;

            self.add_reduce_node(
                "ReduceProd",
                middle_out,
                prod_out.clone(),
                Some(&[0]),
                1,
                &format!("flatten_prod_{}", out_id),
            );
            shape_parts.push(prod_out);

            if normalized_stop_axis + 1 < rank {
                let suffix_out = format!("flatten_suffix_{}", out_id);
                self.add_slice_node(
                    shape_out.clone(),
                    suffix_out.clone(),
                    &[normalized_stop_axis + 1],
                    &[rank],
                    Some(&[0]),
                    None,
                    &format!("flatten_suffix_{}", out_id),
                )?;
                shape_parts.push(suffix_out);
            }

            let reshape_shape = if shape_parts.len() == 1 {
                shape_parts[0].clone()
            } else {
                let concat_out = format!("flatten_shape_{}", out_id);
                let mut concat_node = onnx::NodeProto {
                    op_type: "Concat".to_string(),
                    input: shape_parts,
                    output: vec![concat_out.clone()],
                    ..Default::default()
                };
                concat_node.attribute.push(helper::attr_int("axis", 0));
                self.onnx_graph.node.push(concat_node);
                concat_out
            };

            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Reshape".to_string(),
                input: vec![input_name, reshape_shape],
                output: vec![self.get_tensor_name(out_id)?],
                ..Default::default()
            });
            return Ok(());
        }

        if target_shape.is_empty() && stop_axis != -1 {
            bail!(
                "flatten with explicit stop_axis={} is missing output shape metadata",
                stop_axis
            );
        }

        let shape_name = format!("flatten_shape_{}", out_id);
        let mut shape_tensor = onnx::TensorProto {
            name: shape_name.clone(),
            dims: vec![],
            data_type: dt::INT64,
            raw_data: vec![],
            ..Default::default()
        };

        if !target_shape.is_empty() {
            shape_tensor.dims = vec![target_shape.len() as i64];
            for dim in target_shape {
                shape_tensor.raw_data.extend_from_slice(&dim.to_le_bytes());
            }
        } else {
            shape_tensor.dims = vec![start_axis + 1];
            for _ in 0..start_axis {
                shape_tensor.raw_data.extend_from_slice(&0i64.to_le_bytes());
            }
            shape_tensor
                .raw_data
                .extend_from_slice(&(-1i64).to_le_bytes());
        }
        self.onnx_graph.initializer.push(shape_tensor);

        let mut reshape_node = onnx::NodeProto {
            op_type: "Reshape".to_string(),
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        };
        reshape_node.input.push(input_name);
        reshape_node.input.push(shape_name);
        self.onnx_graph.node.push(reshape_node);
        Ok(())
    }

    pub fn op_stack(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let axis = helper::attr(op, "axis")
            .and_then(|d| d.as_i64())
            .unwrap_or(0);

        let inputs = helper::op_input_ids(op);
        let list_id = inputs.first().copied().unwrap_or(-1);
        let mut concat_inputs = Vec::new();

        if let Some(tensors) = self.state.combines.get(&list_id).cloned() {
            self.state.stack_parts.insert(out_id, tensors.clone());
            for (idx, t) in tensors.into_iter().enumerate() {
                let unsqueeze_out = format!("stack_unsqueezed_{}_{}", out_id, idx);
                self.add_unsqueeze_node(
                    self.get_tensor_name(t)?,
                    unsqueeze_out.clone(),
                    &[axis],
                    format!("stack_axes_{}_{}", out_id, idx),
                );
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
        concat_node.attribute.push(helper::attr_int("axis", axis));
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
            if let Some(tensors) = self.state.combines.get(&list_id) {
                for &t in tensors {
                    onnx_node.input.push(self.get_tensor_name(t)?);
                }
            } else {
                bail!("Combine list_id {} not found for concat", list_id);
            }
            let axis = self
                .state
                .constants
                .get(&axis_id)
                .and_then(|v| v.first())
                .copied()
                .ok_or_else(|| {
                    anyhow::anyhow!("concat: missing axis constant for id {}", axis_id)
                })? as i64;
            onnx_node.attribute.push(helper::attr_int("axis", axis));
        } else {
            bail!("Concat op missing enough inputs");
        }
        let out_id = helper::op_out_id(op)?;
        onnx_node.output.push(self.get_tensor_name(out_id)?);
        self.onnx_graph.node.push(onnx_node);
        Ok(())
    }

    pub fn op_split_family(&mut self, op_type: &str, op: &Value) -> anyhow::Result<()> {
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("split op missing inputs");
        }

        let vec_out_id = helper::op_out_id(op)?;
        let split_out_ids = self.state.splits.get(&vec_out_id).cloned().ok_or_else(|| {
            anyhow::anyhow!("split op missing 0.split metadata for {}", vec_out_id)
        })?;

        let mut node = onnx::NodeProto {
            op_type: "Split".to_string(),
            input: vec![self.get_tensor_name(inputs[0])?],
            output: split_out_ids
                .into_iter()
                .map(|id| self.get_tensor_name(id))
                .collect::<anyhow::Result<Vec<_>>>()?,
            ..Default::default()
        };

        let axis = if op_type == helper::paddle_op::SPLIT {
            if inputs.len() > 1 {
                node.input.push(self.get_tensor_name(inputs[1])?);
            }
            inputs
                .get(2)
                .and_then(|id| self.state.constants.get(id))
                .and_then(|vals| vals.first())
                .copied()
                .unwrap_or(0.0) as i64
        } else {
            inputs
                .get(1)
                .and_then(|id| self.state.constants.get(id))
                .and_then(|vals| vals.first())
                .copied()
                .unwrap_or(0.0) as i64
        };

        node.attribute.push(helper::attr_int("axis", axis));
        if self.target_opset >= 18 && node.input.len() == 1 {
            node.attribute
                .push(helper::attr_int("num_outputs", node.output.len() as i64));
        }
        self.onnx_graph.node.push(node);
        Ok(())
    }
}
