use crate::converter::Converter;
use crate::helper::{self, dt};
use crate::proto::onnx;
use anyhow::bail;
use serde_json::Value;

impl Converter {
    pub fn op_index_put(&mut self, op: &Value) -> anyhow::Result<()> {
        self.require_opset(11, "index_put")?;
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 3 {
            bail!("index_put missing inputs");
        }
        let mask_ids = self
            .state
            .combines
            .get(&inputs[1])
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("index_put expects combined mask input"))?;
        let data_shape = self
            .state
            .tensor_shapes
            .get(&inputs[0])
            .cloned()
            .unwrap_or_default();
        let value_shape = self
            .state
            .tensor_shapes
            .get(&inputs[2])
            .cloned()
            .unwrap_or_default();

        if mask_ids.len() == 1 {
            let mask_shape = self
                .state
                .tensor_shapes
                .get(&mask_ids[0])
                .cloned()
                .unwrap_or_default();
            let mask_type = self
                .state
                .tensor_types
                .get(&mask_ids[0])
                .cloned()
                .unwrap_or_default();

            if mask_type == helper::paddle_tt::BOOL && mask_shape == data_shape {
                let shape_name = format!("index_put_shape_{}", out_id);
                self.onnx_graph.node.push(onnx::NodeProto {
                    op_type: "Shape".to_string(),
                    input: vec![self.get_tensor_name(inputs[0])?],
                    output: vec![shape_name.clone()],
                    ..Default::default()
                });

                let expanded_value = format!("index_put_value_{}", out_id);
                self.onnx_graph.node.push(onnx::NodeProto {
                    op_type: "Expand".to_string(),
                    input: vec![self.get_tensor_name(inputs[2])?, shape_name],
                    output: vec![expanded_value.clone()],
                    ..Default::default()
                });

                self.onnx_graph.node.push(onnx::NodeProto {
                    op_type: "Where".to_string(),
                    input: vec![
                        self.get_tensor_name(mask_ids[0])?,
                        expanded_value,
                        self.get_tensor_name(inputs[0])?,
                    ],
                    output: vec![self.get_tensor_name(out_id)?],
                    ..Default::default()
                });
                return Ok(());
            }

            if mask_shape.len() == 1 && data_shape.len() == 2 && value_shape.len() <= 1 {
                let indices_unsqueezed = format!("index_put_indices_unsqueezed_{}", out_id);
                let axis_name = format!("index_put_axis_{}", out_id);
                self.add_unsqueeze_node(
                    self.get_tensor_name(mask_ids[0])?,
                    indices_unsqueezed.clone(),
                    &[0],
                    axis_name,
                );

                let shape_name = format!("index_put_shape_{}", out_id);
                self.onnx_graph.node.push(onnx::NodeProto {
                    op_type: "Shape".to_string(),
                    input: vec![indices_unsqueezed.clone()],
                    output: vec![shape_name.clone()],
                    ..Default::default()
                });

                let expanded_value = format!("index_put_value_{}", out_id);
                self.onnx_graph.node.push(onnx::NodeProto {
                    op_type: "Expand".to_string(),
                    input: vec![self.get_tensor_name(inputs[2])?, shape_name],
                    output: vec![expanded_value.clone()],
                    ..Default::default()
                });

                let mut scatter = onnx::NodeProto {
                    op_type: "ScatterElements".to_string(),
                    input: vec![
                        self.get_tensor_name(inputs[0])?,
                        indices_unsqueezed,
                        expanded_value,
                    ],
                    output: vec![self.get_tensor_name(out_id)?],
                    ..Default::default()
                };
                scatter.attribute.push(helper::attr_int("axis", 0));
                self.onnx_graph.node.push(scatter);
                return Ok(());
            }
        }

        if mask_ids.len() == 2 && data_shape.len() == 2 && value_shape == data_shape {
            let axis_name = format!("index_put_stack_axis_{}", out_id);
            let row_unsqueezed = format!("index_put_row_unsqueezed_{}", out_id);
            let col_unsqueezed = format!("index_put_col_unsqueezed_{}", out_id);
            for (input_id, output_name) in [
                (mask_ids[0], row_unsqueezed.clone()),
                (mask_ids[1], col_unsqueezed.clone()),
            ] {
                self.add_unsqueeze_node(
                    self.get_tensor_name(input_id)?,
                    output_name,
                    &[-1],
                    axis_name.clone(),
                );
            }

            let indices_name = format!("index_put_indices_{}", out_id);
            let mut concat = onnx::NodeProto {
                op_type: "Concat".to_string(),
                input: vec![row_unsqueezed, col_unsqueezed],
                output: vec![indices_name.clone()],
                ..Default::default()
            };
            concat.attribute.push(helper::attr_int("axis", -1));
            self.onnx_graph.node.push(concat);

            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "ScatterND".to_string(),
                input: vec![
                    self.get_tensor_name(inputs[0])?,
                    indices_name,
                    self.get_tensor_name(inputs[2])?,
                ],
                output: vec![self.get_tensor_name(out_id)?],
                ..Default::default()
            });
            return Ok(());
        }

        bail!("unsupported index_put pattern for output {}", out_id)
    }

    /// Supported `set_value_` patterns:
    /// - `axes=[1]`: constant scalar writes over a contiguous or strided rank-2 slice
    /// - `axes=[1,2]`: constant scalar writes over a static rectangular block
    pub fn op_set_value(&mut self, op: &Value) -> anyhow::Result<()> {
        self.require_opset(11, "set_value_")?;
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 4 {
            bail!("set_value_ missing inputs");
        }
        let axes = helper::attr(op, "axes")
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
        let values = helper::attr(op, "values")
            .and_then(|d| d.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.get("D").and_then(|d| d.as_f64()))
            .unwrap_or(0.0);

        if axes == vec![1] {
            let start = self
                .state
                .constants
                .get(&inputs[1])
                .and_then(|v| v.first())
                .copied()
                .unwrap_or(0.0) as i64;
            let end = self
                .state
                .constants
                .get(&inputs[2])
                .and_then(|v| v.first())
                .copied()
                .unwrap_or((start + 1) as f64) as i64;
            let step = self
                .state
                .constants
                .get(&inputs[3])
                .and_then(|v| v.first())
                .copied()
                .unwrap_or(1.0) as i64;
            if step == 0 {
                bail!("set_value_ step cannot be zero");
            }
            if step < 0 {
                bail!("set_value_ axes=[1] currently only supports positive step");
            }
            if start < 0 || end < 0 {
                bail!(
                    "set_value_ axes=[1] currently requires non-negative start/end (negative indices not yet supported)"
                );
            }
            let indices: Vec<i64> = (start..end).step_by(step as usize).collect();
            if indices.is_empty() {
                self.onnx_graph.node.push(onnx::NodeProto {
                    op_type: "Identity".to_string(),
                    input: vec![self.get_tensor_name(inputs[0])?],
                    output: vec![self.get_tensor_name(out_id)?],
                    ..Default::default()
                });
                return Ok(());
            }
            let axis_len = indices.len() as i64;

            let shape_name = format!("set_value_shape_{}", out_id);
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Shape".to_string(),
                input: vec![self.get_tensor_name(inputs[0])?],
                output: vec![shape_name.clone()],
                ..Default::default()
            });

            let batch_axis_name = format!("set_value_batch_axis_{}", out_id);
            self.push_i64_initializer(batch_axis_name.clone(), vec![1], &[0]);

            let batch_dim_name = format!("set_value_batch_dim_{}", out_id);
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Gather".to_string(),
                input: vec![shape_name.clone(), batch_axis_name],
                output: vec![batch_dim_name.clone()],
                ..Default::default()
            });

            let slice_len_name = format!("set_value_slice_len_{}", out_id);
            self.push_i64_initializer(slice_len_name.clone(), vec![1], &[axis_len]);

            let update_shape_name = format!("set_value_update_shape_{}", out_id);
            let mut concat_shape = onnx::NodeProto {
                op_type: "Concat".to_string(),
                input: vec![batch_dim_name, slice_len_name.clone()],
                output: vec![update_shape_name.clone()],
                ..Default::default()
            };
            concat_shape.attribute.push(helper::attr_int("axis", 0));
            self.onnx_graph.node.push(concat_shape);

            let indices_name = format!("set_value_indices_{}", out_id);
            let mut indices_tensor = onnx::TensorProto {
                name: indices_name.clone(),
                dims: vec![1, axis_len],
                data_type: dt::INT64,
                ..Default::default()
            };
            for &idx in &indices {
                indices_tensor
                    .raw_data
                    .extend_from_slice(&idx.to_le_bytes());
            }
            self.onnx_graph.initializer.push(indices_tensor);

            let expanded_indices_name = format!("set_value_indices_expanded_{}", out_id);
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Expand".to_string(),
                input: vec![indices_name, update_shape_name.clone()],
                output: vec![expanded_indices_name.clone()],
                ..Default::default()
            });

            let updates_name = format!("set_value_updates_{}", out_id);
            let updates_dtype = self
                .maybe_onnx_dtype_for_tensor_id(inputs[0])?
                .unwrap_or(dt::FLOAT);
            self.push_numeric_initializer(updates_name.clone(), vec![1], updates_dtype, &[values])?;

            let expanded_updates_name = format!("set_value_updates_expanded_{}", out_id);
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Expand".to_string(),
                input: vec![updates_name, update_shape_name],
                output: vec![expanded_updates_name.clone()],
                ..Default::default()
            });

            let mut scatter = onnx::NodeProto {
                op_type: "ScatterElements".to_string(),
                input: vec![
                    self.get_tensor_name(inputs[0])?,
                    expanded_indices_name,
                    expanded_updates_name,
                ],
                output: vec![self.get_tensor_name(out_id)?],
                ..Default::default()
            };
            scatter.attribute.push(helper::attr_int("axis", 1));
            self.onnx_graph.node.push(scatter);
            return Ok(());
        }

        if axes == vec![1, 2] {
            let data_shape = self
                .state
                .tensor_shapes
                .get(&inputs[0])
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("set_value_: missing shape metadata"))?;
            if data_shape.iter().any(|&dim| dim <= 0) {
                bail!("set_value_ axes=[1,2] requires static positive dims");
            }

            let starts = self
                .state
                .constants
                .get(&inputs[1])
                .map(|v| v.iter().map(|&x| x as i64).collect::<Vec<_>>())
                .ok_or_else(|| anyhow::anyhow!("set_value_: missing constant starts"))?;
            let ends = self
                .state
                .constants
                .get(&inputs[2])
                .map(|v| v.iter().map(|&x| x as i64).collect::<Vec<_>>())
                .ok_or_else(|| anyhow::anyhow!("set_value_: missing constant ends"))?;
            let steps = self
                .state
                .constants
                .get(&inputs[3])
                .map(|v| v.iter().map(|&x| x as i64).collect::<Vec<_>>())
                .ok_or_else(|| anyhow::anyhow!("set_value_: missing constant steps"))?;
            if starts.len() != axes.len() || ends.len() != axes.len() || steps.len() != axes.len() {
                bail!("set_value_ axes=[1,2] expects starts/ends/steps to match axes rank");
            }

            let build_positions =
                |dim: i64, start: i64, end: i64, step: i64| -> anyhow::Result<Vec<i64>> {
                    if step == 0 {
                        bail!("set_value_ step cannot be zero");
                    }
                    let mut start = start;
                    let mut end = end;
                    if start < 0 {
                        start += dim;
                    }
                    if end < 0 {
                        end += dim;
                    }

                    let mut positions = Vec::new();
                    if step > 0 {
                        let start = start.clamp(0, dim);
                        let end = end.clamp(0, dim);
                        let mut idx = start;
                        while idx < end {
                            positions.push(idx);
                            idx += step;
                        }
                    } else {
                        let start = start.clamp(-1, dim - 1);
                        let end = end.clamp(-1, dim - 1);
                        let mut idx = start;
                        while idx > end {
                            positions.push(idx);
                            idx += step;
                        }
                    }
                    Ok(positions)
                };

            let rank = data_shape.len();
            let axis_positions = axes
                .iter()
                .enumerate()
                .map(|(i, &axis)| {
                    let axis = if axis < 0 { axis + rank as i64 } else { axis } as usize;
                    if axis >= rank {
                        bail!("set_value_: axis {} out of range for rank {}", axis, rank);
                    }
                    Ok((
                        axis,
                        build_positions(data_shape[axis], starts[i], ends[i], steps[i])?,
                    ))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            let per_axis_positions = (0..rank)
                .map(|dim| {
                    axis_positions
                        .iter()
                        .find(|(axis, _)| *axis == dim)
                        .map(|(_, positions)| positions.clone())
                        .unwrap_or_else(|| (0..data_shape[dim]).collect())
                })
                .collect::<Vec<_>>();
            let update_count = per_axis_positions.iter().map(Vec::len).product::<usize>();
            if update_count == 0 {
                self.onnx_graph.node.push(onnx::NodeProto {
                    op_type: "Identity".to_string(),
                    input: vec![self.get_tensor_name(inputs[0])?],
                    output: vec![self.get_tensor_name(out_id)?],
                    ..Default::default()
                });
                return Ok(());
            }

            let indices_name = format!("set_value_indices_{}", out_id);
            let mut indices_tensor = onnx::TensorProto {
                name: indices_name.clone(),
                dims: vec![update_count as i64, rank as i64],
                data_type: dt::INT64,
                ..Default::default()
            };
            let mut cursor = vec![0usize; rank];
            loop {
                for dim in 0..rank {
                    indices_tensor
                        .raw_data
                        .extend_from_slice(&per_axis_positions[dim][cursor[dim]].to_le_bytes());
                }
                let mut carry = true;
                for dim in (0..rank).rev() {
                    if carry && cursor[dim] + 1 < per_axis_positions[dim].len() {
                        cursor[dim] += 1;
                        for item in cursor.iter_mut().skip(dim + 1) {
                            *item = 0;
                        }
                        carry = false;
                        break;
                    }
                }
                if carry {
                    break;
                }
            }
            self.onnx_graph.initializer.push(indices_tensor);

            let updates_name = format!("set_value_updates_{}", out_id);
            let data_type = self
                .maybe_onnx_dtype_for_tensor_id(inputs[0])?
                .unwrap_or(dt::FLOAT);
            self.push_numeric_initializer(
                updates_name.clone(),
                vec![update_count as i64],
                data_type,
                &vec![values; update_count],
            )?;

            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "ScatterND".to_string(),
                input: vec![self.get_tensor_name(inputs[0])?, indices_name, updates_name],
                output: vec![self.get_tensor_name(out_id)?],
                ..Default::default()
            });
            return Ok(());
        }

        bail!("set_value_ currently only supports axes=[1] or axes=[1,2] constant block writes");
    }

    pub fn op_set_value_with_tensor(&mut self, op: &Value) -> anyhow::Result<()> {
        self.require_opset(11, "set_value_with_tensor_")?;
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("set_value_with_tensor_ missing inputs");
        }
        let axes = helper::attr(op, "axes")
            .and_then(|d| d.as_array())
            .cloned()
            .unwrap_or_default();
        if axes.is_empty() {
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Identity".to_string(),
                input: vec![self.get_tensor_name(inputs[1])?],
                output: vec![self.get_tensor_name(out_id)?],
                ..Default::default()
            });
            return Ok(());
        }
        let decrease_axes = helper::attr(op, "decrease_axes")
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
        if axes.len() != 1 || axes[0].get("D").and_then(|v| v.as_i64()) != Some(1) {
            bail!("set_value_with_tensor_ currently only supports axes=[1]");
        }
        if !matches!(decrease_axes.as_slice(), [] | [1]) {
            bail!("set_value_with_tensor_ currently only supports decrease_axes=[] or [1]");
        }
        if inputs.len() < 5 {
            bail!("set_value_with_tensor_ axes=[1] expects data, values, starts, ends, steps");
        }
        let data_rank = self
            .state
            .tensor_shapes
            .get(&inputs[0])
            .map(|shape| shape.len())
            .ok_or_else(|| anyhow::anyhow!("set_value_with_tensor_: missing rank metadata"))?;
        if data_rank < 2 {
            bail!("set_value_with_tensor_ axes=[1] requires rank >= 2 inputs");
        }

        let data_name = self.get_tensor_name(inputs[0])?;
        let values_name = self.get_tensor_name(inputs[1])?;
        let start_name =
            self.ensure_scalar_i64_input(inputs[2], out_id, "set_value_tensor_start")?;
        let end_name = self.ensure_scalar_i64_input(inputs[3], out_id, "set_value_tensor_end")?;
        let step_name = self.ensure_scalar_i64_input(inputs[4], out_id, "set_value_tensor_step")?;
        let range_name = format!("set_value_tensor_range_{}", out_id);
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Range".to_string(),
            input: vec![start_name, end_name, step_name],
            output: vec![range_name.clone()],
            ..Default::default()
        });

        let (updates_name, updates_shape_name) = if decrease_axes.is_empty() {
            let updates_shape_name = format!("set_value_tensor_updates_shape_{}", out_id);
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Shape".to_string(),
                input: vec![values_name.clone()],
                output: vec![updates_shape_name.clone()],
                ..Default::default()
            });
            (values_name, updates_shape_name)
        } else {
            let expanded_updates_name = format!("set_value_tensor_updates_{}", out_id);
            self.add_unsqueeze_node(
                values_name,
                expanded_updates_name.clone(),
                &[1],
                format!("set_value_tensor_unsqueeze_axes_{}", out_id),
            );
            let updates_shape_name = format!("set_value_tensor_updates_shape_{}", out_id);
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Shape".to_string(),
                input: vec![expanded_updates_name.clone()],
                output: vec![updates_shape_name.clone()],
                ..Default::default()
            });
            if let (Some(start), Some(end), Some(step)) = (
                self.state
                    .constants
                    .get(&inputs[2])
                    .and_then(|values| values.first())
                    .copied(),
                self.state
                    .constants
                    .get(&inputs[3])
                    .and_then(|values| values.first())
                    .copied(),
                self.state
                    .constants
                    .get(&inputs[4])
                    .and_then(|values| values.first())
                    .copied(),
            ) {
                let start = start as i64;
                let end = end as i64;
                let step = step as i64;
                if step == 0 {
                    bail!("set_value_with_tensor_ step cannot be zero");
                }
                let len = if step > 0 {
                    if end <= start {
                        0
                    } else {
                        (end - start + step - 1) / step
                    }
                } else if start <= end {
                    0
                } else {
                    (start - end + (-step) - 1) / (-step)
                };
                if len != 1 {
                    bail!(
                        "set_value_with_tensor_ decrease_axes=[1] requires a single indexed position"
                    );
                }
            }
            (expanded_updates_name, updates_shape_name)
        };

        let index_shape_name = format!("set_value_tensor_index_shape_{}", out_id);
        let mut index_shape = Vec::with_capacity(data_rank);
        for dim_idx in 0..data_rank {
            index_shape.push(if dim_idx == 1 { -1 } else { 1 });
        }
        self.push_i64_initializer(
            index_shape_name.clone(),
            vec![data_rank as i64],
            &index_shape,
        );

        let shaped_range_name = format!("set_value_tensor_range_shaped_{}", out_id);
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Reshape".to_string(),
            input: vec![range_name, index_shape_name],
            output: vec![shaped_range_name.clone()],
            ..Default::default()
        });

        let expanded_indices_name = format!("set_value_tensor_indices_{}", out_id);
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Expand".to_string(),
            input: vec![shaped_range_name, updates_shape_name],
            output: vec![expanded_indices_name.clone()],
            ..Default::default()
        });

        let mut scatter = onnx::NodeProto {
            op_type: "ScatterElements".to_string(),
            input: vec![data_name, expanded_indices_name, updates_name],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        };
        scatter.attribute.push(helper::attr_int("axis", 1));
        self.onnx_graph.node.push(scatter);
        Ok(())
    }

    pub fn op_repeat_interleave(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("repeat_interleave missing inputs");
        }
        let input_id = inputs[0];
        let axis = self.normalize_axis(
            input_id,
            helper::attr(op, "axis")
                .and_then(|d| d.as_i64())
                .unwrap_or(0),
        )?;
        let rank = self
            .state
            .tensor_shapes
            .get(&input_id)
            .map(|dims| dims.len() as i64)
            .ok_or_else(|| anyhow::anyhow!("repeat_interleave: missing rank metadata"))?;

        let input_name = self.get_tensor_name(input_id)?;
        let expanded_name = format!("repeat_interleave_expanded_{}", out_id);
        self.add_unsqueeze_node(
            input_name.clone(),
            expanded_name.clone(),
            &[axis + 1],
            format!("repeat_interleave_axes_{}", out_id),
        );

        let repeats_dim_name = format!("repeat_interleave_repeats_dim_{}", out_id);
        if let Some(repeats) = helper::attr(op, "repeats").and_then(|d| d.as_i64()) {
            self.push_i64_initializer(repeats_dim_name.clone(), vec![1], &[repeats]);
        } else if let Some(&repeats_id) = inputs.get(1) {
            let repeats_scalar_name =
                self.ensure_scalar_i64_input(repeats_id, out_id, "repeat_interleave_repeats")?;
            self.add_unsqueeze_node(
                repeats_scalar_name,
                repeats_dim_name.clone(),
                &[0],
                format!("repeat_interleave_repeats_axes_{}", out_id),
            );
        } else {
            bail!("repeat_interleave: missing repeats");
        }

        let mut repeats_parts = Vec::with_capacity((rank + 1) as usize);
        for idx in 0..=rank {
            if idx == axis + 1 {
                repeats_parts.push(repeats_dim_name.clone());
                continue;
            }
            let one_name = format!("repeat_interleave_tile_repeat_{}_{}", out_id, idx);
            self.push_i64_initializer(one_name.clone(), vec![1], &[1]);
            repeats_parts.push(one_name);
        }
        let repeats_name = if repeats_parts.len() == 1 {
            repeats_parts[0].clone()
        } else {
            let name = format!("repeat_interleave_tile_repeats_{}", out_id);
            let mut concat = onnx::NodeProto {
                op_type: "Concat".to_string(),
                input: repeats_parts,
                output: vec![name.clone()],
                ..Default::default()
            };
            concat.attribute.push(helper::attr_int("axis", 0));
            self.onnx_graph.node.push(concat);
            name
        };

        let tiled_name = format!("repeat_interleave_tiled_{}", out_id);
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Tile".to_string(),
            input: vec![expanded_name, repeats_name],
            output: vec![tiled_name.clone()],
            ..Default::default()
        });

        let input_shape = format!("repeat_interleave_input_shape_{}", out_id);
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Shape".to_string(),
            input: vec![input_name],
            output: vec![input_shape.clone()],
            ..Default::default()
        });

        let axis_dim_name = format!("repeat_interleave_axis_dim_{}", out_id);
        self.add_slice_node(
            input_shape.clone(),
            axis_dim_name.clone(),
            &[axis],
            &[axis + 1],
            Some(&[0]),
            None,
            &format!("repeat_interleave_axis_dim_{}", out_id),
        )?;

        let repeated_dim_name = format!("repeat_interleave_repeated_dim_{}", out_id);
        self.add_binary_node(
            "Mul",
            axis_dim_name,
            repeats_dim_name,
            repeated_dim_name.clone(),
        );

        let mut shape_parts = Vec::new();
        if axis > 0 {
            let prefix_name = format!("repeat_interleave_prefix_{}", out_id);
            self.add_slice_node(
                input_shape.clone(),
                prefix_name.clone(),
                &[0],
                &[axis],
                Some(&[0]),
                None,
                &format!("repeat_interleave_prefix_{}", out_id),
            )?;
            shape_parts.push(prefix_name);
        }
        shape_parts.push(repeated_dim_name);
        if axis + 1 < rank {
            let suffix_name = format!("repeat_interleave_suffix_{}", out_id);
            self.add_slice_node(
                input_shape,
                suffix_name.clone(),
                &[axis + 1],
                &[rank],
                Some(&[0]),
                None,
                &format!("repeat_interleave_suffix_{}", out_id),
            )?;
            shape_parts.push(suffix_name);
        }

        let output_shape = if shape_parts.len() == 1 {
            shape_parts[0].clone()
        } else {
            let concat_name = format!("repeat_interleave_shape_{}", out_id);
            let mut concat = onnx::NodeProto {
                op_type: "Concat".to_string(),
                input: shape_parts,
                output: vec![concat_name.clone()],
                ..Default::default()
            };
            concat.attribute.push(helper::attr_int("axis", 0));
            self.onnx_graph.node.push(concat);
            concat_name
        };

        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Reshape".to_string(),
            input: vec![tiled_name, output_shape],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        });
        Ok(())
    }

    pub fn op_put_along_axis(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 3 {
            bail!("put_along_axis missing inputs");
        }

        let reduce = helper::attr(op, "reduce")
            .and_then(|d| d.as_str())
            .unwrap_or("assign");
        if reduce != "assign" {
            bail!("put_along_axis only supports reduce=assign");
        }
        if !helper::attr(op, "include_self")
            .and_then(|d| d.as_bool())
            .unwrap_or(true)
        {
            bail!("put_along_axis only supports include_self=true");
        }
        let axis = self.normalize_axis(
            inputs[0],
            helper::attr(op, "axis")
                .and_then(|d| d.as_i64())
                .unwrap_or(0),
        )?;

        let mut node = onnx::NodeProto {
            op_type: "ScatterElements".to_string(),
            input: vec![
                self.get_tensor_name(inputs[0])?,
                self.ensure_i64_input(inputs[1], out_id, "put_along_axis_indices")?,
                self.get_tensor_name(inputs[2])?,
            ],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        };
        node.attribute.push(helper::attr_int("axis", axis));
        self.onnx_graph.node.push(node);
        Ok(())
    }
}
