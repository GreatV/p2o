use crate::helper::{self, dt};
use crate::proto::onnx;
use anyhow::bail;
use serde_json::Value;

#[derive(Clone, Copy)]
enum ExpandShapePart {
    Tensor(i64),
    Literal(i64),
}

impl super::super::Converter {
    fn aligned_expand_input_axis(
        &self,
        data_id: i64,
        output_rank: usize,
        output_axis: usize,
    ) -> anyhow::Result<i64> {
        let input_rank = self
            .state
            .tensor_shapes
            .get(&data_id)
            .map(|shape| shape.len())
            .ok_or_else(|| {
                anyhow::anyhow!("expand: missing rank metadata for tensor {}", data_id)
            })?;
        if output_rank < input_rank {
            bail!(
                "expand: output rank {} smaller than input rank {} for tensor {}",
                output_rank,
                input_rank,
                data_id
            );
        }
        let leading = output_rank - input_rank;
        if output_axis < leading {
            bail!(
                "expand: -1 at axis {} cannot map to leading broadcast axis for tensor {}",
                output_axis,
                data_id
            );
        }
        Ok((output_axis - leading) as i64)
    }

    fn ensure_expand_shape_part(
        &mut self,
        data_id: i64,
        out_id: i64,
        output_rank: usize,
        output_axis: usize,
        part: ExpandShapePart,
        input_shape_name: &mut Option<String>,
    ) -> anyhow::Result<String> {
        match part {
            ExpandShapePart::Literal(value) => self.ensure_expand_shape_literal_part(
                data_id,
                out_id,
                output_rank,
                output_axis,
                value,
                input_shape_name,
            ),
            ExpandShapePart::Tensor(part_id) => {
                if let Some(values) = self.state.constants.get(&part_id)
                    && let Some(&value) = values.first()
                {
                    return self.ensure_expand_shape_literal_part(
                        data_id,
                        out_id,
                        output_rank,
                        output_axis,
                        value as i64,
                        input_shape_name,
                    );
                }

                let shape = self
                    .state
                    .tensor_shapes
                    .get(&part_id)
                    .cloned()
                    .unwrap_or_default();
                if shape.is_empty() {
                    let mut source_name = self.get_tensor_name(part_id)?;
                    if !matches!(
                        self.state.tensor_types.get(&part_id).map(String::as_str),
                        Some(crate::helper::paddle_tt::I64)
                    ) {
                        let cast_name = format!("expand_shape_part_i64_{}_{}", out_id, output_axis);
                        self.add_cast_node(source_name, cast_name.clone(), dt::INT64);
                        source_name = cast_name;
                    }
                    let part_name = format!("expand_shape_part_{}_{}", out_id, output_axis);
                    self.add_unsqueeze_node(
                        source_name,
                        part_name.clone(),
                        &[0],
                        format!("expand_shape_axes_{}_{}", out_id, output_axis),
                    );
                    return Ok(part_name);
                }
                if shape == vec![1] {
                    if matches!(
                        self.state.tensor_types.get(&part_id).map(String::as_str),
                        Some(crate::helper::paddle_tt::I64)
                    ) {
                        return self.get_tensor_name(part_id);
                    }
                    let cast_name = format!("expand_shape_part_i64_{}_{}", out_id, output_axis);
                    self.add_cast_node(
                        self.get_tensor_name(part_id)?,
                        cast_name.clone(),
                        dt::INT64,
                    );
                    return Ok(cast_name);
                }

                bail!(
                    "expand: unsupported dynamic shape part {} with rank {}",
                    part_id,
                    shape.len()
                );
            }
        }
    }

    fn ensure_expand_shape_literal_part(
        &mut self,
        data_id: i64,
        out_id: i64,
        output_rank: usize,
        output_axis: usize,
        value: i64,
        input_shape_name: &mut Option<String>,
    ) -> anyhow::Result<String> {
        if value < 0 {
            let shape_name = if let Some(name) = input_shape_name.clone() {
                name
            } else {
                let name = format!("expand_input_shape_{}", out_id);
                self.onnx_graph.node.push(onnx::NodeProto {
                    op_type: "Shape".to_string(),
                    input: vec![self.get_tensor_name(data_id)?],
                    output: vec![name.clone()],
                    ..Default::default()
                });
                *input_shape_name = Some(name.clone());
                name
            };
            let input_axis = self.aligned_expand_input_axis(data_id, output_rank, output_axis)?;
            let part_name = format!("expand_shape_part_{}_{}", out_id, output_axis);
            self.add_slice_node(
                shape_name,
                part_name.clone(),
                &[input_axis],
                &[input_axis + 1],
                Some(&[0]),
                None,
                &format!("expand_shape_slice_{}_{}", out_id, output_axis),
            )?;
            return Ok(part_name);
        }

        let part_name = format!("expand_shape_part_{}_{}", out_id, output_axis);
        let mut tensor = onnx::TensorProto {
            name: part_name.clone(),
            dims: vec![1],
            data_type: dt::INT64,
            ..Default::default()
        };
        tensor.raw_data.extend_from_slice(&value.to_le_bytes());
        self.onnx_graph.initializer.push(tensor);
        Ok(part_name)
    }
    fn build_expand_shape(
        &mut self,
        data_id: i64,
        out_id: i64,
        parts: &[ExpandShapePart],
    ) -> anyhow::Result<String> {
        let mut input_shape_name = None;
        let mut shape_parts = Vec::with_capacity(parts.len());
        for (axis, &part) in parts.iter().enumerate() {
            shape_parts.push(self.ensure_expand_shape_part(
                data_id,
                out_id,
                parts.len(),
                axis,
                part,
                &mut input_shape_name,
            )?);
        }

        if shape_parts.len() == 1 {
            return Ok(shape_parts[0].clone());
        }

        let concat_name = format!("expand_shape_{}", out_id);
        let mut concat = onnx::NodeProto {
            op_type: "Concat".to_string(),
            input: shape_parts,
            output: vec![concat_name.clone()],
            ..Default::default()
        };
        concat.attribute.push(helper::attr_int("axis", 0));
        self.onnx_graph.node.push(concat);
        Ok(concat_name)
    }

    pub(crate) fn build_expand_shape_from_parts(
        &mut self,
        data_id: i64,
        out_id: i64,
        part_ids: &[i64],
    ) -> anyhow::Result<String> {
        let parts = part_ids
            .iter()
            .copied()
            .map(ExpandShapePart::Tensor)
            .collect::<Vec<_>>();
        self.build_expand_shape(data_id, out_id, &parts)
    }

    fn build_expand_shape_from_literals(
        &mut self,
        data_id: i64,
        out_id: i64,
        values: &[i64],
    ) -> anyhow::Result<String> {
        let parts = values
            .iter()
            .copied()
            .map(ExpandShapePart::Literal)
            .collect::<Vec<_>>();
        self.build_expand_shape(data_id, out_id, &parts)
    }

    pub(crate) fn normalize_axis(&self, input_id: i64, axis: i64) -> anyhow::Result<i64> {
        let rank = self
            .state
            .tensor_shapes
            .get(&input_id)
            .map(|dims| dims.len() as i64)
            .ok_or_else(|| anyhow::anyhow!("missing rank metadata for tensor {}", input_id))?;
        Ok(if axis < 0 { axis + rank } else { axis })
    }

    pub(crate) fn ensure_scalar_i64_input(
        &mut self,
        input_id: i64,
        out_id: i64,
        label: &str,
    ) -> anyhow::Result<String> {
        let mut name = self.get_tensor_name(input_id)?;
        if matches!(self.state.tensor_shapes.get(&input_id), Some(shape) if shape == &vec![1]) {
            let squeezed = format!("{}_scalar_{}", label, out_id);
            self.add_squeeze_node(
                name,
                squeezed.clone(),
                Some(&[0]),
                Some(format!("{}_scalar_axes_{}", label, out_id)),
            );
            name = squeezed;
        }
        if !matches!(
            self.state.tensor_types.get(&input_id).map(String::as_str),
            Some(helper::paddle_tt::I64)
        ) {
            let casted = format!("{}_i64_{}", label, out_id);
            self.add_cast_node(name, casted.clone(), dt::INT64);
            name = casted;
        }
        Ok(name)
    }

    pub(crate) fn ensure_i64_input(
        &mut self,
        input_id: i64,
        out_id: i64,
        label: &str,
    ) -> anyhow::Result<String> {
        let name = self.get_tensor_name(input_id)?;
        if matches!(
            self.state.tensor_types.get(&input_id).map(String::as_str),
            Some(helper::paddle_tt::I64)
        ) {
            return Ok(name);
        }

        let casted = format!("{}_i64_{}", label, out_id);
        self.add_cast_node(name, casted.clone(), dt::INT64);
        Ok(casted)
    }

    pub(crate) fn scalar_reduce_passthrough(
        &mut self,
        input_id: i64,
        out_id: i64,
    ) -> anyhow::Result<bool> {
        if !matches!(self.state.tensor_shapes.get(&input_id), Some(shape) if shape.is_empty()) {
            return Ok(false);
        }

        let input_name = self.get_tensor_name(input_id)?;
        let output_name = self.get_tensor_name(out_id)?;
        if matches!(self.state.tensor_shapes.get(&out_id), Some(shape) if shape == &vec![1]) {
            self.add_unsqueeze_node(
                input_name,
                output_name,
                &[0],
                format!("scalar_reduce_axes_{}", out_id),
            );
        } else {
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Identity".to_string(),
                input: vec![input_name],
                output: vec![output_name],
                ..Default::default()
            });
        }
        Ok(true)
    }

    pub fn op_assign_value(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let dtype = helper::attr(op, "dtype")
            .and_then(|d| d.as_str())
            .ok_or_else(|| anyhow::anyhow!("assign_value_: missing dtype"))?;
        let onnx_dtype = helper::paddle_dtype_to_onnx(dtype)
            .ok_or_else(|| anyhow::anyhow!("assign_value_: unsupported dtype {}", dtype))?;
        let shape = helper::attr(op, "shape")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter()
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
            .map(|arr| {
                arr.iter()
                    .filter_map(helper::value_as_f64)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let mut tensor = onnx::TensorProto {
            name: self.get_tensor_name(out_id)?,
            dims: shape.clone(),
            data_type: onnx_dtype,
            ..Default::default()
        };
        if tensor.dims.is_empty() && values.len() > 1 {
            tensor.dims = vec![values.len() as i64];
        }
        let expected_value_count = if tensor.dims.is_empty() {
            1
        } else if tensor.dims.contains(&0) {
            0
        } else {
            tensor.dims.iter().product::<i64>() as usize
        };
        if values.len() != expected_value_count {
            bail!(
                "assign_value_ shape {:?} expects {} values, got {}",
                tensor.dims,
                expected_value_count,
                values.len()
            );
        }

        for &value in &values {
            tensor
                .raw_data
                .extend_from_slice(&self.encode_scalar_f64_as_raw_data(value, onnx_dtype)?);
        }

        self.onnx_graph.initializer.push(tensor);
        Ok(())
    }

    pub fn op_arange(&mut self, op: &Value) -> anyhow::Result<()> {
        self.require_opset(11, "arange")?;
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 3 {
            bail!("arange missing inputs");
        }

        let mut range_inputs = Vec::new();
        for (idx, input_id) in inputs.iter().take(3).enumerate() {
            let mut input_name = self.get_tensor_name(*input_id)?;
            if matches!(self.state.tensor_shapes.get(input_id), Some(shape) if shape == &vec![1]) {
                let squeezed_name = format!("arange_squeezed_{}_{}", out_id, idx);
                self.add_squeeze_node(
                    input_name,
                    squeezed_name.clone(),
                    Some(&[0]),
                    Some(format!("arange_squeeze_axes_{}_{}", out_id, idx)),
                );
                input_name = squeezed_name;
            }
            range_inputs.push(input_name);
        }

        let range_out = if let Some(dtype) = helper::attr(op, "dtype").and_then(|d| d.as_str())
            && helper::paddle_dtype_to_onnx(dtype).is_some()
        {
            format!("arange_raw_{}", out_id)
        } else {
            self.get_tensor_name(out_id)?
        };

        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Range".to_string(),
            input: range_inputs,
            output: vec![range_out.clone()],
            ..Default::default()
        });

        if range_out != self.get_tensor_name(out_id)? {
            let dtype = helper::attr(op, "dtype")
                .and_then(|d| d.as_str())
                .and_then(helper::paddle_dtype_to_onnx)
                .ok_or_else(|| anyhow::anyhow!("arange: unsupported dtype"))?;
            self.add_cast_node(range_out, self.get_tensor_name(out_id)?, dtype);
        }
        Ok(())
    }

    pub fn op_cast(&mut self, op: &Value) -> anyhow::Result<()> {
        let dtype = helper::attr(op, "dtype")
            .and_then(|d| d.as_str())
            .ok_or_else(|| anyhow::anyhow!("cast: missing dtype"))?;
        let to = helper::paddle_dtype_to_onnx(dtype)
            .ok_or_else(|| anyhow::anyhow!("cast: unsupported dtype {}", dtype))?;

        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("cast missing inputs");
        }

        let out_id = helper::op_out_id(op)?;
        self.add_cast_node(
            self.get_tensor_name(inputs[0])?,
            self.get_tensor_name(out_id)?,
            to,
        );
        Ok(())
    }

    pub fn op_clip(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("clip missing inputs");
        }

        let target_dtype = self.maybe_onnx_dtype_for_tensor_id(inputs[0])?;

        let mut node_inputs = vec![self.get_tensor_name(inputs[0])?];
        for (idx, input_id) in inputs.iter().enumerate().skip(1) {
            let mut input_name = self.get_tensor_name(*input_id)?;
            if let Some(dtype) = target_dtype
                && let Some(input_type) = self.state.tensor_types.get(input_id).map(String::as_str)
                && matches!(input_type, helper::paddle_tt::F32 | helper::paddle_tt::F64)
                && dtype != dt::FLOAT
            {
                let cast_name = format!("clip_cast_{}_{}", out_id, idx);
                self.add_cast_node(input_name, cast_name.clone(), dtype);
                input_name = cast_name;
            }
            node_inputs.push(input_name);
        }

        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Clip".to_string(),
            input: node_inputs,
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        });
        Ok(())
    }

    pub fn op_slice(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let mut axes = vec![];
        let mut decrease_axis = vec![];

        if let Some(d_arr) = helper::attr(op, "axes").and_then(|d| d.as_array()) {
            axes = d_arr
                .iter()
                .filter_map(|x| x.get("D").and_then(|d| d.as_i64()).or_else(|| x.as_i64()))
                .collect();
        }
        if let Some(d_arr) = helper::attr(op, "decrease_axis").and_then(|d| d.as_array()) {
            decrease_axis = d_arr
                .iter()
                .filter_map(|x| x.get("D").and_then(|d| d.as_i64()))
                .collect();
        }

        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("slice missing inputs");
        }

        if self.target_opset < 10 {
            if let Some(&start_id) = inputs.get(1)
                && !self.state.constants.contains_key(&start_id)
            {
                bail!("slice requires constant starts input for opset < 10");
            }
            if let Some(&end_id) = inputs.get(2)
                && !self.state.constants.contains_key(&end_id)
            {
                bail!("slice requires constant ends input for opset < 10");
            }
            let starts = inputs
                .get(1)
                .and_then(|id| self.state.constants.get(id))
                .map(|values| values.iter().map(|&value| value as i64).collect::<Vec<_>>())
                .or_else(|| {
                    helper::attr(op, "starts").and_then(|d| {
                        d.as_array().map(|items| {
                            items
                                .iter()
                                .filter_map(|item| {
                                    item.get("D")
                                        .and_then(|v| v.as_i64())
                                        .or_else(|| item.as_i64())
                                })
                                .collect::<Vec<_>>()
                        })
                    })
                })
                .ok_or_else(|| anyhow::anyhow!("slice requires constant starts for opset < 10"))?;
            let ends = inputs
                .get(2)
                .and_then(|id| self.state.constants.get(id))
                .map(|values| values.iter().map(|&value| value as i64).collect::<Vec<_>>())
                .or_else(|| {
                    helper::attr(op, "ends").and_then(|d| {
                        d.as_array().map(|items| {
                            items
                                .iter()
                                .filter_map(|item| {
                                    item.get("D")
                                        .and_then(|v| v.as_i64())
                                        .or_else(|| item.as_i64())
                                })
                                .collect::<Vec<_>>()
                        })
                    })
                })
                .ok_or_else(|| anyhow::anyhow!("slice requires constant ends for opset < 10"))?;

            let slice_out_name = if decrease_axis.is_empty() {
                self.get_tensor_name(out_id)?
            } else {
                format!("slice_out_{}", out_id)
            };
            self.add_slice_node(
                self.get_tensor_name(inputs[0])?,
                slice_out_name.clone(),
                &starts,
                &ends,
                if axes.is_empty() { None } else { Some(&axes) },
                None,
                &format!("slice_{}", out_id),
            )?;

            if !decrease_axis.is_empty() {
                self.add_squeeze_node(
                    slice_out_name,
                    self.get_tensor_name(out_id)?,
                    Some(&decrease_axis),
                    Some(format!("slice_decrease_axes_{}", out_id)),
                );
            }
            return Ok(());
        }

        let mut onnx_node = onnx::NodeProto {
            op_type: "Slice".to_string(),
            ..Default::default()
        };

        for input_id in inputs.iter().take(3) {
            onnx_node.input.push(self.get_tensor_name(*input_id)?);
        }

        if !axes.is_empty() {
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
        }

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
                squeeze_axes_tensor
                    .raw_data
                    .extend_from_slice(&a.to_le_bytes());
            }
            self.onnx_graph.initializer.push(squeeze_axes_tensor);

            self.add_squeeze_node(
                slice_out_name,
                self.get_tensor_name(out_id)?,
                Some(&decrease_axis),
                Some(squeeze_axes_name),
            );
        }
        Ok(())
    }

    pub fn op_strided_slice(&mut self, op: &Value) -> anyhow::Result<()> {
        self.require_opset(10, "strided_slice")?;
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 4 {
            bail!("strided_slice missing inputs");
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
        let decrease_axis = helper::attr(op, "decrease_axis")
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

        let mut node = onnx::NodeProto {
            op_type: "Slice".to_string(),
            input: vec![
                self.get_tensor_name(inputs[0])?,
                self.ensure_i64_input(inputs[1], out_id, "strided_slice_starts")?,
                self.ensure_i64_input(inputs[2], out_id, "strided_slice_ends")?,
            ],
            output: vec![if decrease_axis.is_empty() {
                self.get_tensor_name(out_id)?
            } else {
                format!("strided_slice_out_{}", out_id)
            }],
            ..Default::default()
        };

        if !axes.is_empty() {
            let axes_name = format!("strided_slice_axes_{}", out_id);
            let mut axes_tensor = onnx::TensorProto {
                name: axes_name.clone(),
                dims: vec![axes.len() as i64],
                data_type: dt::INT64,
                ..Default::default()
            };
            for axis in axes {
                axes_tensor.raw_data.extend_from_slice(&axis.to_le_bytes());
            }
            self.onnx_graph.initializer.push(axes_tensor);
            node.input.push(axes_name);
        }

        node.input
            .push(self.ensure_i64_input(inputs[3], out_id, "strided_slice_steps")?);
        let slice_out_name = node.output[0].clone();
        self.onnx_graph.node.push(node);

        if !decrease_axis.is_empty() {
            self.add_squeeze_node(
                slice_out_name,
                self.get_tensor_name(out_id)?,
                Some(&decrease_axis),
                Some(format!("strided_slice_decrease_axes_{}", out_id)),
            );
        }
        Ok(())
    }

    pub fn op_full(&mut self, op_type: &str, op: &Value) -> anyhow::Result<()> {
        if let Ok(out_id) = helper::op_out_id(op)
            && let Some(vals) = self.state.constants.get(&out_id).cloned()
        {
            let mut tensor_dtype = dt::INT64;
            let mut shape_dims = vec![];
            if let Some(d_arr) = helper::attr(op, "shape").and_then(|d| d.as_array()) {
                shape_dims = d_arr
                    .iter()
                    .filter_map(|x| x.get("D").and_then(|d| d.as_i64()).or_else(|| x.as_i64()))
                    .collect();
            }
            if op_type == helper::paddle_op::FULL
                && let Some(dtype) = helper::attr(op, "dtype").and_then(|d| d.as_str())
            {
                tensor_dtype = helper::paddle_dtype_to_onnx(dtype)
                    .ok_or_else(|| anyhow::anyhow!("full: unsupported dtype {}", dtype))?;
            }
            if op_type == helper::paddle_op::FULL_INT_ARRAY
                && let Some(dtype) = helper::attr(op, "dtype").and_then(|d| d.as_str())
                && let Some(onnx_dt) = helper::paddle_dtype_to_onnx(dtype)
            {
                tensor_dtype = onnx_dt;
            }

            let mut tensor = onnx::TensorProto {
                name: self.get_tensor_name(out_id)?,
                dims: shape_dims.clone(),
                ..Default::default()
            };

            let has_negative_dim = tensor.dims.iter().any(|&dim| dim < 0);
            let has_zero_dim = tensor.dims.contains(&0);
            let expected_volume = if tensor.dims.is_empty() {
                1
            } else if has_negative_dim {
                -1
            } else {
                tensor.dims.iter().product::<i64>()
            };
            let materialized_vals = if has_negative_dim {
                log::warn!(
                    "full initializer {} has dynamic shape {:?}; materializing as 1D fallback",
                    tensor.name,
                    tensor.dims
                );
                tensor.dims = vec![vals.len() as i64];
                vals.clone()
            } else if has_zero_dim {
                Vec::new()
            } else if expected_volume > 1 && vals.len() == 1 {
                vec![vals[0]; expected_volume as usize]
            } else {
                vals.clone()
            };
            if op_type == helper::paddle_op::FULL_INT_ARRAY
                && (tensor.dims.is_empty() || expected_volume != materialized_vals.len() as i64)
            {
                tensor.dims = vec![vals.len() as i64];
            }
            let final_expected_volume = if tensor.dims.is_empty() {
                1
            } else if tensor.dims.contains(&0) {
                0
            } else {
                tensor.dims.iter().product::<i64>()
            };
            if !tensor.dims.iter().any(|&dim| dim < 0)
                && final_expected_volume != materialized_vals.len() as i64
            {
                bail!(
                    "{} initializer shape {:?} expects {} values, got {}",
                    op_type,
                    tensor.dims,
                    final_expected_volume,
                    materialized_vals.len()
                );
            }
            tensor.data_type = tensor_dtype;
            for &value in &materialized_vals {
                tensor
                    .raw_data
                    .extend_from_slice(&self.encode_scalar_f64_as_raw_data(value, tensor_dtype)?);
            }
            self.onnx_graph.initializer.push(tensor);
        }
        Ok(())
    }

    pub fn op_full_like(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("full_like missing inputs");
        }

        let shape_name = format!("full_like_shape_{}", out_id);
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Shape".to_string(),
            input: vec![self.get_tensor_name(inputs[0])?],
            output: vec![shape_name.clone()],
            ..Default::default()
        });

        let mut value_name = self.get_tensor_name(inputs[1])?;
        if let Some(dtype) = helper::attr(op, "dtype").and_then(|d| d.as_str())
            && let Some(to) = helper::paddle_dtype_to_onnx(dtype)
        {
            let cast_name = format!("full_like_cast_{}", out_id);
            self.add_cast_node(value_name, cast_name.clone(), to);
            value_name = cast_name;
        }

        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Expand".to_string(),
            input: vec![value_name, shape_name],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        });
        Ok(())
    }

    pub fn op_expand(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("expand missing inputs");
        }

        let data_id = inputs[0];
        let data_name = self.get_tensor_name(inputs[0])?;
        if let Some(shape_values) = self.state.constants.get(&inputs[1]).cloned() {
            let shape_dims: Vec<i64> = shape_values.iter().map(|&value| value as i64).collect();
            if shape_dims.iter().any(|&dim| dim < 0) {
                let shape_name =
                    self.build_expand_shape_from_literals(data_id, out_id, &shape_dims)?;

                self.onnx_graph.node.push(onnx::NodeProto {
                    op_type: "Expand".to_string(),
                    input: vec![data_name, shape_name],
                    output: vec![self.get_tensor_name(out_id)?],
                    ..Default::default()
                });
                return Ok(());
            }
        }

        if let Some(shape_part_ids) = self.state.stack_parts.get(&inputs[1]).cloned()
            && shape_part_ids.iter().any(|part_id| {
                self.state
                    .constants
                    .get(part_id)
                    .and_then(|values| values.first())
                    .is_some_and(|&value| value < 0.0)
            })
        {
            let shape_name =
                self.build_expand_shape_from_parts(data_id, out_id, &shape_part_ids)?;
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Expand".to_string(),
                input: vec![data_name, shape_name],
                output: vec![self.get_tensor_name(out_id)?],
                ..Default::default()
            });
            return Ok(());
        }

        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Expand".to_string(),
            input: vec![data_name, self.get_tensor_name(inputs[1])?],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        });
        Ok(())
    }

    pub fn op_full_with_tensor(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("full_with_tensor missing inputs");
        }

        let mut value_name = self.get_tensor_name(inputs[0])?;
        if let Some(dtype) = helper::attr(op, "dtype").and_then(|d| d.as_str())
            && let Some(to) = helper::paddle_dtype_to_onnx(dtype)
        {
            let cast_name = format!("full_with_tensor_cast_{}", out_id);
            self.add_cast_node(value_name, cast_name.clone(), to);
            value_name = cast_name;
        }

        let mut shape_name = self.get_tensor_name(inputs[1])?;
        if matches!(self.state.tensor_shapes.get(&inputs[1]), Some(shape) if shape.is_empty()) {
            let unsqueezed_shape = format!("full_with_tensor_shape_{}", out_id);
            self.add_unsqueeze_node(
                shape_name,
                unsqueezed_shape.clone(),
                &[0],
                format!("full_with_tensor_shape_axes_{}", out_id),
            );
            shape_name = unsqueezed_shape;
        }

        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Expand".to_string(),
            input: vec![value_name, shape_name],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        });
        Ok(())
    }
}
