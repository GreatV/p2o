use crate::converter::Converter;
use crate::helper::{self, dt};
use crate::proto::onnx;
use anyhow::bail;
use serde_json::Value;

impl Converter {
    fn reduce_via_const_axes(
        &mut self,
        op: &Value,
        missing_input_msg: &str,
        onnx_op: &str,
        node_prefix: &str,
    ) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("{}", missing_input_msg);
        }
        if self.scalar_reduce_passthrough(inputs[0], out_id)? {
            return Ok(());
        }
        // Paddle uses tensor id 0 as the "absent input" placeholder; treat an omitted
        // axes input as reduce-all rather than as a dynamic-axes tensor.
        let axes_input_id = inputs.get(1).copied().filter(|&id| id != 0);
        let axes = axes_input_id
            .and_then(|axis_id| self.state.constants.get(&axis_id))
            .map(|values| values.iter().map(|&value| value as i64).collect::<Vec<_>>());
        let keepdims = i64::from(
            helper::attr(op, "keepdim")
                .and_then(|d| d.as_bool())
                .unwrap_or(true),
        );
        if let Some(axis_id) = axes_input_id
            && axes.is_none()
        {
            if self.target_opset < 18 {
                bail!("{onnx_op} with dynamic axes requires opset >= 18");
            }
            let mut axis_name = self.get_tensor_name(axis_id)?;
            if !matches!(
                self.state.tensor_types.get(&axis_id).map(String::as_str),
                Some(helper::paddle_tt::I64)
            ) {
                let cast_output = format!("{}_axes_i64_{}", node_prefix, out_id);
                self.add_cast_node(axis_name, cast_output.clone(), dt::INT64);
                axis_name = cast_output;
            }
            let mut node = onnx::NodeProto {
                op_type: onnx_op.to_string(),
                input: vec![self.get_tensor_name(inputs[0])?, axis_name],
                output: vec![self.get_tensor_name(out_id)?],
                ..Default::default()
            };
            node.attribute.push(helper::attr_int("keepdims", keepdims));
            self.onnx_graph.node.push(node);
        } else {
            self.add_reduce_node(
                onnx_op,
                self.get_tensor_name(inputs[0])?,
                self.get_tensor_name(out_id)?,
                axes.as_deref(),
                keepdims,
                &format!("{}_{}", node_prefix, out_id),
            );
        }
        Ok(())
    }

    pub fn op_reduce_max(&mut self, op: &Value) -> anyhow::Result<()> {
        self.reduce_via_const_axes(op, "max missing inputs", "ReduceMax", "reduce_max")
    }

    pub fn op_reduce_mean(&mut self, op: &Value) -> anyhow::Result<()> {
        self.reduce_via_const_axes(op, "mean missing inputs", "ReduceMean", "reduce_mean")
    }

    pub fn op_reduce_prod(&mut self, op: &Value) -> anyhow::Result<()> {
        self.reduce_via_const_axes(op, "prod missing inputs", "ReduceProd", "reduce_prod")
    }

    pub fn op_reduce_min(&mut self, op: &Value) -> anyhow::Result<()> {
        self.reduce_via_const_axes(op, "min missing inputs", "ReduceMin", "reduce_min")
    }

    pub fn op_reduce_sum(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("sum missing inputs");
        }
        if self.scalar_reduce_passthrough(inputs[0], out_id)? {
            return Ok(());
        }
        let mut node = onnx::NodeProto {
            op_type: "ReduceSum".to_string(),
            input: vec![self.get_tensor_name(inputs[0])?],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        };
        // Tensor id 0 is Paddle's absent-input placeholder; skip it so an omitted axes
        // input reduces over all dimensions instead of referencing a bogus tensor.
        if inputs.len() > 1 && inputs[1] != 0 {
            node.input.push(self.get_tensor_name(inputs[1])?);
        }
        node.attribute.push(helper::attr_int(
            "keepdims",
            i64::from(
                helper::attr(op, "keepdim")
                    .and_then(|d| d.as_bool())
                    .unwrap_or(true),
            ),
        ));
        self.onnx_graph.node.push(node);
        Ok(())
    }

    pub fn op_topk(&mut self, op: &Value) -> anyhow::Result<()> {
        let outputs = op
            .get("O")
            .and_then(|o| o.as_array())
            .ok_or_else(|| anyhow::anyhow!("topk missing outputs"))?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("topk missing inputs");
        }
        let primary_out_id = outputs
            .first()
            .and_then(|output| output.get("%"))
            .and_then(|id| id.as_i64())
            .ok_or_else(|| anyhow::anyhow!("topk missing primary output id"))?;

        let mut k_input_name = self.get_tensor_name(inputs[1])?;
        if !matches!(
            self.state.tensor_types.get(&inputs[1]).map(String::as_str),
            Some(helper::paddle_tt::I64)
        ) {
            let cast_output = format!("topk_k_i64_{}", primary_out_id);
            self.add_cast_node(k_input_name, cast_output.clone(), dt::INT64);
            k_input_name = cast_output;
        }

        let output_names = outputs
            .iter()
            .map(|output| {
                output
                    .get("%")
                    .and_then(|id| id.as_i64())
                    .ok_or_else(|| anyhow::anyhow!("topk output missing id"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .map(|id| self.get_tensor_name(id))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut node = onnx::NodeProto {
            op_type: "TopK".to_string(),
            input: vec![self.get_tensor_name(inputs[0])?],
            output: output_names,
            ..Default::default()
        };
        if self.target_opset >= 11 {
            node.input.push(k_input_name);
        } else {
            let k = self
                .state
                .constants
                .get(&inputs[1])
                .and_then(|values| values.first())
                .copied()
                .ok_or_else(|| anyhow::anyhow!("topk: opset <= 10 requires constant k"))?
                as i64;
            node.attribute.push(helper::attr_int("k", k));
        }
        for (name, value) in [
            (
                "axis",
                helper::attr(op, "axis")
                    .and_then(|d| d.as_i64())
                    .unwrap_or(-1),
            ),
            (
                "largest",
                if helper::attr(op, "largest")
                    .and_then(|d| d.as_bool())
                    .unwrap_or(true)
                {
                    1
                } else {
                    0
                },
            ),
            (
                "sorted",
                if helper::attr(op, "sorted")
                    .and_then(|d| d.as_bool())
                    .unwrap_or(true)
                {
                    1
                } else {
                    0
                },
            ),
        ] {
            node.attribute.push(helper::attr_int(name, value));
        }
        self.onnx_graph.node.push(node);
        Ok(())
    }

    pub fn op_argmax(&mut self, op: &Value) -> anyhow::Result<()> {
        if helper::attr(op, "flatten")
            .and_then(|d| d.as_bool())
            .unwrap_or(false)
        {
            bail!("argmax with flatten=true is not supported");
        }

        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("argmax missing inputs");
        }

        let output_name = self.get_tensor_name(out_id)?;
        let target_dtype = self.state.tensor_types.get(&out_id).cloned();
        let argmax_output_name = if matches!(target_dtype.as_deref(), Some(helper::paddle_tt::I32))
        {
            format!("argmax_i64_{}", out_id)
        } else {
            output_name.clone()
        };

        let mut node = onnx::NodeProto {
            op_type: "ArgMax".to_string(),
            input: vec![self.get_tensor_name(inputs[0])?],
            output: vec![argmax_output_name.clone()],
            ..Default::default()
        };
        if inputs.len() > 1
            && let Some(axis) = self
                .state
                .constants
                .get(&inputs[1])
                .and_then(|vals| vals.first())
                .copied()
        {
            node.attribute.push(helper::attr_int("axis", axis as i64));
        }
        node.attribute.push(helper::attr_int(
            "keepdims",
            i64::from(
                helper::attr(op, "keepdims")
                    .and_then(|d| d.as_bool())
                    .unwrap_or(true),
            ),
        ));
        self.onnx_graph.node.push(node);
        if matches!(target_dtype.as_deref(), Some(helper::paddle_tt::I32)) {
            self.add_cast_node(argmax_output_name, output_name, dt::INT32);
        }
        Ok(())
    }

    pub fn op_argsort(&mut self, op: &Value) -> anyhow::Result<()> {
        let outputs = op
            .get("O")
            .and_then(|o| o.as_array())
            .ok_or_else(|| anyhow::anyhow!("argsort missing outputs"))?;
        if outputs.len() < 2 {
            bail!("argsort expects two outputs");
        }
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("argsort missing inputs");
        }
        let input_id = inputs[0];
        let input_name = self.get_tensor_name(input_id)?;
        let axis = helper::attr(op, "axis")
            .and_then(|d| d.as_i64())
            .unwrap_or(-1);
        let descending = helper::attr(op, "descending")
            .and_then(|d| d.as_bool())
            .unwrap_or(false);
        let stable = helper::attr(op, "stable")
            .and_then(|d| d.as_bool())
            .unwrap_or(false);
        let output_names = outputs
            .iter()
            .filter_map(|output| output.get("%").and_then(|id| id.as_i64()))
            .map(|id| self.get_tensor_name(id))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let normalized_axis = self.normalize_axis(input_id, axis).ok();
        let gather_axis = if let Some(normalized_axis) = normalized_axis {
            normalized_axis
        } else {
            if axis < 0 {
                self.require_opset(13, "argsort negative axis gather")?;
            }
            axis
        };

        let shape_name = format!(
            "argsort_shape_{}",
            outputs[0].get("%").and_then(|v| v.as_i64()).unwrap_or(0)
        );
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Shape".to_string(),
            input: vec![input_name.clone()],
            output: vec![shape_name.clone()],
            ..Default::default()
        });

        let axis_name = format!("argsort_axis_{}", shape_name);
        self.push_i64_initializer(axis_name.clone(), vec![1], &[gather_axis]);

        let k_name = format!("argsort_k_{}", shape_name);
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Gather".to_string(),
            input: vec![shape_name.clone(), axis_name],
            output: vec![k_name.clone()],
            ..Default::default()
        });

        let input_type = self.state.tensor_types.get(&input_id).map(String::as_str);
        let input_shape = self.state.tensor_shapes.get(&input_id);
        let axis_len = normalized_axis
            .and_then(|normalized| {
                input_shape
                    .and_then(|shape| shape.get(normalized as usize))
                    .copied()
            })
            .filter(|&dim| dim > 0);
        let use_integer_tie_break = !stable
            && matches!(
                input_type,
                Some(helper::paddle_tt::I32 | helper::paddle_tt::I64)
            )
            && normalized_axis.is_some()
            && axis_len.is_some();
        if !stable
            && matches!(
                input_type,
                Some(helper::paddle_tt::F16 | helper::paddle_tt::F32 | helper::paddle_tt::F64)
            )
        {
            log::warn!(
                "argsort with stable=false on floating inputs has backend-dependent tie order"
            );
        }

        if use_integer_tie_break {
            let normalized_axis = normalized_axis.unwrap_or(axis);
            let axis_len = axis_len.unwrap_or_default();
            let rank = input_shape.map_or(0, Vec::len);
            let out_id = outputs[0].get("%").and_then(|v| v.as_i64()).unwrap_or(0);

            let keyed_input_name = if matches!(input_type, Some(helper::paddle_tt::I32)) {
                let cast_name = format!("argsort_input_i64_{}", out_id);
                self.add_cast_node(input_name.clone(), cast_name.clone(), dt::INT64);
                cast_name
            } else {
                input_name.clone()
            };

            let scale_name = format!("argsort_tie_scale_{}", out_id);
            self.push_i64_initializer(scale_name.clone(), vec![1], &[axis_len + 1]);

            let tie_name = format!("argsort_tie_break_{}", out_id);
            let mut tie_dims = vec![1; rank];
            tie_dims[normalized_axis as usize] = axis_len;
            let mut tie_tensor = onnx::TensorProto {
                name: tie_name.clone(),
                dims: tie_dims,
                data_type: dt::INT64,
                ..Default::default()
            };
            for idx in 0..axis_len {
                let tie_value = if descending { axis_len - idx } else { idx };
                tie_tensor
                    .raw_data
                    .extend_from_slice(&tie_value.to_le_bytes());
            }
            self.onnx_graph.initializer.push(tie_tensor);

            let scaled_name = format!("argsort_scaled_input_{}", out_id);
            self.add_binary_node(
                "Mul",
                keyed_input_name.clone(),
                scale_name,
                scaled_name.clone(),
            );

            let key_name = format!("argsort_key_{}", out_id);
            self.add_binary_node("Add", scaled_name, tie_name, key_name.clone());

            let key_values_name = format!("argsort_key_values_{}", out_id);
            let mut topk = onnx::NodeProto {
                op_type: "TopK".to_string(),
                input: vec![key_name, k_name],
                output: vec![key_values_name, output_names[1].clone()],
                ..Default::default()
            };
            topk.attribute
                .push(helper::attr_int("axis", normalized_axis));
            topk.attribute
                .push(helper::attr_int("largest", i64::from(descending)));
            topk.attribute.push(helper::attr_int("sorted", 1));
            self.onnx_graph.node.push(topk);

            let mut gather = onnx::NodeProto {
                op_type: "GatherElements".to_string(),
                input: vec![input_name, output_names[1].clone()],
                output: vec![output_names[0].clone()],
                ..Default::default()
            };
            gather
                .attribute
                .push(helper::attr_int("axis", normalized_axis));
            self.onnx_graph.node.push(gather);
            return Ok(());
        }

        let mut node = onnx::NodeProto {
            op_type: "TopK".to_string(),
            input: vec![input_name.clone(), k_name],
            output: output_names.clone(),
            ..Default::default()
        };
        node.attribute
            .push(helper::attr_int("axis", normalized_axis.unwrap_or(axis)));
        node.attribute
            .push(helper::attr_int("largest", i64::from(descending)));
        node.attribute.push(helper::attr_int("sorted", 1));
        self.onnx_graph.node.push(node);
        Ok(())
    }
}
