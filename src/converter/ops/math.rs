use crate::helper::{self, dt};
use crate::proto::onnx;
use anyhow::bail;
use serde_json::Value;

impl super::super::Converter {
    fn encode_scale_attr_value_as_target_dtype(
        &self,
        value: f64,
        data_type: i32,
    ) -> anyhow::Result<Vec<u8>> {
        Ok(match data_type {
            dt::FLOAT => (value as f32).to_le_bytes().to_vec(),
            dt::DOUBLE => value.to_le_bytes().to_vec(),
            dt::INT8 => (value as i8).to_le_bytes().to_vec(),
            dt::UINT8 => (value as u8).to_le_bytes().to_vec(),
            dt::INT16 => (value as i16).to_le_bytes().to_vec(),
            dt::INT32 => (value as i32).to_le_bytes().to_vec(),
            dt::INT64 => (value as i64).to_le_bytes().to_vec(),
            _ => {
                bail!(
                    "scale attr lowering does not support target dtype {}",
                    helper::onnx_dtype_name(data_type)
                )
            }
        })
    }

    fn push_scale_attr_initializer(
        &mut self,
        name: String,
        dims: Vec<i64>,
        data_type: i32,
        value: f64,
    ) -> anyhow::Result<()> {
        let mut tensor = onnx::TensorProto {
            name,
            dims,
            data_type,
            ..Default::default()
        };
        tensor
            .raw_data
            .extend_from_slice(&self.encode_scale_attr_value_as_target_dtype(value, data_type)?);
        self.onnx_graph.initializer.push(tensor);
        Ok(())
    }

    fn add_integer_mod_node(&mut self, lhs: String, rhs: String, output: String) {
        let mut node = onnx::NodeProto {
            op_type: "Mod".to_string(),
            input: vec![lhs, rhs],
            output: vec![output],
            ..Default::default()
        };
        node.attribute.push(helper::attr_int("fmod", 0));
        self.onnx_graph.node.push(node);
    }

    fn push_int64_scalar_initializer(&mut self, name: String, value: i64) {
        self.onnx_graph.initializer.push(onnx::TensorProto {
            name,
            dims: vec![],
            data_type: dt::INT64,
            raw_data: value.to_le_bytes().to_vec(),
            ..Default::default()
        });
    }

    fn lower_signed_bitwise_and_before_opset_18(
        &mut self,
        lhs_name: String,
        rhs_name: String,
        output_name: String,
        output_dtype: i32,
        out_id: i64,
    ) -> anyhow::Result<()> {
        let bit_width = match output_dtype {
            dt::INT8 => 8,
            dt::INT16 => 16,
            dt::INT32 => 32,
            dt::INT64 => 64,
            _ => bail!(
                "signed bitwise_and lowering expects int8/int16/int32/int64, got {}",
                helper::onnx_dtype_name(output_dtype)
            ),
        };
        let prefix = format!("bitwise_and_signed_{}", out_id);
        let zero_name = format!("{}_zero", prefix);
        let two_name = format!("{}_two", prefix);
        self.push_int64_scalar_initializer(zero_name.clone(), 0);
        self.push_int64_scalar_initializer(two_name.clone(), 2);

        let mut lhs_current = lhs_name;
        let mut rhs_current = rhs_name;
        if output_dtype != dt::INT64 {
            let lhs_i64 = format!("{}_lhs_i64", prefix);
            let rhs_i64 = format!("{}_rhs_i64", prefix);
            self.add_cast_node(lhs_current, lhs_i64.clone(), dt::INT64);
            self.add_cast_node(rhs_current, rhs_i64.clone(), dt::INT64);
            lhs_current = lhs_i64;
            rhs_current = rhs_i64;
        }

        let mut accumulated_name: Option<String> = None;
        for bit in 0..bit_width {
            let lhs_bit = format!("{}_lhs_bit_{}", prefix, bit);
            let rhs_bit = format!("{}_rhs_bit_{}", prefix, bit);
            let bit_and = format!("{}_bit_and_{}", prefix, bit);
            let weight_name = format!("{}_weight_{}", prefix, bit);
            let weighted = format!("{}_weighted_{}", prefix, bit);
            let lhs_without_bit = format!("{}_lhs_without_bit_{}", prefix, bit);
            let rhs_without_bit = format!("{}_rhs_without_bit_{}", prefix, bit);

            self.add_integer_mod_node(lhs_current.clone(), two_name.clone(), lhs_bit.clone());
            self.add_integer_mod_node(rhs_current.clone(), two_name.clone(), rhs_bit.clone());
            self.add_binary_node("Mul", lhs_bit.clone(), rhs_bit.clone(), bit_and.clone());

            let weight = if bit == bit_width - 1 {
                if bit_width == 64 {
                    i64::MIN
                } else {
                    -(1_i64 << (bit_width - 1))
                }
            } else {
                1_i64 << bit
            };
            self.push_int64_scalar_initializer(weight_name.clone(), weight);
            self.add_binary_node("Mul", bit_and, weight_name, weighted.clone());

            let next_acc = if let Some(current_acc) = accumulated_name {
                let acc = format!("{}_acc_{}", prefix, bit);
                self.add_binary_node("Add", current_acc, weighted, acc.clone());
                acc
            } else {
                weighted
            };
            accumulated_name = Some(next_acc);

            if bit + 1 < bit_width {
                let lhs_shifted = format!("{}_lhs_shifted_{}", prefix, bit);
                let rhs_shifted = format!("{}_rhs_shifted_{}", prefix, bit);
                self.add_binary_node("Sub", lhs_current, lhs_bit, lhs_without_bit.clone());
                self.add_binary_node("Sub", rhs_current, rhs_bit, rhs_without_bit.clone());
                self.add_binary_node(
                    "Div",
                    lhs_without_bit,
                    two_name.clone(),
                    lhs_shifted.clone(),
                );
                self.add_binary_node(
                    "Div",
                    rhs_without_bit,
                    two_name.clone(),
                    rhs_shifted.clone(),
                );
                lhs_current = lhs_shifted;
                rhs_current = rhs_shifted;
            }
        }

        let accumulated_name = accumulated_name.expect("bitwise_and lowering requires bits");
        if output_dtype == dt::INT64 {
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Identity".to_string(),
                input: vec![accumulated_name],
                output: vec![output_name],
                ..Default::default()
            });
        } else {
            self.add_cast_node(accumulated_name, output_name, output_dtype);
        }
        Ok(())
    }

    pub fn op_multiply(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("multiply missing inputs");
        }

        let op_type = if matches!(
            self.state.tensor_types.get(&inputs[0]).map(String::as_str),
            Some(helper::paddle_tt::BOOL)
        ) && matches!(
            self.state.tensor_types.get(&inputs[1]).map(String::as_str),
            Some(helper::paddle_tt::BOOL)
        ) {
            "And"
        } else {
            "Mul"
        };

        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: op_type.to_string(),
            input: vec![
                self.get_tensor_name(inputs[0])?,
                self.get_tensor_name(inputs[1])?,
            ],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        });
        Ok(())
    }

    pub fn op_matmul(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("matmul missing inputs");
        }

        let mut lhs = self.get_tensor_name(inputs[0])?;
        let mut rhs = self.get_tensor_name(inputs[1])?;

        for (input_id, enabled, side) in [
            (
                inputs[0],
                helper::attr(op, "transpose_x")
                    .and_then(|d| d.as_bool())
                    .unwrap_or(false),
                "x",
            ),
            (
                inputs[1],
                helper::attr(op, "transpose_y")
                    .and_then(|d| d.as_bool())
                    .unwrap_or(false),
                "y",
            ),
        ] {
            if !enabled {
                continue;
            }

            let rank = self
                .state
                .tensor_shapes
                .get(&input_id)
                .map(|dims| dims.len())
                .unwrap_or(0);
            if rank < 2 {
                // Paddle matmul treats transpose on 1-D operands as a no-op on the
                // implicitly lifted view, so skipping an explicit ONNX transpose here
                // preserves vector semantics.
                continue;
            }

            let mut perm: Vec<i64> = (0..rank as i64).collect();
            perm.swap(rank - 2, rank - 1);
            let transposed_name = format!("matmul_transpose_{}_{}", side, out_id);
            let mut transpose = onnx::NodeProto {
                op_type: "Transpose".to_string(),
                input: vec![self.get_tensor_name(input_id)?],
                output: vec![transposed_name.clone()],
                ..Default::default()
            };
            transpose.attribute.push(helper::attr_ints("perm", &perm));
            self.onnx_graph.node.push(transpose);

            if side == "x" {
                lhs = transposed_name;
            } else {
                rhs = transposed_name;
            }
        }

        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "MatMul".to_string(),
            input: vec![lhs, rhs],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        });
        Ok(())
    }

    pub fn op_scale(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("scale op missing inputs");
        }
        let in_name = self.get_tensor_name(inputs[0])?;
        let output_dtype = self
            .maybe_onnx_dtype_for_tensor_id(out_id)?
            .unwrap_or(dt::FLOAT);
        let output_is_scalar = self
            .state
            .tensor_shapes
            .get(&out_id)
            .map(|dims| dims.is_empty())
            .unwrap_or(false);

        let scale_name = if inputs.len() > 1 {
            let mut scale_name = self.get_tensor_name(inputs[1])?;
            if output_dtype != dt::FLOAT
                && matches!(
                    self.state.tensor_types.get(&inputs[1]).map(String::as_str),
                    Some(helper::paddle_tt::F32) | Some(helper::paddle_tt::F64)
                )
            {
                let cast_name = format!("scale_input_cast_{}", out_id);
                self.add_cast_node(scale_name, cast_name.clone(), output_dtype);
                scale_name = cast_name;
            }
            if output_is_scalar
                && matches!(self.state.tensor_shapes.get(&inputs[1]), Some(shape) if shape == &vec![1])
            {
                let squeezed_name = format!("scale_input_squeezed_{}", out_id);
                self.add_squeeze_node(
                    scale_name,
                    squeezed_name.clone(),
                    Some(&[0]),
                    Some(format!("scale_input_squeeze_axes_{}", out_id)),
                );
                scale_name = squeezed_name;
            }
            scale_name
        } else {
            let scale = helper::attr(op, "scale")
                .and_then(|d| d.as_f64())
                .unwrap_or(1.0);
            let s_name = format!("scale_factor_{}", out_id);
            self.push_scale_attr_initializer(
                s_name.clone(),
                if output_is_scalar { vec![] } else { vec![1] },
                output_dtype,
                scale,
            )?;
            s_name
        };

        let bias = helper::attr(op, "bias")
            .and_then(|d| d.as_f64())
            .unwrap_or(0.0);
        let bias_after_scale = helper::attr(op, "bias_after_scale")
            .and_then(|d| d.as_bool())
            .unwrap_or(true);

        let bias_name = format!("scale_bias_{}", out_id);
        self.push_scale_attr_initializer(
            bias_name.clone(),
            if output_is_scalar { vec![] } else { vec![1] },
            output_dtype,
            bias,
        )?;

        if bias_after_scale {
            // Y = scale * X + bias
            let mul_out = format!("scale_mul_{}", out_id);
            self.add_binary_node("Mul", in_name, scale_name, mul_out.clone());
            self.add_binary_node("Add", mul_out, bias_name, self.get_tensor_name(out_id)?);
        } else {
            // Y = scale * (X + bias)
            let add_out = format!("scale_add_{}", out_id);
            self.add_binary_node("Add", in_name, bias_name, add_out.clone());
            self.add_binary_node("Mul", add_out, scale_name, self.get_tensor_name(out_id)?);
        }
        Ok(())
    }

    pub fn op_swish(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("swish missing inputs");
        }

        let in_name = self.get_tensor_name(inputs[0])?;
        let sigmoid_out = format!("swish_sigmoid_{}", out_id);

        let sigmoid_node = onnx::NodeProto {
            op_type: "Sigmoid".to_string(),
            input: vec![in_name.clone()],
            output: vec![sigmoid_out.clone()],
            ..Default::default()
        };
        self.onnx_graph.node.push(sigmoid_node);

        self.add_binary_node("Mul", in_name, sigmoid_out, self.get_tensor_name(out_id)?);
        Ok(())
    }

    pub fn op_silu(&mut self, op: &Value) -> anyhow::Result<()> {
        self.op_swish(op)
    }

    pub fn op_hardswish(&mut self, op: &Value) -> anyhow::Result<()> {
        if self.target_opset >= 14 {
            self.convert_generic_op("hardswish", op)?;
            return Ok(());
        }

        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("hardswish missing inputs");
        }

        let input_name = self.get_tensor_name(inputs[0])?;
        let hard_sigmoid_out = format!("hardswish_hardsigmoid_{}", out_id);
        let mut hard_sigmoid = onnx::NodeProto {
            op_type: "HardSigmoid".to_string(),
            input: vec![input_name.clone()],
            output: vec![hard_sigmoid_out.clone()],
            ..Default::default()
        };
        hard_sigmoid
            .attribute
            .push(helper::attr_float("alpha", 1.0 / 6.0));
        hard_sigmoid.attribute.push(helper::attr_float("beta", 0.5));
        self.onnx_graph.node.push(hard_sigmoid);

        self.add_binary_node(
            "Mul",
            input_name,
            hard_sigmoid_out,
            self.get_tensor_name(out_id)?,
        );
        Ok(())
    }

    pub fn op_gelu(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("gelu missing inputs");
        }

        let in_name = self.get_tensor_name(inputs[0])?;
        let approximate = matches!(
            helper::attr(op, "approximate"),
            Some(value) if value.as_bool() == Some(true) || value.as_str() == Some("tanh")
        );
        if approximate {
            let half_name = format!("gelu_half_{}", out_id);
            let one_name = format!("gelu_one_{}", out_id);
            let coeff_name = format!("gelu_coeff_{}", out_id);
            let sqrt_2_over_pi_name = format!("gelu_sqrt_2_over_pi_{}", out_id);
            self.push_f32_initializer(half_name.clone(), vec![1], &[0.5]);
            self.push_f32_initializer(one_name.clone(), vec![1], &[1.0]);
            self.push_f32_initializer(coeff_name.clone(), vec![1], &[0.044_715]);
            self.push_f32_initializer(sqrt_2_over_pi_name.clone(), vec![1], &[0.797_884_6]);

            let x_sq = format!("gelu_x_sq_{}", out_id);
            let x_cube = format!("gelu_x_cube_{}", out_id);
            let cubic_term = format!("gelu_cubic_term_{}", out_id);
            let inner_sum = format!("gelu_inner_sum_{}", out_id);
            let scaled_inner = format!("gelu_scaled_inner_{}", out_id);
            let tanh_out = format!("gelu_tanh_{}", out_id);
            let tanh_plus_one = format!("gelu_tanh_plus_one_{}", out_id);
            let half_x = format!("gelu_half_x_{}", out_id);

            self.add_binary_node("Mul", in_name.clone(), in_name.clone(), x_sq.clone());
            self.add_binary_node("Mul", x_sq, in_name.clone(), x_cube.clone());
            self.add_binary_node("Mul", x_cube, coeff_name, cubic_term.clone());
            self.add_binary_node("Add", in_name.clone(), cubic_term, inner_sum.clone());
            self.add_binary_node("Mul", inner_sum, sqrt_2_over_pi_name, scaled_inner.clone());
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Tanh".to_string(),
                input: vec![scaled_inner],
                output: vec![tanh_out.clone()],
                ..Default::default()
            });
            self.add_binary_node("Add", tanh_out, one_name, tanh_plus_one.clone());
            self.add_binary_node("Mul", in_name, half_name, half_x.clone());
            self.add_binary_node("Mul", half_x, tanh_plus_one, self.get_tensor_name(out_id)?);
            return Ok(());
        }

        let sqrt2_name = format!("gelu_sqrt2_{}", out_id);
        let half_name = format!("gelu_half_{}", out_id);
        let one_name = format!("gelu_one_{}", out_id);

        self.push_f32_initializer(sqrt2_name.clone(), vec![1], &[std::f32::consts::SQRT_2]);
        self.push_f32_initializer(half_name.clone(), vec![1], &[0.5]);
        self.push_f32_initializer(one_name.clone(), vec![1], &[1.0]);

        let div_out = format!("gelu_div_{}", out_id);
        let erf_out = format!("gelu_erf_{}", out_id);
        let add_out = format!("gelu_add_{}", out_id);
        let mul_half_out = format!("gelu_mul_half_{}", out_id);

        self.add_binary_node("Div", in_name.clone(), sqrt2_name, div_out.clone());
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Erf".to_string(),
            input: vec![div_out],
            output: vec![erf_out.clone()],
            ..Default::default()
        });
        self.add_binary_node("Add", erf_out, one_name, add_out.clone());
        self.add_binary_node("Mul", in_name, half_name, mul_half_out.clone());
        self.add_binary_node("Mul", mul_half_out, add_out, self.get_tensor_name(out_id)?);
        Ok(())
    }

    pub fn op_prelu(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("prelu missing slope input");
        }

        let data_name = self.get_tensor_name(inputs[0])?;
        let mut slope_name = self.get_tensor_name(inputs[1])?;
        let input_rank = self
            .state
            .tensor_shapes
            .get(&inputs[0])
            .map(|shape| shape.len())
            .unwrap_or(0);
        let slope_shape = self
            .state
            .tensor_shapes
            .get(&inputs[1])
            .cloned()
            .unwrap_or_default();
        let mode = helper::attr(op, "mode")
            .and_then(|d| d.as_str())
            .unwrap_or("channel");
        let data_format = helper::attr(op, "data_format")
            .and_then(|d| d.as_str())
            .unwrap_or("NCHW");
        if mode == "element" && matches!(data_format, "NHWC" | "NDHWC") {
            bail!("prelu mode=element only supports NCHW-style layouts");
        }

        if mode != "all"
            && slope_shape.len() == 1
            && slope_shape[0] > 1
            && matches!(data_format, "NHWC" | "NDHWC")
            && input_rank >= 2
        {
            let reshape_shape_name = format!("prelu_slope_shape_{}", out_id);
            let mut reshape_shape = vec![1_i64; input_rank];
            reshape_shape[input_rank - 1] = slope_shape[0];

            let mut shape_tensor = onnx::TensorProto {
                name: reshape_shape_name.clone(),
                dims: vec![input_rank as i64],
                data_type: dt::INT64,
                ..Default::default()
            };
            for dim in reshape_shape {
                shape_tensor.raw_data.extend_from_slice(&dim.to_le_bytes());
            }
            self.onnx_graph.initializer.push(shape_tensor);

            let reshaped_slope_name = format!("prelu_slope_reshaped_{}", out_id);
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Reshape".to_string(),
                input: vec![slope_name, reshape_shape_name],
                output: vec![reshaped_slope_name.clone()],
                ..Default::default()
            });
            slope_name = reshaped_slope_name;
        }

        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "PRelu".to_string(),
            input: vec![data_name, slope_name],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        });
        Ok(())
    }

    pub fn op_floor_divide(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("floor_divide missing inputs");
        }

        let output_tt = op
            .get("O")
            .and_then(|o| o.as_array())
            .and_then(|o| o.first())
            .and_then(|o| o.get("TT"))
            .and_then(|tt| tt.get("D"))
            .and_then(|d| d.as_array())
            .and_then(|d| d.first())
            .and_then(|t| t.get("#"))
            .and_then(|t| t.as_str())
            .unwrap_or(helper::paddle_tt::I64);
        let output_dtype = helper::paddle_elem_type_to_onnx(output_tt).unwrap_or(dt::INT64);
        let lhs_dtype = self.onnx_dtype_for_tensor_id(inputs[0])?;
        let rhs_dtype = self.onnx_dtype_for_tensor_id(inputs[1])?;
        let lhs_name = self.get_tensor_name(inputs[0])?;
        let rhs_name = self.get_tensor_name(inputs[1])?;

        let is_integer_floor_divide = matches!(
            lhs_dtype,
            dt::INT8 | dt::UINT8 | dt::INT16 | dt::INT32 | dt::INT64
        ) && matches!(
            rhs_dtype,
            dt::INT8 | dt::UINT8 | dt::INT16 | dt::INT32 | dt::INT64
        );
        if is_integer_floor_divide {
            let div_out = format!("floor_divide_div_{}", out_id);
            let mod_out = format!("floor_divide_mod_{}", out_id);
            let zero_name = format!("floor_divide_zero_{}", out_id);
            let lhs_negative = format!("floor_divide_lhs_negative_{}", out_id);
            let rhs_negative = format!("floor_divide_rhs_negative_{}", out_id);
            let signs_differ = format!("floor_divide_signs_differ_{}", out_id);
            let remainder_nonzero = format!("floor_divide_remainder_nonzero_{}", out_id);
            let needs_adjust = format!("floor_divide_needs_adjust_{}", out_id);
            let adjust_int = format!("floor_divide_adjust_i64_{}", out_id);
            let adjust_name = format!("floor_divide_adjust_{}", out_id);
            let one_name = format!("floor_divide_one_{}", out_id);

            self.push_numeric_initializer(zero_name.clone(), vec![], output_dtype, &[0.0])?;
            self.push_numeric_initializer(one_name.clone(), vec![], output_dtype, &[1.0])?;

            self.add_binary_node("Div", lhs_name.clone(), rhs_name.clone(), div_out.clone());
            self.add_integer_mod_node(lhs_name.clone(), rhs_name.clone(), mod_out.clone());
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Less".to_string(),
                input: vec![lhs_name, zero_name.clone()],
                output: vec![lhs_negative.clone()],
                ..Default::default()
            });
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Less".to_string(),
                input: vec![rhs_name, zero_name.clone()],
                output: vec![rhs_negative.clone()],
                ..Default::default()
            });
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Xor".to_string(),
                input: vec![lhs_negative, rhs_negative],
                output: vec![signs_differ.clone()],
                ..Default::default()
            });
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Equal".to_string(),
                input: vec![mod_out, zero_name],
                output: vec![format!("floor_divide_remainder_zero_{}", out_id)],
                ..Default::default()
            });
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Not".to_string(),
                input: vec![format!("floor_divide_remainder_zero_{}", out_id)],
                output: vec![remainder_nonzero.clone()],
                ..Default::default()
            });
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "And".to_string(),
                input: vec![signs_differ, remainder_nonzero],
                output: vec![needs_adjust.clone()],
                ..Default::default()
            });
            self.add_cast_node(needs_adjust, adjust_int.clone(), output_dtype);
            self.add_binary_node("Mul", adjust_int, one_name, adjust_name.clone());
            self.add_binary_node("Sub", div_out, adjust_name, self.get_tensor_name(out_id)?);
            return Ok(());
        }

        let cast_a = format!("floor_divide_a_{}", out_id);
        let cast_b = format!("floor_divide_b_{}", out_id);
        let div_out = format!("floor_divide_div_{}", out_id);
        let floor_out = format!("floor_divide_floor_{}", out_id);
        for (input_name, output_name) in [
            (self.get_tensor_name(inputs[0])?, cast_a.clone()),
            (self.get_tensor_name(inputs[1])?, cast_b.clone()),
        ] {
            self.add_cast_node(input_name, output_name, dt::FLOAT);
        }

        self.add_binary_node("Div", cast_a, cast_b, div_out.clone());
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Floor".to_string(),
            input: vec![div_out],
            output: vec![floor_out.clone()],
            ..Default::default()
        });

        self.add_cast_node(floor_out, self.get_tensor_name(out_id)?, output_dtype);
        Ok(())
    }

    pub fn op_grid_sample(&mut self, op: &Value) -> anyhow::Result<()> {
        self.require_opset(16, "grid_sample")?;
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("grid_sample missing inputs");
        }

        let mut node = onnx::NodeProto {
            op_type: "GridSample".to_string(),
            input: vec![
                self.get_tensor_name(inputs[0])?,
                self.get_tensor_name(inputs[1])?,
            ],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        };

        if let Some(mode) = helper::attr(op, "mode").and_then(|d| d.as_str()) {
            // GridSample renamed bilinear/bicubic to linear/cubic in opset 20.
            let onnx_mode = match mode {
                "bilinear" if self.target_opset >= 20 => "linear",
                "bicubic" if self.target_opset >= 20 => "cubic",
                other => other,
            };
            node.attribute.push(helper::attr_str("mode", onnx_mode));
        }
        if let Some(mode) = helper::attr(op, "padding_mode").and_then(|d| d.as_str()) {
            node.attribute.push(helper::attr_str("padding_mode", mode));
        }
        node.attribute.push(helper::attr_int(
            "align_corners",
            i64::from(
                helper::attr(op, "align_corners")
                    .and_then(|d| d.as_bool())
                    .unwrap_or(false),
            ),
        ));
        self.onnx_graph.node.push(node);
        Ok(())
    }

    pub fn op_trilu(&mut self, op: &Value, upper: bool) -> anyhow::Result<()> {
        self.require_opset(14, if upper { "triu" } else { "tril" })?;
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("trilu missing inputs");
        }
        let diagonal = helper::attr(op, "diagonal")
            .and_then(|d| d.as_i64())
            .unwrap_or(0);
        let k_name = format!("trilu_k_{}", out_id);
        self.push_i64_initializer(k_name.clone(), vec![1], &[diagonal]);

        let mut node = onnx::NodeProto {
            op_type: "Trilu".to_string(),
            input: vec![self.get_tensor_name(inputs[0])?, k_name],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        };
        node.attribute
            .push(helper::attr_int("upper", i64::from(upper)));
        self.onnx_graph.node.push(node);
        Ok(())
    }

    pub fn op_pow(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("pow missing inputs");
        }

        let mut node_inputs = vec![self.get_tensor_name(inputs[0])?];
        if inputs.len() > 1 {
            node_inputs.push(self.get_tensor_name(inputs[1])?);
        } else {
            let exponent = helper::attr(op, "y")
                .and_then(|d| d.as_f64())
                .ok_or_else(|| anyhow::anyhow!("pow: missing exponent"))?;
            let exponent_name = format!("pow_exponent_{}", out_id);
            let data_type = self
                .maybe_onnx_dtype_for_tensor_id(inputs[0])?
                .unwrap_or(dt::FLOAT);
            self.push_numeric_initializer(exponent_name.clone(), vec![], data_type, &[exponent])?;
            node_inputs.push(exponent_name);
        }

        self.add_binary_node(
            "Pow",
            node_inputs[0].clone(),
            node_inputs[1].clone(),
            self.get_tensor_name(out_id)?,
        );
        Ok(())
    }

    pub fn op_leaky_relu(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("leaky_relu missing inputs");
        }

        // Use attr_f64 which handles string-encoded floats, integers,
        // and special float representations (inf/nan) robustly
        let alpha = helper::attr_f64(op, "alpha")
            .or_else(|| helper::attr_f64(op, "negative_slope"))
            .unwrap_or(0.01);

        let mut node = onnx::NodeProto {
            op_type: "LeakyRelu".to_string(),
            input: vec![self.get_tensor_name(inputs[0])?],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        };
        node.attribute
            .push(helper::attr_float("alpha", alpha as f32));
        self.onnx_graph.node.push(node);
        Ok(())
    }

    pub fn op_linear_v2(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("linear_v2 needs at least input and weight");
        }

        let in_name = self.get_tensor_name(inputs[0])?;
        let weight_name = self.get_tensor_name(inputs[1])?;

        // PIR weight is stored as [in_dim, out_dim] (unlike PyTorch [out_dim, in_dim])
        // So ONNX MatMul: X[B,T,in_dim] @ W[in_dim,out_dim] = [B,T,out_dim]
        // No transpose needed! Just MatMul + optional Add bias.
        //
        // In PIR, an optional bias that is omitted has input ID 0.
        let has_bias = inputs.len() > 2 && inputs[2] != 0;
        let mm_output = if has_bias {
            format!("linear_mm_{}", out_id)
        } else {
            self.get_tensor_name(out_id)?
        };

        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "MatMul".to_string(),
            input: vec![in_name, weight_name],
            output: vec![mm_output.clone()],
            ..Default::default()
        });

        if has_bias {
            let bias_name = self.get_tensor_name(inputs[2])?;
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Add".to_string(),
                input: vec![mm_output, bias_name],
                output: vec![self.get_tensor_name(out_id)?],
                ..Default::default()
            });
        }
        Ok(())
    }

    pub fn op_multinomial(&mut self, op: &Value) -> anyhow::Result<()> {
        self.require_opset(7, "multinomial")?;

        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("multinomial missing inputs");
        }
        let sample_size = self
            .state
            .constants
            .get(&inputs[1])
            .and_then(|values| values.first())
            .copied()
            .ok_or_else(|| anyhow::anyhow!("multinomial requires constant sample size"))?
            as i64;
        if sample_size <= 0 {
            bail!("multinomial requires positive sample size");
        }

        let replacement = helper::attr(op, "replacement")
            .and_then(|d| d.as_bool())
            .unwrap_or(false);
        if sample_size != 1 && !replacement {
            bail!("multinomial only supports replacement=false when sample_size=1");
        }
        if self.strict {
            bail!("strict mode rejects deterministic multinomial -> ArgMax lowering");
        }

        if !self.warned_multinomial_degraded {
            log::warn!(
                "multinomial is lowered to deterministic ArgMax for inference; stochastic sampling is not preserved"
            );
            self.warned_multinomial_degraded = true;
        }

        let argmax_output = if sample_size == 1 {
            self.get_tensor_name(out_id)?
        } else {
            format!("multinomial_argmax_{}", out_id)
        };
        let mut node = onnx::NodeProto {
            op_type: "ArgMax".to_string(),
            input: vec![self.get_tensor_name(inputs[0])?],
            output: vec![argmax_output.clone()],
            ..Default::default()
        };
        // Deterministic inference contract: lower to argmax instead of stochastic sampling.
        node.attribute.push(helper::attr_int("axis", -1));
        node.attribute.push(helper::attr_int("keepdims", 1));
        node.attribute
            .push(helper::attr_int("select_last_index", 0));
        self.onnx_graph.node.push(node);

        if sample_size > 1 {
            let repeats_name = format!("multinomial_repeats_{}", out_id);
            self.push_i64_initializer(repeats_name.clone(), vec![2], &[1, sample_size]);
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Tile".to_string(),
                input: vec![argmax_output, repeats_name],
                output: vec![self.get_tensor_name(out_id)?],
                ..Default::default()
            });
        }
        Ok(())
    }

    pub fn op_unbind(&mut self, op: &Value) -> anyhow::Result<()> {
        let vec_out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("unbind missing inputs");
        }
        let input_id = inputs[0];
        let output_ids =
            self.state.splits.get(&vec_out_id).cloned().ok_or_else(|| {
                anyhow::anyhow!("unbind missing 0.split metadata for {}", vec_out_id)
            })?;
        let axis = self.normalize_axis(
            input_id,
            helper::attr(op, "axis")
                .and_then(|d| d.as_i64())
                .unwrap_or(0),
        )?;

        let split_outputs = output_ids
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("unbind_split_{}_{}", vec_out_id, idx))
            .collect::<Vec<_>>();
        let mut split = onnx::NodeProto {
            op_type: "Split".to_string(),
            input: vec![self.get_tensor_name(input_id)?],
            output: split_outputs.clone(),
            ..Default::default()
        };
        split.attribute.push(helper::attr_int("axis", axis));
        if self.target_opset >= 18 {
            split
                .attribute
                .push(helper::attr_int("num_outputs", split_outputs.len() as i64));
        }
        self.onnx_graph.node.push(split);

        for (idx, output_id) in output_ids.iter().enumerate() {
            self.add_squeeze_node(
                split_outputs[idx].clone(),
                self.get_tensor_name(*output_id)?,
                Some(&[axis]),
                Some(format!("unbind_axes_{}_{}", vec_out_id, idx)),
            );
        }
        Ok(())
    }

    pub fn op_bitwise_not(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            bail!("bitwise_not missing inputs");
        }
        let input_name = self.get_tensor_name(inputs[0])?;
        let input_dtype = self.onnx_dtype_for_tensor_id(inputs[0])?;
        if input_dtype == dt::BOOL {
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Not".to_string(),
                input: vec![input_name],
                output: vec![self.get_tensor_name(out_id)?],
                ..Default::default()
            });
            return Ok(());
        }

        match input_dtype {
            dt::INT8 | dt::INT16 | dt::INT32 | dt::INT64 => {
                if self.target_opset >= 18 {
                    self.onnx_graph.node.push(onnx::NodeProto {
                        op_type: "BitwiseNot".to_string(),
                        input: vec![input_name],
                        output: vec![self.get_tensor_name(out_id)?],
                        ..Default::default()
                    });
                    return Ok(());
                }

                // Paddle integer tensors are materialized as standard signed
                // two's-complement integers, so `~x == -(x + 1)` preserves the
                // bitwise-not semantics even before ONNX added BitwiseNot.
                let one_name = format!("bitwise_not_one_{}", out_id);
                self.push_numeric_initializer(one_name.clone(), vec![], input_dtype, &[1.0])?;
                let plus_one_name = format!("bitwise_not_plus_one_{}", out_id);
                self.add_binary_node("Add", input_name, one_name, plus_one_name.clone());
                self.onnx_graph.node.push(onnx::NodeProto {
                    op_type: "Neg".to_string(),
                    input: vec![plus_one_name],
                    output: vec![self.get_tensor_name(out_id)?],
                    ..Default::default()
                });
                Ok(())
            }
            dt::UINT8 => {
                self.require_opset(18, "bitwise_not on integer tensors")?;
                self.onnx_graph.node.push(onnx::NodeProto {
                    op_type: "BitwiseNot".to_string(),
                    input: vec![input_name],
                    output: vec![self.get_tensor_name(out_id)?],
                    ..Default::default()
                });
                Ok(())
            }
            dt::UINT16 | dt::BFLOAT16 => bail!(
                "bitwise_not does not support {} in this converter; Paddle PIR is not expected to emit this integer type here",
                helper::onnx_dtype_name(input_dtype)
            ),
            _ => bail!(
                "bitwise_not only supports bool/int8/uint8/int16/int32/int64 tensors in the current converter, got {}",
                helper::onnx_dtype_name(input_dtype)
            ),
        }
    }

    pub fn op_bitwise_and(&mut self, op: &Value) -> anyhow::Result<()> {
        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("bitwise_and requires two inputs");
        }

        let lhs_name = self.get_tensor_name(inputs[0])?;
        let rhs_name = self.get_tensor_name(inputs[1])?;
        let output_name = self.get_tensor_name(out_id)?;
        let lhs_dtype = self.onnx_dtype_for_tensor_id(inputs[0])?;
        let rhs_dtype = self.onnx_dtype_for_tensor_id(inputs[1])?;
        if lhs_dtype != rhs_dtype {
            bail!(
                "bitwise_and requires matching input dtypes, got {} and {}",
                helper::onnx_dtype_name(lhs_dtype),
                helper::onnx_dtype_name(rhs_dtype)
            );
        }

        match lhs_dtype {
            dt::BOOL => {
                self.add_binary_node("And", lhs_name, rhs_name, output_name);
                Ok(())
            }
            dt::INT8 | dt::INT16 | dt::INT32 | dt::INT64 => {
                if self.target_opset >= 18 {
                    self.add_binary_node("BitwiseAnd", lhs_name, rhs_name, output_name);
                    return Ok(());
                }

                self.lower_signed_bitwise_and_before_opset_18(
                    lhs_name,
                    rhs_name,
                    output_name,
                    lhs_dtype,
                    out_id,
                )
            }
            dt::UINT8 => {
                self.require_opset(18, "bitwise_and on integer tensors")?;
                self.add_binary_node("BitwiseAnd", lhs_name, rhs_name, output_name);
                Ok(())
            }
            dt::UINT16 | dt::BFLOAT16 => bail!(
                "bitwise_and does not support {} in this converter; Paddle PIR is not expected to emit this integer type here",
                helper::onnx_dtype_name(lhs_dtype)
            ),
            _ => bail!(
                "bitwise_and only supports bool/int8/uint8/int16/int32/int64 tensors in the current converter, got {}",
                helper::onnx_dtype_name(lhs_dtype)
            ),
        }
    }
}
