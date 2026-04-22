use crate::converter::Converter;
use crate::helper;
use crate::proto::onnx;
use anyhow::bail;
use serde_json::Value;

impl Converter {
    pub(crate) fn op_batch_norm(&mut self, op: &Value) -> anyhow::Result<()> {
        if matches!(
            helper::attr(op, "is_test").and_then(|d| d.as_bool()),
            Some(false)
        ) {
            bail!("batch_norm_ only supports inference-style lowering with is_test=true");
        }
        let data_format = helper::attr(op, "data_format")
            .and_then(|d| d.as_str())
            .unwrap_or("NCHW");
        if data_format != "NCHW" {
            bail!("batch_norm_ only supports NCHW");
        }

        let inputs = helper::op_input_ids(op);
        if inputs.len() < 5 {
            bail!("batch_norm_ missing inputs");
        }
        let mut onnx_node = onnx::NodeProto {
            op_type: "BatchNormalization".to_string(),
            input: vec![
                self.get_tensor_name(inputs[0])?,
                self.get_tensor_name(inputs[3])?,
                self.get_tensor_name(inputs[4])?,
                self.get_tensor_name(inputs[1])?,
                self.get_tensor_name(inputs[2])?,
            ],
            ..Default::default()
        };
        if let Some(attrs) = op.get("A") {
            onnx_node.attribute = self.extract_attributes("batch_norm_", attrs);
        }
        let out_id = helper::op_out_id(op)?;
        onnx_node.output.push(self.get_tensor_name(out_id)?);
        self.onnx_graph.node.push(onnx_node);
        Ok(())
    }

    pub fn op_group_norm(&mut self, op: &Value) -> anyhow::Result<()> {
        self.require_opset(6, "group_norm")?;

        let inputs = helper::op_input_ids(op);
        if inputs.len() < 3 {
            bail!("group_norm missing inputs");
        }
        let outputs = op
            .get("O")
            .and_then(|o| o.as_array())
            .ok_or_else(|| anyhow::anyhow!("group_norm missing outputs"))?;
        let out_id = outputs[0]
            .get("%")
            .and_then(|id| id.as_i64())
            .ok_or_else(|| anyhow::anyhow!("group_norm missing output id"))?;
        let groups = helper::attr(op, "groups")
            .and_then(|d| d.as_i64())
            .ok_or_else(|| anyhow::anyhow!("group_norm missing groups"))?;
        let epsilon = helper::attr(op, "epsilon")
            .and_then(|d| d.as_f64())
            .unwrap_or(1e-5) as f32;
        let data_format = helper::attr(op, "data_format")
            .and_then(|d| d.as_str())
            .unwrap_or("NCHW");
        if data_format != "NCHW" {
            bail!("group_norm only supports NCHW");
        }

        let input_id = inputs[0];
        let input_rank = self
            .state
            .tensor_shapes
            .get(&input_id)
            .map(|shape| shape.len())
            .ok_or_else(|| anyhow::anyhow!("group_norm missing input rank metadata"))?;
        if input_rank < 3 {
            bail!("group_norm expects rank >= 3, got {}", input_rank);
        }

        let input_name = self.get_tensor_name(input_id)?;
        let input_shape_name = format!("group_norm_input_shape_{}", out_id);
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Shape".to_string(),
            input: vec![input_name.clone()],
            output: vec![input_shape_name.clone()],
            ..Default::default()
        });

        let batch_dim_name = format!("group_norm_batch_dim_{}", out_id);
        self.add_slice_node(
            input_shape_name.clone(),
            batch_dim_name.clone(),
            &[0],
            &[1],
            Some(&[0]),
            None,
            &format!("group_norm_batch_dim_{}", out_id),
        )?;

        let groups_name = format!("group_norm_groups_{}", out_id);
        self.push_i64_initializer(groups_name.clone(), vec![1], &[groups]);
        let minus_one_name = format!("group_norm_minus_one_{}", out_id);
        self.push_i64_initializer(minus_one_name.clone(), vec![1], &[-1]);

        let reshape_shape_name = format!("group_norm_reshape_shape_{}", out_id);
        let mut reshape_shape = onnx::NodeProto {
            op_type: "Concat".to_string(),
            input: vec![batch_dim_name, groups_name, minus_one_name],
            output: vec![reshape_shape_name.clone()],
            ..Default::default()
        };
        reshape_shape.attribute.push(helper::attr_int("axis", 0));
        self.onnx_graph.node.push(reshape_shape);

        let reshaped_name = format!("group_norm_reshaped_{}", out_id);
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Reshape".to_string(),
            input: vec![input_name, reshape_shape_name],
            output: vec![reshaped_name.clone()],
            ..Default::default()
        });

        let scale_name = format!("group_norm_scale_{}", out_id);
        let bias_name = format!("group_norm_bias_{}", out_id);
        self.push_f32_initializer(
            scale_name.clone(),
            vec![groups],
            &vec![1.0; groups as usize],
        );
        self.push_f32_initializer(bias_name.clone(), vec![groups], &vec![0.0; groups as usize]);

        let normalized_name = format!("group_norm_normalized_{}", out_id);
        let mut instance_norm = onnx::NodeProto {
            op_type: "InstanceNormalization".to_string(),
            input: vec![reshaped_name, scale_name, bias_name],
            output: vec![normalized_name.clone()],
            ..Default::default()
        };
        instance_norm
            .attribute
            .push(helper::attr_float("epsilon", epsilon));
        self.onnx_graph.node.push(instance_norm);

        let restored_name = format!("group_norm_restored_{}", out_id);
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Reshape".to_string(),
            input: vec![normalized_name, input_shape_name],
            output: vec![restored_name.clone()],
            ..Default::default()
        });

        let affine_axes = std::iter::once(0)
            .chain(2..input_rank as i64)
            .collect::<Vec<_>>();
        let scale_broadcast_name = format!("group_norm_affine_scale_{}", out_id);
        let bias_broadcast_name = format!("group_norm_affine_bias_{}", out_id);
        self.add_unsqueeze_node(
            self.get_tensor_name(inputs[1])?,
            scale_broadcast_name.clone(),
            &affine_axes,
            format!("group_norm_affine_scale_axes_{}", out_id),
        );
        self.add_unsqueeze_node(
            self.get_tensor_name(inputs[2])?,
            bias_broadcast_name.clone(),
            &affine_axes,
            format!("group_norm_affine_bias_axes_{}", out_id),
        );

        let scaled_name = format!("group_norm_scaled_{}", out_id);
        self.add_binary_node(
            "Mul",
            restored_name,
            scale_broadcast_name,
            scaled_name.clone(),
        );
        self.add_binary_node(
            "Add",
            scaled_name,
            bias_broadcast_name,
            self.get_tensor_name(out_id)?,
        );
        Ok(())
    }
}
