use crate::helper::{self, dt};
use crate::proto::onnx;
use serde_json::Value;

impl super::super::Converter {
    fn add_empty_tensor_initializer(&mut self, name: String, data_type: i32) -> String {
        self.onnx_graph.initializer.push(onnx::TensorProto {
            name: name.clone(),
            dims: vec![0],
            data_type,
            ..Default::default()
        });
        name
    }

    fn push_resize_optional_input(
        &mut self,
        node: &mut onnx::NodeProto,
        out_id: i64,
        slot: &str,
        data_type: i32,
    ) {
        // ONNX Resize uses an empty-string sentinel for omitted optional inputs from opset 13
        // onward. Older opsets require a typed empty initializer in the same input slot.
        if self.target_opset >= 13 {
            node.input.push(String::new());
        } else {
            node.input.push(self.add_empty_tensor_initializer(
                format!("resize_empty_{}_{}", slot, out_id),
                data_type,
            ));
        }
    }

    fn build_resize_sizes_from_spatial_tensor(
        &mut self,
        out_id: i64,
        data_input_id: i64,
        spatial_name: String,
    ) -> anyhow::Result<String> {
        let input_name = self.get_tensor_name(data_input_id)?;
        let shape_name = format!("resize_shape_{}", out_id);
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Shape".to_string(),
            input: vec![input_name],
            output: vec![shape_name.clone()],
            ..Default::default()
        });

        let prefix_indices_name = format!("resize_prefix_indices_{}", out_id);
        self.push_i64_initializer(prefix_indices_name.clone(), vec![2], &[0, 1]);

        let prefix_name = format!("resize_prefix_{}", out_id);
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Gather".to_string(),
            input: vec![shape_name, prefix_indices_name],
            output: vec![prefix_name.clone()],
            ..Default::default()
        });

        let sizes_name = format!("resize_sizes_{}", out_id);
        let mut concat = onnx::NodeProto {
            op_type: "Concat".to_string(),
            input: vec![prefix_name, spatial_name],
            output: vec![sizes_name.clone()],
            ..Default::default()
        };
        concat.attribute.push(helper::attr_int("axis", 0));
        self.onnx_graph.node.push(concat);
        Ok(sizes_name)
    }

    fn build_resize_sizes_from_spatial_list(
        &mut self,
        out_id: i64,
        data_input_id: i64,
        size_input_id: i64,
    ) -> anyhow::Result<Option<String>> {
        let spatial_ids = match self.state.combines.get(&size_input_id) {
            Some(ids) if !ids.is_empty() => ids.clone(),
            _ => return Ok(None),
        };

        let mut spatial_inputs = Vec::with_capacity(spatial_ids.len());
        for (index, spatial_id) in spatial_ids.iter().enumerate() {
            let spatial_tensor_name = self.get_tensor_name(*spatial_id)?;
            let spatial_shape = self
                .state
                .tensor_shapes
                .get(spatial_id)
                .cloned()
                .unwrap_or_default();
            if spatial_shape.is_empty() {
                let expanded_name = format!("resize_spatial_{}_{}", out_id, index);
                self.add_unsqueeze_node(
                    spatial_tensor_name,
                    expanded_name.clone(),
                    &[0],
                    format!("resize_unsqueeze_axes_{}", out_id),
                );
                spatial_inputs.push(expanded_name);
            } else {
                spatial_inputs.push(spatial_tensor_name);
            }
        }

        let spatial_name = if spatial_inputs.len() == 1 {
            spatial_inputs.remove(0)
        } else {
            let spatial_name = format!("resize_hw_{}", out_id);
            let mut concat = onnx::NodeProto {
                op_type: "Concat".to_string(),
                input: spatial_inputs,
                output: vec![spatial_name.clone()],
                ..Default::default()
            };
            concat.attribute.push(helper::attr_int("axis", 0));
            self.onnx_graph.node.push(concat);
            spatial_name
        };

        self.build_resize_sizes_from_spatial_tensor(out_id, data_input_id, spatial_name)
            .map(Some)
    }

    fn op_resize_interp(&mut self, op: &Value, mode: &str) -> anyhow::Result<()> {
        self.require_opset(11, mode)?;
        let data_format = helper::attr(op, "data_format")
            .and_then(|d| d.as_str())
            .unwrap_or("NCHW");
        if data_format != "NCHW" {
            anyhow::bail!(
                "{} currently only supports data_format=NCHW (got {})",
                mode,
                data_format
            );
        }
        let align_corners = helper::attr(op, "align_corners")
            .and_then(|d| d.as_bool())
            .unwrap_or(false);
        let align_mode = helper::attr(op, "align_mode")
            .and_then(|d| d.as_i64())
            .unwrap_or(1);
        // Paddle nearest_interp matches ORT Resize with asymmetric coordinates and floor
        // rounding for the models we convert here, including SLANet's odd-sized CSPPAN
        // upsampling path (e.g. 16 -> 31). The half_pixel/round_prefer_ceil mapping
        // shifts samples and causes large feature drift.
        let coordinate_mode = if align_corners {
            "align_corners"
        } else if mode == "nearest" {
            "asymmetric"
        } else if align_mode == 0 {
            "half_pixel"
        } else {
            "asymmetric"
        };
        let nearest_mode = "floor";

        let mut onnx_node = onnx::NodeProto {
            op_type: "Resize".to_string(),
            ..Default::default()
        };

        onnx_node.attribute.push(helper::attr_str("mode", mode));
        onnx_node.attribute.push(helper::attr_str(
            "coordinate_transformation_mode",
            coordinate_mode,
        ));
        if mode == "nearest" {
            onnx_node
                .attribute
                .push(helper::attr_str("nearest_mode", nearest_mode));
        }

        let out_id = helper::op_out_id(op)?;
        let inputs = helper::op_input_ids(op);
        if !inputs.is_empty() {
            onnx_node.input.push(self.get_tensor_name(inputs[0])?);
        }
        self.push_resize_optional_input(&mut onnx_node, out_id, "roi", dt::FLOAT);
        if inputs.len() > 2
            && inputs[2] != 0
            && let Some(sizes_name) =
                self.build_resize_sizes_from_spatial_list(out_id, inputs[0], inputs[2])?
        {
            self.push_resize_optional_input(&mut onnx_node, out_id, "scales", dt::FLOAT);
            onnx_node.input.push(sizes_name);
        } else if let Some(output_shape) = self.state.tensor_shapes.get(&out_id).cloned()
            && output_shape.len() == 4
            && output_shape[2] > 0
            && output_shape[3] > 0
        {
            self.push_resize_optional_input(&mut onnx_node, out_id, "scales", dt::FLOAT);

            let hw_name = format!("resize_static_hw_{}", out_id);
            self.push_i64_initializer(hw_name.clone(), vec![2], &output_shape[2..4]);

            let sizes_name =
                self.build_resize_sizes_from_spatial_tensor(out_id, inputs[0], hw_name)?;
            onnx_node.input.push(sizes_name);
        } else {
            let scale_name = format!("scales_{}", out_id);
            onnx_node.input.push(scale_name.clone());
            let mut scale_vals = vec![1.0f32, 1.0, 2.0, 2.0];
            if let Some(d_arr) = helper::attr(op, "scale").and_then(|d| d.as_array())
                && d_arr.len() >= 2
            {
                scale_vals[2] = d_arr[0].get("D").and_then(|v| v.as_f64()).unwrap_or(2.0) as f32;
                scale_vals[3] = d_arr[1].get("D").and_then(|v| v.as_f64()).unwrap_or(2.0) as f32;
            }

            self.push_f32_initializer(scale_name, vec![4], &scale_vals);
        }

        onnx_node.output.push(self.get_tensor_name(out_id)?);
        self.onnx_graph.node.push(onnx_node);
        Ok(())
    }

    pub fn op_nearest_interp(&mut self, op: &Value) -> anyhow::Result<()> {
        self.op_resize_interp(op, "nearest")
    }

    pub fn op_bilinear_interp(&mut self, op: &Value) -> anyhow::Result<()> {
        self.op_resize_interp(op, "linear")
    }
}
