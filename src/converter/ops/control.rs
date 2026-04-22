use crate::helper::{self, dt};
use crate::proto::onnx;
use serde_json::Value;
use std::collections::HashSet;

impl super::super::Converter {
    fn convert_region_to_graph(
        &self,
        region: &Value,
        graph_name: &str,
        extra_inputs: Vec<onnx::ValueInfoProto>,
        block_arg_names: &[String],
    ) -> anyhow::Result<onnx::GraphProto> {
        let block = region
            .get("blocks")
            .and_then(|blocks| blocks.as_array())
            .and_then(|blocks| blocks.first())
            .ok_or_else(|| anyhow::anyhow!("Region is missing blocks"))?;

        let args = block
            .get("args")
            .and_then(|args| args.as_array())
            .cloned()
            .unwrap_or_default();
        if args.len() != block_arg_names.len() {
            anyhow::bail!(
                "Block arg count mismatch for {}: expected {}, got {}",
                graph_name,
                block_arg_names.len(),
                args.len()
            );
        }

        let mut sub = self.sub_converter();
        let mut graph = onnx::GraphProto {
            name: graph_name.to_string(),
            input: extra_inputs,
            ..Default::default()
        };

        for (arg, input_name) in args.iter().zip(block_arg_names.iter()) {
            let arg_id = arg
                .get("#")
                .and_then(|id| id.as_i64())
                .ok_or_else(|| anyhow::anyhow!("Block arg missing id"))?;
            sub.state.id_to_name.insert(arg_id, input_name.clone());
            if let Some(tt) = arg
                .get("TT")
                .and_then(|tt| tt.get("D"))
                .and_then(|d| d.as_array())
                && let Some(elem_type_str) =
                    tt.first().and_then(|t| t.get("#")).and_then(|t| t.as_str())
            {
                let dims = tt
                    .get(1)
                    .and_then(|dims| dims.as_array())
                    .map(|dims| {
                        dims.iter()
                            .filter_map(|dim| dim.as_i64())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                graph.input.push(sub.build_value_info_from_meta(
                    input_name.clone(),
                    elem_type_str,
                    &dims,
                )?);
            }
        }

        let ops = block
            .get("ops")
            .and_then(|ops| ops.as_array())
            .ok_or_else(|| anyhow::anyhow!("Block is missing ops"))?;

        // Subgraphs need their own pass1 so local constants/metadata produced
        // inside the region are visible to pass2 lowering in that same region.
        sub.collect_pass1_from_ops(ops)?;

        let mut yield_ids: Option<Vec<i64>> = None;
        for op in ops {
            match helper::op_type(op) {
                Some("2.yield") => yield_ids = Some(helper::op_input_ids(op)),
                Some(op_type) => sub.process_pass2_op(op_type, op)?,
                None => {}
            }
        }

        let yield_ids =
            yield_ids.ok_or_else(|| anyhow::anyhow!("Region {} is missing 2.yield", graph_name))?;
        let mut defined = HashSet::new();
        for input in &graph.input {
            defined.insert(input.name.clone());
        }
        for init in &sub.onnx_graph.initializer {
            defined.insert(init.name.clone());
        }
        for node in &sub.onnx_graph.node {
            for output in &node.output {
                defined.insert(output.clone());
            }
        }
        let mut extra_output_nodes = Vec::new();
        for (index, id) in yield_ids.into_iter().enumerate() {
            let name = sub.get_tensor_name(id)?;
            let output_name = if defined.contains(&name) {
                name
            } else {
                let aliased = format!("{}_yield_{}", graph_name, index);
                extra_output_nodes.push(onnx::NodeProto {
                    op_type: "Identity".to_string(),
                    input: vec![name],
                    output: vec![aliased.clone()],
                    ..Default::default()
                });
                aliased
            };
            graph
                .output
                .push(sub.build_value_info_for_id(id, output_name)?);
        }
        graph.node = sub.onnx_graph.node;
        graph.node.extend(extra_output_nodes);
        graph.initializer = sub.onnx_graph.initializer;
        graph.value_info = sub.onnx_graph.value_info;
        Ok(graph)
    }

    pub fn op_any_all(&mut self, op: &Value, is_any: bool) -> anyhow::Result<()> {
        let inputs = helper::op_input_ids(op);
        let input_id = *inputs
            .first()
            .ok_or_else(|| anyhow::anyhow!("any/all requires an input tensor"))?;
        let out_id = helper::op_out_id(op)?;

        let input_name = self.get_tensor_name(input_id)?;
        let cast_name = format!("{}_cast_{}", if is_any { "any" } else { "all" }, out_id);
        let reduced_name = format!("{}_reduced_{}", if is_any { "any" } else { "all" }, out_id);

        self.add_cast_node(input_name, cast_name.clone(), dt::INT64);

        let reduce_op_type = if is_any { "ReduceMax" } else { "ReduceMin" };
        let parsed_axes = helper::attr(op, "axis")
            .and_then(|axis| axis.as_array())
            .map(|axes| {
                axes.iter()
                    .filter_map(|axis| {
                        axis.get("D")
                            .and_then(|d| d.as_i64())
                            .or_else(|| axis.as_i64())
                    })
                    .collect::<Vec<_>>()
            });
        self.add_reduce_node(
            reduce_op_type,
            cast_name,
            reduced_name.clone(),
            parsed_axes.as_deref().filter(|axes| !axes.is_empty()),
            if helper::attr(op, "keepdim")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                1
            } else {
                0
            },
            &format!("{}_reduce_{}", if is_any { "any" } else { "all" }, out_id),
        );

        self.add_cast_node(reduced_name, self.get_tensor_name(out_id)?, dt::BOOL);
        Ok(())
    }

    pub fn op_one_hot(&mut self, op: &Value) -> anyhow::Result<()> {
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            anyhow::bail!("one_hot requires indices and depth inputs");
        }
        let out_id = helper::op_out_id(op)?;
        let indices_rank = self
            .state
            .tensor_shapes
            .get(&inputs[0])
            .map(|shape| shape.len())
            .unwrap_or(1);
        let indices_name = self.get_tensor_name(inputs[0])?;
        let indices_cast_name = format!("one_hot_indices_i64_{}", out_id);
        let depth_cast_name = format!("one_hot_depth_i64_{}", out_id);
        let depth_scalar_name = format!("one_hot_depth_scalar_{}", out_id);
        let range_start_name = format!("one_hot_start_{}", out_id);
        let range_step_name = format!("one_hot_step_{}", out_id);
        let range_squeeze_axes_name = format!("one_hot_depth_squeeze_axes_{}", out_id);
        let range_name = format!("one_hot_range_{}", out_id);
        let axes_name = format!("one_hot_unsqueeze_axes_{}", out_id);
        let expanded_indices_name = format!("one_hot_indices_expanded_{}", out_id);
        let equal_name = format!("one_hot_equal_{}", out_id);
        let output_name = self.get_tensor_name(out_id)?;

        self.add_cast_node(indices_name, indices_cast_name.clone(), dt::INT64);

        let mut depth_scalar_input =
            if let Some(depth_values) = self.state.constants.get(&inputs[1]).cloned() {
                let Some(&depth) = depth_values.first() else {
                    anyhow::bail!("one_hot depth constant is empty");
                };
                if !depth.is_finite() || depth.trunc() != depth {
                    anyhow::bail!("one_hot depth must be an integer scalar, got {}", depth);
                }
                self.push_i64_initializer(depth_cast_name.clone(), vec![], &[depth as i64]);
                depth_cast_name
            } else {
                let depth_name = self.get_tensor_name(inputs[1])?;
                self.add_cast_node(depth_name, depth_cast_name.clone(), dt::INT64);
                depth_cast_name
            };
        if !self.state.constants.contains_key(&inputs[1])
            && matches!(self.state.tensor_shapes.get(&inputs[1]), Some(shape) if shape == &vec![1])
        {
            self.add_squeeze_node(
                depth_scalar_input.clone(),
                depth_scalar_name.clone(),
                Some(&[0]),
                Some(range_squeeze_axes_name),
            );
            depth_scalar_input = depth_scalar_name;
        }

        for (name, value) in [(&range_start_name, 0_i64), (&range_step_name, 1_i64)] {
            let mut tensor = onnx::TensorProto {
                name: name.to_string(),
                dims: vec![],
                data_type: dt::INT64,
                ..Default::default()
            };
            tensor.raw_data.extend_from_slice(&value.to_le_bytes());
            self.onnx_graph.initializer.push(tensor);
        }

        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Range".to_string(),
            input: vec![range_start_name, depth_scalar_input, range_step_name],
            output: vec![range_name.clone()],
            ..Default::default()
        });

        self.add_unsqueeze_node(
            indices_cast_name,
            expanded_indices_name.clone(),
            &[indices_rank as i64],
            axes_name,
        );

        self.add_binary_node(
            "Equal",
            expanded_indices_name,
            range_name,
            equal_name.clone(),
        );

        let output_dtype = self
            .state
            .tensor_types
            .get(&out_id)
            .map(|dtype| self.onnx_elem_type_from_paddle(dtype))
            .transpose()?
            .unwrap_or(dt::FLOAT);
        self.add_cast_node(equal_name, output_name, output_dtype);
        Ok(())
    }

    pub fn op_if(&mut self, op: &Value) -> anyhow::Result<()> {
        let inputs = helper::op_input_ids(op);
        let cond_name = self.get_tensor_name(
            *inputs
                .first()
                .ok_or_else(|| anyhow::anyhow!("if requires a condition input"))?,
        )?;
        let regions = op
            .get("regions")
            .and_then(|regions| regions.as_array())
            .ok_or_else(|| anyhow::anyhow!("if is missing regions"))?;
        if regions.len() != 2 {
            anyhow::bail!("if expects 2 regions, got {}", regions.len());
        }

        let then_graph =
            self.convert_region_to_graph(&regions[0], "if_then_branch", vec![], &[])?;
        let else_graph =
            self.convert_region_to_graph(&regions[1], "if_else_branch", vec![], &[])?;

        let mut node = onnx::NodeProto {
            op_type: "If".to_string(),
            input: vec![cond_name],
            ..Default::default()
        };
        node.attribute
            .push(helper::attr_graph("then_branch", then_graph));
        node.attribute
            .push(helper::attr_graph("else_branch", else_graph));

        if let Some(outputs) = op.get("O").and_then(|outputs| outputs.as_array()) {
            for output in outputs {
                if let Some(id) = output.get("%").and_then(|id| id.as_i64()) {
                    node.output.push(self.get_tensor_name(id)?);
                }
            }
        }
        self.onnx_graph.node.push(node);
        Ok(())
    }

    pub fn op_while(&mut self, op: &Value) -> anyhow::Result<()> {
        let inputs = helper::op_input_ids(op);
        if inputs.is_empty() {
            anyhow::bail!("while requires condition and loop-carried inputs");
        }

        let cond_name = self.get_tensor_name(inputs[0])?;
        let loop_var_input_names = inputs[1..]
            .iter()
            .map(|&id| self.get_tensor_name(id))
            .collect::<anyhow::Result<Vec<_>>>()?;

        let regions = op
            .get("regions")
            .and_then(|regions| regions.as_array())
            .ok_or_else(|| anyhow::anyhow!("while is missing regions"))?;
        let body_region = regions
            .first()
            .ok_or_else(|| anyhow::anyhow!("while is missing body region"))?;
        let block_arg_count = body_region
            .get("blocks")
            .and_then(|blocks| blocks.as_array())
            .and_then(|blocks| blocks.first())
            .and_then(|block| block.get("args"))
            .and_then(|args| args.as_array())
            .map(|args| args.len())
            .ok_or_else(|| anyhow::anyhow!("while body region is missing block args"))?;
        if block_arg_count != loop_var_input_names.len() {
            anyhow::bail!(
                "while body arg count mismatch: expected {}, got {}",
                loop_var_input_names.len(),
                block_arg_count
            );
        }

        // Loop body formals are positional and live in the body graph scope.
        // They can safely differ from captured outer names such as cond_name.
        // They must also stay unique when the same outer value is passed twice.
        let body_arg_names = inputs[1..]
            .iter()
            .enumerate()
            .map(|(index, input_id)| format!("while_arg_{}_{}", index, input_id))
            .collect::<Vec<_>>();

        let body_graph = self.convert_region_to_graph(
            body_region,
            "while_body",
            vec![
                self.build_value_info_from_meta(
                    "__p2o_loop_iter".to_string(),
                    crate::helper::paddle_tt::I64,
                    &[],
                )?,
                self.build_value_info_from_meta(
                    "__p2o_loop_cond".to_string(),
                    crate::helper::paddle_tt::BOOL,
                    &[],
                )?,
            ],
            &body_arg_names,
        )?;

        let mut node = onnx::NodeProto {
            op_type: "Loop".to_string(),
            input: vec![String::new(), cond_name],
            ..Default::default()
        };
        node.input.extend(loop_var_input_names);
        node.attribute.push(helper::attr_graph("body", body_graph));

        if let Some(outputs) = op.get("O").and_then(|outputs| outputs.as_array()) {
            for output in outputs {
                if let Some(id) = output.get("%").and_then(|id| id.as_i64()) {
                    node.output.push(self.get_tensor_name(id)?);
                }
            }
        }
        self.onnx_graph.node.push(node);
        Ok(())
    }
}
