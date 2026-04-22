use crate::converter::Converter;
use crate::helper::{self, dt};
use crate::proto::onnx;
use anyhow::bail;
use serde_json::Value;

impl Converter {
    pub fn op_multiclass_nms3(&mut self, op: &Value) -> anyhow::Result<()> {
        self.require_opset(11, "multiclass_nms3")?;

        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            bail!("multiclass_nms3 missing inputs");
        }
        let outputs = op
            .get("O")
            .and_then(|o| o.as_array())
            .ok_or_else(|| anyhow::anyhow!("multiclass_nms3 missing outputs"))?;
        if outputs.len() < 3 {
            bail!("multiclass_nms3 expects three outputs");
        }

        let out_id = outputs[0]
            .get("%")
            .and_then(|id| id.as_i64())
            .ok_or_else(|| anyhow::anyhow!("multiclass_nms3 missing output id"))?;
        let index_out_id = outputs[1]
            .get("%")
            .and_then(|id| id.as_i64())
            .ok_or_else(|| anyhow::anyhow!("multiclass_nms3 missing index output id"))?;
        let rois_num_out_id = outputs[2]
            .get("%")
            .and_then(|id| id.as_i64())
            .ok_or_else(|| anyhow::anyhow!("multiclass_nms3 missing rois_num output id"))?;

        let background_label = helper::attr(op, "background_label")
            .and_then(|d| d.as_i64())
            .unwrap_or(-1);
        if background_label != -1 {
            bail!("multiclass_nms3 currently only supports background_label=-1");
        }
        let nms_eta = helper::attr(op, "nms_eta")
            .and_then(|d| d.as_f64())
            .unwrap_or(1.0);
        if (nms_eta - 1.0).abs() > f64::EPSILON {
            bail!("multiclass_nms3 currently only supports nms_eta=1.0");
        }

        let nms_top_k = helper::attr(op, "nms_top_k")
            .and_then(|d| d.as_i64())
            .unwrap_or(1000);
        let keep_top_k = helper::attr(op, "keep_top_k")
            .and_then(|d| d.as_i64())
            .unwrap_or(-1);
        let score_threshold = helper::attr(op, "score_threshold")
            .and_then(|d| d.as_f64())
            .unwrap_or(0.0) as f32;
        let nms_threshold = helper::attr(op, "nms_threshold")
            .and_then(|d| d.as_f64())
            .unwrap_or(0.5) as f32;

        let make_i64_tensor = |graph: &mut onnx::GraphProto, name: String, values: &[i64]| {
            let mut tensor = onnx::TensorProto {
                name,
                dims: vec![values.len() as i64],
                data_type: dt::INT64,
                ..Default::default()
            };
            for value in values {
                tensor.raw_data.extend_from_slice(&value.to_le_bytes());
            }
            graph.initializer.push(tensor);
        };
        let make_f32_tensor = |graph: &mut onnx::GraphProto, name: String, values: &[f32]| {
            let mut tensor = onnx::TensorProto {
                name,
                dims: vec![values.len() as i64],
                data_type: dt::FLOAT,
                ..Default::default()
            };
            for value in values {
                tensor.raw_data.extend_from_slice(&value.to_le_bytes());
            }
            graph.initializer.push(tensor);
        };

        let max_boxes_name = format!("nms_max_boxes_{}", out_id);
        make_i64_tensor(&mut self.onnx_graph, max_boxes_name.clone(), &[nms_top_k]);
        let iou_name = format!("nms_iou_threshold_{}", out_id);
        make_f32_tensor(&mut self.onnx_graph, iou_name.clone(), &[nms_threshold]);
        let score_name = format!("nms_score_threshold_{}", out_id);
        make_f32_tensor(&mut self.onnx_graph, score_name.clone(), &[score_threshold]);

        let selected_indices_name = format!("nms_selected_indices_{}", out_id);
        let mut nms = onnx::NodeProto {
            op_type: "NonMaxSuppression".to_string(),
            input: vec![
                self.get_tensor_name(inputs[0])?,
                self.get_tensor_name(inputs[1])?,
                max_boxes_name,
                iou_name,
                score_name,
            ],
            output: vec![selected_indices_name.clone()],
            ..Default::default()
        };
        nms.attribute.push(helper::attr_int("center_point_box", 0));
        self.onnx_graph.node.push(nms);

        let shape_name = format!("nms_selected_shape_{}", out_id);
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Shape".to_string(),
            input: vec![selected_indices_name.clone()],
            output: vec![shape_name.clone()],
            ..Default::default()
        });
        let selected_count_index = format!("nms_selected_count_index_{}", out_id);
        make_i64_tensor(&mut self.onnx_graph, selected_count_index.clone(), &[0]);
        let selected_count_name = format!("nms_selected_count_{}", out_id);
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Gather".to_string(),
            input: vec![shape_name, selected_count_index],
            output: vec![selected_count_name.clone()],
            ..Default::default()
        });

        let keep_top_k_name = format!("nms_keep_top_k_{}", out_id);
        let take_k_name = format!("nms_take_k_{}", out_id);
        if keep_top_k > 0 {
            make_i64_tensor(&mut self.onnx_graph, keep_top_k_name.clone(), &[keep_top_k]);
            let take_cond_name = format!("nms_take_cond_{}", out_id);
            self.add_binary_node(
                "Less",
                selected_count_name.clone(),
                keep_top_k_name.clone(),
                take_cond_name.clone(),
            );
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Where".to_string(),
                input: vec![take_cond_name, selected_count_name.clone(), keep_top_k_name],
                output: vec![take_k_name.clone()],
                ..Default::default()
            });
        } else {
            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Identity".to_string(),
                input: vec![selected_count_name.clone()],
                output: vec![take_k_name.clone()],
                ..Default::default()
            });
        }
        let take_k_scalar_name = format!("nms_take_k_scalar_{}", out_id);
        self.add_squeeze_node(
            take_k_name.clone(),
            take_k_scalar_name.clone(),
            Some(&[0]),
            Some(format!("nms_take_k_axes_{}", out_id)),
        );

        let gather_batch_index = format!("nms_gather_batch_index_{}", out_id);
        let gather_class_index = format!("nms_gather_class_index_{}", out_id);
        let gather_box_index = format!("nms_gather_box_index_{}", out_id);
        let gather_batch_box_index = format!("nms_gather_batch_box_index_{}", out_id);
        make_i64_tensor(&mut self.onnx_graph, gather_batch_index.clone(), &[0]);
        make_i64_tensor(&mut self.onnx_graph, gather_class_index.clone(), &[1]);
        make_i64_tensor(&mut self.onnx_graph, gather_box_index.clone(), &[2]);
        make_i64_tensor(
            &mut self.onnx_graph,
            gather_batch_box_index.clone(),
            &[0, 2],
        );

        let batch_indices_name = format!("nms_batch_indices_{}", out_id);
        let class_indices_name = format!("nms_class_indices_{}", out_id);
        let box_indices_name = format!("nms_box_indices_{}", out_id);
        let batch_box_indices_name = format!("nms_batch_box_indices_{}", out_id);
        for (index_name, output_name) in [
            (gather_batch_index, batch_indices_name.clone()),
            (gather_class_index, class_indices_name.clone()),
            (gather_box_index, box_indices_name.clone()),
            (gather_batch_box_index, batch_box_indices_name.clone()),
        ] {
            let mut gather = onnx::NodeProto {
                op_type: "Gather".to_string(),
                input: vec![selected_indices_name.clone(), index_name],
                output: vec![output_name],
                ..Default::default()
            };
            gather.attribute.push(helper::attr_int("axis", 1));
            self.onnx_graph.node.push(gather);
        }
        let batch_indices_scalar_name = format!("nms_batch_indices_scalar_{}", out_id);
        self.add_squeeze_node(
            batch_indices_name.clone(),
            batch_indices_scalar_name.clone(),
            Some(&[1]),
            Some(format!("nms_batch_indices_axes_{}", out_id)),
        );
        let class_indices_scalar_name = format!("nms_class_indices_scalar_{}", out_id);
        self.add_squeeze_node(
            class_indices_name.clone(),
            class_indices_scalar_name.clone(),
            Some(&[1]),
            Some(format!("nms_class_indices_axes_{}", out_id)),
        );
        let box_indices_scalar_name = format!("nms_box_indices_scalar_{}", out_id);
        self.add_squeeze_node(
            box_indices_name.clone(),
            box_indices_scalar_name.clone(),
            Some(&[1]),
            Some(format!("nms_box_indices_axes_{}", out_id)),
        );

        let selected_scores_name = format!("nms_selected_scores_{}", out_id);
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "GatherND".to_string(),
            input: vec![self.get_tensor_name(inputs[1])?, selected_indices_name],
            output: vec![selected_scores_name.clone()],
            ..Default::default()
        });
        let selected_boxes_name = format!("nms_selected_boxes_{}", out_id);
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "GatherND".to_string(),
            input: vec![self.get_tensor_name(inputs[0])?, batch_box_indices_name],
            output: vec![selected_boxes_name.clone()],
            ..Default::default()
        });

        let boxes_shape_name = format!("nms_boxes_shape_{}", out_id);
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Shape".to_string(),
            input: vec![self.get_tensor_name(inputs[0])?],
            output: vec![boxes_shape_name.clone()],
            ..Default::default()
        });
        let num_boxes_index_name = format!("nms_num_boxes_index_{}", out_id);
        make_i64_tensor(&mut self.onnx_graph, num_boxes_index_name.clone(), &[1]);
        let num_boxes_name = format!("nms_num_boxes_{}", out_id);
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Gather".to_string(),
            input: vec![boxes_shape_name, num_boxes_index_name],
            output: vec![num_boxes_name.clone()],
            ..Default::default()
        });

        let absolute_index_scaled_name = format!("nms_absolute_index_scaled_{}", out_id);
        self.add_binary_node(
            "Mul",
            batch_indices_scalar_name.clone(),
            num_boxes_name,
            absolute_index_scaled_name.clone(),
        );
        let absolute_index_name = format!("nms_absolute_index_{}", out_id);
        self.add_binary_node(
            "Add",
            absolute_index_scaled_name,
            box_indices_scalar_name.clone(),
            absolute_index_name.clone(),
        );

        let limited_scores_name = format!("nms_limited_scores_{}", out_id);
        let limited_class_name = format!("nms_limited_class_{}", out_id);
        let limited_index_name = format!("nms_limited_index_{}", out_id);
        let limited_boxes_name = format!("nms_limited_boxes_{}", out_id);
        if keep_top_k > 0 {
            let topk_scores_name = format!("nms_topk_scores_{}", out_id);
            let topk_indices_name = format!("nms_topk_indices_{}", out_id);
            let mut topk = onnx::NodeProto {
                op_type: "TopK".to_string(),
                input: vec![selected_scores_name.clone(), take_k_name.clone()],
                output: vec![topk_scores_name.clone(), topk_indices_name.clone()],
                ..Default::default()
            };
            topk.attribute.push(helper::attr_int("axis", -1));
            topk.attribute.push(helper::attr_int("largest", 1));
            topk.attribute.push(helper::attr_int("sorted", 1));
            self.onnx_graph.node.push(topk);

            self.onnx_graph.node.push(onnx::NodeProto {
                op_type: "Identity".to_string(),
                input: vec![topk_scores_name],
                output: vec![limited_scores_name.clone()],
                ..Default::default()
            });
            for (input_name, output_name) in [
                (
                    class_indices_scalar_name.clone(),
                    limited_class_name.clone(),
                ),
                (absolute_index_name.clone(), limited_index_name.clone()),
                (selected_boxes_name.clone(), limited_boxes_name.clone()),
            ] {
                let mut gather = onnx::NodeProto {
                    op_type: "Gather".to_string(),
                    input: vec![input_name, topk_indices_name.clone()],
                    output: vec![output_name],
                    ..Default::default()
                };
                gather.attribute.push(helper::attr_int("axis", 0));
                self.onnx_graph.node.push(gather);
            }
        } else {
            for (input_name, output_name) in [
                (selected_scores_name.clone(), limited_scores_name.clone()),
                (
                    class_indices_scalar_name.clone(),
                    limited_class_name.clone(),
                ),
                (absolute_index_name.clone(), limited_index_name.clone()),
                (selected_boxes_name.clone(), limited_boxes_name.clone()),
            ] {
                self.onnx_graph.node.push(onnx::NodeProto {
                    op_type: "Identity".to_string(),
                    input: vec![input_name],
                    output: vec![output_name],
                    ..Default::default()
                });
            }
        }

        let sort_positions_start_name = format!("nms_sort_positions_start_{}", out_id);
        let sort_positions_delta_name = format!("nms_sort_positions_delta_{}", out_id);
        make_i64_tensor(
            &mut self.onnx_graph,
            sort_positions_start_name.clone(),
            &[0],
        );
        make_i64_tensor(
            &mut self.onnx_graph,
            sort_positions_delta_name.clone(),
            &[1],
        );
        let sort_positions_name = format!("nms_sort_positions_{}", out_id);
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: "Range".to_string(),
            input: vec![
                sort_positions_start_name,
                take_k_scalar_name,
                sort_positions_delta_name,
            ],
            output: vec![sort_positions_name.clone()],
            ..Default::default()
        });

        let sort_scale_one_name = format!("nms_sort_scale_one_{}", out_id);
        make_i64_tensor(&mut self.onnx_graph, sort_scale_one_name.clone(), &[1]);
        let sort_scale_name = format!("nms_sort_scale_{}", out_id);
        self.add_binary_node(
            "Add",
            take_k_name.clone(),
            sort_scale_one_name,
            sort_scale_name.clone(),
        );

        let class_sort_key_scaled_name = format!("nms_class_sort_key_scaled_{}", out_id);
        self.add_binary_node(
            "Mul",
            limited_class_name.clone(),
            sort_scale_name,
            class_sort_key_scaled_name.clone(),
        );
        let class_sort_key_name = format!("nms_class_sort_key_{}", out_id);
        self.add_binary_node(
            "Add",
            class_sort_key_scaled_name,
            sort_positions_name,
            class_sort_key_name.clone(),
        );

        let class_sort_values_name = format!("nms_class_sort_values_{}", out_id);
        let class_sort_indices_name = format!("nms_class_sort_indices_{}", out_id);
        let mut class_sort = onnx::NodeProto {
            op_type: "TopK".to_string(),
            input: vec![class_sort_key_name, take_k_name.clone()],
            output: vec![class_sort_values_name, class_sort_indices_name.clone()],
            ..Default::default()
        };
        class_sort.attribute.push(helper::attr_int("axis", -1));
        class_sort.attribute.push(helper::attr_int("largest", 0));
        class_sort.attribute.push(helper::attr_int("sorted", 1));
        self.onnx_graph.node.push(class_sort);

        let selected_class_name = format!("nms_selected_class_{}", out_id);
        let selected_index_name = format!("nms_selected_index_{}", out_id);
        let selected_boxes_sorted_name = format!("nms_selected_boxes_sorted_{}", out_id);
        let selected_scores_sorted_name = format!("nms_selected_scores_sorted_{}", out_id);
        for (input_name, output_name) in [
            (limited_class_name, selected_class_name.clone()),
            (limited_index_name, selected_index_name.clone()),
            (limited_boxes_name, selected_boxes_sorted_name.clone()),
            (limited_scores_name, selected_scores_sorted_name.clone()),
        ] {
            let mut gather = onnx::NodeProto {
                op_type: "Gather".to_string(),
                input: vec![input_name, class_sort_indices_name.clone()],
                output: vec![output_name],
                ..Default::default()
            };
            gather.attribute.push(helper::attr_int("axis", 0));
            self.onnx_graph.node.push(gather);
        }

        let selected_class_float_name = format!("nms_selected_class_float_{}", out_id);
        self.add_cast_node(
            selected_class_name,
            selected_class_float_name.clone(),
            dt::FLOAT,
        );

        let class_expanded_name = format!("nms_class_expanded_{}", out_id);
        self.add_unsqueeze_node(
            selected_class_float_name,
            class_expanded_name.clone(),
            &[1],
            format!("nms_class_expand_axes_{}", out_id),
        );

        let scores_expanded_name = format!("nms_scores_expanded_{}", out_id);
        self.add_unsqueeze_node(
            selected_scores_sorted_name,
            scores_expanded_name.clone(),
            &[1],
            format!("nms_scores_expand_axes_{}", out_id),
        );

        let mut concat = onnx::NodeProto {
            op_type: "Concat".to_string(),
            input: vec![
                class_expanded_name,
                scores_expanded_name,
                selected_boxes_sorted_name,
            ],
            output: vec![self.get_tensor_name(out_id)?],
            ..Default::default()
        };
        concat.attribute.push(helper::attr_int("axis", 1));
        self.onnx_graph.node.push(concat);

        self.add_cast_node(
            selected_index_name,
            format!("nms_selected_index_i32_{}", out_id),
            dt::INT32,
        );
        self.add_unsqueeze_node(
            format!("nms_selected_index_i32_{}", out_id),
            self.get_tensor_name(index_out_id)?,
            &[1],
            format!("nms_selected_index_axes_{}", out_id),
        );

        self.add_cast_node(
            take_k_name,
            self.get_tensor_name(rois_num_out_id)?,
            dt::INT32,
        );
        Ok(())
    }
}
