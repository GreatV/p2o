#![allow(unused_imports)]

use crate::converter::Converter;
use crate::proto::{PaddleDataType, TensorDesc};
use prost::Message;
use serde_json::json;
use std::fs;
use std::io::Write;

#[test]
fn test_multiclass_nms3_reorders_final_output_by_class_and_emits_rank2_index() {
    let mut converter = Converter::new();

    let op_json = json!({
        "#": "1.multiclass_nms3",
        "A": [
            { "AT": { "D": 0.025 }, "N": "score_threshold" },
            { "AT": { "D": 1000 }, "N": "nms_top_k" },
            { "AT": { "D": 100 }, "N": "keep_top_k" },
            { "AT": { "D": 0.6 }, "N": "nms_threshold" },
            { "AT": { "D": true }, "N": "normalized" },
            { "AT": { "D": 1.0 }, "N": "nms_eta" },
            { "AT": { "D": -1 }, "N": "background_label" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 0 }
        ],
        "O": [
            {
                "%": 30,
                "TT": {
                    "D": [
                        { "#": "0.t_f32" },
                        [-1, 6]
                    ]
                }
            },
            {
                "%": 31,
                "TT": {
                    "D": [
                        { "#": "0.t_i32" },
                        [-1, 1]
                    ]
                }
            },
            {
                "%": 32,
                "TT": {
                    "D": [
                        { "#": "0.t_i32" },
                        [-1]
                    ]
                }
            }
        ]
    });

    converter
        .process_pass2_op("1.multiclass_nms3", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    let topk_nodes = graph
        .node
        .iter()
        .filter(|node| node.op_type == "TopK")
        .collect::<Vec<_>>();
    assert_eq!(topk_nodes.len(), 2);
    assert!(
        graph
            .node
            .iter()
            .any(|node| node.op_type == "Range" && node.output == vec!["nms_sort_positions_30"])
    );
    assert!(
        graph
            .node
            .iter()
            .any(|node| node.op_type == "Shape" && node.output == vec!["nms_boxes_shape_30"])
    );
    assert!(
        graph
            .node
            .iter()
            .any(|node| node.op_type == "Unsqueeze" && node.output == vec!["tensor_31"])
    );
    let concat = graph
        .node
        .iter()
        .find(|node| node.op_type == "Concat" && node.output == vec!["tensor_30"])
        .unwrap();
    assert_eq!(
        concat.input,
        vec![
            "nms_class_expanded_30",
            "nms_scores_expanded_30",
            "nms_selected_boxes_sorted_30"
        ]
    );
}

#[test]
fn test_multiclass_nms3_without_keep_top_k_uses_selected_count_for_range_end() {
    let mut converter = Converter::new();

    let op_json = json!({
        "#": "1.multiclass_nms3",
        "A": [
            { "AT": { "D": 0.025 }, "N": "score_threshold" },
            { "AT": { "D": 1000 }, "N": "nms_top_k" },
            { "AT": { "D": -1 }, "N": "keep_top_k" },
            { "AT": { "D": 0.6 }, "N": "nms_threshold" },
            { "AT": { "D": true }, "N": "normalized" },
            { "AT": { "D": 1.0 }, "N": "nms_eta" },
            { "AT": { "D": -1 }, "N": "background_label" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 0 }
        ],
        "O": [
            { "%": 30, "TT": { "D": [{ "#": "0.t_f32" }, [-1, 6]] } },
            { "%": 31, "TT": { "D": [{ "#": "0.t_i32" }, [-1, 1]] } },
            { "%": 32, "TT": { "D": [{ "#": "0.t_i32" }, [-1]] } }
        ]
    });

    converter
        .process_pass2_op("1.multiclass_nms3", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    let range = graph
        .node
        .iter()
        .find(|node| node.op_type == "Range" && node.output == vec!["nms_sort_positions_30"])
        .unwrap();
    assert_eq!(range.input[1], "nms_take_k_scalar_30");
    let topk_nodes = graph
        .node
        .iter()
        .filter(|node| node.op_type == "TopK")
        .collect::<Vec<_>>();
    assert_eq!(topk_nodes.len(), 1);
    assert!(
        graph
            .node
            .iter()
            .any(|node| node.op_type == "Squeeze" && node.output == vec!["nms_take_k_scalar_30"])
    );
}

#[test]
fn test_group_norm_lowers_to_instance_norm_and_affine_scale_bias() {
    let mut converter = Converter::new();
    converter
        .state
        .tensor_shapes
        .insert(10, vec![-1, 64, -1, -1]);

    let op_json = json!({
        "#": "1.group_norm",
        "A": [
            { "AT": { "D": 1e-5 }, "N": "epsilon" },
            { "AT": { "D": 32 }, "N": "groups" },
            { "AT": { "D": "NCHW" }, "N": "data_format" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 12 }
        ],
        "O": [
            { "%": 20, "TT": { "D": [{ "#": "0.t_f32" }, [-1, 64, -1, -1]] } },
            { "%": 21, "TT": { "D": [{ "#": "0.t_f32" }, [-1, 32]] } },
            { "%": 22, "TT": { "D": [{ "#": "0.t_f32" }, [-1, 32]] } }
        ]
    });

    converter
        .process_pass2_op("1.group_norm", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    let node_types = graph
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        node_types,
        vec![
            "Shape",
            "Slice",
            "Concat",
            "Reshape",
            "InstanceNormalization",
            "Reshape",
            "Unsqueeze",
            "Unsqueeze",
            "Mul",
            "Add"
        ]
    );
    assert_eq!(graph.node[9].output, vec!["tensor_20"]);
}

#[test]
fn test_reduce_max_on_scalar_becomes_identity() {
    let mut converter = Converter::new();
    converter.state.tensor_shapes.insert(10, vec![]);
    converter.state.tensor_shapes.insert(12, vec![]);
    converter.state.constants.insert(11, vec![0.0]);

    let op_json = json!({
        "#": "1.max",
        "A": [
            { "AT": { "D": false }, "N": "keepdim" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12, "TT": { "D": [{ "#": "0.t_f32" }, []] } }
        ]
    });

    converter.process_pass2_op("1.max", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Identity");
    assert_eq!(graph.node[0].input, vec!["tensor_10"]);
    assert_eq!(graph.node[0].output, vec!["tensor_12"]);
}

#[test]
fn test_repeat_interleave_with_tensor_index_uses_dynamic_repeats_input() {
    let mut converter = Converter::new();
    converter.state.tensor_shapes.insert(10, vec![-1, 1]);
    converter.state.tensor_shapes.insert(11, vec![1]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_i32".to_string());

    let op_json = json!({
        "#": "1.repeat_interleave_with_tensor_index",
        "A": [
            { "AT": { "D": 1 }, "N": "axis" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12, "TT": { "D": [{ "#": "0.t_i64" }, [-1, -1]] } }
        ]
    });

    converter
        .process_pass2_op("1.repeat_interleave_with_tensor_index", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    assert!(graph.node.iter().any(|node| node.op_type == "Cast"));
    assert!(
        graph.node.iter().any(|node| node.op_type == "Concat"
            && node.output == vec!["repeat_interleave_tile_repeats_12"])
    );
    assert!(graph.node.iter().any(|node| node.op_type == "Tile"));
    assert!(
        graph
            .node
            .iter()
            .any(|node| node.op_type == "Reshape" && node.output == vec!["tensor_12"])
    );
}

#[test]
fn test_put_along_axis_assign_uses_scatterelements() {
    let mut converter = Converter::new();
    converter.state.tensor_shapes.insert(10, vec![-1, 8000]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_i32".to_string());

    let op_json = json!({
        "#": "1.put_along_axis",
        "A": [
            { "AT": { "D": 1 }, "N": "axis" },
            { "AT": { "D": "assign" }, "N": "reduce" },
            { "AT": { "D": true }, "N": "include_self" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 12 }
        ],
        "O": [
            { "%": 13, "TT": { "D": [{ "#": "0.t_f32" }, [-1, 8000]] } }
        ]
    });

    converter
        .process_pass2_op("1.put_along_axis", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    let node_types = graph
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect::<Vec<_>>();
    assert_eq!(node_types, vec!["Cast", "ScatterElements"]);
    assert_eq!(graph.node[1].output, vec!["tensor_13"]);
}

#[test]
fn test_multiply_on_bool_mask_lowers_to_and() {
    let mut converter = Converter::new();
    converter
        .state
        .tensor_types
        .insert(10, "0.t_bool".to_string());
    converter
        .state
        .tensor_types
        .insert(11, "0.t_bool".to_string());

    let op_json = json!({
        "#": "1.multiply",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12, "TT": { "D": [{ "#": "0.t_bool" }, [-1, 1, -1, -1]] } }
        ]
    });

    converter.process_pass2_op("1.multiply", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "And");
    assert_eq!(graph.node[0].output, vec!["tensor_12"]);
}

#[test]
fn test_multinomial_sample_size_one_lowers_to_argmax() {
    let mut converter = Converter::new();
    converter.state.constants.insert(11, vec![1.0]);

    let op_json = json!({
        "#": "1.multinomial",
        "A": [
            { "AT": { "D": false }, "N": "replacement" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12, "TT": { "D": [{ "#": "0.t_i64" }, [-1, 1]] } }
        ]
    });

    converter
        .process_pass2_op("1.multinomial", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "ArgMax");
    assert_eq!(graph.node[0].output, vec!["tensor_12"]);
    assert!(
        graph.node[0]
            .attribute
            .iter()
            .any(|attr| attr.name == "axis" && attr.i == -1)
    );
    assert!(
        graph.node[0]
            .attribute
            .iter()
            .any(|attr| attr.name == "keepdims" && attr.i == 1)
    );
    assert!(
        graph.node[0]
            .attribute
            .iter()
            .any(|attr| attr.name == "select_last_index" && attr.i == 0)
    );
}

#[test]
fn test_multinomial_sample_size_gt_one_tiles_argmax_to_preserve_shape() {
    let mut converter = Converter::new();
    converter.state.constants.insert(11, vec![3.0]);

    let op_json = json!({
        "#": "1.multinomial",
        "A": [
            { "AT": { "D": true }, "N": "replacement" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12, "TT": { "D": [{ "#": "0.t_i64" }, [-1, 3]] } }
        ]
    });

    converter
        .process_pass2_op("1.multinomial", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 2);
    assert_eq!(graph.node[0].op_type, "ArgMax");
    assert_eq!(graph.node[0].output, vec!["multinomial_argmax_12"]);
    assert_eq!(graph.node[1].op_type, "Tile");
    assert_eq!(
        graph.node[1].input,
        vec!["multinomial_argmax_12", "multinomial_repeats_12"]
    );
    assert_eq!(graph.node[1].output, vec!["tensor_12"]);
    let repeats = graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "multinomial_repeats_12")
        .unwrap();
    let values = repeats
        .raw_data
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(values, vec![1, 3]);
}

#[test]
fn test_multinomial_strict_mode_rejects_argmax_lowering() {
    let mut converter = Converter::new();
    converter.set_strict(true);
    converter.state.constants.insert(11, vec![1.0]);

    let op_json = json!({
        "#": "1.multinomial",
        "A": [
            { "AT": { "D": false }, "N": "replacement" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12, "TT": { "D": [{ "#": "0.t_i64" }, [-1, 1]] } }
        ]
    });

    let err = converter
        .process_pass2_op("1.multinomial", &op_json)
        .unwrap_err();
    assert!(err.to_string().contains("strict mode rejects"));
}

#[test]
fn test_argsort_int64_uses_stable_tie_break_keys() {
    let mut converter = Converter::new();
    converter.state.tensor_shapes.insert(10, vec![-1, 300]);
    converter
        .state
        .tensor_types
        .insert(10, "0.t_i64".to_string());

    let op_json = json!({
        "#": "1.argsort",
        "A": [
            { "AT": { "D": 1 }, "N": "axis" },
            { "AT": { "D": true }, "N": "descending" },
            { "AT": { "D": false }, "N": "stable" }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            {
                "%": 11,
                "TT": {
                    "D": [
                        { "#": "0.t_i64" },
                        [-1, 300]
                    ]
                }
            },
            {
                "%": 12,
                "TT": {
                    "D": [
                        { "#": "0.t_i64" },
                        [-1, 300]
                    ]
                }
            }
        ]
    });

    converter.process_pass2_op("1.argsort", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    let node_types: Vec<&str> = graph
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect();
    assert_eq!(
        node_types,
        vec!["Shape", "Gather", "Mul", "Add", "TopK", "GatherElements"]
    );
    let topk = graph
        .node
        .iter()
        .find(|node| node.op_type == "TopK")
        .unwrap();
    assert_eq!(topk.output, vec!["argsort_key_values_11", "tensor_12"]);
    let gather = graph
        .node
        .iter()
        .find(|node| node.op_type == "GatherElements")
        .unwrap();
    assert_eq!(gather.input, vec!["tensor_10", "tensor_12"]);
    assert_eq!(gather.output, vec!["tensor_11"]);
    assert!(
        graph
            .initializer
            .iter()
            .any(|tensor| tensor.name == "argsort_tie_scale_11")
    );
    assert!(
        graph
            .initializer
            .iter()
            .any(|tensor| tensor.name == "argsort_tie_break_11")
    );
    let tie_break = graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "argsort_tie_break_11")
        .unwrap();
    let tie_values = tie_break
        .raw_data
        .chunks_exact(8)
        .take(5)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(tie_values, vec![300, 299, 298, 297, 296]);
}

#[test]
fn test_argsort_normalizes_negative_axis_for_legacy_gather() {
    let mut converter = Converter::new();
    converter.set_target_opset(10);
    converter.state.tensor_shapes.insert(10, vec![2, 3, 4]);

    let op_json = json!({
        "#": "1.argsort",
        "A": [
            { "AT": { "D": -1 }, "N": "axis" }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 },
            { "%": 12 }
        ]
    });

    converter.process_pass2_op("1.argsort", &op_json).unwrap();

    let axis = converter
        .onnx_graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "argsort_axis_argsort_shape_11")
        .unwrap();
    let value = i64::from_le_bytes(axis.raw_data[..8].try_into().unwrap());
    assert_eq!(value, 2);
}

#[test]
fn test_argsort_negative_axis_without_rank_requires_opset_13() {
    let mut converter = Converter::new();
    converter.set_target_opset(12);

    let op_json = json!({
        "#": "1.argsort",
        "A": [
            { "AT": { "D": -1 }, "N": "axis" }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 },
            { "%": 12 }
        ]
    });

    let err = converter
        .process_pass2_op("1.argsort", &op_json)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("argsort negative axis gather requires opset >= 13")
    );
}

#[test]
fn test_argsort_without_ppdoclayout_tie_fallback_stays_topk_only() {
    let mut converter = Converter::new();
    converter.state.tensor_shapes.insert(10, vec![-1, 128]);

    let op_json = json!({
        "#": "1.argsort",
        "A": [
            { "AT": { "D": 1 }, "N": "axis" },
            { "AT": { "D": true }, "N": "descending" },
            { "AT": { "D": false }, "N": "stable" }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            {
                "%": 11,
                "TT": {
                    "D": [
                        { "#": "0.t_i64" },
                        [-1, 128]
                    ]
                }
            },
            {
                "%": 12,
                "TT": {
                    "D": [
                        { "#": "0.t_i64" },
                        [-1, 128]
                    ]
                }
            }
        ]
    });

    converter.process_pass2_op("1.argsort", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 3);
    assert_eq!(graph.node[0].op_type, "Shape");
    assert_eq!(graph.node[1].op_type, "Gather");
    assert_eq!(graph.node[2].op_type, "TopK");
    assert_eq!(graph.node[2].output, vec!["tensor_11", "tensor_12"]);
}

#[test]
fn test_argsort_int32_casts_to_i64_before_stable_tie_break() {
    let mut converter = Converter::new();
    converter.state.tensor_shapes.insert(10, vec![-1, 32]);
    converter
        .state
        .tensor_types
        .insert(10, "0.t_i32".to_string());

    let op_json = json!({
        "#": "1.argsort",
        "A": [
            { "AT": { "D": 1 }, "N": "axis" },
            { "AT": { "D": false }, "N": "descending" },
            { "AT": { "D": false }, "N": "stable" }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            {
                "%": 11,
                "TT": {
                    "D": [
                        { "#": "0.t_i32" },
                        [-1, 32]
                    ]
                }
            },
            {
                "%": 12,
                "TT": {
                    "D": [
                        { "#": "0.t_i64" },
                        [-1, 32]
                    ]
                }
            }
        ]
    });

    converter.process_pass2_op("1.argsort", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    let node_types: Vec<&str> = graph
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect();
    assert_eq!(
        node_types,
        vec![
            "Shape",
            "Gather",
            "Cast",
            "Mul",
            "Add",
            "TopK",
            "GatherElements"
        ]
    );
    assert_eq!(graph.node[2].input, vec!["tensor_10"]);
    assert_eq!(graph.node[2].output, vec!["argsort_input_i64_11"]);
    assert_eq!(
        graph.node[5].output,
        vec!["argsort_key_values_11", "tensor_12"]
    );
    assert_eq!(graph.node[6].output, vec!["tensor_11"]);
    let tie_break = graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "argsort_tie_break_11")
        .unwrap();
    let tie_values = tie_break
        .raw_data
        .chunks_exact(8)
        .take(5)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(tie_values, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_pool2d_rejects_ceil_mode_below_opset_10() {
    let mut converter = Converter::new();
    converter.set_target_opset(9);

    let op_json = json!({
        "#": "1.pool2d",
        "A": [
            { "AT": { "D": "max" }, "N": "pooling_type" },
            { "AT": { "D": false }, "N": "adaptive" },
            { "AT": { "D": false }, "N": "global_pooling" },
            { "AT": { "D": true }, "N": "ceil_mode" }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    let err = converter
        .process_pass2_op("1.pool2d", &op_json)
        .unwrap_err();
    assert!(err.to_string().contains("ceil_mode requires opset >= 10"));
}

#[test]
fn test_grid_sample_reports_opset_requirement() {
    let mut converter = Converter::new();
    converter.set_target_opset(15);

    let op_json = json!({
        "#": "1.grid_sample",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12 }
        ]
    });

    let err = converter
        .process_pass2_op("1.grid_sample", &op_json)
        .unwrap_err();
    assert!(err.to_string().contains("grid_sample requires opset >= 16"));
}

#[test]
fn test_grid_sample_renames_modes_for_opset_20() {
    let mut converter = Converter::new();
    converter.set_target_opset(20);

    let op_json = json!({
        "#": "1.grid_sample",
        "A": [
            { "AT": { "D": "bilinear" }, "N": "mode" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12 }
        ]
    });

    converter
        .process_pass2_op("1.grid_sample", &op_json)
        .unwrap();

    let mode = converter.onnx_graph.node[0]
        .attribute
        .iter()
        .find(|attr| attr.name == "mode")
        .unwrap();
    assert_eq!(mode.s, b"linear");
}
