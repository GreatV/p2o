#![allow(unused_imports)]

use crate::converter::Converter;
use crate::helper::dt;
use crate::proto::{PaddleDataType, TensorDesc};
use prost::Message;
use serde_json::json;
use std::fs;
use std::io::Write;

#[test]
fn test_topk_casts_int32_k_to_int64() {
    let mut converter = Converter::new();
    converter
        .state
        .tensor_types
        .insert(11, "0.t_i32".to_string());

    let op_json = json!({
        "#": "1.topk",
        "A": [
            { "AT": { "D": 1 }, "N": "axis" },
            { "AT": { "D": true }, "N": "largest" },
            { "AT": { "D": true }, "N": "sorted" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            {
                "%": 12,
                "TT": {
                    "D": [
                        { "#": "0.t_f32" },
                        [-1, 5]
                    ]
                }
            },
            {
                "%": 13,
                "TT": {
                    "D": [
                        { "#": "0.t_i64" },
                        [-1, 5]
                    ]
                }
            }
        ]
    });

    converter.process_pass2_op("1.topk", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 2);
    assert_eq!(graph.node[0].op_type, "Cast");
    assert_eq!(graph.node[0].input, vec!["tensor_11"]);
    assert_eq!(graph.node[0].output, vec!["topk_k_i64_12"]);
    assert_eq!(graph.node[1].op_type, "TopK");
    assert_eq!(graph.node[1].input, vec!["tensor_10", "topk_k_i64_12"]);
    assert_eq!(graph.node[1].output, vec!["tensor_12", "tensor_13"]);
}

#[test]
fn test_eye_requires_opset_11() {
    let mut converter = Converter::new();
    converter.set_target_opset(10);
    converter.state.tensor_shapes.insert(10, vec![]);
    converter.state.tensor_shapes.insert(11, vec![]);
    converter
        .state
        .tensor_types
        .insert(10, "0.t_i64".to_string());
    converter
        .state
        .tensor_types
        .insert(11, "0.t_i64".to_string());

    let op_json = json!({
        "#": "1.eye",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12 }
        ]
    });

    let err = converter.process_pass2_op("1.eye", &op_json).unwrap_err();
    assert!(err.to_string().contains("eye requires opset >= 11"));
}

#[test]
fn test_pool2d_adaptive_avg_uses_global_average_pool() {
    let mut converter = Converter::new();
    converter.state.constants.insert(11, vec![1.0, 1.0]);

    let op_json = json!({
        "#": "1.pool2d",
        "A": [
            { "AT": { "D": true }, "N": "adaptive" },
            { "AT": { "D": false }, "N": "global_pooling" },
            { "AT": { "D": "avg" }, "N": "pooling_type" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            {
                "%": 12,
                "TT": {
                    "D": [
                        { "#": "0.t_f32" },
                        [-1, 512, 1, 1]
                    ]
                }
            }
        ]
    });

    converter.process_pass2_op("1.pool2d", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "GlobalAveragePool");
    assert_eq!(graph.node[0].input, vec!["tensor_10"]);
    assert_eq!(graph.node[0].output, vec!["tensor_12"]);
    assert!(graph.node[0].attribute.is_empty());
}

#[test]
fn test_pool2d_rejects_non_global_adaptive_output_size() {
    let mut converter = Converter::new();
    converter.state.constants.insert(11, vec![2.0, 3.0]);

    let op_json = json!({
        "#": "1.pool2d",
        "A": [
            { "AT": { "D": true }, "N": "adaptive" },
            { "AT": { "D": false }, "N": "global_pooling" },
            { "AT": { "D": "avg" }, "N": "pooling_type" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12 }
        ]
    });

    let err = converter
        .process_pass2_op("1.pool2d", &op_json)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("adaptive pool2d only supports global")
    );
}

#[test]
fn test_pool2d_rejects_same_padding_with_non_adaptive_output_size_hint() {
    let mut converter = Converter::new();
    converter.state.constants.insert(11, vec![1.0, 1.0]);

    let op_json = json!({
        "#": "1.pool2d",
        "A": [
            { "AT": { "D": false }, "N": "adaptive" },
            { "AT": { "D": false }, "N": "global_pooling" },
            { "AT": { "D": "avg" }, "N": "pooling_type" },
            { "AT": { "D": "SAME" }, "N": "padding_algorithm" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12 }
        ]
    });

    let err = converter
        .process_pass2_op("1.pool2d", &op_json)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("does not support output_size=[1,1] unless adaptive=true")
    );
}

#[test]
fn test_pool2d_avg_exclusive_false_sets_count_include_pad() {
    let mut converter = Converter::new();

    let op_json = json!({
        "#": "1.pool2d",
        "A": [
            { "AT": { "D": false }, "N": "adaptive" },
            { "AT": { "D": false }, "N": "global_pooling" },
            { "AT": { "D": "avg" }, "N": "pooling_type" },
            { "AT": { "D": false }, "N": "exclusive" }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 12 }
        ]
    });

    converter.process_pass2_op("1.pool2d", &op_json).unwrap();

    let node = &converter.onnx_graph.node[0];
    let attr = node
        .attribute
        .iter()
        .find(|attr| attr.name == "count_include_pad")
        .unwrap();
    assert_eq!(attr.i, 1);
}

#[test]
fn test_expand_rewrites_negative_shape_dims() {
    let mut converter = Converter::new();
    converter.state.tensor_shapes.insert(10, vec![2, 1, 4]);
    converter.state.constants.insert(11, vec![-1.0, 300.0, 4.0]);

    let op_json = json!({
        "#": "1.expand",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            {
                "%": 12,
                "TT": {
                    "D": [
                        { "#": "0.t_f32" },
                        [-1, 300, 4]
                    ]
                }
            }
        ]
    });

    converter.process_pass2_op("1.expand", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    let node_types: Vec<&str> = graph
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect();
    assert!(node_types.contains(&"Shape"));
    assert!(node_types.contains(&"Slice"));
    assert!(node_types.contains(&"Concat"));

    let expand = graph
        .node
        .iter()
        .find(|node| node.op_type == "Expand")
        .unwrap();
    assert_eq!(expand.input, vec!["tensor_10", "expand_shape_12"]);
    assert_eq!(expand.output, vec!["tensor_12"]);
    assert!(!converter.state.constants.keys().any(|id| *id < 0));
}

#[test]
fn test_squeeze_keeps_singleton_axis_from_tensor_input() {
    let mut converter = Converter::new();
    converter
        .state
        .tensor_shapes
        .insert(10, vec![-1, 120, 1, -1]);
    converter.state.tensor_shapes.insert(12, vec![-1, 120, -1]);
    converter.state.constants.insert(11, vec![2.0]);

    let op_json = json!({
        "#": "1.squeeze",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            {
                "%": 12,
                "TT": {
                    "D": [
                        { "#": "0.t_f32" },
                        [-1, 120, -1]
                    ]
                }
            }
        ]
    });

    converter.process_pass2_op("1.squeeze", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Squeeze");
    assert_eq!(graph.node[0].input, vec!["tensor_10", "squeeze_axes_12"]);
    assert_eq!(graph.node[0].output, vec!["tensor_12"]);

    let axes = graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "squeeze_axes_12")
        .unwrap();
    assert_eq!(axes.dims, vec![1]);
    assert_eq!(axes.raw_data, 2_i64.to_le_bytes());
}

#[test]
fn test_squeeze_drops_non_singleton_axis_to_identity() {
    let mut converter = Converter::new();
    converter.state.tensor_shapes.insert(10, vec![-1, 300, 512]);
    converter.state.tensor_shapes.insert(12, vec![-1, 300, 512]);
    converter.state.constants.insert(11, vec![-1.0]);

    let op_json = json!({
        "#": "1.squeeze",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            {
                "%": 12,
                "TT": {
                    "D": [
                        { "#": "0.t_f32" },
                        [-1, 300, 512]
                    ]
                }
            }
        ]
    });

    converter.process_pass2_op("1.squeeze", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Identity");
    assert_eq!(graph.node[0].input, vec!["tensor_10"]);
    assert_eq!(graph.node[0].output, vec!["tensor_12"]);
    assert!(
        graph
            .initializer
            .iter()
            .all(|tensor| tensor.name != "squeeze_axes_12")
    );
}

#[test]
fn test_pad3d_reorders_spatial_paddings_for_onnx_pad() {
    let mut converter = Converter::new();
    converter
        .state
        .tensor_shapes
        .insert(10, vec![-1, 32, 1, 45, 31]);
    converter.state.tensor_shapes.insert(11, vec![6]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_i32".to_string());

    let op_json = json!({
        "#": "1.pad3d",
        "A": [
            { "AT": { "D": "reflect" }, "N": "mode" },
            { "AT": { "D": "NCDHW" }, "N": "data_format" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            {
                "%": 12,
                "TT": {
                    "D": [
                        { "#": "0.t_f32" },
                        [-1, 32, 1, 49, 35]
                    ]
                }
            }
        ]
    });

    converter.process_pass2_op("1.pad3d", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    let starts = graph
        .node
        .iter()
        .find(|node| node.output == vec!["pad3d_starts_12"])
        .unwrap();
    assert_eq!(starts.op_type, "Concat");
    assert_eq!(
        starts.input,
        vec![
            "pad3d_zero_prefix_12",
            "pad3d_d_begin_12_i64",
            "pad3d_h_begin_12_i64",
            "pad3d_w_begin_12_i64"
        ]
    );

    let ends = graph
        .node
        .iter()
        .find(|node| node.output == vec!["pad3d_ends_12"])
        .unwrap();
    assert_eq!(ends.op_type, "Concat");
    assert_eq!(
        ends.input,
        vec![
            "pad3d_zero_prefix_12",
            "pad3d_d_end_12_i64",
            "pad3d_h_end_12_i64",
            "pad3d_w_end_12_i64"
        ]
    );

    let pad = graph
        .node
        .iter()
        .find(|node| node.op_type == "Pad")
        .unwrap();
    assert_eq!(pad.input, vec!["tensor_10", "pad3d_pads_12"]);
    assert_eq!(pad.output, vec!["tensor_12"]);
    assert_eq!(pad.attribute.len(), 1);
    assert_eq!(pad.attribute[0].name, "mode");
    assert_eq!(pad.attribute[0].s, b"reflect");
}

#[test]
fn test_pad3d_reorders_spatial_paddings_for_ndhwc() {
    let mut converter = Converter::new();
    converter
        .state
        .tensor_shapes
        .insert(10, vec![-1, 45, 31, 1, 32]);
    converter.state.tensor_shapes.insert(11, vec![6]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_i32".to_string());

    let op_json = json!({
        "#": "1.pad3d",
        "A": [
            { "AT": { "D": "constant" }, "N": "mode" },
            { "AT": { "D": "NDHWC" }, "N": "data_format" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12 }
        ]
    });

    converter.process_pass2_op("1.pad3d", &op_json).unwrap();

    let starts = converter
        .onnx_graph
        .node
        .iter()
        .find(|node| node.output == vec!["pad3d_starts_12"])
        .unwrap();
    assert_eq!(
        starts.input,
        vec![
            "pad3d_zero_prefix_12",
            "pad3d_h_begin_12_i64",
            "pad3d_w_begin_12_i64",
            "pad3d_d_begin_12_i64"
        ]
    );
}

#[test]
fn test_prelu_maps_to_onnx_without_paddle_attrs() {
    let mut converter = Converter::new();
    converter
        .state
        .tensor_shapes
        .insert(10, vec![-1, 32, 45, 31]);
    converter.state.tensor_shapes.insert(11, vec![1]);

    let op_json = json!({
        "#": "1.prelu",
        "A": [
            { "AT": { "D": "NCHW" }, "N": "data_format" },
            { "AT": { "D": "all" }, "N": "mode" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            {
                "%": 12,
                "TT": {
                    "D": [
                        { "#": "0.t_f32" },
                        [-1, 32, 45, 31]
                    ]
                }
            }
        ]
    });

    converter.process_pass2_op("1.prelu", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "PRelu");
    assert_eq!(graph.node[0].input, vec!["tensor_10", "tensor_11"]);
    assert_eq!(graph.node[0].output, vec!["tensor_12"]);
    assert!(graph.node[0].attribute.is_empty());
}

#[test]
fn test_meshgrid_supports_three_inputs() {
    let mut converter = Converter::new();
    converter.state.combines.insert(20, vec![21, 22, 23]);
    converter.state.splits.insert(30, vec![31, 32, 33]);

    let op_json = json!({
        "#": "1.meshgrid",
        "I": [
            { "%": 20 }
        ],
        "O": [
            { "%": 30 }
        ]
    });

    converter.process_pass2_op("1.meshgrid", &op_json).unwrap();

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
            "Shape",
            "Shape",
            "Concat",
            "Unsqueeze",
            "Expand",
            "Unsqueeze",
            "Expand",
            "Unsqueeze",
            "Expand"
        ]
    );
    assert_eq!(graph.node[5].output, vec!["tensor_31"]);
    assert_eq!(graph.node[7].output, vec!["tensor_32"]);
    assert_eq!(graph.node[9].output, vec!["tensor_33"]);
}

#[test]
fn test_prelu_element_nhwc_bails() {
    let mut converter = Converter::new();
    converter
        .state
        .tensor_shapes
        .insert(10, vec![-1, 45, 31, 32]);
    converter.state.tensor_shapes.insert(11, vec![45, 31, 32]);

    let op_json = json!({
        "#": "1.prelu",
        "A": [
            { "AT": { "D": "NHWC" }, "N": "data_format" },
            { "AT": { "D": "element" }, "N": "mode" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12 }
        ]
    });

    let err = converter.process_pass2_op("1.prelu", &op_json).unwrap_err();
    assert!(
        err.to_string()
            .contains("mode=element only supports NCHW-style layouts")
    );
}

#[test]
fn test_roll_constant_shifts_decomposes_to_slice_concat() {
    let mut converter = Converter::new();
    converter
        .state
        .tensor_shapes
        .insert(10, vec![-1, 50, 170, 128]);
    converter.state.constants.insert(11, vec![1.0, 2.0]);

    let op_json = json!({
        "#": "1.roll",
        "A": [
            {
                "AT": {
                    "#": "0.a_array",
                    "D": [
                        { "#": "0.a_i64", "D": 1 },
                        { "#": "0.a_i64", "D": 2 }
                    ]
                },
                "N": "axis"
            }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            {
                "%": 12,
                "TT": {
                    "D": [
                        { "#": "0.t_f32" },
                        [-1, 50, 170, 128]
                    ]
                }
            }
        ]
    });

    converter.process_pass2_op("1.roll", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    let node_types: Vec<&str> = graph
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect();
    assert_eq!(
        node_types,
        vec!["Slice", "Slice", "Concat", "Slice", "Slice", "Concat"]
    );
    assert_eq!(graph.node[0].output, vec!["roll_tail_12_0"]);
    assert_eq!(graph.node[1].output, vec!["roll_head_12_0"]);
    assert_eq!(graph.node[2].output, vec!["roll_axis_12_0"]);
    assert_eq!(graph.node[5].output, vec!["tensor_12"]);
}

#[test]
fn test_strided_slice_wires_steps_and_decrease_axis() {
    let mut converter = Converter::new();
    converter
        .state
        .tensor_types
        .insert(11, "0.t_i32".to_string());
    converter
        .state
        .tensor_types
        .insert(12, "0.t_i64".to_string());
    converter
        .state
        .tensor_types
        .insert(13, "0.t_i64".to_string());
    converter.state.tensor_shapes.insert(11, vec![2]);
    converter.state.tensor_shapes.insert(12, vec![2]);
    converter.state.tensor_shapes.insert(13, vec![2]);

    let op_json = json!({
        "#": "1.strided_slice",
        "A": [
            {
                "AT": {
                    "#": "0.a_array",
                    "D": [
                        { "#": "0.a_i32", "D": 1 },
                        { "#": "0.a_i32", "D": 2 }
                    ]
                },
                "N": "axes"
            },
            {
                "AT": {
                    "#": "0.a_array",
                    "D": [
                        { "#": "0.a_i32", "D": 1 }
                    ]
                },
                "N": "decrease_axis"
            }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 12 },
            { "%": 13 }
        ],
        "O": [
            {
                "%": 14,
                "TT": {
                    "D": [
                        { "#": "0.t_f32" },
                        [-1, 84, 128]
                    ]
                }
            }
        ]
    });

    converter
        .process_pass2_op("1.strided_slice", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    let node_types: Vec<&str> = graph
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect();
    assert_eq!(node_types, vec!["Cast", "Slice", "Squeeze"]);
    assert_eq!(
        graph.node[1].input,
        vec![
            "tensor_10",
            "strided_slice_starts_i64_14",
            "tensor_12",
            "strided_slice_axes_14",
            "tensor_13"
        ]
    );
    assert_eq!(graph.node[1].output, vec!["strided_slice_out_14"]);
    assert_eq!(
        graph.node[2].input,
        vec!["strided_slice_out_14", "strided_slice_decrease_axes_14"]
    );
}

#[test]
fn test_strided_slice_requires_opset_10() {
    let mut converter = Converter::new();
    converter.set_target_opset(9);

    let op_json = json!({
        "#": "1.strided_slice",
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 12 },
            { "%": 13 }
        ],
        "O": [
            { "%": 14 }
        ]
    });

    let err = converter
        .process_pass2_op("1.strided_slice", &op_json)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("strided_slice requires opset >= 10")
    );
}

#[test]
fn test_slice_without_axes_does_not_append_empty_axes_input() {
    let mut converter = Converter::new();

    let op_json = json!({
        "#": "1.slice",
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 12 }
        ],
        "O": [
            { "%": 13 }
        ]
    });

    converter.process_pass2_op("1.slice", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Slice");
    assert_eq!(
        graph.node[0].input,
        vec!["tensor_10", "tensor_11", "tensor_12"]
    );
    assert!(graph.initializer.is_empty());
}

#[test]
fn test_slice_rejects_non_constant_starts_for_opset_9() {
    let mut converter = Converter::new();
    converter.set_target_opset(9);

    let op_json = json!({
        "#": "1.slice",
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 12 }
        ],
        "O": [
            { "%": 13 }
        ]
    });

    let err = converter.process_pass2_op("1.slice", &op_json).unwrap_err();
    assert!(
        err.to_string()
            .contains("slice requires constant starts input for opset < 10")
    );
}

#[test]
fn test_set_value_axes_1_2_constant_block_uses_scatternd() {
    let mut converter = Converter::new();
    converter.state.tensor_shapes.insert(10, vec![1, 4, 5, 1]);
    converter
        .state
        .tensor_types
        .insert(10, "0.t_f32".to_string());
    converter.state.constants.insert(11, vec![0.0, 0.0]);
    converter.state.constants.insert(12, vec![-2.0, -3.0]);
    converter.state.constants.insert(13, vec![1.0, 1.0]);

    let op_json = json!({
        "#": "1.set_value_",
        "A": [
            {
                "AT": {
                    "#": "0.a_array",
                    "D": [
                        { "#": "0.a_i64", "D": 1 },
                        { "#": "0.a_i64", "D": 2 }
                    ]
                },
                "N": "axes"
            },
            {
                "AT": {
                    "#": "0.a_array",
                    "D": [
                        { "#": "0.a_f64", "D": 7.0 }
                    ]
                },
                "N": "values"
            }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 12 },
            { "%": 13 }
        ],
        "O": [
            {
                "%": 14,
                "TT": {
                    "D": [
                        { "#": "0.t_f32" },
                        [1, 4, 5, 1]
                    ]
                }
            }
        ]
    });

    converter
        .process_pass2_op("1.set_value_", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "ScatterND");
    assert_eq!(
        graph.node[0].input,
        vec!["tensor_10", "set_value_indices_14", "set_value_updates_14"]
    );
    let indices = graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "set_value_indices_14")
        .unwrap();
    assert_eq!(indices.dims, vec![4, 4]);
    let updates = graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "set_value_updates_14")
        .unwrap();
    assert_eq!(updates.dims, vec![4]);
}

#[test]
fn test_set_value_axes_1_2_constant_block_preserves_uint8_updates_dtype() {
    let mut converter = Converter::new();
    converter.state.tensor_shapes.insert(10, vec![1, 4, 5, 1]);
    converter
        .state
        .tensor_types
        .insert(10, "0.t_ui8".to_string());
    converter.state.constants.insert(11, vec![0.0, 0.0]);
    converter.state.constants.insert(12, vec![-2.0, -3.0]);
    converter.state.constants.insert(13, vec![1.0, 1.0]);

    let op_json = json!({
        "#": "1.set_value_",
        "A": [
            {
                "AT": {
                    "#": "0.a_array",
                    "D": [
                        { "#": "0.a_i64", "D": 1 },
                        { "#": "0.a_i64", "D": 2 }
                    ]
                },
                "N": "axes"
            },
            {
                "AT": {
                    "#": "0.a_array",
                    "D": [
                        { "#": "0.a_f64", "D": 255.0 }
                    ]
                },
                "N": "values"
            }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 12 },
            { "%": 13 }
        ],
        "O": [
            {
                "%": 14,
                "TT": {
                    "D": [
                        { "#": "0.t_ui8" },
                        [1, 4, 5, 1]
                    ]
                }
            }
        ]
    });

    converter
        .process_pass2_op("1.set_value_", &op_json)
        .unwrap();

    let updates = converter
        .onnx_graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "set_value_updates_14")
        .unwrap();
    assert_eq!(updates.data_type, dt::UINT8);
    assert_eq!(updates.raw_data, vec![255; 4]);
}
