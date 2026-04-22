#![allow(unused_imports)]

use crate::converter::Converter;
use crate::helper::dt;
use crate::proto::{PaddleDataType, TensorDesc};
use prost::Message;
use serde_json::json;
use std::fs;
use std::io::Write;

#[test]
fn test_scale_with_input_tensor_and_bias_before_scale_uses_input_scale_tensor() {
    let mut converter = Converter::new();
    converter
        .state
        .tensor_types
        .insert(10, "0.t_f32".to_string());
    converter.state.tensor_shapes.insert(10, vec![2]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_f32".to_string());
    converter.state.tensor_shapes.insert(11, vec![1]);
    converter
        .state
        .tensor_types
        .insert(12, "0.t_f32".to_string());
    converter.state.tensor_shapes.insert(12, vec![2]);

    let op_json = json!({
        "#": "1.scale",
        "A": [
            { "AT": { "D": 99.0 }, "N": "scale" },
            { "AT": { "D": 2.0 }, "N": "bias" },
            { "AT": { "D": false }, "N": "bias_after_scale" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12 }
        ]
    });

    converter.process_pass2_op("1.scale", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 2);
    assert_eq!(graph.node[0].op_type, "Add");
    assert_eq!(graph.node[0].input, vec!["tensor_10", "scale_bias_12"]);
    assert_eq!(graph.node[1].op_type, "Mul");
    assert_eq!(graph.node[1].input, vec!["scale_add_12", "tensor_11"]);
    assert!(
        !graph
            .initializer
            .iter()
            .any(|tensor| tensor.name == "scale_factor_12")
    );
}

#[test]
fn test_scale_with_input_tensor_and_bias_after_scale_multiplies_before_add() {
    let mut converter = Converter::new();
    converter
        .state
        .tensor_types
        .insert(10, "0.t_f32".to_string());
    converter.state.tensor_shapes.insert(10, vec![2]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_f32".to_string());
    converter.state.tensor_shapes.insert(11, vec![1]);
    converter
        .state
        .tensor_types
        .insert(12, "0.t_f32".to_string());
    converter.state.tensor_shapes.insert(12, vec![2]);

    let op_json = json!({
        "#": "1.scale",
        "A": [
            { "AT": { "D": 42.0 }, "N": "scale" },
            { "AT": { "D": 1.5 }, "N": "bias" },
            { "AT": { "D": true }, "N": "bias_after_scale" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12 }
        ]
    });

    converter.process_pass2_op("1.scale", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 2);
    assert_eq!(graph.node[0].op_type, "Mul");
    assert_eq!(graph.node[0].input, vec!["tensor_10", "tensor_11"]);
    assert_eq!(graph.node[1].op_type, "Add");
    assert_eq!(graph.node[1].input, vec!["scale_mul_12", "scale_bias_12"]);
    assert!(
        !graph
            .initializer
            .iter()
            .any(|tensor| tensor.name == "scale_factor_12")
    );
}

#[test]
fn test_scale_integer_output_truncates_fractional_bias_to_tensor_dtype() {
    let mut converter = Converter::new();
    converter
        .state
        .tensor_types
        .insert(10, "0.t_i32".to_string());
    converter.state.tensor_shapes.insert(10, vec![2]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_i32".to_string());
    converter.state.tensor_shapes.insert(11, vec![2]);

    let op_json = json!({
        "#": "1.scale",
        "A": [
            { "AT": { "D": 2.0 }, "N": "scale" },
            { "AT": { "D": 1.5 }, "N": "bias" },
            { "AT": { "D": true }, "N": "bias_after_scale" }
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
                        [2]
                    ]
                }
            }
        ]
    });

    converter.process_pass2_op("1.scale", &op_json).unwrap();

    let bias = converter
        .onnx_graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "scale_bias_11")
        .unwrap();
    assert_eq!(bias.data_type, dt::INT32);
    let bias_value = i32::from_le_bytes(bias.raw_data[..4].try_into().unwrap());
    assert_eq!(bias_value, 1);
}

#[test]
fn test_clip_uint8_casts_float_bounds_to_uint8() {
    let mut converter = Converter::new();
    converter
        .state
        .tensor_types
        .insert(10, "0.t_ui8".to_string());
    converter.state.tensor_shapes.insert(10, vec![4]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_f32".to_string());
    converter.state.tensor_shapes.insert(11, vec![]);
    converter
        .state
        .tensor_types
        .insert(12, "0.t_f32".to_string());
    converter.state.tensor_shapes.insert(12, vec![]);
    converter
        .state
        .tensor_types
        .insert(13, "0.t_ui8".to_string());
    converter.state.tensor_shapes.insert(13, vec![4]);

    let op_json = json!({
        "#": "1.clip",
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 12 }
        ],
        "O": [
            { "%": 13 }
        ]
    });

    converter.process_pass2_op("1.clip", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 3);
    assert_eq!(graph.node[0].op_type, "Cast");
    assert_eq!(graph.node[1].op_type, "Cast");
    assert_eq!(graph.node[2].op_type, "Clip");
    let min_to = graph.node[0]
        .attribute
        .iter()
        .find(|attr| attr.name == "to")
        .unwrap();
    let max_to = graph.node[1]
        .attribute
        .iter()
        .find(|attr| attr.name == "to")
        .unwrap();
    assert_eq!(min_to.i, dt::UINT8 as i64);
    assert_eq!(max_to.i, dt::UINT8 as i64);
}

#[test]
fn test_floor_divide_int16_casts_back_to_int16_output() {
    let mut converter = Converter::new();
    converter
        .state
        .tensor_types
        .insert(10, "0.t_i16".to_string());
    converter.state.tensor_shapes.insert(10, vec![4]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_i16".to_string());
    converter.state.tensor_shapes.insert(11, vec![4]);
    converter
        .state
        .tensor_types
        .insert(12, "0.t_i16".to_string());
    converter.state.tensor_shapes.insert(12, vec![4]);

    let op_json = json!({
        "#": "1.floor_divide",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            {
                "%": 12,
                "TT": {
                    "D": [
                        { "#": "0.t_i16" },
                        [4]
                    ]
                }
            }
        ]
    });

    converter
        .process_pass2_op("1.floor_divide", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    let node_types = graph
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect::<Vec<_>>();
    assert!(node_types.contains(&"Div"));
    assert!(node_types.contains(&"Mod"));
    assert!(node_types.contains(&"Less"));
    assert!(node_types.contains(&"Equal"));
    assert!(node_types.contains(&"And"));
    assert_eq!(graph.node.last().unwrap().op_type, "Sub");
    assert_eq!(graph.node.last().unwrap().output, vec!["tensor_12"]);
    assert!(
        !node_types.contains(&"Floor"),
        "integer floor_divide should not route through float Floor"
    );
}

#[test]
fn test_floor_divide_float_inputs_still_use_floor_path() {
    let mut converter = Converter::new();
    converter
        .state
        .tensor_types
        .insert(10, "0.t_f32".to_string());
    converter.state.tensor_shapes.insert(10, vec![4]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_f32".to_string());
    converter.state.tensor_shapes.insert(11, vec![4]);
    converter
        .state
        .tensor_types
        .insert(12, "0.t_f32".to_string());
    converter.state.tensor_shapes.insert(12, vec![4]);

    let op_json = json!({
        "#": "1.floor_divide",
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
                        [4]
                    ]
                }
            }
        ]
    });

    converter
        .process_pass2_op("1.floor_divide", &op_json)
        .unwrap();

    let node_types = converter
        .onnx_graph
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect::<Vec<_>>();
    assert_eq!(node_types, vec!["Cast", "Cast", "Div", "Floor", "Cast"]);
}

#[test]
fn test_flip_multiple_axes_chains_gather_nodes() {
    let mut converter = Converter::new();
    converter.state.tensor_shapes.insert(10, vec![2, 3]);

    let op_json = json!({
        "#": "1.flip",
        "A": [
            {
                "AT": {
                    "D": [
                        { "D": 0 },
                        { "D": 1 }
                    ]
                },
                "N": "axis"
            }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            {
                "%": 11,
                "TT": {
                    "D": [
                        { "#": "0.t_f32" },
                        [2, 3]
                    ]
                }
            }
        ]
    });

    converter.process_pass2_op("1.flip", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    let node_types = graph
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect::<Vec<_>>();
    assert_eq!(node_types, vec!["Gather", "Gather"]);
    assert_eq!(graph.node[0].input, vec!["tensor_10", "flip_indices_11_0"]);
    assert_eq!(graph.node[0].output, vec!["flip_axis_11_0"]);
    assert_eq!(
        graph.node[1].input,
        vec!["flip_axis_11_0", "flip_indices_11_1"]
    );
    assert_eq!(graph.node[1].output, vec!["tensor_11"]);
    let axes = graph
        .node
        .iter()
        .map(|node| {
            node.attribute
                .iter()
                .find(|attr| attr.name == "axis")
                .map(|attr| attr.i)
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(axes, vec![0, 1]);
}

#[test]
fn test_gather_rejects_non_constant_axis_input() {
    let mut converter = Converter::new();

    let op_json = json!({
        "#": "1.gather",
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 12 }
        ],
        "O": [
            { "%": 13 }
        ]
    });

    let err = converter
        .process_pass2_op("1.gather", &op_json)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("gather requires constant axis input")
    );
}
