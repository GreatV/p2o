#![allow(unused_imports)]

use crate::converter::Converter;
use crate::helper::dt;
use crate::proto::{PaddleDataType, TensorDesc};
use prost::Message;
use serde_json::json;
use std::fs;
use std::io::Write;

fn scalar_f32_initializer(converter: &Converter, name: &str) -> f32 {
    let tensor = converter
        .onnx_graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap();
    let bytes: [u8; 4] = tensor.raw_data.as_slice().try_into().unwrap();
    f32::from_le_bytes(bytes)
}

#[test]
fn test_softplus_lowers_beta_and_threshold_semantics() {
    let mut converter = Converter::new();
    converter
        .state
        .tensor_types
        .insert(10, "0.t_f32".to_string());

    let op_json = json!({
        "#": "1.softplus",
        "A": [
            { "AT": { "D": 2.0 }, "N": "beta" },
            { "AT": { "D": 4.0 }, "N": "threshold" }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    converter.process_pass2_op("1.softplus", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    let node_types = graph
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        node_types,
        vec!["Mul", "Softplus", "Div", "Greater", "Where"]
    );
    assert_eq!(graph.node[0].input, vec!["tensor_10", "softplus_beta_11"]);
    assert_eq!(graph.node[1].input, vec!["softplus_scaled_11"]);
    assert_eq!(
        graph.node[2].input,
        vec!["softplus_value_11", "softplus_beta_11"]
    );
    assert_eq!(
        graph.node[3].input,
        vec!["softplus_scaled_11", "softplus_threshold_11"]
    );
    assert_eq!(
        graph.node[4].input,
        vec![
            "softplus_linear_cond_11",
            "tensor_10",
            "softplus_unscaled_11"
        ]
    );
    assert_eq!(graph.node[4].output, vec!["tensor_11"]);
    assert_eq!(scalar_f32_initializer(&converter, "softplus_beta_11"), 2.0);
    assert_eq!(
        scalar_f32_initializer(&converter, "softplus_threshold_11"),
        4.0
    );
}

#[test]
fn test_softplus_rejects_zero_beta() {
    let mut converter = Converter::new();

    let op_json = json!({
        "#": "1.softplus",
        "A": [
            { "AT": { "D": 0.0 }, "N": "beta" },
            { "AT": { "D": 20.0 }, "N": "threshold" }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    let err = converter
        .process_pass2_op("1.softplus", &op_json)
        .unwrap_err();
    assert!(err.to_string().contains("finite beta != 0"));
}

#[test]
fn test_softshrink_sets_shrink_bias_to_threshold() {
    let mut converter = Converter::new();

    let op_json = json!({
        "#": "1.softshrink",
        "A": [
            { "AT": { "D": 0.75 }, "N": "threshold" }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    converter
        .process_pass2_op("1.softshrink", &op_json)
        .unwrap();

    let node = &converter.onnx_graph.node[0];
    assert_eq!(node.op_type, "Shrink");
    let lambd = node
        .attribute
        .iter()
        .find(|attr| attr.name == "lambd")
        .unwrap()
        .f;
    let bias = node
        .attribute
        .iter()
        .find(|attr| attr.name == "bias")
        .unwrap()
        .f;
    assert_eq!(lambd, 0.75);
    assert_eq!(bias, 0.75);
}

#[test]
fn test_round_requires_opset_11() {
    let mut converter = Converter::new();
    converter.set_target_opset(10);

    let op_json = json!({
        "#": "1.round",
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    let err = converter.process_pass2_op("1.round", &op_json).unwrap_err();
    assert!(err.to_string().contains("round requires opset >= 11"));
}

#[test]
fn test_round_emits_direct_onnx_node_at_opset_11() {
    let mut converter = Converter::new();
    converter.set_target_opset(11);

    let op_json = json!({
        "#": "1.round",
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    converter.process_pass2_op("1.round", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Round");
    assert_eq!(graph.node[0].input, vec!["tensor_10"]);
    assert_eq!(graph.node[0].output, vec!["tensor_11"]);
}

#[test]
fn test_hardswish_opset_14_does_not_forward_paddle_attrs() {
    let mut converter = Converter::new();
    converter.set_target_opset(14);

    let op_json = json!({
        "#": "1.hardswish",
        "A": [
            { "AT": { "D": "/scope/Hardswish/" }, "N": "struct_name" }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    converter.process_pass2_op("1.hardswish", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "HardSwish");
    assert!(graph.node[0].attribute.is_empty());
    assert_eq!(graph.node[0].input, vec!["tensor_10"]);
    assert_eq!(graph.node[0].output, vec!["tensor_11"]);
}

#[test]
fn test_layer_norm_requires_opset_17() {
    let mut converter = Converter::new();
    converter.set_target_opset(16);

    let op_json = json!({
        "#": "1.layer_norm",
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
        .process_pass2_op("1.layer_norm", &op_json)
        .unwrap_err();
    assert!(err.to_string().contains("layer_norm requires opset >= 17"));
}

#[test]
fn test_layer_norm_emits_onnx_node_at_opset_17() {
    let mut converter = Converter::new();
    converter.set_target_opset(17);

    let op_json = json!({
        "#": "1.layer_norm",
        "A": [
            { "AT": { "#": "0.a_f32", "D": 1e-5 }, "N": "epsilon" },
            { "AT": { "#": "0.a_i32", "D": 2 }, "N": "begin_norm_axis" },
            { "AT": { "#": "0.a_str", "D": "/scope/LayerNorm/" }, "N": "struct_name" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 12 }
        ],
        "O": [
            { "%": 13 }
        ]
    });

    converter
        .process_pass2_op("1.layer_norm", &op_json)
        .unwrap();

    let node = &converter.onnx_graph.node[0];
    assert_eq!(node.op_type, "LayerNormalization");
    assert_eq!(node.input, vec!["tensor_10", "tensor_11", "tensor_12"]);
    assert_eq!(node.output, vec!["tensor_13"]);
    assert!(
        node.attribute
            .iter()
            .any(|attr| attr.name == "axis" && attr.i == 2)
    );
    assert!(!node.attribute.iter().any(|attr| attr.name == "struct_name"));
}

#[test]
fn test_layer_norm_reconstructs_variance_output() {
    let mut converter = Converter::new();
    converter.set_target_opset(17);
    converter
        .state
        .tensor_types
        .insert(10, "0.t_f32".to_string());

    let op_json = json!({
        "#": "1.layer_norm",
        "A": [
            { "AT": { "#": "0.a_f32", "D": 0.001 }, "N": "epsilon" },
            { "AT": { "#": "0.a_i32", "D": 2 }, "N": "begin_norm_axis" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 12 }
        ],
        "O": [
            { "%": 13 },
            { "%": 14 },
            { "%": 15 }
        ]
    });

    converter
        .process_pass2_op("1.layer_norm", &op_json)
        .unwrap();

    assert_eq!(converter.onnx_graph.node.len(), 4);
    assert_eq!(converter.onnx_graph.node[0].op_type, "LayerNormalization");
    assert_eq!(
        converter.onnx_graph.node[0].output,
        vec!["tensor_13", "tensor_14", "layer_norm_inv_stddev_15"]
    );
    assert_eq!(converter.onnx_graph.node[1].op_type, "Mul");
    assert_eq!(converter.onnx_graph.node[2].op_type, "Reciprocal");
    assert_eq!(converter.onnx_graph.node[3].op_type, "Sub");
    assert_eq!(converter.onnx_graph.node[3].output, vec!["tensor_15"]);
    assert_eq!(
        scalar_f32_initializer(&converter, "layer_norm_epsilon_15"),
        0.001
    );
    // float32 input: epsilon is float32 and no cast is inserted.
    let epsilon = converter
        .onnx_graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "layer_norm_epsilon_15")
        .unwrap();
    assert_eq!(epsilon.data_type, dt::FLOAT);
    assert!(
        !converter
            .onnx_graph
            .node
            .iter()
            .any(|node| node.op_type == "Cast")
    );
}

#[test]
fn test_layer_norm_variance_casts_to_float16_output_dtype() {
    let mut converter = Converter::new();
    converter.set_target_opset(17);
    // float16 input/variance: ONNX InvStdDev stays float32 (stash type), so the
    // reconstruction must run in float32 and cast the variance to float16.
    converter
        .state
        .tensor_types
        .insert(10, "0.t_f16".to_string());
    converter
        .state
        .tensor_types
        .insert(15, "0.t_f16".to_string());

    let op_json = json!({
        "#": "1.layer_norm",
        "A": [
            { "AT": { "#": "0.a_f32", "D": 0.001 }, "N": "epsilon" },
            { "AT": { "#": "0.a_i32", "D": 2 }, "N": "begin_norm_axis" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 12 }
        ],
        "O": [
            { "%": 13 },
            { "%": 14 },
            { "%": 15 }
        ]
    });

    converter
        .process_pass2_op("1.layer_norm", &op_json)
        .unwrap();

    // LayerNormalization, Mul, Reciprocal, Sub, Cast.
    assert_eq!(converter.onnx_graph.node.len(), 5);
    let sub = &converter.onnx_graph.node[3];
    assert_eq!(sub.op_type, "Sub");
    assert_eq!(sub.output, vec!["layer_norm_variance_stash_15"]);
    let cast = &converter.onnx_graph.node[4];
    assert_eq!(cast.op_type, "Cast");
    assert_eq!(cast.input, vec!["layer_norm_variance_stash_15"]);
    assert_eq!(cast.output, vec!["tensor_15"]);
    assert!(
        cast.attribute
            .iter()
            .any(|attr| attr.name == "to" && attr.i == i64::from(dt::FLOAT16))
    );
    // epsilon is built in the float32 stash dtype, not float16.
    let epsilon = converter
        .onnx_graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "layer_norm_epsilon_15")
        .unwrap();
    assert_eq!(epsilon.data_type, dt::FLOAT);
}

#[test]
fn test_greater_equal_lowers_before_opset_12() {
    let mut converter = Converter::new();
    converter.set_target_opset(11);

    let op_json = json!({
        "#": "1.greater_equal",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12 }
        ]
    });

    converter
        .process_pass2_op("1.greater_equal", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 3);
    assert_eq!(graph.node[0].op_type, "Greater");
    assert_eq!(graph.node[0].input, vec!["tensor_10", "tensor_11"]);
    assert_eq!(graph.node[1].op_type, "Equal");
    assert_eq!(graph.node[1].input, vec!["tensor_10", "tensor_11"]);
    assert_eq!(graph.node[2].op_type, "Or");
    assert_eq!(graph.node[2].output, vec!["tensor_12"]);
}

#[test]
fn test_less_equal_lowers_before_opset_12_with_nan_safe_ordered_compare() {
    let mut converter = Converter::new();
    converter.set_target_opset(11);

    let op_json = json!({
        "#": "1.less_equal",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12 }
        ]
    });

    converter
        .process_pass2_op("1.less_equal", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 3);
    assert_eq!(graph.node[0].op_type, "Less");
    assert_eq!(graph.node[0].input, vec!["tensor_10", "tensor_11"]);
    assert_eq!(graph.node[1].op_type, "Equal");
    assert_eq!(graph.node[1].input, vec!["tensor_10", "tensor_11"]);
    assert_eq!(graph.node[2].op_type, "Or");
    assert_eq!(graph.node[2].output, vec!["tensor_12"]);
}

#[test]
fn test_less_equal_uses_direct_onnx_op_at_opset_12() {
    let mut converter = Converter::new();
    converter.set_target_opset(12);

    let op_json = json!({
        "#": "1.less_equal",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12 }
        ]
    });

    converter
        .process_pass2_op("1.less_equal", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "LessOrEqual");
    assert_eq!(graph.node[0].input, vec!["tensor_10", "tensor_11"]);
    assert_eq!(graph.node[0].output, vec!["tensor_12"]);
}

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
