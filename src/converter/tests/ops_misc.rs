#![allow(unused_imports)]

use crate::converter::Converter;
use crate::helper::dt;
use crate::proto::{PaddleDataType, TensorDesc};
use prost::Message;
use serde_json::json;
use std::fs;
use std::io::Write;
use std::sync::Arc;

#[test]
fn test_full_with_dynamic_shape_falls_back_to_non_negative_initializer_dims() {
    let mut converter = Converter::new();
    converter.state.constants.insert(11, vec![3.0]);

    let op_json = json!({
        "#": "1.full",
        "A": [
            { "AT": { "D": [-1, 2] }, "N": "shape" },
            { "AT": { "D": 3.0 }, "N": "value" },
            { "AT": { "D": "float32" }, "N": "dtype" }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    converter.process_pass2_op("1.full", &op_json).unwrap();

    let tensor = converter
        .onnx_graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "tensor_11")
        .unwrap();
    assert!(!tensor.dims.iter().any(|&dim| dim < 0));
    assert_eq!(tensor.dims, vec![1]);
}

#[test]
fn test_full_with_zero_dim_emits_empty_initializer() {
    let mut converter = Converter::new();
    converter.state.constants.insert(11, vec![7.0]);

    let op_json = json!({
        "#": "1.full",
        "A": [
            { "AT": { "D": [0, 3] }, "N": "shape" },
            { "AT": { "D": 7.0 }, "N": "value" },
            { "AT": { "D": "float32" }, "N": "dtype" }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    converter.process_pass2_op("1.full", &op_json).unwrap();

    let tensor = converter
        .onnx_graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "tensor_11")
        .unwrap();
    assert_eq!(tensor.dims, vec![0, 3]);
    assert!(tensor.raw_data.is_empty());
}

#[test]
fn test_full_supports_small_integer_initializer_dtypes() {
    for (dtype, value, expected_dtype, expected_raw) in [
        ("int8", -3.0, dt::INT8, (-3_i8).to_le_bytes().to_vec()),
        ("uint8", 255.0, dt::UINT8, 255_u8.to_le_bytes().to_vec()),
        ("int16", 1024.0, dt::INT16, 1024_i16.to_le_bytes().to_vec()),
    ] {
        let mut converter = Converter::new();
        converter.state.constants.insert(11, vec![value]);

        let op_json = json!({
            "#": "1.full",
            "A": [
                { "AT": { "D": [1] }, "N": "shape" },
                { "AT": { "D": value }, "N": "value" },
                { "AT": { "D": dtype }, "N": "dtype" }
            ],
            "O": [
                { "%": 11 }
            ]
        });

        converter.process_pass2_op("1.full", &op_json).unwrap();

        let tensor = converter
            .onnx_graph
            .initializer
            .iter()
            .find(|tensor| tensor.name == "tensor_11")
            .unwrap();
        assert_eq!(
            tensor.data_type, expected_dtype,
            "unexpected dtype for {}",
            dtype
        );
        assert_eq!(
            tensor.raw_data, expected_raw,
            "unexpected raw_data for {}",
            dtype
        );
    }
}

#[test]
fn test_assign_value_supports_uint8_raw_data() {
    let mut converter = Converter::new();

    let op_json = json!({
        "#": "1.assign_value_",
        "A": [
            { "AT": { "D": "uint8" }, "N": "dtype" },
            { "AT": { "D": [2] }, "N": "shape" },
            {
                "AT": {
                    "D": [
                        { "D": 1.0 },
                        { "D": 255.0 }
                    ]
                },
                "N": "values"
            }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    converter
        .process_pass2_op("1.assign_value_", &op_json)
        .unwrap();

    let tensor = converter
        .onnx_graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "tensor_11")
        .unwrap();
    assert_eq!(tensor.data_type, dt::UINT8);
    assert_eq!(tensor.dims, vec![2]);
    assert_eq!(tensor.raw_data, vec![1, 255]);
}

#[test]
fn test_assign_value_supports_special_float_strings() {
    let mut converter = Converter::new();

    let op_json = json!({
        "#": "1.assign_value_",
        "A": [
            { "AT": { "D": "float32" }, "N": "dtype" },
            { "AT": { "D": [2] }, "N": "shape" },
            {
                "AT": {
                    "D": [
                        { "VD": "inf" },
                        { "VD": "-nan" }
                    ]
                },
                "N": "values"
            }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    converter
        .process_pass2_op("1.assign_value_", &op_json)
        .unwrap();

    let tensor = converter
        .onnx_graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "tensor_11")
        .unwrap();
    let values = tensor
        .raw_data
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert!(values[0].is_infinite() && values[0].is_sign_positive());
    assert!(values[1].is_nan());
}

#[test]
fn test_assign_value_rejects_shape_value_count_mismatch() {
    let mut converter = Converter::new();

    let op_json = json!({
        "#": "1.assign_value_",
        "A": [
            { "AT": { "D": "float32" }, "N": "dtype" },
            { "AT": { "D": [2, 3] }, "N": "shape" },
            {
                "AT": {
                    "D": [
                        { "D": 1.0 },
                        { "D": 2.0 },
                        { "D": 3.0 },
                        { "D": 4.0 }
                    ]
                },
                "N": "values"
            }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    let err = converter
        .process_pass2_op("1.assign_value_", &op_json)
        .unwrap_err();
    assert!(err.to_string().contains("expects 6 values, got 4"));
}

#[test]
fn test_full_rejects_initializer_volume_mismatch() {
    let mut converter = Converter::new();
    converter.state.constants.insert(11, vec![1.0, 2.0]);

    let op_json = json!({
        "#": "1.full",
        "A": [
            { "AT": { "D": [2, 2] }, "N": "shape" },
            { "AT": { "D": "float32" }, "N": "dtype" }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    let err = converter.process_pass2_op("1.full", &op_json).unwrap_err();
    assert!(err.to_string().contains("expects 4 values, got 2"));
}

#[test]
fn test_full_like_casts_value_to_small_integer_dtype_before_expand() {
    let mut converter = Converter::new();
    converter.state.tensor_shapes.insert(10, vec![2, 3]);
    converter.state.tensor_shapes.insert(11, vec![]);

    let op_json = json!({
        "#": "1.full_like",
        "A": [
            { "AT": { "D": "int8" }, "N": "dtype" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12 }
        ]
    });

    converter.process_pass2_op("1.full_like", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 3);
    assert_eq!(graph.node[0].op_type, "Shape");
    assert_eq!(graph.node[1].op_type, "Cast");
    assert_eq!(graph.node[1].input, vec!["tensor_11"]);
    assert_eq!(graph.node[1].output, vec!["full_like_cast_12"]);
    let cast_to = graph.node[1]
        .attribute
        .iter()
        .find(|attr| attr.name == "to")
        .unwrap();
    assert_eq!(cast_to.i, dt::INT8 as i64);
    assert_eq!(graph.node[2].op_type, "Expand");
    assert_eq!(
        graph.node[2].input,
        vec!["full_like_cast_12", "full_like_shape_12"]
    );
}

#[test]
fn test_full_with_tensor_casts_value_and_unsqueezes_scalar_shape_input() {
    let mut converter = Converter::new();
    converter.state.tensor_shapes.insert(10, vec![]);
    converter.state.tensor_shapes.insert(11, vec![]);

    let op_json = json!({
        "#": "1.full_with_tensor",
        "A": [
            { "AT": { "D": "uint8" }, "N": "dtype" }
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
        .process_pass2_op("1.full_with_tensor", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 3);
    assert_eq!(graph.node[0].op_type, "Cast");
    assert_eq!(graph.node[0].output, vec!["full_with_tensor_cast_12"]);
    let cast_to = graph.node[0]
        .attribute
        .iter()
        .find(|attr| attr.name == "to")
        .unwrap();
    assert_eq!(cast_to.i, dt::UINT8 as i64);
    assert_eq!(graph.node[1].op_type, "Unsqueeze");
    assert_eq!(graph.node[1].output, vec!["full_with_tensor_shape_12"]);
    assert_eq!(graph.node[2].op_type, "Expand");
    assert_eq!(
        graph.node[2].input,
        vec!["full_with_tensor_cast_12", "full_with_tensor_shape_12"]
    );
}

#[test]
fn test_batch_norm_rejects_training_mode_attrs() {
    let mut converter = Converter::new();

    let op_json = json!({
        "#": "1.batch_norm_",
        "A": [
            { "AT": { "D": false }, "N": "is_test" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 12 },
            { "%": 13 },
            { "%": 14 }
        ],
        "O": [
            { "%": 15 },
            { "%": 16 },
            { "%": 17 }
        ]
    });

    let err = converter
        .process_pass2_op("1.batch_norm_", &op_json)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("only supports inference-style lowering with is_test=true")
    );
}

#[test]
fn test_batch_norm_reorders_paddle_inputs_for_onnx() {
    let mut converter = Converter::new();

    let op_json = json!({
        "#": "1.batch_norm_",
        "A": [
            { "AT": { "D": true }, "N": "is_test" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 12 },
            { "%": 13 },
            { "%": 14 }
        ],
        "O": [
            { "%": 15 },
            { "%": 16 },
            { "%": 17 }
        ]
    });

    converter
        .process_pass2_op("1.batch_norm_", &op_json)
        .unwrap();

    let node = &converter.onnx_graph.node[0];
    assert_eq!(node.op_type, "BatchNormalization");
    assert_eq!(
        node.input,
        vec![
            "tensor_10",
            "tensor_13",
            "tensor_14",
            "tensor_11",
            "tensor_12"
        ]
    );
    assert_eq!(node.output, vec!["tensor_15"]);
}

#[test]
fn test_batch_norm_rejects_nhwc() {
    let mut converter = Converter::new();

    let op_json = json!({
        "#": "1.batch_norm_",
        "A": [
            { "AT": { "D": true }, "N": "is_test" },
            { "AT": { "D": "NHWC" }, "N": "data_format" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 12 },
            { "%": 13 },
            { "%": 14 }
        ],
        "O": [
            { "%": 15 }
        ]
    });

    let err = converter
        .process_pass2_op("1.batch_norm_", &op_json)
        .unwrap_err();
    assert!(err.to_string().contains("only supports NCHW"));
}

#[test]
fn test_one_hot_casts_depth_and_materializes_values_tensor() {
    let mut converter = Converter::new();
    converter
        .state
        .tensor_types
        .insert(12, "0.t_f32".to_string());

    let op_json = json!({
        "#": "1.one_hot",
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
                        [-1, 50]
                    ]
                }
            }
        ]
    });

    converter.process_pass2_op("1.one_hot", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 6);
    assert_eq!(graph.node[0].op_type, "Cast");
    assert_eq!(graph.node[1].op_type, "Cast");
    assert_eq!(
        graph.node[4].input,
        vec!["one_hot_indices_expanded_12", "one_hot_range_12"]
    );
    assert_eq!(graph.node[5].op_type, "Cast");
    assert!(
        graph
            .initializer
            .iter()
            .any(|tensor| tensor.name == "one_hot_unsqueeze_axes_12")
    );
}

#[test]
fn test_one_hot_supports_rank2_indices() {
    let mut converter = Converter::new();
    converter.state.tensor_shapes.insert(10, vec![2, 3]);
    converter
        .state
        .tensor_types
        .insert(12, "0.t_f32".to_string());

    let op_json = json!({
        "#": "1.one_hot",
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
                        [2, 3, 50]
                    ]
                }
            }
        ]
    });

    converter.process_pass2_op("1.one_hot", &op_json).unwrap();

    let equal = converter
        .onnx_graph
        .node
        .iter()
        .find(|node| node.op_type == "Equal")
        .unwrap();
    assert_eq!(
        equal.input,
        vec!["one_hot_indices_expanded_12", "one_hot_range_12"]
    );

    let axes = converter
        .onnx_graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "one_hot_unsqueeze_axes_12")
        .unwrap();
    assert_eq!(axes.dims, vec![1]);
    assert_eq!(axes.raw_data, 2_i64.to_le_bytes().to_vec());
}

#[test]
fn test_one_hot_materializes_constant_depth_input() {
    let mut converter = Converter::new();
    converter.state.constants.insert(11, vec![4.0]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_i64".to_string());
    converter.state.tensor_shapes.insert(11, vec![]);
    converter
        .state
        .tensor_types
        .insert(12, "0.t_f32".to_string());

    let op_json = json!({
        "#": "1.one_hot",
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
                        [3, 4]
                    ]
                }
            }
        ]
    });

    converter.process_pass2_op("1.one_hot", &op_json).unwrap();

    assert!(
        converter
            .onnx_graph
            .node
            .iter()
            .all(|node| !node.input.contains(&"tensor_11".to_string()))
    );
    let depth = converter
        .onnx_graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "one_hot_depth_i64_12")
        .unwrap();
    assert_eq!(depth.dims, Vec::<i64>::new());
    assert_eq!(depth.raw_data, 4_i64.to_le_bytes().to_vec());
}

#[test]
fn test_bitwise_and_on_bool_uses_logical_and() {
    let mut converter = Converter::new();
    converter
        .state
        .tensor_types
        .insert(10, "0.t_bool".to_string());
    converter.state.tensor_shapes.insert(10, vec![4]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_bool".to_string());
    converter.state.tensor_shapes.insert(11, vec![4]);
    converter
        .state
        .tensor_types
        .insert(12, "0.t_bool".to_string());
    converter.state.tensor_shapes.insert(12, vec![4]);

    let op_json = json!({
        "#": "1.bitwise_and",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12 }
        ]
    });

    converter
        .process_pass2_op("1.bitwise_and", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "And");
    assert_eq!(graph.node[0].output, vec!["tensor_12"]);
}

#[test]
fn test_bitwise_and_on_int64_uses_onnx_bitwiseand_at_opset_18() {
    let mut converter = Converter::new();
    converter.set_target_opset(18);
    converter
        .state
        .tensor_types
        .insert(10, "0.t_i64".to_string());
    converter.state.tensor_shapes.insert(10, vec![4]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_i64".to_string());
    converter.state.tensor_shapes.insert(11, vec![4]);
    converter
        .state
        .tensor_types
        .insert(12, "0.t_i64".to_string());
    converter.state.tensor_shapes.insert(12, vec![4]);

    let op_json = json!({
        "#": "1.bitwise_and",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12 }
        ]
    });

    converter
        .process_pass2_op("1.bitwise_and", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "BitwiseAnd");
    assert_eq!(graph.node[0].output, vec!["tensor_12"]);
}

#[test]
fn test_bitwise_and_on_int64_lowers_before_opset_18() {
    let mut converter = Converter::new();
    converter.set_target_opset(17);
    converter
        .state
        .tensor_types
        .insert(10, "0.t_i64".to_string());
    converter.state.tensor_shapes.insert(10, vec![4]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_i64".to_string());
    converter.state.tensor_shapes.insert(11, vec![4]);
    converter
        .state
        .tensor_types
        .insert(12, "0.t_i64".to_string());
    converter.state.tensor_shapes.insert(12, vec![4]);

    let op_json = json!({
        "#": "1.bitwise_and",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12 }
        ]
    });

    converter
        .process_pass2_op("1.bitwise_and", &op_json)
        .unwrap();
    let graph = &converter.onnx_graph;
    assert!(graph.node.len() > 10);
    assert_eq!(graph.node.last().unwrap().output, vec!["tensor_12"]);
}

#[test]
fn test_bitwise_and_on_uint8_uses_onnx_bitwiseand_at_opset_18() {
    let mut converter = Converter::new();
    converter.set_target_opset(18);
    converter
        .state
        .tensor_types
        .insert(10, "0.t_ui8".to_string());
    converter.state.tensor_shapes.insert(10, vec![4]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_ui8".to_string());
    converter.state.tensor_shapes.insert(11, vec![4]);
    converter
        .state
        .tensor_types
        .insert(12, "0.t_ui8".to_string());
    converter.state.tensor_shapes.insert(12, vec![4]);

    let op_json = json!({
        "#": "1.bitwise_and",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12 }
        ]
    });

    converter
        .process_pass2_op("1.bitwise_and", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "BitwiseAnd");
    assert_eq!(graph.node[0].output, vec!["tensor_12"]);
}

#[test]
fn test_bitwise_not_on_uint8_uses_onnx_bitwisenot_at_opset_18() {
    let mut converter = Converter::new();
    converter.set_target_opset(18);
    converter
        .state
        .tensor_types
        .insert(10, "0.t_ui8".to_string());
    converter.state.tensor_shapes.insert(10, vec![4]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_ui8".to_string());
    converter.state.tensor_shapes.insert(11, vec![4]);

    let op_json = json!({
        "#": "1.bitwise_not",
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    converter
        .process_pass2_op("1.bitwise_not", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "BitwiseNot");
    assert_eq!(graph.node[0].output, vec!["tensor_11"]);
}

#[test]
fn test_bitwise_not_on_int8_uses_onnx_bitwisenot_at_opset_18() {
    let mut converter = Converter::new();
    converter.set_target_opset(18);
    converter
        .state
        .tensor_types
        .insert(10, "0.t_i8".to_string());
    converter.state.tensor_shapes.insert(10, vec![4]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_i8".to_string());
    converter.state.tensor_shapes.insert(11, vec![4]);

    let op_json = json!({
        "#": "1.bitwise_not",
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    converter
        .process_pass2_op("1.bitwise_not", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "BitwiseNot");
}

#[test]
fn test_bitwise_not_on_int16_uses_onnx_bitwisenot_at_opset_18() {
    let mut converter = Converter::new();
    converter.set_target_opset(18);
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

    let op_json = json!({
        "#": "1.bitwise_not",
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    converter
        .process_pass2_op("1.bitwise_not", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "BitwiseNot");
}

#[test]
fn test_bitwise_not_on_uint8_requires_opset_18() {
    let mut converter = Converter::new();
    converter.set_target_opset(17);
    converter
        .state
        .tensor_types
        .insert(10, "0.t_ui8".to_string());
    converter.state.tensor_shapes.insert(10, vec![4]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_ui8".to_string());
    converter.state.tensor_shapes.insert(11, vec![4]);

    let op_json = json!({
        "#": "1.bitwise_not",
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    let err = converter
        .process_pass2_op("1.bitwise_not", &op_json)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("bitwise_not on integer tensors requires opset >= 18")
    );
}

#[test]
fn test_bitwise_not_on_int64_lowers_before_opset_18() {
    let mut converter = Converter::new();
    converter.set_target_opset(17);
    converter
        .state
        .tensor_types
        .insert(10, "0.t_i64".to_string());
    converter.state.tensor_shapes.insert(10, vec![4]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_i64".to_string());
    converter.state.tensor_shapes.insert(11, vec![4]);

    let op_json = json!({
        "#": "1.bitwise_not",
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    converter
        .process_pass2_op("1.bitwise_not", &op_json)
        .unwrap();
    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 2);
    assert_eq!(graph.node[0].op_type, "Add");
    assert_eq!(graph.node[1].op_type, "Neg");
    assert_eq!(graph.node[1].output, vec!["tensor_11"]);
}

#[test]
fn test_pow_uses_input_dtype_for_exponent_constant() {
    let mut converter = Converter::new();
    converter
        .state
        .tensor_types
        .insert(10, "0.t_i16".to_string());
    converter.state.tensor_shapes.insert(10, vec![4]);

    let op_json = json!({
        "#": "1.pow",
        "A": [
            { "AT": { "D": 2.0 }, "N": "y" }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    converter.process_pass2_op("1.pow", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Pow");
    assert_eq!(graph.node[0].input[1], "pow_exponent_11");

    let exponent_init = graph
        .initializer
        .iter()
        .find(|t| t.name == "pow_exponent_11")
        .expect("exponent initializer not found");
    assert_eq!(exponent_init.data_type, dt::INT16);
    assert_eq!(exponent_init.dims, Vec::<i64>::new());
    assert_eq!(exponent_init.raw_data, 2_i16.to_le_bytes().to_vec());
}

#[test]
fn test_bitwise_and_on_int8_lowers_before_opset_18() {
    let mut converter = Converter::new();
    converter.set_target_opset(17);
    converter
        .state
        .tensor_types
        .insert(10, "0.t_i8".to_string());
    converter.state.tensor_shapes.insert(10, vec![4]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_i8".to_string());
    converter.state.tensor_shapes.insert(11, vec![4]);
    converter
        .state
        .tensor_types
        .insert(12, "0.t_i8".to_string());
    converter.state.tensor_shapes.insert(12, vec![4]);

    let op_json = json!({
        "#": "1.bitwise_and",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12 }
        ]
    });

    converter
        .process_pass2_op("1.bitwise_and", &op_json)
        .unwrap();
    assert!(converter.onnx_graph.node.len() > 10);
    assert_eq!(
        converter.onnx_graph.node.last().unwrap().output,
        vec!["tensor_12"]
    );
}

#[test]
fn test_bitwise_and_on_int16_lowers_before_opset_18() {
    let mut converter = Converter::new();
    converter.set_target_opset(17);
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
        "#": "1.bitwise_and",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12 }
        ]
    });

    converter
        .process_pass2_op("1.bitwise_and", &op_json)
        .unwrap();
    assert!(converter.onnx_graph.node.len() > 10);
    assert_eq!(
        converter.onnx_graph.node.last().unwrap().output,
        vec!["tensor_12"]
    );
}

#[test]
fn test_bitwise_and_on_int16_uses_onnx_bitwiseand_at_opset_18() {
    let mut converter = Converter::new();
    converter.set_target_opset(18);
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
        "#": "1.bitwise_and",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12 }
        ]
    });

    converter
        .process_pass2_op("1.bitwise_and", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "BitwiseAnd");
    assert_eq!(graph.node[0].output, vec!["tensor_12"]);
}

#[test]
fn test_cast_supports_small_integer_dtypes() {
    for (dtype, expected_to) in [
        ("int8", dt::INT8),
        ("uint8", dt::UINT8),
        ("int16", dt::INT16),
    ] {
        let mut converter = Converter::new();
        let op_json = json!({
            "#": "1.cast",
            "A": [
                { "AT": { "D": dtype }, "N": "dtype" }
            ],
            "I": [
                { "%": 10 }
            ],
            "O": [
                { "%": 11 }
            ]
        });

        converter.process_pass2_op("1.cast", &op_json).unwrap();

        let node = &converter.onnx_graph.node[0];
        let to_attr = node
            .attribute
            .iter()
            .find(|attr| attr.name == "to")
            .unwrap();
        assert_eq!(node.op_type, "Cast");
        assert_eq!(
            to_attr.i, expected_to as i64,
            "unexpected cast target for {}",
            dtype
        );
    }
}

#[test]
fn test_conv_transpose_bails_on_4th_input_bias() {
    let mut converter = Converter::new();

    let op_json = json!({
        "#": "1.conv2d_transpose_",
        "A": [],
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

    let result = converter.process_pass2_op("1.conv2d_transpose_", &op_json);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("ConvTranspose") && msg.contains("bias"),
        "expected error about ConvTranspose bias, got: {}",
        msg
    );
}

#[test]
fn test_conv_transpose_with_3_inputs_takes_only_2() {
    let mut converter = Converter::new();
    converter.state.constants.insert(12, vec![32.0, 48.0]);

    let op_json = json!({
        "#": "1.conv2d_transpose_",
        "A": [],
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 12 }
        ],
        "O": [
            { "%": 14 }
        ]
    });

    converter
        .process_pass2_op("1.conv2d_transpose_", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    let node = &graph.node[0];
    assert_eq!(node.op_type, "ConvTranspose");
    assert_eq!(node.input.len(), 2);
    let output_shape = node
        .attribute
        .iter()
        .find(|attr| attr.name == "output_shape")
        .unwrap();
    assert_eq!(output_shape.ints, vec![32, 48]);
}

#[test]
fn test_fetch_without_input_bails() {
    let mut converter = Converter::new();

    let op_json = json!({
        "#": "1.fetch",
        "A": [
            { "AT": { "D": "out" }, "N": "name" }
        ],
        "I": [],
        "O": [
            {
                "%": 10,
                "TT": {
                    "D": [
                        { "#": "0.t_f32" },
                        [1]
                    ]
                }
            }
        ]
    });

    let result = converter.process_pass2_op("1.fetch", &op_json);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("no input"));
}

#[test]
fn test_bool_nan_encoding_bails() {
    let converter = Converter::new();
    let result = converter.encode_scalar_f64_as_raw_data(f64::NAN, dt::BOOL);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("NaN"));
}

#[test]
fn test_float32_encoding_rejects_out_of_range_value() {
    let converter = Converter::new();
    let result = converter.encode_scalar_f64_as_raw_data(1.0e300, dt::FLOAT);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("float32"));
}

#[test]
fn test_add_slice_node_rejects_mismatched_lengths() {
    let mut converter = Converter::new();
    let result = converter.add_slice_node(
        "input".to_string(),
        "output".to_string(),
        &[0],
        &[1],
        None,
        Some(&[1, 1]),
        "bad_slice",
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("steps length"));
}

#[test]
fn test_sub_converter_shares_state_until_copy_on_write() {
    let mut converter = Converter::new();
    converter.state.id_to_name.insert(10, "cond".to_string());
    converter
        .state
        .tensor_types
        .insert(10, "0.t_bool".to_string());
    converter.state.constants.insert(11, vec![1.0]);

    let mut sub = converter.sub_converter();

    assert!(Arc::ptr_eq(
        &converter.state.id_to_name.inner,
        &sub.state.id_to_name.inner
    ));
    assert!(Arc::ptr_eq(
        &converter.state.tensor_types.inner,
        &sub.state.tensor_types.inner
    ));
    assert!(Arc::ptr_eq(
        &converter.state.constants.inner,
        &sub.state.constants.inner
    ));

    sub.state.id_to_name.insert(20, "loop_var".to_string());
    sub.state.constants.insert(12, vec![2.0]);

    assert!(!converter.state.id_to_name.contains_key(&20));
    assert!(!converter.state.constants.contains_key(&12));
    assert!(!Arc::ptr_eq(
        &converter.state.id_to_name.inner,
        &sub.state.id_to_name.inner
    ));
    assert!(Arc::ptr_eq(
        &converter.state.tensor_types.inner,
        &sub.state.tensor_types.inner
    ));
    assert!(!Arc::ptr_eq(
        &converter.state.constants.inner,
        &sub.state.constants.inner
    ));
}
