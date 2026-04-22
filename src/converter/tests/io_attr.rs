#![allow(unused_imports)]

use crate::converter::{Converter, ParamMeta};
use crate::helper::dt;
use crate::proto::{PaddleDataType, TensorDesc, onnx};
use prost::Message;
use serde_json::json;
use std::fs;
use std::io::Write;

#[test]
fn test_convert_generic_op_relu() {
    let mut converter = Converter::new();

    let op_json = json!({
        "#": "1.relu",
        "A": [],
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    converter.process_pass2_op("1.relu", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    let node = &graph.node[0];
    assert_eq!(node.op_type, "Relu");
    assert_eq!(node.input, vec!["tensor_10"]);
    assert_eq!(node.output, vec!["tensor_11"]);
}

#[test]
fn test_convert_generic_op_relu_inplace_variant_reuses_relu_mapping() {
    let mut converter = Converter::new();

    let op_json = json!({
        "#": "1.relu_",
        "A": [],
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    converter.process_pass2_op("1.relu_", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Relu");
    assert_eq!(graph.node[0].input, vec!["tensor_10"]);
    assert_eq!(graph.node[0].output, vec!["tensor_11"]);
}

#[test]
fn test_load_paddle_model_synthetic() {
    let synthetic_json = json!({
        "program": {
            "regions": [
                {
                    "blocks": [
                        {
                            "ops": [
                                {
                                    "#": "1.data",
                                    "A": [
                                        { "AT": { "D": "input_0" }, "N": "name" }
                                    ],
                                    "I": [],
                                    "O": [
                                        {
                                            "%": 1,
                                            "TT": {
                                                "D": [
                                                    { "#": "0.t_f32" },
                                                    [1, 3, 224, 224]
                                                ]
                                            }
                                        }
                                    ]
                                },
                                {
                                    "#": "1.relu",
                                    "I": [ { "%": 1 } ],
                                    "O": [ { "%": 2 } ]
                                },
                                {
                                    "#": "1.fetch",
                                    "A": [
                                        { "AT": { "D": "output_0" }, "N": "name" }
                                    ],
                                    "I": [ { "%": 2 } ],
                                    "O": []
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    });

    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    temp_file
        .write_all(synthetic_json.to_string().as_bytes())
        .unwrap();

    let mut converter = Converter::new();
    converter
        .load_paddle_model(temp_file.path().to_str().unwrap())
        .unwrap();

    let graph = &converter.onnx_graph;

    assert_eq!(graph.input.len(), 1);
    assert_eq!(graph.input[0].name, "input_0");

    assert_eq!(graph.output.len(), 1);
    assert_eq!(graph.output[0].name, "output_0");

    assert_eq!(graph.node.len(), 2);
    assert_eq!(graph.node[0].op_type, "Relu");
    assert_eq!(graph.node[1].op_type, "Identity");
    assert_eq!(graph.node[1].output[0], "output_0");
}

#[test]
fn test_load_paddle_model_rejects_file_over_soft_limit() {
    let temp_file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(temp_file.path(), b"{}").unwrap();
    temp_file
        .as_file()
        .set_len((256_u64 * 1024 * 1024) + 1)
        .unwrap();

    let mut converter = Converter::new();
    let err = converter
        .load_paddle_model(temp_file.path().to_str().unwrap())
        .unwrap_err();
    assert!(err.to_string().contains("model JSON is too large"));
}

#[test]
fn test_build_value_info_rejects_unsupported_paddle_dtype() {
    let converter = Converter::new();
    let err = converter
        .build_value_info_from_meta("bad".to_string(), "0.t_bf16", &[1, 2])
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("Unsupported Paddle element type: 0.t_bf16")
    );
}

#[test]
fn test_build_value_info_supports_small_integer_dtypes() {
    let converter = Converter::new();
    for (dtype, expected) in [
        ("0.t_i8", dt::INT8),
        ("0.t_ui8", dt::UINT8),
        ("0.t_i16", dt::INT16),
    ] {
        let value_info = converter
            .build_value_info_from_meta("small_int".to_string(), dtype, &[2, 3])
            .unwrap();
        let elem_type = value_info
            .r#type
            .as_ref()
            .and_then(|ty| ty.value.as_ref())
            .and_then(|value| match value {
                crate::proto::onnx::type_proto::Value::TensorType(tensor) => Some(tensor.elem_type),
                _ => None,
            })
            .unwrap();
        assert_eq!(elem_type, expected, "unexpected elem_type for {}", dtype);
    }
}

#[test]
fn test_build_value_info_preserves_zero_dim() {
    let converter = Converter::new();
    let value_info = converter
        .build_value_info_from_meta("empty".to_string(), "0.t_f32", &[0, 4])
        .unwrap();

    let dims = &value_info
        .r#type
        .as_ref()
        .and_then(|ty| ty.value.as_ref())
        .and_then(|value| match value {
            crate::proto::onnx::type_proto::Value::TensorType(tensor) => tensor.shape.as_ref(),
            _ => None,
        })
        .unwrap()
        .dim;
    assert!(matches!(
        dims[0].value,
        Some(crate::proto::onnx::tensor_shape_proto::dimension::Value::DimValue(0))
    ));
    assert!(matches!(
        dims[1].value,
        Some(crate::proto::onnx::tensor_shape_proto::dimension::Value::DimValue(4))
    ));
}

#[test]
fn test_build_value_info_preserves_dynamic_dims_for_small_integer_dtype() {
    let converter = Converter::new();
    let value_info = converter
        .build_value_info_from_meta("dyn".to_string(), "0.t_i8", &[-1, 4])
        .unwrap();

    let dims = &value_info
        .r#type
        .as_ref()
        .and_then(|ty| ty.value.as_ref())
        .and_then(|value| match value {
            crate::proto::onnx::type_proto::Value::TensorType(tensor) => tensor.shape.as_ref(),
            _ => None,
        })
        .unwrap()
        .dim;
    assert!(dims[0].value.is_none());
    assert!(matches!(
        dims[1].value,
        Some(crate::proto::onnx::tensor_shape_proto::dimension::Value::DimValue(4))
    ));
}

#[test]
fn test_build_value_info_defaults_use_unique_input_output_names() {
    let mut converter = Converter::new();

    let data_op = json!({
        "#": "1.data",
        "I": [],
        "O": [
            {
                "%": 42,
                "TT": {
                    "D": [
                        { "#": "0.t_f32" },
                        [1, 3]
                    ]
                }
            }
        ]
    });
    let fetch_op = json!({
        "#": "1.fetch",
        "I": [
            {
                "%": 42,
                "TT": {
                    "D": [
                        { "#": "0.t_f32" },
                        [1, 3]
                    ]
                }
            }
        ],
        "O": []
    });

    let input_vi = converter.build_value_info(&data_op, true).unwrap();
    let output_vi = converter.build_value_info(&fetch_op, false).unwrap();
    assert_eq!(input_vi.name, "input_42");
    assert_eq!(output_vi.name, "output_42");
}

#[test]
fn test_extract_attributes_reorders_asymmetric_conv_pads() {
    let converter = Converter::new();
    let attrs = json!([
        {
            "N": "paddings",
            "AT": {
                "#": "0.a_array",
                "D": [
                    { "#": "0.a_i64", "D": 1 },
                    { "#": "0.a_i64", "D": 2 },
                    { "#": "0.a_i64", "D": 3 },
                    { "#": "0.a_i64", "D": 4 }
                ]
            }
        }
    ]);

    let onnx_attrs = converter.extract_attributes("conv2d", &attrs);
    let pads = onnx_attrs.iter().find(|attr| attr.name == "pads").unwrap();
    assert_eq!(pads.ints, vec![1, 3, 2, 4]);
}

#[test]
fn test_extract_attributes_maps_same_padding_to_auto_pad() {
    let converter = Converter::new();
    let attrs = json!([
        {
            "N": "padding_algorithm",
            "AT": {
                "#": "0.a_str",
                "D": "SAME"
            }
        },
        {
            "N": "paddings",
            "AT": {
                "#": "0.a_array",
                "D": [
                    { "#": "0.a_i64", "D": 1 },
                    { "#": "0.a_i64", "D": 2 }
                ]
            }
        }
    ]);

    let onnx_attrs = converter.extract_attributes("conv2d", &attrs);
    assert!(
        onnx_attrs
            .iter()
            .any(|attr| { attr.name == "auto_pad" && attr.s == b"SAME_UPPER" })
    );
    assert!(!onnx_attrs.iter().any(|attr| attr.name == "pads"));
}

#[test]
fn test_extract_attributes_renames_begin_norm_axis_to_axis() {
    let converter = Converter::new();
    let attrs = json!([
        {
            "N": "begin_norm_axis",
            "AT": {
                "#": "0.a_i64",
                "D": 2
            }
        }
    ]);

    let onnx_attrs = converter.extract_attributes("layer_norm", &attrs);
    assert_eq!(onnx_attrs.len(), 1);
    assert_eq!(onnx_attrs[0].name, "axis");
    assert_eq!(onnx_attrs[0].i, 2);
}

#[test]
fn test_extract_attributes_renames_hardsigmoid_slope_and_offset() {
    let converter = Converter::new();
    let attrs = json!([
        {
            "N": "slope",
            "AT": {
                "#": "0.a_f32",
                "D": 0.2
            }
        },
        {
            "N": "offset",
            "AT": {
                "#": "0.a_f32",
                "D": 0.5
            }
        }
    ]);

    let onnx_attrs = converter.extract_attributes("hardsigmoid", &attrs);
    assert!(onnx_attrs.iter().any(|attr| attr.name == "alpha"));
    assert!(onnx_attrs.iter().any(|attr| attr.name == "beta"));
}

#[test]
fn test_per_op_attr_skip_allows_scale_for_non_scale_op() {
    let converter = Converter::new();

    // "scale" attr is only skipped for "scale" op, not other ops
    let attrs = json!([
        {
            "N": "scale",
            "AT": { "#": "0.a_f32", "D": 2.0 }
        }
    ]);

    let onnx_attrs = converter.extract_attributes("some_custom_op", &attrs);
    assert!(
        onnx_attrs.iter().any(|attr| attr.name == "scale"),
        "scale attr should pass through for non-scale ops"
    );
}

#[test]
fn test_per_op_attr_skip_blocks_scale_for_scale_op() {
    let converter = Converter::new();

    let attrs = json!([
        {
            "N": "scale",
            "AT": { "#": "0.a_f32", "D": 2.0 }
        }
    ]);

    let onnx_attrs = converter.extract_attributes("scale", &attrs);
    assert!(
        !onnx_attrs.iter().any(|attr| attr.name == "scale"),
        "scale attr should be skipped for scale ops"
    );
}

#[test]
fn test_per_op_attr_skip_padding_algorithm_only_for_conv() {
    let converter = Converter::new();

    let attrs = json!([
        {
            "N": "padding_algorithm",
            "AT": { "#": "0.a_str", "D": "EXPLICIT" }
        }
    ]);

    // Should be skipped for conv ops
    let conv_attrs = converter.extract_attributes("conv2d", &attrs);
    assert!(
        !conv_attrs
            .iter()
            .any(|attr| attr.name == "padding_algorithm"),
        "padding_algorithm should be skipped for conv ops"
    );

    // Should pass through for unknown ops
    let other_attrs = converter.extract_attributes("some_custom_op", &attrs);
    assert!(
        other_attrs
            .iter()
            .any(|attr| attr.name == "padding_algorithm"),
        "padding_algorithm should pass through for non-conv ops"
    );
}
