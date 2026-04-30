#![allow(unused_imports)]

use crate::converter::Converter;
use crate::helper::dt;
use crate::proto::onnx;
use crate::proto::{PaddleDataType, TensorDesc};
use prost::Message;
use serde_json::json;
use std::collections::HashSet;
use std::fs;
use std::io::Write;

fn test_i64_tensor(name: &str, values: &[i64]) -> onnx::TensorProto {
    let mut tensor = onnx::TensorProto {
        name: name.to_string(),
        dims: vec![values.len() as i64],
        data_type: dt::INT64,
        ..Default::default()
    };
    for value in values {
        tensor.raw_data.extend_from_slice(&value.to_le_bytes());
    }
    tensor
}

fn test_cast_node(input: &str, output: &str, to: i32) -> onnx::NodeProto {
    onnx::NodeProto {
        op_type: "Cast".to_string(),
        input: vec![input.to_string()],
        output: vec![output.to_string()],
        attribute: vec![crate::helper::attr_int("to", to as i64)],
        ..Default::default()
    }
}

fn test_value_info(name: &str) -> onnx::ValueInfoProto {
    onnx::ValueInfoProto {
        name: name.to_string(),
        ..Default::default()
    }
}

#[test]
fn test_flatten_with_explicit_stop_axis_uses_output_shape() {
    let mut converter = Converter::new();

    let op_json = json!({
        "#": "1.flatten",
        "A": [
            { "AT": { "D": 2 }, "N": "start_axis" },
            { "AT": { "D": 3 }, "N": "stop_axis" }
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
                        [-1, 256, 625]
                    ]
                }
            }
        ]
    });

    converter.process_pass2_op("1.flatten", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.initializer.len(), 1);
    assert_eq!(graph.initializer[0].name, "flatten_shape_11");
    assert_eq!(graph.initializer[0].dims, vec![3]);

    let mut expected_raw = Vec::new();
    for dim in [-1_i64, 256, 625] {
        expected_raw.extend_from_slice(&dim.to_le_bytes());
    }
    assert_eq!(graph.initializer[0].raw_data, expected_raw);

    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Reshape");
    assert_eq!(graph.node[0].input, vec!["tensor_10", "flatten_shape_11"]);
    assert_eq!(graph.node[0].output, vec!["tensor_11"]);
}

#[test]
fn test_flatten_with_multiple_dynamic_dims_builds_shape_subgraph() {
    let mut converter = Converter::new();
    converter
        .state
        .tensor_shapes
        .insert(10, vec![-1, 120, 8, 15]);

    let op_json = json!({
        "#": "1.flatten",
        "A": [
            { "AT": { "D": 2 }, "N": "start_axis" },
            { "AT": { "D": 3 }, "N": "stop_axis" }
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
                        [-1, 120, -1]
                    ]
                }
            }
        ]
    });

    converter.process_pass2_op("1.flatten", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    let node_types: Vec<&str> = graph
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect();
    assert!(node_types.contains(&"Shape"));
    assert!(node_types.iter().filter(|&&op| op == "Slice").count() >= 2);
    assert!(node_types.contains(&"ReduceProd"));
    assert!(node_types.contains(&"Concat"));

    let reshape = graph
        .node
        .iter()
        .find(|node| node.op_type == "Reshape")
        .unwrap();
    assert_eq!(reshape.input, vec!["tensor_10", "flatten_shape_11"]);
    assert_eq!(reshape.output, vec!["tensor_11"]);

    assert!(
        graph
            .initializer
            .iter()
            .any(|tensor| tensor.name.starts_with("flatten_middle_11"))
    );
}

#[test]
fn test_gelu_approximate_uses_tanh_path() {
    let mut converter = Converter::new();

    let op_json = json!({
        "#": "1.gelu",
        "A": [
            { "AT": { "D": true }, "N": "approximate" }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    });

    converter.process_pass2_op("1.gelu", &op_json).unwrap();

    let node_types = converter
        .onnx_graph
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect::<Vec<_>>();
    assert!(node_types.contains(&"Tanh"));
    assert!(!node_types.contains(&"Erf"));
}

#[test]
fn test_tile_unsqueezes_scalar_input() {
    let mut converter = Converter::new();
    converter.state.tensor_shapes.insert(10, vec![]);

    let op_json = json!({
        "#": "1.tile",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            {
                "%": 12,
                "TT": {
                    "D": [
                        { "#": "0.t_i32" },
                        [-1]
                    ]
                }
            }
        ]
    });

    converter.process_pass2_op("1.tile", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.node.len(), 2);
    assert_eq!(graph.node[0].op_type, "Unsqueeze");
    assert_eq!(
        graph.node[0].input,
        vec!["tensor_10", "tile_unsqueeze_axes_12"]
    );
    assert_eq!(graph.node[1].op_type, "Tile");
    assert_eq!(graph.node[1].input, vec!["tile_unsqueezed_12", "tensor_11"]);
    assert_eq!(graph.node[1].output, vec!["tensor_12"]);
}

#[test]
fn test_nearest_interp_prefers_dynamic_size_tensor_list() {
    let mut converter = Converter::new();
    converter.state.combines.insert(20, vec![21, 22]);
    converter.state.tensor_shapes.insert(21, vec![]);
    converter.state.tensor_shapes.insert(22, vec![]);

    let op_json = json!({
        "#": "1.nearest_interp",
        "A": [
            { "AT": { "D": [] }, "N": "scale" },
            { "AT": { "D": false }, "N": "align_corners" },
            { "AT": { "D": 0 }, "N": "align_mode" }
        ],
        "I": [
            { "%": 10 },
            { "%": 0 },
            { "%": 20 },
            { "%": 0 }
        ],
        "O": [
            {
                "%": 11,
                "TT": {
                    "D": [
                        { "#": "0.t_f32" },
                        [-1, 96, -1, -1]
                    ]
                }
            }
        ]
    });

    converter
        .process_pass2_op("1.nearest_interp", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    let node_types: Vec<&str> = graph
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect();
    assert_eq!(
        node_types,
        vec![
            "Unsqueeze",
            "Unsqueeze",
            "Concat",
            "Shape",
            "Gather",
            "Concat",
            "Resize"
        ]
    );
    assert_eq!(
        graph.node[6].input,
        vec!["tensor_10", "", "", "resize_sizes_11"]
    );
    assert!(
        graph
            .initializer
            .iter()
            .any(|tensor| tensor.name == "resize_unsqueeze_axes_11")
    );
    let resize = &graph.node[6];
    let coordinate_mode = resize
        .attribute
        .iter()
        .find(|attr| attr.name == "coordinate_transformation_mode")
        .map(|attr| String::from_utf8(attr.s.clone()).unwrap())
        .unwrap();
    let nearest_mode = resize
        .attribute
        .iter()
        .find(|attr| attr.name == "nearest_mode")
        .map(|attr| String::from_utf8(attr.s.clone()).unwrap())
        .unwrap();
    assert_eq!(coordinate_mode, "asymmetric");
    assert_eq!(nearest_mode, "floor");
}

#[test]
fn test_sanitize_graph_dedupes_initializers_and_removes_initializer_inputs() {
    let mut graph = crate::proto::onnx::GraphProto {
        input: vec![
            crate::proto::onnx::ValueInfoProto {
                name: "dup_axes".to_string(),
                ..Default::default()
            },
            crate::proto::onnx::ValueInfoProto {
                name: "data".to_string(),
                ..Default::default()
            },
        ],
        initializer: vec![
            crate::proto::onnx::TensorProto {
                name: "dup_axes".to_string(),
                dims: vec![1],
                data_type: crate::helper::dt::INT64,
                raw_data: 0_i64.to_le_bytes().to_vec(),
                ..Default::default()
            },
            crate::proto::onnx::TensorProto {
                name: "dup_axes".to_string(),
                dims: vec![1],
                data_type: crate::helper::dt::INT64,
                raw_data: 1_i64.to_le_bytes().to_vec(),
                ..Default::default()
            },
        ],
        node: vec![crate::proto::onnx::NodeProto {
            op_type: "Unsqueeze".to_string(),
            input: vec!["data".to_string(), "dup_axes".to_string()],
            output: vec!["out".to_string()],
            ..Default::default()
        }],
        ..Default::default()
    };

    Converter::sanitize_graph(&mut graph, true);

    assert_eq!(graph.initializer.len(), 1);
    assert_eq!(graph.initializer[0].name, "dup_axes");
    assert_eq!(graph.input.len(), 1);
    assert_eq!(graph.input[0].name, "data");
}

#[test]
fn test_sanitize_graph_prunes_unused_initializers() {
    let mut graph = crate::proto::onnx::GraphProto {
        initializer: vec![
            crate::proto::onnx::TensorProto {
                name: "used".to_string(),
                dims: vec![1],
                data_type: crate::helper::dt::INT64,
                raw_data: 1_i64.to_le_bytes().to_vec(),
                ..Default::default()
            },
            crate::proto::onnx::TensorProto {
                name: "unused".to_string(),
                dims: vec![1],
                data_type: crate::helper::dt::INT64,
                raw_data: 2_i64.to_le_bytes().to_vec(),
                ..Default::default()
            },
        ],
        node: vec![crate::proto::onnx::NodeProto {
            op_type: "Identity".to_string(),
            input: vec!["used".to_string()],
            output: vec!["out".to_string()],
            ..Default::default()
        }],
        ..Default::default()
    };

    Converter::sanitize_graph(&mut graph, true);

    assert_eq!(graph.initializer.len(), 1);
    assert_eq!(graph.initializer[0].name, "used");
}

#[test]
fn test_sanitize_graph_keeps_parent_initializers_used_by_subgraphs() {
    let body = crate::proto::onnx::GraphProto {
        initializer: vec![
            crate::proto::onnx::TensorProto {
                name: "body_used".to_string(),
                dims: vec![1],
                data_type: crate::helper::dt::INT64,
                raw_data: 2_i64.to_le_bytes().to_vec(),
                ..Default::default()
            },
            crate::proto::onnx::TensorProto {
                name: "body_unused".to_string(),
                dims: vec![1],
                data_type: crate::helper::dt::INT64,
                raw_data: 3_i64.to_le_bytes().to_vec(),
                ..Default::default()
            },
        ],
        node: vec![crate::proto::onnx::NodeProto {
            op_type: "Concat".to_string(),
            input: vec!["outer_used".to_string(), "body_used".to_string()],
            output: vec!["body_out".to_string()],
            ..Default::default()
        }],
        output: vec![crate::proto::onnx::ValueInfoProto {
            name: "body_out".to_string(),
            ..Default::default()
        }],
        ..Default::default()
    };
    let mut graph = crate::proto::onnx::GraphProto {
        initializer: vec![
            crate::proto::onnx::TensorProto {
                name: "outer_used".to_string(),
                dims: vec![1],
                data_type: crate::helper::dt::INT64,
                raw_data: 1_i64.to_le_bytes().to_vec(),
                ..Default::default()
            },
            crate::proto::onnx::TensorProto {
                name: "outer_unused".to_string(),
                dims: vec![1],
                data_type: crate::helper::dt::INT64,
                raw_data: 4_i64.to_le_bytes().to_vec(),
                ..Default::default()
            },
        ],
        node: vec![crate::proto::onnx::NodeProto {
            op_type: "Loop".to_string(),
            attribute: vec![crate::proto::onnx::AttributeProto {
                name: "body".to_string(),
                g: Some(body),
                ..Default::default()
            }],
            ..Default::default()
        }],
        ..Default::default()
    };

    Converter::sanitize_graph(&mut graph, true);

    let parent_initializer_names = graph
        .initializer
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(parent_initializer_names, vec!["outer_used"]);

    let body = graph.node[0].attribute[0].g.as_ref().unwrap();
    let body_initializer_names = body
        .initializer
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(body_initializer_names, vec!["body_used"]);
}

#[test]
fn test_sanitize_graph_materializes_initializer_only_outputs() {
    let mut graph = crate::proto::onnx::GraphProto {
        initializer: vec![crate::proto::onnx::TensorProto {
            name: "const_out".to_string(),
            dims: vec![1],
            data_type: crate::helper::dt::INT64,
            raw_data: 5_i64.to_le_bytes().to_vec(),
            ..Default::default()
        }],
        output: vec![crate::proto::onnx::ValueInfoProto {
            name: "const_out".to_string(),
            ..Default::default()
        }],
        ..Default::default()
    };

    Converter::sanitize_graph(&mut graph, true);

    assert!(graph.initializer.is_empty());
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Constant");
    assert_eq!(graph.node[0].output, vec!["const_out"]);
}

#[test]
fn test_canonicalize_graph_removes_identity_chain_without_changing_output_name() {
    let mut graph = onnx::GraphProto {
        input: vec![test_value_info("x")],
        node: vec![
            onnx::NodeProto {
                op_type: "Identity".to_string(),
                input: vec!["x".to_string()],
                output: vec!["tmp".to_string()],
                ..Default::default()
            },
            onnx::NodeProto {
                op_type: "Relu".to_string(),
                input: vec!["tmp".to_string()],
                output: vec!["y".to_string()],
                ..Default::default()
            },
        ],
        output: vec![test_value_info("y")],
        ..Default::default()
    };

    Converter::canonicalize_graph(&mut graph);

    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Relu");
    assert_eq!(graph.node[0].input, vec!["x"]);
    assert_eq!(graph.output[0].name, "y");
}

#[test]
fn test_canonicalize_graph_keeps_identity_when_output_is_external() {
    let mut graph = onnx::GraphProto {
        input: vec![test_value_info("x")],
        node: vec![onnx::NodeProto {
            op_type: "Identity".to_string(),
            input: vec!["x".to_string()],
            output: vec!["y".to_string()],
            ..Default::default()
        }],
        output: vec![test_value_info("y")],
        ..Default::default()
    };

    Converter::canonicalize_graph(&mut graph);

    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Identity");
    assert_eq!(graph.node[0].output, vec!["y"]);
}

#[test]
fn test_canonicalize_graph_collapses_duplicate_cast_chain() {
    let mut graph = onnx::GraphProto {
        input: vec![test_value_info("x")],
        node: vec![
            test_cast_node("x", "tmp", dt::INT64),
            test_cast_node("tmp", "y", dt::INT64),
        ],
        output: vec![test_value_info("y")],
        ..Default::default()
    };

    Converter::canonicalize_graph(&mut graph);

    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Cast");
    assert_eq!(graph.node[0].input, vec!["x"]);
    assert_eq!(graph.node[0].output, vec!["y"]);
}

#[test]
fn test_canonicalize_graph_does_not_collapse_different_cast_dtypes() {
    let mut graph = onnx::GraphProto {
        input: vec![test_value_info("x")],
        node: vec![
            test_cast_node("x", "tmp", dt::FLOAT),
            test_cast_node("tmp", "y", dt::INT64),
        ],
        output: vec![test_value_info("y")],
        ..Default::default()
    };

    Converter::canonicalize_graph(&mut graph);

    assert_eq!(graph.node.len(), 2);
    assert_eq!(graph.node[1].input, vec!["tmp"]);
}

#[test]
fn test_canonicalize_graph_removes_canceling_unsqueeze_squeeze_pair() {
    let mut graph = onnx::GraphProto {
        input: vec![test_value_info("x")],
        initializer: vec![test_i64_tensor("axes", &[0])],
        node: vec![
            onnx::NodeProto {
                op_type: "Unsqueeze".to_string(),
                input: vec!["x".to_string(), "axes".to_string()],
                output: vec!["tmp".to_string()],
                ..Default::default()
            },
            onnx::NodeProto {
                op_type: "Squeeze".to_string(),
                input: vec!["tmp".to_string(), "axes".to_string()],
                output: vec!["y".to_string()],
                ..Default::default()
            },
            onnx::NodeProto {
                op_type: "Relu".to_string(),
                input: vec!["y".to_string()],
                output: vec!["z".to_string()],
                ..Default::default()
            },
        ],
        output: vec![test_value_info("z")],
        ..Default::default()
    };

    Converter::canonicalize_graph(&mut graph);

    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Relu");
    assert_eq!(graph.node[0].input, vec!["x"]);
}

#[test]
fn test_canonicalize_graph_keeps_shape_adapter_pair_when_temp_has_extra_consumer() {
    let mut graph = onnx::GraphProto {
        input: vec![test_value_info("x")],
        initializer: vec![test_i64_tensor("axes", &[0])],
        node: vec![
            onnx::NodeProto {
                op_type: "Unsqueeze".to_string(),
                input: vec!["x".to_string(), "axes".to_string()],
                output: vec!["tmp".to_string()],
                ..Default::default()
            },
            onnx::NodeProto {
                op_type: "Squeeze".to_string(),
                input: vec!["tmp".to_string(), "axes".to_string()],
                output: vec!["y".to_string()],
                ..Default::default()
            },
            onnx::NodeProto {
                op_type: "Relu".to_string(),
                input: vec!["tmp".to_string()],
                output: vec!["z".to_string()],
                ..Default::default()
            },
        ],
        output: vec![test_value_info("y"), test_value_info("z")],
        ..Default::default()
    };

    Converter::canonicalize_graph(&mut graph);

    assert_eq!(graph.node.len(), 3);
    assert_eq!(graph.node[1].input[0], "tmp");
    assert_eq!(graph.node[2].input[0], "tmp");
}

#[test]
fn test_canonicalize_graph_deduplicates_byte_identical_initializers() {
    let mut graph = onnx::GraphProto {
        input: vec![test_value_info("x")],
        initializer: vec![
            test_i64_tensor("axes_a", &[0]),
            test_i64_tensor("axes_b", &[0]),
        ],
        node: vec![onnx::NodeProto {
            op_type: "Unsqueeze".to_string(),
            input: vec!["x".to_string(), "axes_b".to_string()],
            output: vec!["y".to_string()],
            ..Default::default()
        }],
        output: vec![test_value_info("y")],
        ..Default::default()
    };

    Converter::canonicalize_graph(&mut graph);

    assert_eq!(graph.initializer.len(), 1);
    assert_eq!(graph.initializer[0].name, "axes_a");
    assert_eq!(graph.node[0].input, vec!["x", "axes_a"]);
}

#[test]
fn test_prepare_graph_for_export_prunes_dead_helper_nodes() {
    let mut graph = onnx::GraphProto {
        input: vec![test_value_info("x")],
        node: vec![
            onnx::NodeProto {
                op_type: "Shape".to_string(),
                input: vec!["x".to_string()],
                output: vec!["dead_shape".to_string()],
                ..Default::default()
            },
            onnx::NodeProto {
                op_type: "Relu".to_string(),
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                ..Default::default()
            },
        ],
        output: vec![test_value_info("y")],
        ..Default::default()
    };

    Converter::prepare_graph_for_export(&mut graph).unwrap();

    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Relu");
}

#[test]
fn test_prepare_graph_for_export_prunes_initializers_orphaned_by_dead_helpers() {
    let mut graph = onnx::GraphProto {
        input: vec![test_value_info("x")],
        initializer: vec![test_i64_tensor("dead_axes", &[0])],
        node: vec![
            onnx::NodeProto {
                op_type: "Unsqueeze".to_string(),
                input: vec!["x".to_string(), "dead_axes".to_string()],
                output: vec!["dead_unsqueeze".to_string()],
                ..Default::default()
            },
            onnx::NodeProto {
                op_type: "Relu".to_string(),
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                ..Default::default()
            },
        ],
        output: vec![test_value_info("y")],
        ..Default::default()
    };

    Converter::prepare_graph_for_export(&mut graph).unwrap();

    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Relu");
    assert!(graph.initializer.is_empty());
}

#[test]
fn test_prepare_graph_for_export_keeps_unreviewed_dead_node() {
    let mut graph = onnx::GraphProto {
        input: vec![test_value_info("x")],
        node: vec![
            onnx::NodeProto {
                op_type: "Add".to_string(),
                input: vec!["x".to_string(), "x".to_string()],
                output: vec!["dead_add".to_string()],
                ..Default::default()
            },
            onnx::NodeProto {
                op_type: "Relu".to_string(),
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                ..Default::default()
            },
        ],
        output: vec![test_value_info("y")],
        ..Default::default()
    };

    Converter::prepare_graph_for_export(&mut graph).unwrap();

    assert_eq!(graph.node.len(), 2);
    assert_eq!(graph.node[0].op_type, "Add");
}

#[test]
fn test_prune_dead_nodes_is_idempotent() {
    let mut graph = onnx::GraphProto {
        input: vec![test_value_info("x")],
        node: vec![
            onnx::NodeProto {
                op_type: "Shape".to_string(),
                input: vec!["x".to_string()],
                output: vec!["dead_shape".to_string()],
                ..Default::default()
            },
            onnx::NodeProto {
                op_type: "Identity".to_string(),
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                ..Default::default()
            },
        ],
        output: vec![test_value_info("y")],
        ..Default::default()
    };

    Converter::prune_dead_nodes(&mut graph);
    let once = graph.clone();
    Converter::prune_dead_nodes(&mut graph);

    assert_eq!(graph, once);
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].output, vec!["y"]);
}

#[test]
fn test_prepare_graph_for_export_applies_passes_to_subgraphs() {
    let then_graph = onnx::GraphProto {
        input: vec![test_value_info("sub_x")],
        initializer: vec![
            test_i64_tensor("sub_axes_a", &[0]),
            test_i64_tensor("sub_axes_b", &[0]),
        ],
        node: vec![
            onnx::NodeProto {
                op_type: "Identity".to_string(),
                input: vec!["sub_x".to_string()],
                output: vec!["sub_tmp".to_string()],
                ..Default::default()
            },
            onnx::NodeProto {
                op_type: "Unsqueeze".to_string(),
                input: vec!["sub_tmp".to_string(), "sub_axes_b".to_string()],
                output: vec!["sub_out".to_string()],
                ..Default::default()
            },
        ],
        output: vec![test_value_info("sub_out")],
        ..Default::default()
    };
    let mut graph = onnx::GraphProto {
        input: vec![test_value_info("cond"), test_value_info("sub_x")],
        node: vec![onnx::NodeProto {
            op_type: "If".to_string(),
            input: vec!["cond".to_string()],
            output: vec!["y".to_string()],
            attribute: vec![crate::helper::attr_graph("then_branch", then_graph)],
            ..Default::default()
        }],
        output: vec![test_value_info("y")],
        ..Default::default()
    };

    Converter::prepare_graph_for_export(&mut graph).unwrap();

    let subgraph = graph.node[0].attribute[0].g.as_ref().unwrap();
    assert_eq!(subgraph.node.len(), 1);
    assert_eq!(subgraph.node[0].input, vec!["sub_x", "sub_axes_b"]);
    assert_eq!(subgraph.initializer.len(), 1);
    assert_eq!(subgraph.initializer[0].name, "sub_axes_b");
}

#[test]
fn test_prepare_graph_for_export_is_idempotent() {
    let mut graph = onnx::GraphProto {
        input: vec![test_value_info("x")],
        initializer: vec![
            test_i64_tensor("axes_a", &[0]),
            test_i64_tensor("axes_b", &[0]),
        ],
        node: vec![
            onnx::NodeProto {
                op_type: "Identity".to_string(),
                input: vec!["x".to_string()],
                output: vec!["id_tmp".to_string()],
                ..Default::default()
            },
            test_cast_node("id_tmp", "cast_tmp", dt::INT64),
            test_cast_node("cast_tmp", "y", dt::INT64),
            onnx::NodeProto {
                op_type: "Shape".to_string(),
                input: vec!["x".to_string()],
                output: vec!["dead_shape".to_string()],
                ..Default::default()
            },
        ],
        output: vec![test_value_info("y")],
        ..Default::default()
    };

    Converter::prepare_graph_for_export(&mut graph).unwrap();
    let once = graph.clone();
    Converter::prepare_graph_for_export(&mut graph).unwrap();

    assert_eq!(graph, once);
}

#[test]
fn test_prepare_graph_for_export_depth_guard_reports_clear_error() {
    let mut nested = onnx::GraphProto {
        output: vec![test_value_info("leaf")],
        node: vec![onnx::NodeProto {
            op_type: "Identity".to_string(),
            input: vec!["leaf".to_string()],
            output: vec!["leaf".to_string()],
            ..Default::default()
        }],
        ..Default::default()
    };
    for depth in 0..130 {
        nested = onnx::GraphProto {
            input: vec![test_value_info(&format!("x_{depth}"))],
            node: vec![onnx::NodeProto {
                op_type: "If".to_string(),
                input: vec![format!("x_{depth}")],
                output: vec![format!("y_{depth}")],
                attribute: vec![crate::helper::attr_graph("then_branch", nested)],
                ..Default::default()
            }],
            output: vec![test_value_info(&format!("y_{depth}"))],
            ..Default::default()
        };
    }

    let err = Converter::prepare_graph_for_export(&mut nested).unwrap_err();
    assert!(
        err.to_string()
            .contains("maximum ONNX subgraph cleanup depth")
    );
}

#[test]
fn test_set_value_with_tensor_range_update_builds_indices_from_range() {
    let mut converter = Converter::new();
    converter.state.tensor_shapes.insert(10, vec![4, 12]);
    converter.state.constants.insert(12, vec![7.0]);
    converter.state.constants.insert(13, vec![10.0]);
    converter.state.constants.insert(14, vec![1.0]);

    let op_json = json!({
        "#": "1.set_value_with_tensor_",
        "A": [
            { "AT": { "D": [{ "D": 1 }] }, "N": "axes" },
            { "AT": { "D": [] }, "N": "decrease_axes" },
            { "AT": { "D": [] }, "N": "none_axes" }
        ],
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 12 },
            { "%": 13 },
            { "%": 14 }
        ],
        "O": [
            {
                "%": 15,
                "TT": {
                    "D": [
                        { "#": "0.t_i32" },
                        [-1, 501]
                    ]
                }
            }
        ]
    });

    converter
        .process_pass2_op("1.set_value_with_tensor_", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    let node_types: Vec<&str> = graph
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect();
    assert_eq!(node_types.iter().filter(|&&ty| ty == "Cast").count(), 3);
    assert!(node_types.contains(&"Range"));
    assert!(node_types.contains(&"Shape"));
    assert!(node_types.contains(&"Reshape"));
    assert!(node_types.contains(&"Expand"));
    assert!(node_types.contains(&"ScatterElements"));
    let range = graph
        .node
        .iter()
        .find(|node| node.op_type == "Range")
        .unwrap();
    assert_eq!(range.output, vec!["set_value_tensor_range_15"]);
    let scatter = graph
        .node
        .iter()
        .find(|node| node.op_type == "ScatterElements")
        .unwrap();
    assert_eq!(
        scatter.input,
        vec!["tensor_10", "set_value_tensor_indices_15", "tensor_11"]
    );
}

#[test]
fn test_set_value_with_tensor_decrease_axes_single_index_uses_range_expand() {
    let mut converter = Converter::new();
    converter.state.tensor_shapes.insert(10, vec![4, 12]);
    converter.state.constants.insert(12, vec![2.0]);
    converter.state.constants.insert(13, vec![3.0]);
    converter.state.constants.insert(14, vec![1.0]);

    let op_json = json!({
        "#": "1.set_value_with_tensor_",
        "A": [
            { "AT": { "D": [{ "D": 1 }] }, "N": "axes" },
            { "AT": { "D": [{ "D": 1 }] }, "N": "decrease_axes" },
            { "AT": { "D": [] }, "N": "none_axes" }
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

    converter
        .process_pass2_op("1.set_value_with_tensor_", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    let node_types: Vec<&str> = graph
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect();
    assert_eq!(node_types.iter().filter(|&&ty| ty == "Cast").count(), 3);
    assert!(node_types.contains(&"Range"));
    assert_eq!(
        node_types.iter().filter(|&&ty| ty == "Unsqueeze").count(),
        1
    );
    assert!(node_types.contains(&"Reshape"));
    assert!(node_types.contains(&"Shape"));
    assert!(node_types.contains(&"Expand"));
    assert!(node_types.contains(&"ScatterElements"));
    let scatter = graph
        .node
        .iter()
        .find(|node| node.op_type == "ScatterElements")
        .unwrap();
    assert_eq!(
        scatter.input,
        vec![
            "tensor_10",
            "set_value_tensor_indices_15",
            "set_value_tensor_updates_15"
        ]
    );
}

#[test]
fn test_set_value_with_tensor_rank4_axis1_update_expands_indices() {
    let mut converter = Converter::new();
    converter.state.tensor_shapes.insert(10, vec![2, 5, 3, 4]);
    converter.state.tensor_shapes.insert(11, vec![2, 2, 3, 4]);
    converter.state.constants.insert(12, vec![1.0]);
    converter.state.constants.insert(13, vec![3.0]);
    converter.state.constants.insert(14, vec![1.0]);

    let op_json = json!({
        "#": "1.set_value_with_tensor_",
        "A": [
            { "AT": { "D": [{ "D": 1 }] }, "N": "axes" },
            { "AT": { "D": [] }, "N": "decrease_axes" },
            { "AT": { "D": [] }, "N": "none_axes" }
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

    converter
        .process_pass2_op("1.set_value_with_tensor_", &op_json)
        .unwrap();

    let graph = &converter.onnx_graph;
    let reshape = graph
        .node
        .iter()
        .find(|node| node.op_type == "Reshape")
        .unwrap();
    assert_eq!(
        reshape.input,
        vec![
            "set_value_tensor_range_15",
            "set_value_tensor_index_shape_15"
        ]
    );

    let index_shape = graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "set_value_tensor_index_shape_15")
        .unwrap();
    assert_eq!(index_shape.dims, vec![4]);
    let mut expected_raw = Vec::new();
    for value in [1_i64, -1, 1, 1] {
        expected_raw.extend_from_slice(&value.to_le_bytes());
    }
    assert_eq!(index_shape.raw_data, expected_raw);

    let expand = graph
        .node
        .iter()
        .find(|node| node.op_type == "Expand")
        .unwrap();
    assert_eq!(
        expand.input,
        vec![
            "set_value_tensor_range_shaped_15",
            "set_value_tensor_updates_shape_15"
        ]
    );

    let scatter = graph
        .node
        .iter()
        .find(|node| node.op_type == "ScatterElements")
        .unwrap();
    assert_eq!(scatter.input[1], "set_value_tensor_indices_15");
    assert_eq!(
        scatter
            .attribute
            .iter()
            .find(|attr| attr.name == "axis")
            .unwrap()
            .i,
        1
    );
}

#[test]
fn test_set_value_with_tensor_rejects_multi_index_update_when_decreasing_axis() {
    let mut converter = Converter::new();
    converter.state.tensor_shapes.insert(10, vec![4, 12]);
    converter.state.constants.insert(12, vec![2.0]);
    converter.state.constants.insert(13, vec![5.0]);
    converter.state.constants.insert(14, vec![1.0]);

    let op_json = json!({
        "#": "1.set_value_with_tensor_",
        "A": [
            { "AT": { "D": [{ "D": 1 }] }, "N": "axes" },
            { "AT": { "D": [{ "D": 1 }] }, "N": "decrease_axes" },
            { "AT": { "D": [] }, "N": "none_axes" }
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
        .process_pass2_op("1.set_value_with_tensor_", &op_json)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("decrease_axes=[1] requires a single indexed position")
    );
}

#[test]
fn test_op_data_defaults_to_unique_names_without_attr() {
    let mut converter = Converter::new();

    let op1 = json!({
        "#": "1.data",
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
    let op2 = json!({
        "#": "1.data",
        "I": [],
        "O": [
            {
                "%": 11,
                "TT": {
                    "D": [
                        { "#": "0.t_f32" },
                        [1]
                    ]
                }
            }
        ]
    });

    converter.process_pass2_op("1.data", &op1).unwrap();
    converter.process_pass2_op("1.data", &op2).unwrap();

    let input_names = converter
        .onnx_graph
        .input
        .iter()
        .map(|vi| vi.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(input_names, vec!["input_10", "input_11"]);
}

#[test]
fn test_validate_catches_undefined_tensor_in_subgraph() {
    let mut converter = Converter::new();

    // Build a graph with a subgraph that references an undefined tensor
    converter.onnx_graph.input.push(onnx::ValueInfoProto {
        name: "x".to_string(),
        ..Default::default()
    });

    let mut if_node = onnx::NodeProto {
        op_type: "If".to_string(),
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        ..Default::default()
    };

    let then_graph = onnx::GraphProto {
        name: "then".to_string(),
        node: vec![onnx::NodeProto {
            op_type: "Identity".to_string(),
            input: vec!["nonexistent_tensor".to_string()],
            output: vec!["then_out".to_string()],
            ..Default::default()
        }],
        output: vec![onnx::ValueInfoProto {
            name: "then_out".to_string(),
            ..Default::default()
        }],
        ..Default::default()
    };

    if_node.attribute.push(onnx::AttributeProto {
        name: "then_branch".to_string(),
        g: Some(then_graph),
        ..Default::default()
    });

    converter.onnx_graph.node.push(if_node);
    converter.onnx_graph.output.push(onnx::ValueInfoProto {
        name: "y".to_string(),
        ..Default::default()
    });

    let result = converter.validate();
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("nonexistent_tensor")
    );
}

#[test]
fn test_validate_catches_duplicate_node_outputs() {
    let mut converter = Converter::new();
    converter.onnx_graph.input.push(onnx::ValueInfoProto {
        name: "x".to_string(),
        ..Default::default()
    });
    converter.onnx_graph.node.push(onnx::NodeProto {
        op_type: "Identity".to_string(),
        input: vec!["x".to_string()],
        output: vec!["dup".to_string()],
        ..Default::default()
    });
    converter.onnx_graph.node.push(onnx::NodeProto {
        op_type: "Identity".to_string(),
        input: vec!["x".to_string()],
        output: vec!["dup".to_string()],
        ..Default::default()
    });

    let err = converter.validate().unwrap_err();
    assert!(err.to_string().contains("produced by more than one node"));
}

#[test]
fn test_validate_allows_subgraph_to_capture_outer_tensor() {
    let mut converter = Converter::new();

    converter.onnx_graph.input.push(onnx::ValueInfoProto {
        name: "x".to_string(),
        ..Default::default()
    });

    // outer_val is produced by a node in the outer graph
    converter.onnx_graph.node.push(onnx::NodeProto {
        op_type: "Identity".to_string(),
        input: vec!["x".to_string()],
        output: vec!["outer_val".to_string()],
        ..Default::default()
    });

    let then_graph = onnx::GraphProto {
        name: "then".to_string(),
        node: vec![onnx::NodeProto {
            op_type: "Identity".to_string(),
            input: vec!["outer_val".to_string()],
            output: vec!["then_out".to_string()],
            ..Default::default()
        }],
        output: vec![onnx::ValueInfoProto {
            name: "then_out".to_string(),
            ..Default::default()
        }],
        ..Default::default()
    };

    let mut if_node = onnx::NodeProto {
        op_type: "If".to_string(),
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        ..Default::default()
    };
    if_node.attribute.push(onnx::AttributeProto {
        name: "then_branch".to_string(),
        g: Some(then_graph),
        ..Default::default()
    });

    converter.onnx_graph.node.push(if_node);
    converter.onnx_graph.output.push(onnx::ValueInfoProto {
        name: "y".to_string(),
        ..Default::default()
    });

    // Should pass: subgraph captures "outer_val" from outer scope
    converter.validate().unwrap();
}

#[test]
fn test_validate_allows_subgraph_to_use_outer_initializer() {
    let mut converter = Converter::new();

    converter.onnx_graph.input.push(onnx::ValueInfoProto {
        name: "x".to_string(),
        ..Default::default()
    });

    // Add an initializer in the outer graph
    converter.onnx_graph.initializer.push(onnx::TensorProto {
        name: "weight".to_string(),
        dims: vec![1],
        data_type: dt::FLOAT,
        ..Default::default()
    });

    let then_graph = onnx::GraphProto {
        name: "then".to_string(),
        node: vec![onnx::NodeProto {
            op_type: "Add".to_string(),
            input: vec!["x".to_string(), "weight".to_string()],
            output: vec!["then_out".to_string()],
            ..Default::default()
        }],
        output: vec![onnx::ValueInfoProto {
            name: "then_out".to_string(),
            ..Default::default()
        }],
        ..Default::default()
    };

    let mut if_node = onnx::NodeProto {
        op_type: "If".to_string(),
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        ..Default::default()
    };
    if_node.attribute.push(onnx::AttributeProto {
        name: "then_branch".to_string(),
        g: Some(then_graph),
        ..Default::default()
    });

    converter.onnx_graph.node.push(if_node);
    converter.onnx_graph.output.push(onnx::ValueInfoProto {
        name: "y".to_string(),
        ..Default::default()
    });

    converter.validate().unwrap();
}

#[test]
fn test_cast_constant_folding_rejects_non_integer_to_int() {
    let mut converter = Converter::new();

    // Simulate pass1 with a float constant followed by a cast to int32
    converter.state.constants.insert(10, vec![3.7, -1.2, 0.0]);

    let ops = vec![json!({
        "#": "1.cast",
        "A": [
            { "AT": { "D": "int32" }, "N": "dtype" }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    })];
    let err = converter.collect_pass1_from_ops(&ops).unwrap_err();
    assert!(err.to_string().contains("without losing precision"));
}

#[test]
fn test_cast_constant_folding_rejects_integer_overflow() {
    let mut converter = Converter::new();
    converter.state.constants.insert(10, vec![128.0]);

    let ops = vec![json!({
        "#": "1.cast",
        "A": [
            { "AT": { "D": "int8" }, "N": "dtype" }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    })];

    let err = converter.collect_pass1_from_ops(&ops).unwrap_err();
    assert!(err.to_string().contains("out of range for int8"));
}

#[test]
fn test_cast_constant_folding_rejects_non_finite_to_int() {
    let mut converter = Converter::new();
    converter.state.constants.insert(10, vec![f64::INFINITY]);

    let ops = vec![json!({
        "#": "1.cast",
        "A": [
            { "AT": { "D": "int16" }, "N": "dtype" }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    })];

    let err = converter.collect_pass1_from_ops(&ops).unwrap_err();
    assert!(err.to_string().contains("cannot cast non-finite constant"));
}

#[test]
fn test_cast_constant_folding_rejects_i64_overflow() {
    let mut converter = Converter::new();
    converter
        .state
        .constants
        .insert(10, vec![i64::MAX as f64 * 2.0]);

    let ops = vec![json!({
        "#": "1.cast",
        "A": [
            { "AT": { "D": "int64" }, "N": "dtype" }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    })];

    let err = converter.collect_pass1_from_ops(&ops).unwrap_err();
    assert!(err.to_string().contains("out of range for int64"));
}

#[test]
fn test_cast_constant_folding_allows_exact_integer_to_int() {
    let mut converter = Converter::new();
    converter.state.constants.insert(10, vec![3.0, -1.0, 0.0]);

    let ops = vec![json!({
        "#": "1.cast",
        "A": [
            { "AT": { "D": "int32" }, "N": "dtype" }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    })];
    converter.collect_pass1_from_ops(&ops).unwrap();

    let folded = converter.state.constants.get(&11).unwrap();
    assert_eq!(folded, &[3.0, -1.0, 0.0]);
}

#[test]
fn test_constant_folding_rejects_consumer_before_constant_producer() {
    let mut converter = Converter::new();

    let ops = vec![
        json!({
            "#": "1.cast",
            "A": [
                { "AT": { "D": "int32" }, "N": "dtype" }
            ],
            "I": [
                { "%": 10 }
            ],
            "O": [
                { "%": 11 }
            ]
        }),
        json!({
            "#": "1.full",
            "A": [
                { "AT": { "D": 3.0 }, "N": "value" }
            ],
            "O": [
                { "%": 10 }
            ]
        }),
    ];

    let err = converter.collect_pass1_from_ops(&ops).unwrap_err();
    assert!(err.to_string().contains("before its constant producer"));
}

#[test]
fn test_pass1_sub_block_constants_do_not_leak_to_outer_scope() {
    let mut converter = Converter::new();

    let ops = vec![
        json!({
            "#": "1.full",
            "A": [
                { "AT": { "D": 7.0 }, "N": "value" }
            ],
            "O": [
                { "%": 10 }
            ]
        }),
        json!({
            "#": "1.if",
            "regions": [
                {
                    "blocks": [
                        {
                            "args": [],
                            "ops": [
                                {
                                    "#": "1.cast",
                                    "A": [
                                        { "AT": { "D": "int32" }, "N": "dtype" }
                                    ],
                                    "I": [
                                        { "%": 10 }
                                    ],
                                    "O": [
                                        { "%": 99 }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ],
            "O": [
                { "%": 11 }
            ]
        }),
    ];

    converter.collect_pass1_from_ops(&ops).unwrap();

    assert_eq!(converter.state.constants.get(&10).unwrap(), &[7.0]);
    assert!(!converter.state.constants.contains_key(&99));
}

#[test]
fn test_cast_constant_folding_converts_float_to_bool() {
    let mut converter = Converter::new();

    converter.state.constants.insert(10, vec![0.0, 1.5, -0.1]);

    let ops = vec![json!({
        "#": "1.cast",
        "A": [
            { "AT": { "D": "bool" }, "N": "dtype" }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    })];
    converter.collect_pass1_from_ops(&ops).unwrap();

    let folded = converter.state.constants.get(&11).unwrap();
    assert_eq!(folded, &[0.0, 1.0, 1.0]);
}

#[test]
fn test_cast_constant_folding_without_dtype_copies_values() {
    let mut converter = Converter::new();

    converter.state.constants.insert(10, vec![3.7, -1.2]);

    let ops = vec![json!({
        "#": "1.cast",
        "A": [],
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11 }
        ]
    })];
    converter.collect_pass1_from_ops(&ops).unwrap();

    let folded = converter.state.constants.get(&11).unwrap();
    assert_eq!(folded, &[3.7, -1.2]);
}

#[test]
fn test_squeeze_propagates_constants_unchanged() {
    let mut converter = Converter::new();
    converter.state.constants.insert(10, vec![1.0, 2.0, 3.0]);

    let ops = vec![json!({
        "#": "1.squeeze",
        "A": [],
        "I": [{ "%": 10 }],
        "O": [{ "%": 11 }]
    })];
    converter.collect_pass1_from_ops(&ops).unwrap();

    let folded = converter.state.constants.get(&11).unwrap();
    assert_eq!(folded, &[1.0, 2.0, 3.0]);
}

#[test]
fn test_reshape_propagates_constants_unchanged() {
    let mut converter = Converter::new();
    converter
        .state
        .constants
        .insert(10, vec![1.0, 2.0, 3.0, 4.0]);

    let ops = vec![json!({
        "#": "1.reshape",
        "A": [],
        "I": [{ "%": 10 }, { "%": 20 }],
        "O": [{ "%": 11 }]
    })];
    converter.collect_pass1_from_ops(&ops).unwrap();

    let folded = converter.state.constants.get(&11).unwrap();
    assert_eq!(folded, &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_scale_constant_folding_bias_after_scale() {
    let mut converter = Converter::new();
    converter.state.constants.insert(10, vec![2.0, 4.0]);

    let ops = vec![json!({
        "#": "1.scale",
        "A": [
            { "AT": { "#": "0.a_f32", "D": 3.0 }, "N": "scale" },
            { "AT": { "#": "0.a_f32", "D": 1.0 }, "N": "bias" },
            { "AT": { "#": "0.a_bool", "D": true }, "N": "bias_after_scale" }
        ],
        "I": [{ "%": 10 }],
        "O": [{ "%": 11 }]
    })];
    converter.collect_pass1_from_ops(&ops).unwrap();

    // bias_after_scale=true: output = x * scale + bias = [2*3+1, 4*3+1] = [7, 13]
    let folded = converter.state.constants.get(&11).unwrap();
    assert_eq!(folded, &[7.0, 13.0]);
}

#[test]
fn test_scale_constant_folding_bias_before_scale() {
    let mut converter = Converter::new();
    converter.state.constants.insert(10, vec![2.0, 4.0]);

    let ops = vec![json!({
        "#": "1.scale",
        "A": [
            { "AT": { "#": "0.a_f32", "D": 3.0 }, "N": "scale" },
            { "AT": { "#": "0.a_f32", "D": 1.0 }, "N": "bias" },
            { "AT": { "#": "0.a_bool", "D": false }, "N": "bias_after_scale" }
        ],
        "I": [{ "%": 10 }],
        "O": [{ "%": 11 }]
    })];
    converter.collect_pass1_from_ops(&ops).unwrap();

    // bias_after_scale=false: output = (x + bias) * scale = [(2+1)*3, (4+1)*3] = [9, 15]
    let folded = converter.state.constants.get(&11).unwrap();
    assert_eq!(folded, &[9.0, 15.0]);
}

#[test]
fn test_scale_constant_folding_skips_when_scale_from_tensor_input() {
    let mut converter = Converter::new();
    converter.state.constants.insert(10, vec![2.0]);

    // When scale comes from a 2nd input tensor (not attr), don't fold
    let ops = vec![json!({
        "#": "1.scale",
        "A": [],
        "I": [{ "%": 10 }, { "%": 20 }],
        "O": [{ "%": 11 }]
    })];
    converter.collect_pass1_from_ops(&ops).unwrap();

    // Should NOT fold because inputs.len() > 1 (scale from tensor)
    assert!(!converter.state.constants.contains_key(&11));
}

#[test]
fn test_full_int_array_reads_dtype_attr() {
    let mut converter = Converter::new();
    converter.state.constants.insert(10, vec![1.0, 2.0, 3.0]);

    let op_json = json!({
        "#": "1.full_int_array",
        "A": [
            { "AT": { "#": "0.a_str", "D": "int32" }, "N": "dtype" }
        ],
        "I": [],
        "O": [{
            "%": 10,
            "TT": { "D": [{ "#": "0.t_i32" }, [3]] }
        }]
    });

    converter
        .process_pass2_op("1.full_int_array", &op_json)
        .unwrap();

    let init = &converter.onnx_graph.initializer[0];
    assert_eq!(init.data_type, dt::INT32);
}

#[test]
fn test_stack_uses_unique_unsqueeze_axes_initializers() {
    let mut converter = Converter::new();
    converter.state.combines.insert(20, vec![21, 22]);

    let op_json = json!({
        "#": "1.stack",
        "A": [
            { "AT": { "D": 0 }, "N": "axis" }
        ],
        "I": [
            { "%": 20 }
        ],
        "O": [
            { "%": 30 }
        ]
    });

    converter.process_pass2_op("1.stack", &op_json).unwrap();

    let axis_inits = converter
        .onnx_graph
        .initializer
        .iter()
        .filter(|tensor| tensor.name.starts_with("stack_axes_30_"))
        .map(|tensor| tensor.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(axis_inits, vec!["stack_axes_30_0", "stack_axes_30_1"]);
}
