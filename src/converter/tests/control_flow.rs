#![allow(unused_imports)]

use crate::converter::Converter;
use serde_json::json;

#[test]
fn test_if_builds_then_and_else_subgraphs() {
    let mut converter = Converter::new();
    converter.state.id_to_name.insert(10, "cond".to_string());
    converter
        .state
        .tensor_types
        .insert(10, crate::helper::paddle_tt::BOOL.to_string());
    converter.state.tensor_shapes.insert(10, vec![]);

    let op_json = json!({
        "#": "1.if",
        "I": [
            { "%": 10 }
        ],
        "O": [
            {
                "%": 11,
                "TT": {
                    "D": [
                        { "#": "0.t_bool" },
                        []
                    ]
                }
            }
        ],
        "regions": [
            {
                "blocks": [
                    {
                        "args": [],
                        "ops": [
                            {
                                "#": "1.full",
                                "A": [
                                    { "AT": { "D": [] }, "N": "shape" },
                                    { "AT": { "D": 1.0 }, "N": "value" },
                                    { "AT": { "D": "bool" }, "N": "dtype" },
                                    { "AT": { "D": [0, 0, ""] }, "N": "place" }
                                ],
                                "I": [],
                                "O": [
                                    {
                                        "%": 12,
                                        "TT": {
                                            "D": [
                                                { "#": "0.t_bool" },
                                                []
                                            ]
                                        }
                                    }
                                ]
                            },
                            {
                                "#": "2.yield",
                                "I": [
                                    { "%": 12 }
                                ],
                                "O": []
                            }
                        ]
                    }
                ]
            },
            {
                "blocks": [
                    {
                        "args": [],
                        "ops": [
                            {
                                "#": "2.yield",
                                "I": [
                                    { "%": 10 }
                                ],
                                "O": []
                            }
                        ]
                    }
                ]
            }
        ]
    });

    converter.process_pass2_op("1.if", &op_json).unwrap();

    let node = &converter.onnx_graph.node[0];
    assert_eq!(node.op_type, "If");
    assert_eq!(node.output, vec!["tensor_11"]);

    let then_branch = node
        .attribute
        .iter()
        .find(|attr| attr.name == "then_branch")
        .and_then(|attr| attr.g.as_ref())
        .unwrap();
    let else_branch = node
        .attribute
        .iter()
        .find(|attr| attr.name == "else_branch")
        .and_then(|attr| attr.g.as_ref())
        .unwrap();
    assert_eq!(then_branch.output.len(), 1);
    assert_eq!(else_branch.output.len(), 1);
}

#[test]
fn test_if_subgraph_collects_local_constants_before_pass2() {
    let mut converter = Converter::new();
    converter.state.id_to_name.insert(10, "cond".to_string());
    converter
        .state
        .tensor_types
        .insert(10, crate::helper::paddle_tt::BOOL.to_string());
    converter.state.tensor_shapes.insert(10, vec![]);

    let op_json = json!({
        "#": "1.if",
        "I": [
            { "%": 10 }
        ],
        "O": [
            {
                "%": 20,
                "TT": {
                    "D": [
                        { "#": "0.t_i64" },
                        [-1]
                    ]
                }
            }
        ],
        "regions": [
            {
                "blocks": [
                    {
                        "args": [],
                        "ops": [
                            {
                                "#": "1.full",
                                "A": [
                                    { "AT": { "D": [1] }, "N": "shape" },
                                    { "AT": { "D": 7.0 }, "N": "value" },
                                    { "AT": { "D": "int64" }, "N": "dtype" },
                                    { "AT": { "D": [0, 0, ""] }, "N": "place" }
                                ],
                                "I": [],
                                "O": [
                                    {
                                        "%": 11,
                                        "TT": {
                                            "D": [
                                                { "#": "0.t_i64" },
                                                [1]
                                            ]
                                        }
                                    }
                                ]
                            },
                            {
                                "#": "1.full",
                                "A": [
                                    { "AT": { "D": [] }, "N": "shape" },
                                    { "AT": { "D": 0.0 }, "N": "value" },
                                    { "AT": { "D": "int32" }, "N": "dtype" },
                                    { "AT": { "D": [0, 0, ""] }, "N": "place" }
                                ],
                                "I": [],
                                "O": [
                                    {
                                        "%": 12,
                                        "TT": {
                                            "D": [
                                                { "#": "0.t_i32" },
                                                []
                                            ]
                                        }
                                    }
                                ]
                            },
                            {
                                "#": "0.combine",
                                "I": [
                                    { "%": 11 },
                                    { "%": 11 }
                                ],
                                "O": [
                                    { "%": 13 }
                                ]
                            },
                            {
                                "#": "1.concat",
                                "A": [],
                                "I": [
                                    { "%": 13 },
                                    { "%": 12 }
                                ],
                                "O": [
                                    {
                                        "%": 14,
                                        "TT": {
                                            "D": [
                                                { "#": "0.t_i64" },
                                                [2]
                                            ]
                                        }
                                    }
                                ]
                            },
                            {
                                "#": "2.yield",
                                "I": [
                                    { "%": 14 }
                                ],
                                "O": []
                            }
                        ]
                    }
                ]
            },
            {
                "blocks": [
                    {
                        "args": [],
                        "ops": [
                            {
                                "#": "2.yield",
                                "I": [
                                    { "%": 11 }
                                ],
                                "O": []
                            }
                        ]
                    }
                ]
            }
        ]
    });

    converter.process_pass2_op("1.if", &op_json).unwrap();

    let if_node = &converter.onnx_graph.node[0];
    let then_branch = if_node
        .attribute
        .iter()
        .find(|attr| attr.name == "then_branch")
        .and_then(|attr| attr.g.as_ref())
        .unwrap();
    let concat = then_branch
        .node
        .iter()
        .find(|node| node.op_type == "Concat")
        .unwrap();
    assert_eq!(concat.output, vec!["tensor_14"]);
    assert_eq!(concat.attribute[0].name, "axis");
    assert_eq!(concat.attribute[0].i, 0);
}

#[test]
fn test_while_builds_loop_body_subgraph() {
    let mut converter = Converter::new();
    converter.state.id_to_name.insert(10, "cond".to_string());
    converter.state.id_to_name.insert(11, "state".to_string());
    converter
        .state
        .tensor_types
        .insert(10, crate::helper::paddle_tt::BOOL.to_string());
    converter.state.tensor_shapes.insert(10, vec![]);
    converter
        .state
        .tensor_types
        .insert(11, crate::helper::paddle_tt::I64.to_string());
    converter.state.tensor_shapes.insert(11, vec![]);

    let op_json = json!({
        "#": "1.while",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            {
                "%": 12,
                "TT": {
                    "D": [
                        { "#": "0.t_i64" },
                        []
                    ]
                }
            }
        ],
        "regions": [
            {
                "blocks": [
                    {
                        "args": [
                            {
                                "#": -1,
                                "TT": {
                                    "D": [
                                        { "#": "0.t_i64" },
                                        []
                                    ]
                                }
                            }
                        ],
                        "ops": [
                            {
                                "#": "2.yield",
                                "I": [
                                    { "%": 10 },
                                    { "%": -1 }
                                ],
                                "O": []
                            }
                        ]
                    }
                ]
            }
        ]
    });

    converter.process_pass2_op("1.while", &op_json).unwrap();

    let node = &converter.onnx_graph.node[0];
    assert_eq!(node.op_type, "Loop");
    assert_eq!(node.input[0], "");
    assert_eq!(node.input[1], "cond");
    assert_eq!(node.output, vec!["tensor_12"]);

    let body = node
        .attribute
        .iter()
        .find(|attr| attr.name == "body")
        .and_then(|attr| attr.g.as_ref())
        .unwrap();
    assert_eq!(body.input.len(), 3);
    assert_eq!(body.output.len(), 2);
}

#[test]
fn test_while_keeps_duplicate_loop_carried_inputs_distinct() {
    let mut converter = Converter::new();
    converter.state.id_to_name.insert(10, "cond".to_string());
    converter.state.id_to_name.insert(11, "state".to_string());
    converter.state.id_to_name.insert(12, "other".to_string());
    converter
        .state
        .tensor_types
        .insert(10, crate::helper::paddle_tt::BOOL.to_string());
    converter.state.tensor_shapes.insert(10, vec![]);
    converter
        .state
        .tensor_types
        .insert(11, crate::helper::paddle_tt::I64.to_string());
    converter.state.tensor_shapes.insert(11, vec![]);
    converter
        .state
        .tensor_types
        .insert(12, crate::helper::paddle_tt::I64.to_string());
    converter.state.tensor_shapes.insert(12, vec![]);

    let op_json = json!({
        "#": "1.while",
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 11 },
            { "%": 12 }
        ],
        "O": [
            {
                "%": 13,
                "TT": {
                    "D": [
                        { "#": "0.t_i64" },
                        []
                    ]
                }
            },
            {
                "%": 14,
                "TT": {
                    "D": [
                        { "#": "0.t_i64" },
                        []
                    ]
                }
            },
            {
                "%": 15,
                "TT": {
                    "D": [
                        { "#": "0.t_i64" },
                        []
                    ]
                }
            }
        ],
        "regions": [
            {
                "blocks": [
                    {
                        "args": [
                            {
                                "#": -1,
                                "TT": {
                                    "D": [
                                        { "#": "0.t_i64" },
                                        []
                                    ]
                                }
                            },
                            {
                                "#": -2,
                                "TT": {
                                    "D": [
                                        { "#": "0.t_i64" },
                                        []
                                    ]
                                }
                            },
                            {
                                "#": -3,
                                "TT": {
                                    "D": [
                                        { "#": "0.t_i64" },
                                        []
                                    ]
                                }
                            }
                        ],
                        "ops": [
                            {
                                "#": "2.yield",
                                "I": [
                                    { "%": 10 },
                                    { "%": -1 },
                                    { "%": -2 },
                                    { "%": -3 }
                                ],
                                "O": []
                            }
                        ]
                    }
                ]
            }
        ]
    });

    converter.process_pass2_op("1.while", &op_json).unwrap();

    let node = &converter.onnx_graph.node[0];
    let body = node
        .attribute
        .iter()
        .find(|attr| attr.name == "body")
        .and_then(|attr| attr.g.as_ref())
        .unwrap();
    assert_eq!(node.input.len(), 5);
    assert_eq!(body.input.len(), 5);
    let body_input_names = body
        .input
        .iter()
        .map(|vi| vi.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        body_input_names,
        vec![
            "__p2o_loop_iter",
            "__p2o_loop_cond",
            "while_arg_0_11",
            "while_arg_1_11",
            "while_arg_2_12"
        ]
    );
}

#[test]
fn test_while_captures_outer_scope_variable() {
    let mut converter = Converter::new();
    converter.state.id_to_name.insert(10, "cond".to_string());
    converter.state.id_to_name.insert(11, "state".to_string());
    converter
        .state
        .id_to_name
        .insert(20, "outer_val".to_string());
    converter
        .state
        .tensor_types
        .insert(10, crate::helper::paddle_tt::BOOL.to_string());
    converter.state.tensor_shapes.insert(10, vec![]);
    converter
        .state
        .tensor_types
        .insert(11, crate::helper::paddle_tt::I64.to_string());
    converter.state.tensor_shapes.insert(11, vec![]);
    converter
        .state
        .tensor_types
        .insert(20, crate::helper::paddle_tt::F32.to_string());
    converter.state.tensor_shapes.insert(20, vec![3]);

    let op_json = json!({
        "#": "1.while",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            {
                "%": 12,
                "TT": { "D": [{ "#": "0.t_i64" }, []] }
            }
        ],
        "regions": [
            {
                "blocks": [
                    {
                        "args": [
                            {
                                "#": -1,
                                "TT": { "D": [{ "#": "0.t_i64" }, []] }
                            }
                        ],
                        "ops": [
                            {
                                "#": "2.yield",
                                "I": [
                                    { "%": 10 },
                                    { "%": -1 }
                                ],
                                "O": []
                            }
                        ]
                    }
                ]
            }
        ]
    });

    converter.process_pass2_op("1.while", &op_json).unwrap();

    let node = &converter.onnx_graph.node[0];
    assert_eq!(node.op_type, "Loop");
    assert_eq!(node.input.len(), 3);
    assert_eq!(node.input[2], "state");

    let body = node
        .attribute
        .iter()
        .find(|attr| attr.name == "body")
        .and_then(|attr| attr.g.as_ref())
        .unwrap();
    assert_eq!(body.input.len(), 3);
    assert!(
        !body.input.iter().any(|vi| vi.name.contains("outer")),
        "outer_val should not appear as a body formal"
    );
}

#[test]
fn test_while_state_carry_maps_outputs_to_inputs() {
    let mut converter = Converter::new();
    converter.state.id_to_name.insert(10, "cond".to_string());
    converter.state.id_to_name.insert(11, "x".to_string());
    converter.state.id_to_name.insert(12, "y".to_string());
    converter
        .state
        .tensor_types
        .insert(10, crate::helper::paddle_tt::BOOL.to_string());
    converter.state.tensor_shapes.insert(10, vec![]);
    converter
        .state
        .tensor_types
        .insert(11, crate::helper::paddle_tt::F32.to_string());
    converter.state.tensor_shapes.insert(11, vec![2]);
    converter
        .state
        .tensor_types
        .insert(12, crate::helper::paddle_tt::F32.to_string());
    converter.state.tensor_shapes.insert(12, vec![2]);

    let op_json = json!({
        "#": "1.while",
        "I": [
            { "%": 10 },
            { "%": 11 },
            { "%": 12 }
        ],
        "O": [
            { "%": 20, "TT": { "D": [{ "#": "0.t_f32" }, [2]] } },
            { "%": 21, "TT": { "D": [{ "#": "0.t_f32" }, [2]] } }
        ],
        "regions": [
            {
                "blocks": [
                    {
                        "args": [
                            { "#": -1, "TT": { "D": [{ "#": "0.t_f32" }, [2]] } },
                            { "#": -2, "TT": { "D": [{ "#": "0.t_f32" }, [2]] } }
                        ],
                        "ops": [
                            {
                                "#": "2.yield",
                                "I": [
                                    { "%": 10 },
                                    { "%": -1 },
                                    { "%": -2 }
                                ],
                                "O": []
                            }
                        ]
                    }
                ]
            }
        ]
    });

    converter.process_pass2_op("1.while", &op_json).unwrap();

    let node = &converter.onnx_graph.node[0];
    assert_eq!(node.op_type, "Loop");
    assert_eq!(node.input.len(), 4);
    assert_eq!(node.input[2], "x");
    assert_eq!(node.input[3], "y");
    assert_eq!(node.output.len(), 2);

    let body = node
        .attribute
        .iter()
        .find(|attr| attr.name == "body")
        .and_then(|attr| attr.g.as_ref())
        .unwrap();
    assert_eq!(body.input.len(), 4);
    let body_input_names: Vec<&str> = body.input.iter().map(|vi| vi.name.as_str()).collect();
    assert_eq!(body_input_names[0], "__p2o_loop_iter");
    assert_eq!(body_input_names[1], "__p2o_loop_cond");
    assert_eq!(
        body.output.len(),
        3,
        "body outputs = 1 cond + N loop-carried"
    );
}

#[test]
fn test_nested_if_inside_while_keeps_inner_scope_local() {
    let mut converter = Converter::new();
    converter.state.id_to_name.insert(10, "cond".to_string());
    converter.state.id_to_name.insert(11, "state".to_string());
    converter
        .state
        .tensor_types
        .insert(10, crate::helper::paddle_tt::BOOL.to_string());
    converter.state.tensor_shapes.insert(10, vec![]);
    converter
        .state
        .tensor_types
        .insert(11, crate::helper::paddle_tt::I64.to_string());
    converter.state.tensor_shapes.insert(11, vec![]);

    let op_json = json!({
        "#": "1.while",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12, "TT": { "D": [{ "#": "0.t_i64" }, []] } }
        ],
        "regions": [
            {
                "blocks": [
                    {
                        "args": [
                            { "#": -1, "TT": { "D": [{ "#": "0.t_i64" }, []] } }
                        ],
                        "ops": [
                            {
                                "#": "1.if",
                                "I": [
                                    { "%": 10 }
                                ],
                                "O": [
                                    { "%": 20, "TT": { "D": [{ "#": "0.t_i64" }, []] } }
                                ],
                                "regions": [
                                    {
                                        "blocks": [
                                            {
                                                "args": [],
                                                "ops": [
                                                    {
                                                        "#": "1.full",
                                                        "A": [
                                                            { "AT": { "D": [] }, "N": "shape" },
                                                            { "AT": { "D": 5.0 }, "N": "value" },
                                                            { "AT": { "D": "int64" }, "N": "dtype" }
                                                        ],
                                                        "I": [],
                                                        "O": [
                                                            { "%": 21, "TT": { "D": [{ "#": "0.t_i64" }, []] } }
                                                        ]
                                                    },
                                                    {
                                                        "#": "2.yield",
                                                        "I": [
                                                            { "%": 21 }
                                                        ],
                                                        "O": []
                                                    }
                                                ]
                                            }
                                        ]
                                    },
                                    {
                                        "blocks": [
                                            {
                                                "args": [],
                                                "ops": [
                                                    {
                                                        "#": "2.yield",
                                                        "I": [
                                                            { "%": -1 }
                                                        ],
                                                        "O": []
                                                    }
                                                ]
                                            }
                                        ]
                                    }
                                ]
                            },
                            {
                                "#": "2.yield",
                                "I": [
                                    { "%": 10 },
                                    { "%": 20 }
                                ],
                                "O": []
                            }
                        ]
                    }
                ]
            }
        ]
    });

    converter.process_pass2_op("1.while", &op_json).unwrap();

    let loop_body = converter.onnx_graph.node[0]
        .attribute
        .iter()
        .find(|attr| attr.name == "body")
        .and_then(|attr| attr.g.as_ref())
        .unwrap();
    let if_node = loop_body
        .node
        .iter()
        .find(|node| node.op_type == "If")
        .unwrap();
    let then_branch = if_node
        .attribute
        .iter()
        .find(|attr| attr.name == "then_branch")
        .and_then(|attr| attr.g.as_ref())
        .unwrap();
    assert_eq!(then_branch.output.len(), 1);
    assert!(
        loop_body.input.iter().all(|vi| vi.name != "tensor_21"),
        "inner if locals must not leak into while body formals"
    );
}
