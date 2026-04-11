use crate::converter::Converter;
use serde_json::json;
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
    temp_file.write_all(synthetic_json.to_string().as_bytes()).unwrap();

    let mut converter = Converter::new();
    converter.load_paddle_model(temp_file.path().to_str().unwrap()).unwrap();

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
