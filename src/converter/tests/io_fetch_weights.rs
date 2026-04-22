#![allow(unused_imports)]

use crate::converter::{Converter, ParamMeta};
use crate::helper::dt;
use crate::proto::{PaddleDataType, TensorDesc, onnx};
use prost::Message;
use serde_json::json;
use std::fs;
use std::io::Seek;
use std::io::SeekFrom;

#[test]
fn test_fetch_without_name_uses_stable_output_alias() {
    let mut converter = Converter::new();

    let op_json = json!({
        "#": "1.fetch",
        "I": [
            {
                "%": 2,
                "TT": {
                    "D": [
                        { "#": "0.t_f32" },
                        [1, 8]
                    ]
                }
            }
        ],
        "O": []
    });

    converter.process_pass2_op("1.fetch", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.output.len(), 1);
    assert_eq!(graph.output[0].name, "fetch_0");
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Identity");
    assert_eq!(graph.node[0].input, vec!["tensor_2"]);
    assert_eq!(graph.node[0].output, vec!["fetch_0"]);
}

#[test]
fn test_fetch_renames_when_alias_collides_with_node_output() {
    let mut converter = Converter::new();
    converter.onnx_graph.node.push(onnx::NodeProto {
        op_type: "Identity".to_string(),
        output: vec!["fetch_0".to_string()],
        ..Default::default()
    });

    let op_json = json!({
        "#": "1.fetch",
        "I": [
            {
                "%": 2,
                "TT": {
                    "D": [
                        { "#": "0.t_f32" },
                        [1, 8]
                    ]
                }
            }
        ],
        "O": []
    });

    converter.process_pass2_op("1.fetch", &op_json).unwrap();

    let graph = &converter.onnx_graph;
    assert_eq!(graph.output[0].name, "fetch_0_1");
    assert_eq!(graph.node[1].output, vec!["fetch_0_1"]);
}

fn encode_weight_record(desc: TensorDesc, tensor_data: &[u8]) -> Vec<u8> {
    let mut desc_buf = Vec::new();
    desc.encode(&mut desc_buf).unwrap();

    let mut record = Vec::new();
    record.extend_from_slice(&0_u32.to_le_bytes());
    record.extend_from_slice(&0_u64.to_le_bytes());
    record.extend_from_slice(&0_u32.to_le_bytes());
    record.extend_from_slice(&(desc_buf.len() as i32).to_le_bytes());
    record.extend_from_slice(&desc_buf);
    record.extend_from_slice(tensor_data);
    record
}

#[test]
fn test_load_paddle_weights_rejects_truncated_tensor_data() {
    use crate::proto::paddle::framework::proto::var_type::var_type::Type as PdType;

    let desc = TensorDesc {
        data_type: PdType::Fp32 as i32,
        dims: vec![1],
    };
    let weight_bytes = encode_weight_record(desc, &[0x00, 0x00]);

    let temp_file = tempfile::NamedTempFile::new().unwrap();
    fs::write(temp_file.path(), &weight_bytes).unwrap();

    let mut converter = Converter::new();
    converter.state.param_names.push("weight_0".to_string());

    let err = converter
        .load_paddle_weights(temp_file.path().to_str().unwrap())
        .unwrap_err();
    assert!(err.to_string().contains("Truncated tensor data"));
}

#[test]
fn test_load_paddle_weights_rejects_missing_parameter_records() {
    use crate::proto::paddle::framework::proto::var_type::var_type::Type as PdType;

    let desc = TensorDesc {
        data_type: PdType::Fp32 as i32,
        dims: vec![1],
    };
    let weight_bytes = encode_weight_record(desc, &1.0_f32.to_le_bytes());

    let temp_file = tempfile::NamedTempFile::new().unwrap();
    fs::write(temp_file.path(), &weight_bytes).unwrap();

    let mut converter = Converter::new();
    converter.state.param_names = vec!["weight_0".to_string(), "weight_1".to_string()].into();

    let err = converter
        .load_paddle_weights(temp_file.path().to_str().unwrap())
        .unwrap_err();
    assert!(err.to_string().contains("Weight count mismatch"));
}

#[test]
fn test_load_paddle_weights_rejects_shape_mismatch_against_model_metadata() {
    use crate::proto::paddle::framework::proto::var_type::var_type::Type as PdType;

    let desc = TensorDesc {
        data_type: PdType::Fp32 as i32,
        dims: vec![2],
    };
    let weight_bytes =
        encode_weight_record(desc, &[0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40]);

    let temp_file = tempfile::NamedTempFile::new().unwrap();
    fs::write(temp_file.path(), &weight_bytes).unwrap();

    let mut converter = Converter::new();
    converter.state.param_names.push("weight_0".to_string());
    converter.state.param_meta.insert(
        "weight_0".to_string(),
        ParamMeta {
            onnx_dtype: Some(dt::FLOAT),
            dims: vec![1, 2],
        },
    );

    let err = converter
        .load_paddle_weights(temp_file.path().to_str().unwrap())
        .unwrap_err();
    assert!(err.to_string().contains("shape mismatch"));
}

#[test]
fn test_load_paddle_weights_rejects_dtype_mismatch_against_model_metadata() {
    use crate::proto::paddle::framework::proto::var_type::var_type::Type as PdType;

    let desc = TensorDesc {
        data_type: PdType::Fp32 as i32,
        dims: vec![1],
    };
    let weight_bytes = encode_weight_record(desc, &1.0_f32.to_le_bytes());

    let temp_file = tempfile::NamedTempFile::new().unwrap();
    fs::write(temp_file.path(), &weight_bytes).unwrap();

    let mut converter = Converter::new();
    converter.state.param_names.push("weight_0".to_string());
    converter.state.param_meta.insert(
        "weight_0".to_string(),
        ParamMeta {
            onnx_dtype: Some(dt::INT32),
            dims: vec![1],
        },
    );

    let err = converter
        .load_paddle_weights(temp_file.path().to_str().unwrap())
        .unwrap_err();
    assert!(err.to_string().contains("dtype mismatch"));
}

#[test]
fn test_load_paddle_weights_happy_path_for_mixed_dtypes() {
    use crate::proto::paddle::framework::proto::var_type::var_type::Type as PdType;

    let mut weight_bytes = Vec::new();
    weight_bytes.extend_from_slice(&encode_weight_record(
        TensorDesc {
            data_type: PdType::Fp32 as i32,
            dims: vec![2],
        },
        &[0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x20, 0x40],
    ));
    weight_bytes.extend_from_slice(&encode_weight_record(
        TensorDesc {
            data_type: PdType::Int64 as i32,
            dims: vec![1],
        },
        &7_i64.to_le_bytes(),
    ));

    let temp_file = tempfile::NamedTempFile::new().unwrap();
    fs::write(temp_file.path(), &weight_bytes).unwrap();

    let mut converter = Converter::new();
    converter.state.param_names = vec!["weight_f32".to_string(), "weight_i64".to_string()].into();
    converter.state.param_meta.insert(
        "weight_f32".to_string(),
        ParamMeta {
            onnx_dtype: Some(dt::FLOAT),
            dims: vec![2],
        },
    );
    converter.state.param_meta.insert(
        "weight_i64".to_string(),
        ParamMeta {
            onnx_dtype: Some(dt::INT64),
            dims: vec![1],
        },
    );

    converter
        .load_paddle_weights(temp_file.path().to_str().unwrap())
        .unwrap();

    assert_eq!(converter.onnx_graph.initializer.len(), 2);
    assert_eq!(converter.onnx_graph.initializer[0].name, "weight_f32");
    assert_eq!(converter.onnx_graph.initializer[0].data_type, dt::FLOAT);
    assert_eq!(converter.onnx_graph.initializer[0].dims, vec![2]);
    assert_eq!(converter.onnx_graph.initializer[1].name, "weight_i64");
    assert_eq!(converter.onnx_graph.initializer[1].data_type, dt::INT64);
    assert_eq!(converter.onnx_graph.initializer[1].dims, vec![1]);
}

#[test]
fn test_fetch_name_cache_rebuilds_after_graph_growth() {
    let mut converter = Converter::new();

    let first_fetch = json!({
        "#": "1.fetch",
        "I": [
            {
                "%": 2,
                "TT": { "D": [{ "#": "0.t_f32" }, [1]] }
            }
        ],
        "O": []
    });
    converter.process_pass2_op("1.fetch", &first_fetch).unwrap();

    converter.onnx_graph.node.push(onnx::NodeProto {
        op_type: "Identity".to_string(),
        output: vec!["fetch_1".to_string()],
        ..Default::default()
    });

    let second_fetch = json!({
        "#": "1.fetch",
        "I": [
            {
                "%": 3,
                "TT": { "D": [{ "#": "0.t_f32" }, [1]] }
            }
        ],
        "O": []
    });
    converter
        .process_pass2_op("1.fetch", &second_fetch)
        .unwrap();

    assert_eq!(converter.onnx_graph.output[1].name, "fetch_1_1");
}

#[test]
fn test_load_paddle_weights_rejects_file_over_soft_limit() {
    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    temp_file
        .as_file_mut()
        .set_len((8_u64 * 1024 * 1024 * 1024) + 1)
        .unwrap();
    temp_file.as_file_mut().seek(SeekFrom::Start(0)).unwrap();

    let mut converter = Converter::new();
    let err = converter
        .load_paddle_weights(temp_file.path().to_str().unwrap())
        .unwrap_err();
    assert!(err.to_string().contains(".pdiparams file is too large"));
}
