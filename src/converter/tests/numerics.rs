use crate::converter::Converter;
use serde_json::json;
use std::process::Command;

struct IoSpec<'a> {
    id: i64,
    elem_type: &'a str,
    dims: &'a [i64],
}

fn python_onnxruntime_available() -> bool {
    Command::new("python3")
        .arg("-c")
        .arg("import numpy, onnxruntime")
        .status()
        .is_ok_and(|status| status.success())
}

fn export_single_output_model(
    converter: &mut Converter,
    inputs: &[IoSpec<'_>],
    output: IoSpec<'_>,
    output_path: &str,
) -> anyhow::Result<()> {
    for input in inputs {
        converter
            .onnx_graph
            .input
            .push(converter.build_value_info_from_meta(
                converter.get_tensor_name(input.id)?,
                input.elem_type,
                input.dims,
            )?);
    }
    converter
        .onnx_graph
        .output
        .push(converter.build_value_info_from_meta(
            converter.get_tensor_name(output.id)?,
            output.elem_type,
            output.dims,
        )?);
    converter.export_onnx(output_path, converter.target_opset())
}

#[test]
#[ignore]
fn test_gelu_numeric_smoke_with_onnxruntime() {
    if !python_onnxruntime_available() {
        return;
    }

    let mut converter = Converter::new();
    converter
        .state
        .tensor_types
        .insert(10, "0.t_f32".to_string());
    converter.state.tensor_shapes.insert(10, vec![3]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_f32".to_string());
    converter.state.tensor_shapes.insert(11, vec![3]);

    let op_json = json!({
        "#": "1.gelu",
        "A": [
            { "AT": { "D": true }, "N": "approximate" }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11, "TT": { "D": [{ "#": "0.t_f32" }, [3]] } }
        ]
    });
    converter.process_pass2_op("1.gelu", &op_json).unwrap();

    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path().join("gelu.onnx");
    export_single_output_model(
        &mut converter,
        &[IoSpec {
            id: 10,
            elem_type: "0.t_f32",
            dims: &[3],
        }],
        IoSpec {
            id: 11,
            elem_type: "0.t_f32",
            dims: &[3],
        },
        model_path.to_str().unwrap(),
    )
    .unwrap();

    let script = r#"
import json, math, sys
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
got = sess.run(None, {"tensor_10": x})[0]
expected = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
if not np.allclose(got, expected, atol=1e-5):
    raise SystemExit(f"gelu mismatch: {got} vs {expected}")
"#;
    let status = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(model_path)
        .status()
        .unwrap();
    assert!(status.success());
}

#[test]
#[ignore]
fn test_one_hot_numeric_smoke_with_onnxruntime() {
    if !python_onnxruntime_available() {
        return;
    }

    let mut converter = Converter::new();
    converter
        .state
        .tensor_types
        .insert(10, "0.t_i64".to_string());
    converter.state.tensor_shapes.insert(10, vec![3]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_i64".to_string());
    converter.state.tensor_shapes.insert(11, vec![]);
    converter
        .state
        .tensor_types
        .insert(12, "0.t_f32".to_string());
    converter.state.tensor_shapes.insert(12, vec![3, 4]);
    converter.state.constants.insert(11, vec![4.0]);

    let op_json = json!({
        "#": "1.one_hot",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12, "TT": { "D": [{ "#": "0.t_f32" }, [3, 4]] } }
        ]
    });
    converter.process_pass2_op("1.one_hot", &op_json).unwrap();

    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path().join("one_hot.onnx");
    export_single_output_model(
        &mut converter,
        &[IoSpec {
            id: 10,
            elem_type: "0.t_i64",
            dims: &[3],
        }],
        IoSpec {
            id: 12,
            elem_type: "0.t_f32",
            dims: &[3, 4],
        },
        model_path.to_str().unwrap(),
    )
    .unwrap();

    let script = r#"
import sys
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
x = np.array([0, 2, 1], dtype=np.int64)
got = sess.run(None, {"tensor_10": x})[0]
expected = np.eye(4, dtype=np.float32)[x]
if not np.array_equal(got, expected):
    raise SystemExit(f"one_hot mismatch: {got} vs {expected}")
"#;
    let status = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(model_path)
        .status()
        .unwrap();
    assert!(status.success());
}

#[test]
#[ignore]
fn test_one_hot_rank2_numeric_smoke_with_onnxruntime() {
    if !python_onnxruntime_available() {
        return;
    }

    let mut converter = Converter::new();
    converter
        .state
        .tensor_types
        .insert(10, "0.t_i64".to_string());
    converter.state.tensor_shapes.insert(10, vec![2, 3]);
    converter
        .state
        .tensor_types
        .insert(11, "0.t_i64".to_string());
    converter.state.tensor_shapes.insert(11, vec![]);
    converter
        .state
        .tensor_types
        .insert(12, "0.t_f32".to_string());
    converter.state.tensor_shapes.insert(12, vec![2, 3, 4]);
    converter.state.constants.insert(11, vec![4.0]);

    let op_json = json!({
        "#": "1.one_hot",
        "I": [
            { "%": 10 },
            { "%": 11 }
        ],
        "O": [
            { "%": 12, "TT": { "D": [{ "#": "0.t_f32" }, [2, 3, 4]] } }
        ]
    });
    converter.process_pass2_op("1.one_hot", &op_json).unwrap();

    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path().join("one_hot_rank2.onnx");
    export_single_output_model(
        &mut converter,
        &[IoSpec {
            id: 10,
            elem_type: "0.t_i64",
            dims: &[2, 3],
        }],
        IoSpec {
            id: 12,
            elem_type: "0.t_f32",
            dims: &[2, 3, 4],
        },
        model_path.to_str().unwrap(),
    )
    .unwrap();

    let script = r#"
import sys
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
x = np.array([[0, 2, 1], [3, 1, 0]], dtype=np.int64)
got = sess.run(None, {"tensor_10": x})[0]
expected = np.eye(4, dtype=np.float32)[x]
if not np.array_equal(got, expected):
    raise SystemExit(f"one_hot rank2 mismatch: {got} vs {expected}")
"#;
    let status = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(model_path)
        .status()
        .unwrap();
    assert!(status.success());
}

#[test]
#[ignore]
fn test_argsort_integer_tie_break_numeric_smoke_with_onnxruntime() {
    if !python_onnxruntime_available() {
        return;
    }

    let mut converter = Converter::new();
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
        "#": "1.argsort",
        "A": [
            { "AT": { "D": -1 }, "N": "axis" },
            { "AT": { "D": false }, "N": "descending" },
            { "AT": { "D": false }, "N": "stable" }
        ],
        "I": [
            { "%": 10 }
        ],
        "O": [
            { "%": 11, "TT": { "D": [{ "#": "0.t_i64" }, [4]] } },
            { "%": 12, "TT": { "D": [{ "#": "0.t_i64" }, [4]] } }
        ]
    });
    converter.process_pass2_op("1.argsort", &op_json).unwrap();
    converter.onnx_graph.input.push(
        converter
            .build_value_info_from_meta("tensor_10".to_string(), "0.t_i64", &[4])
            .unwrap(),
    );
    converter.onnx_graph.output.push(
        converter
            .build_value_info_from_meta("tensor_11".to_string(), "0.t_i64", &[4])
            .unwrap(),
    );
    converter.onnx_graph.output.push(
        converter
            .build_value_info_from_meta("tensor_12".to_string(), "0.t_i64", &[4])
            .unwrap(),
    );

    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path().join("argsort.onnx");
    converter
        .export_onnx(model_path.to_str().unwrap(), converter.target_opset())
        .unwrap();

    let script = r#"
import sys
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
x = np.array([5, 5, 3, 5], dtype=np.int64)
values, indices = sess.run(None, {"tensor_10": x})
expected_indices = np.array([2, 0, 1, 3], dtype=np.int64)
expected_values = x[expected_indices]
if not np.array_equal(indices, expected_indices):
    raise SystemExit(f"argsort indices mismatch: {indices} vs {expected_indices}")
if not np.array_equal(values, expected_values):
    raise SystemExit(f"argsort values mismatch: {values} vs {expected_values}")
"#;
    let status = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(model_path)
        .status()
        .unwrap();
    assert!(status.success());
}

#[test]
#[ignore]
fn test_bitwise_and_int8_numeric_smoke_with_onnxruntime() {
    if !python_onnxruntime_available() {
        return;
    }

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
            { "%": 12, "TT": { "D": [{ "#": "0.t_i8" }, [4]] } }
        ]
    });
    converter
        .process_pass2_op("1.bitwise_and", &op_json)
        .unwrap();

    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path().join("bitwise_and_int8_opset17.onnx");
    export_single_output_model(
        &mut converter,
        &[
            IoSpec {
                id: 10,
                elem_type: "0.t_i8",
                dims: &[4],
            },
            IoSpec {
                id: 11,
                elem_type: "0.t_i8",
                dims: &[4],
            },
        ],
        IoSpec {
            id: 12,
            elem_type: "0.t_i8",
            dims: &[4],
        },
        model_path.to_str().unwrap(),
    )
    .unwrap();

    let script = r#"
import sys
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
lhs = np.array([7, 6, -3, -1], dtype=np.int8)
rhs = np.array([3, 5, 2, -8], dtype=np.int8)
got = sess.run(None, {"tensor_10": lhs, "tensor_11": rhs})[0]
expected = np.bitwise_and(lhs, rhs)
if not np.array_equal(got, expected):
    raise SystemExit(f"bitwise_and mismatch: {got} vs {expected}")
"#;
    let status = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(model_path)
        .status()
        .unwrap();
    assert!(status.success());
}

#[test]
#[ignore]
fn test_bitwise_not_uint8_numeric_smoke_with_onnxruntime() {
    if !python_onnxruntime_available() {
        return;
    }

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
            { "%": 11, "TT": { "D": [{ "#": "0.t_ui8" }, [4]] } }
        ]
    });
    converter
        .process_pass2_op("1.bitwise_not", &op_json)
        .unwrap();

    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path().join("bitwise_not_uint8.onnx");
    export_single_output_model(
        &mut converter,
        &[IoSpec {
            id: 10,
            elem_type: "0.t_ui8",
            dims: &[4],
        }],
        IoSpec {
            id: 11,
            elem_type: "0.t_ui8",
            dims: &[4],
        },
        model_path.to_str().unwrap(),
    )
    .unwrap();

    let script = r#"
import sys
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
x = np.array([0, 1, 15, 255], dtype=np.uint8)
got = sess.run(None, {"tensor_10": x})[0]
expected = np.bitwise_not(x)
if not np.array_equal(got, expected):
    raise SystemExit(f"bitwise_not mismatch: {got} vs {expected}")
"#;
    let status = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(model_path)
        .status()
        .unwrap();
    assert!(status.success());
}
