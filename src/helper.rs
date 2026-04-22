use crate::proto::onnx;
use serde_json::Value;

pub fn op_type(op: &Value) -> Option<&str> {
    op.get("#").and_then(|t| t.as_str())
}

pub fn op_out_id(op: &Value) -> anyhow::Result<i64> {
    if let Some(o) = op.get("O") {
        if let Some(arr) = o.as_array() {
            if let Some(first) = arr.first()
                && let Some(id) = first.get("%").and_then(|id| id.as_i64())
            {
                return Ok(id);
            }
        } else if let Some(obj) = o.as_object()
            && let Some(id) = obj.get("%").and_then(|id| id.as_i64())
        {
            return Ok(id);
        }
    }
    anyhow::bail!("Missing output ID in op: {:?}", op)
}

pub fn op_input_ids(op: &Value) -> Vec<i64> {
    if let Some(inputs) = op.get("I").and_then(|i| i.as_array()) {
        inputs
            .iter()
            .filter_map(|i| i.get("%").and_then(|id| id.as_i64()))
            .collect()
    } else {
        Vec::new()
    }
}

pub fn attr<'a>(op: &'a Value, name: &str) -> Option<&'a Value> {
    op.get("A")
        .and_then(|a| a.as_array())
        .and_then(|a| {
            a.iter()
                .find(|x| x.get("N").and_then(|n| n.as_str()) == Some(name))
        })
        .and_then(|x| x.get("AT"))
        .and_then(|at| at.get("D"))
}

fn parse_special_f64(text: &str) -> Option<f64> {
    let text = text.trim();
    if text.eq_ignore_ascii_case("inf") || text.eq_ignore_ascii_case("+inf") {
        return Some(f64::INFINITY);
    }
    if text.eq_ignore_ascii_case("-inf") {
        return Some(f64::NEG_INFINITY);
    }
    if text.eq_ignore_ascii_case("nan")
        || text.eq_ignore_ascii_case("+nan")
        || text.eq_ignore_ascii_case("-nan")
    {
        return Some(f64::NAN);
    }
    text.parse::<f64>().ok()
}

pub fn value_as_f64(value: &Value) -> Option<f64> {
    if let Some(v) = value.as_f64() {
        return Some(v);
    }
    if let Some(v) = value.as_i64() {
        return Some(v as f64);
    }
    if let Some(obj) = value.as_object() {
        if let Some(v) = obj.get("D").and_then(value_as_f64) {
            return Some(v);
        }
        if let Some(text) = obj.get("VD").and_then(|v| v.as_str()) {
            return parse_special_f64(text);
        }
    }
    if let Some(text) = value.as_str() {
        return parse_special_f64(text);
    }
    None
}

pub fn attr_f64(op: &Value, name: &str) -> Option<f64> {
    op.get("A")
        .and_then(|a| a.as_array())
        .and_then(|a| {
            a.iter()
                .find(|x| x.get("N").and_then(|n| n.as_str()) == Some(name))
        })
        .and_then(|x| x.get("AT"))
        .and_then(value_as_f64)
}

fn paddle_dtype_token_to_onnx(dtype: &str) -> Option<i32> {
    match dtype {
        "bool" => Some(dt::BOOL),
        "f16" | "float16" => Some(dt::FLOAT16),
        "f32" | "float32" => Some(dt::FLOAT),
        "f64" | "float64" => Some(dt::DOUBLE),
        "i8" | "int8" => Some(dt::INT8),
        "ui8" | "uint8" => Some(dt::UINT8),
        "i16" | "int16" => Some(dt::INT16),
        "i32" | "int32" => Some(dt::INT32),
        "i64" | "int64" => Some(dt::INT64),
        _ => None,
    }
}

pub fn paddle_dtype_to_onnx(dtype: &str) -> Option<i32> {
    paddle_dtype_token_to_onnx(dtype)
}

pub fn paddle_elem_type_to_onnx(elem_type_str: &str) -> Option<i32> {
    elem_type_str
        .strip_prefix("0.t_")
        .and_then(paddle_dtype_token_to_onnx)
}

pub fn onnx_dtype_name(dtype: i32) -> &'static str {
    match dtype {
        dt::BOOL => "bool",
        dt::FLOAT16 => "float16",
        dt::FLOAT => "float32",
        dt::DOUBLE => "float64",
        dt::INT8 => "int8",
        dt::UINT8 => "uint8",
        dt::INT16 => "int16",
        dt::INT32 => "int32",
        dt::INT64 => "int64",
        dt::UINT16 => "uint16",
        dt::BFLOAT16 => "bfloat16",
        _ => "unknown",
    }
}

#[allow(dead_code)]
pub fn onnx_dtype_is_supported_bitwise_integer(dtype: i32) -> bool {
    matches!(
        dtype,
        dt::INT8 | dt::UINT8 | dt::INT16 | dt::INT32 | dt::INT64
    )
}

#[allow(dead_code)]
pub mod dt {
    pub const UNDEFINED: i32 = 0;
    pub const FLOAT: i32 = 1;
    pub const UINT8: i32 = 2;
    pub const INT8: i32 = 3;
    pub const UINT16: i32 = 4;
    pub const INT16: i32 = 5;
    pub const INT32: i32 = 6;
    pub const INT64: i32 = 7;
    pub const STRING: i32 = 8;
    pub const BOOL: i32 = 9;
    pub const FLOAT16: i32 = 10;
    pub const DOUBLE: i32 = 11;
    pub const BFLOAT16: i32 = 16;
}

pub mod at {
    pub const FLOAT: i32 = 1;
    pub const INT: i32 = 2;
    pub const STRING: i32 = 3;
    pub const FLOATS: i32 = 6;
    pub const INTS: i32 = 7;
    pub const STRINGS: i32 = 8;
}

pub fn attr_int(name: &str, value: i64) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
        r#type: at::INT,
        i: value,
        ..Default::default()
    }
}

pub fn attr_ints(name: &str, values: &[i64]) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
        r#type: at::INTS,
        ints: values.to_vec(),
        ..Default::default()
    }
}

pub fn attr_float(name: &str, value: f32) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
        r#type: at::FLOAT,
        f: value,
        ..Default::default()
    }
}

pub fn attr_str(name: &str, value: &str) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
        r#type: at::STRING,
        s: value.as_bytes().to_vec(),
        ..Default::default()
    }
}

pub fn attr_tensor(name: &str, value: onnx::TensorProto) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
        r#type: onnx::attribute_proto::AttributeType::Tensor as i32,
        t: Some(value),
        ..Default::default()
    }
}

pub fn attr_graph(name: &str, value: onnx::GraphProto) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
        r#type: onnx::attribute_proto::AttributeType::Graph as i32,
        g: Some(value),
        ..Default::default()
    }
}

#[allow(dead_code)]
pub mod paddle_tt {
    pub const BOOL: &str = "0.t_bool";
    pub const F16: &str = "0.t_f16";
    pub const F32: &str = "0.t_f32";
    pub const F64: &str = "0.t_f64";
    pub const I8: &str = "0.t_i8";
    pub const UI8: &str = "0.t_ui8";
    pub const I16: &str = "0.t_i16";
    pub const I32: &str = "0.t_i32";
    pub const I64: &str = "0.t_i64";
}

pub mod paddle_op {
    pub const ASSIGN_VALUE: &str = "1.assign_value_";
    pub const CAST: &str = "1.cast";
    pub const FLATTEN: &str = "1.flatten";
    pub const FULL: &str = "1.full";
    pub const FULL_INT_ARRAY: &str = "1.full_int_array";
    pub const SPLIT: &str = "1.split";
    pub const RESHAPE: &str = "1.reshape";
    pub const SCALE: &str = "1.scale";
    pub const SQUEEZE: &str = "1.squeeze";
    pub const SQUEEZE_INPLACE: &str = "1.squeeze_";
    pub const UNSQUEEZE: &str = "1.unsqueeze";
    pub const UNSQUEEZE_INPLACE: &str = "1.unsqueeze_";
}

#[cfg(test)]
mod tests {
    use super::value_as_f64;
    use serde_json::json;

    #[test]
    fn test_value_as_f64_accepts_signed_and_mixed_case_special_values() {
        assert_eq!(value_as_f64(&json!("-Inf")), Some(f64::NEG_INFINITY));
        assert_eq!(value_as_f64(&json!("+iNF")), Some(f64::INFINITY));
        assert!(value_as_f64(&json!({"VD": "nAn"})).unwrap().is_nan());
    }
}
