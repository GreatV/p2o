use serde_json::Value;

pub fn op_type(op: &Value) -> Option<&str> {
    op.get("#").and_then(|t| t.as_str())
}

pub fn op_out_id(op: &Value) -> anyhow::Result<i64> {
    if let Some(o) = op.get("O") {
        if let Some(arr) = o.as_array() {
            if let Some(first) = arr.first()
                && let Some(id) = first.get("%").and_then(|id| id.as_i64()) {
                    return Ok(id);
                }
        } else if let Some(obj) = o.as_object()
            && let Some(id) = obj.get("%").and_then(|id| id.as_i64()) {
                return Ok(id);
            }
    }
    anyhow::bail!("Missing output ID in op: {:?}", op)
}

pub fn op_input_ids(op: &Value) -> Vec<i64> {
    if let Some(inputs) = op.get("I").and_then(|i| i.as_array()) {
        inputs.iter().filter_map(|i| i.get("%").and_then(|id| id.as_i64())).collect()
    } else {
        Vec::new()
    }
}

pub fn attr<'a>(op: &'a Value, name: &str) -> Option<&'a Value> {
    op.get("A")
        .and_then(|a| a.as_array())
        .and_then(|a| a.iter().find(|x| x.get("N").and_then(|n| n.as_str()) == Some(name)))
        .and_then(|x| x.get("AT"))
        .and_then(|at| at.get("D"))
}

pub mod dt {
    pub const UNDEFINED: i32 = 0;
    pub const FLOAT:     i32 = 1;
    pub const UINT8:     i32 = 2;
    pub const INT8:      i32 = 3;
    pub const UINT16:    i32 = 4;
    pub const INT16:     i32 = 5;
    pub const INT32:     i32 = 6;
    pub const INT64:     i32 = 7;
    pub const STRING:    i32 = 8;
    pub const BOOL:      i32 = 9;
    pub const FLOAT16:   i32 = 10;
    pub const DOUBLE:    i32 = 11;
    pub const BFLOAT16:  i32 = 16;
}

pub mod at {
    pub const FLOAT: i32 = 1;
    pub const INT: i32 = 2;
    pub const STRING: i32 = 3;
    pub const FLOATS: i32 = 6;
    pub const INTS: i32 = 7;
    pub const STRINGS: i32 = 8;
}
