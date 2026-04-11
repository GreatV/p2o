use serde_json::Value;
use crate::proto::onnx;
use crate::helper::at;

impl super::Converter {
    pub fn extract_attributes(&self, op_type: &str, attrs: &Value) -> Vec<onnx::AttributeProto> {
        let mut onnx_attrs = Vec::new();
        if let Some(arr) = attrs.as_array() {
            for attr in arr {
                let mut name = attr.get("N").and_then(|n| n.as_str()).unwrap_or("").to_string();
                
                if name == "padding_algorithm"
                    && let Some(at_obj) = attr.get("AT")
                        && let Some(d) = at_obj.get("D").and_then(|d| d.as_str())
                            && d == "SAME" {
                                onnx_attrs.push(onnx::AttributeProto {
                                    name: "auto_pad".to_string(),
                                    r#type: at::STRING,
                                    s: b"SAME_UPPER".to_vec(),
                                    ..Default::default()
                                });
                            }
                
                if name == "ceil_mode"
                    && let Some(at_obj) = attr.get("AT")
                        && let Some(d) = at_obj.get("D").and_then(|d| d.as_bool()) {
                            onnx_attrs.push(onnx::AttributeProto {
                                name: "ceil_mode".to_string(),
                                r#type: at::INT,
                                i: if d { 1 } else { 0 },
                                ..Default::default()
                            });
                        }

                if name == "start_axis" { name = "axis".to_string(); }
                if name == "begin_norm_axis" { name = "axis".to_string(); }

                match name.as_str() {
                    "padding_algorithm" | "data_format" | "struct_name" | "use_global_stats" | 
                    "trainable_statistics" | "is_test" | "global_pooling" | "adaptive" | 
                    "exclusive" | "pooling_type" | "ceil_mode" | "align_corners" | 
                    "align_mode" | "interp_method" | "out_h" | "out_w" | "out_d" | 
                    "stop_gradient" | "bias_after_scale" | "name" | "place" | "dtype" | "col" | "scale" | "bias" | "stop_axis" | "transpose_x" | "transpose_y" => continue,
                    _ => {}
                }

                let at_v = attr.get("AT");
                
                if op_type == "conv2d" || op_type == "pool2d" || op_type == "depthwise_conv2d" || op_type == "conv2d_transpose" {
                    if name == "paddings" { name = "pads".to_string(); }
                    if name == "groups" { name = "group".to_string(); }
                }

                if let Some(at_obj) = at_v {
                    let type_str = at_obj.get("#").and_then(|t| t.as_str()).unwrap_or("");
                    let data = at_obj.get("D");
                    
                    let mut onnx_attr = onnx::AttributeProto {
                        name: name.clone(),
                        ..Default::default()
                    };

                    let mut pushed = false;
                    match type_str {
                        "0.a_i32" | "0.a_i64" => {
                            if let Some(d) = data.and_then(|d| d.as_i64()) {
                                onnx_attr.r#type = at::INT;
                                onnx_attr.i = d;
                                pushed = true;
                            }
                        },
                        "0.a_f32" | "0.a_f64" => {
                            if let Some(d) = data.and_then(|d| d.as_f64()) {
                                onnx_attr.r#type = at::FLOAT;
                                onnx_attr.f = d as f32;
                                pushed = true;
                            }
                        },
                        "0.a_str" => {
                            if let Some(d) = data.and_then(|d| d.as_str()) {
                                onnx_attr.r#type = at::STRING;
                                onnx_attr.s = d.as_bytes().to_vec();
                                pushed = true;
                            }
                        },
                        "0.a_bool" => {
                            if let Some(d) = data.and_then(|d| d.as_bool()) {
                                onnx_attr.r#type = at::INT;
                                onnx_attr.i = if d { 1 } else { 0 };
                                pushed = true;
                            }
                        },
                        "0.a_array" | "1.a_intarray" => {
                            if let Some(d_arr) = data.and_then(|d| d.as_array()) {
                                if let Some(first) = d_arr.first() {
                                    let elem_type = first.get("#").and_then(|t| t.as_str()).unwrap_or("");
                                    match elem_type {
                                        "0.a_i32" | "0.a_i64" => {
                                            onnx_attr.r#type = at::INTS;
                                            onnx_attr.ints = d_arr.iter().filter_map(|x| x.get("D").and_then(|d| d.as_i64())).collect();
                                            if onnx_attr.ints.len() == 2 && onnx_attr.name == "pads" {
                                                let h = onnx_attr.ints[0];
                                                let w = onnx_attr.ints[1];
                                                onnx_attr.ints = vec![h, w, h, w];
                                            }
                                            pushed = true;
                                        },
                                        "0.a_f32" | "0.a_f64" => {
                                            onnx_attr.r#type = at::FLOATS;
                                            onnx_attr.floats = d_arr.iter().filter_map(|x| x.get("D").and_then(|d| d.as_f64()).map(|f| f as f32)).collect();
                                            pushed = true;
                                        },
                                        "0.a_str" => {
                                            onnx_attr.r#type = at::STRINGS;
                                            onnx_attr.strings = d_arr.iter().filter_map(|x| x.get("D").and_then(|d| d.as_str()).map(|s| s.as_bytes().to_vec())).collect();
                                            pushed = true;
                                        },
                                        _ => {
                                            if first.is_i64() || first.is_number() {
                                                onnx_attr.r#type = at::INTS;
                                                onnx_attr.ints = d_arr.iter().filter_map(|x| x.as_i64()).collect();
                                                pushed = true;
                                            }
                                        }
                                    }
                                } else {
                                    onnx_attr.r#type = at::INTS;
                                    pushed = true;
                                }
                            }
                        },
                        _ => {}
                    }
                    if pushed {
                        onnx_attrs.push(onnx_attr);
                    }
                }
            }
        }
        
        let has_auto_pad = onnx_attrs.iter().any(|a| a.name == "auto_pad");
        if has_auto_pad {
            onnx_attrs.retain(|a| a.name != "pads");
        }
        
        onnx_attrs
    }

    pub fn map_op_type(&self, paddle_op: &str) -> anyhow::Result<String> {
        match paddle_op {
            "conv2d" | "depthwise_conv2d" => Ok("Conv".to_string()),
            "add" => Ok("Add".to_string()),
            "relu" => Ok("Relu".to_string()),
            "pool2d" => Ok("MaxPool".to_string()),
            "batch_norm" => Ok("BatchNormalization".to_string()),
            "matmul" | "matmul_v2" => Ok("MatMul".to_string()),
            "reshape" => Ok("Reshape".to_string()),
            "transpose" => Ok("Transpose".to_string()),
            "softmax" => Ok("Softmax".to_string()),
            "conv2d_transpose" => Ok("ConvTranspose".to_string()),
            "nearest_interp" => Ok("Resize".to_string()),
            "sigmoid" => Ok("Sigmoid".to_string()),
            "assign" => Ok("Identity".to_string()),
            "shape64" | "shape" => Ok("Shape".to_string()),
            "flatten" => Ok("Flatten".to_string()),
            "layer_norm" => Ok("LayerNormalization".to_string()),
            "squeeze" | "squeeze_" => Ok("Squeeze".to_string()),
            "unsqueeze" | "unsqueeze_" => Ok("Unsqueeze".to_string()),
            _ => anyhow::bail!("Unsupported operator: {}", paddle_op)
        }
    }
}
