use crate::helper::{self, at};
use crate::proto::onnx;
use serde_json::Value;

include!(concat!(env!("OUT_DIR"), "/op_type_lookup.rs"));

impl super::Converter {
    fn rename_attr_name(&self, op_type: &str, name: &str) -> String {
        match (op_type, name) {
            (_, "start_axis" | "begin_norm_axis") => "axis".to_string(),
            ("hardsigmoid", "slope") => "alpha".to_string(),
            ("hardsigmoid", "offset") => "beta".to_string(),
            _ => name.to_string(),
        }
    }

    fn should_skip_attr(&self, op_type: &str, name: &str) -> bool {
        // Universal: Paddle-internal metadata, never valid ONNX attributes
        if matches!(
            name,
            "stop_gradient" | "name" | "place" | "struct_name" | "dtype" | "col"
        ) {
            return true;
        }
        // Training-mode flags: never needed for inference export
        if matches!(
            name,
            "is_test" | "use_global_stats" | "trainable_statistics"
        ) {
            return true;
        }

        match op_type {
            "conv2d" | "depthwise_conv2d" | "conv2d_transpose" => {
                matches!(name, "padding_algorithm" | "data_format")
            }
            "pool2d" => matches!(
                name,
                "padding_algorithm"
                    | "data_format"
                    | "global_pooling"
                    | "adaptive"
                    | "exclusive"
                    | "pooling_type"
                    | "ceil_mode"
            ),
            "nearest_interp" | "bilinear_interp" => matches!(
                name,
                "data_format"
                    | "align_corners"
                    | "align_mode"
                    | "interp_method"
                    | "out_h"
                    | "out_w"
                    | "out_d"
            ),
            "scale" => matches!(name, "bias_after_scale" | "scale" | "bias"),
            "matmul" | "matmul_v2" | "bmm" => matches!(name, "transpose_x" | "transpose_y"),
            "flatten" => matches!(name, "stop_axis" | "flatten"),
            "batch_norm" | "batch_norm_" => matches!(name, "data_format" | "scale" | "bias"),
            _ => false,
        }
    }

    fn normalize_pads_attr(&self, values: &[i64]) -> anyhow::Result<Vec<i64>> {
        match values {
            [h, w] => Ok(vec![*h, *w, *h, *w]),
            [top, bottom, left, right] => Ok(vec![*top, *left, *bottom, *right]),
            _ => anyhow::bail!("unsupported pads attribute length {}", values.len()),
        }
    }

    fn build_onnx_attribute(&self, name: String, at_obj: &Value) -> Option<onnx::AttributeProto> {
        let type_str = at_obj.get("#").and_then(|t| t.as_str()).unwrap_or("");
        let data = at_obj.get("D");

        let mut onnx_attr = onnx::AttributeProto {
            name,
            ..Default::default()
        };

        match type_str {
            "0.a_i32" | "0.a_i64" => {
                let value = data.and_then(|d| d.as_i64())?;
                onnx_attr.r#type = at::INT;
                onnx_attr.i = value;
                Some(onnx_attr)
            }
            "0.a_f32" | "0.a_f64" => {
                let value = data.and_then(|d| d.as_f64())?;
                onnx_attr.r#type = at::FLOAT;
                onnx_attr.f = value as f32;
                Some(onnx_attr)
            }
            "0.a_str" => {
                let value = data.and_then(|d| d.as_str())?;
                onnx_attr.r#type = at::STRING;
                onnx_attr.s = value.as_bytes().to_vec();
                Some(onnx_attr)
            }
            "0.a_bool" => {
                let value = data.and_then(|d| d.as_bool())?;
                onnx_attr.r#type = at::INT;
                onnx_attr.i = i64::from(value);
                Some(onnx_attr)
            }
            "0.a_array" | "1.a_intarray" => {
                let values = data.and_then(|d| d.as_array())?;
                if let Some(first) = values.first() {
                    let elem_type = first.get("#").and_then(|t| t.as_str()).unwrap_or("");
                    match elem_type {
                        "0.a_i32" | "0.a_i64" => {
                            onnx_attr.r#type = at::INTS;
                            onnx_attr.ints = values
                                .iter()
                                .filter_map(|x| x.get("D").and_then(|d| d.as_i64()))
                                .collect();
                            Some(onnx_attr)
                        }
                        "0.a_f32" | "0.a_f64" => {
                            onnx_attr.r#type = at::FLOATS;
                            onnx_attr.floats = values
                                .iter()
                                .filter_map(|x| {
                                    x.get("D").and_then(|d| d.as_f64()).map(|f| f as f32)
                                })
                                .collect();
                            Some(onnx_attr)
                        }
                        "0.a_str" => {
                            onnx_attr.r#type = at::STRINGS;
                            onnx_attr.strings = values
                                .iter()
                                .filter_map(|x| {
                                    x.get("D")
                                        .and_then(|d| d.as_str())
                                        .map(|s| s.as_bytes().to_vec())
                                })
                                .collect();
                            Some(onnx_attr)
                        }
                        _ if first.is_i64() || first.is_number() => {
                            onnx_attr.r#type = at::INTS;
                            onnx_attr.ints = values.iter().filter_map(|x| x.as_i64()).collect();
                            Some(onnx_attr)
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn apply_op_specific_attr_overrides(
        &self,
        op_type: &str,
        attrs: &Value,
        onnx_attrs: &mut Vec<onnx::AttributeProto>,
    ) {
        let is_conv_like = matches!(
            op_type,
            "conv2d" | "pool2d" | "depthwise_conv2d" | "conv2d_transpose"
        );

        if is_conv_like {
            for attr in onnx_attrs.iter_mut() {
                if attr.name == "paddings" {
                    attr.name = "pads".to_string();
                } else if attr.name == "groups" {
                    attr.name = "group".to_string();
                }

                if attr.name == "pads" && attr.r#type == at::INTS {
                    if let Ok(normalized) = self.normalize_pads_attr(&attr.ints) {
                        attr.ints = normalized;
                    } else {
                        log::warn!(
                            "leaving non-standard pads attribute {:?} unchanged for {}",
                            attr.ints,
                            op_type
                        );
                    }
                }
            }

            let has_same_padding = attrs.as_array().is_some_and(|items| {
                items.iter().any(|attr| {
                    attr.get("N").and_then(|n| n.as_str()) == Some("padding_algorithm")
                        && attr
                            .get("AT")
                            .and_then(|at| at.get("D"))
                            .and_then(|d| d.as_str())
                            == Some("SAME")
                })
            });
            if has_same_padding {
                onnx_attrs.push(helper::attr_str("auto_pad", "SAME_UPPER"));
                onnx_attrs.retain(|attr| attr.name != "pads");
            }
        }

        if op_type == "pool2d"
            && let Some(ceil_mode) = attrs.as_array().and_then(|items| {
                items.iter().find_map(|attr| {
                    (attr.get("N").and_then(|n| n.as_str()) == Some("ceil_mode"))
                        .then(|| {
                            attr.get("AT")
                                .and_then(|at| at.get("D"))
                                .and_then(|d| d.as_bool())
                        })
                        .flatten()
                })
            })
        {
            onnx_attrs.push(helper::attr_int("ceil_mode", i64::from(ceil_mode)));
        }
    }

    pub fn extract_attributes(&self, op_type: &str, attrs: &Value) -> Vec<onnx::AttributeProto> {
        let mut onnx_attrs = Vec::new();
        if let Some(arr) = attrs.as_array() {
            for attr in arr {
                let name = attr.get("N").and_then(|n| n.as_str()).unwrap_or("");
                let name = self.rename_attr_name(op_type, name);
                if self.should_skip_attr(op_type, &name) {
                    continue;
                }

                if let Some(at_obj) = attr.get("AT")
                    && let Some(onnx_attr) = self.build_onnx_attribute(name, at_obj)
                {
                    onnx_attrs.push(onnx_attr);
                }
            }
        }

        self.apply_op_specific_attr_overrides(op_type, attrs, &mut onnx_attrs);
        onnx_attrs
    }

    pub fn map_op_type(&self, paddle_op: &str) -> anyhow::Result<String> {
        if let Some(onnx_op) = generated_op_type(paddle_op) {
            return Ok(onnx_op.to_string());
        }
        anyhow::bail!("Unsupported operator: {}", paddle_op)
    }
}
