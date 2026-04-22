use serde_json::Value;

use crate::helper::{self, paddle_tt};
use crate::proto::onnx;

impl super::Converter {
    pub fn onnx_elem_type_from_paddle(&self, elem_type_str: &str) -> anyhow::Result<i32> {
        helper::paddle_elem_type_to_onnx(elem_type_str)
            .ok_or_else(|| anyhow::anyhow!("Unsupported Paddle element type: {}", elem_type_str))
    }

    pub fn build_value_info_from_meta(
        &self,
        name: String,
        elem_type_str: &str,
        dims: &[i64],
    ) -> anyhow::Result<onnx::ValueInfoProto> {
        let mut vi = onnx::ValueInfoProto {
            name,
            ..Default::default()
        };

        let mut shape_proto = onnx::TensorShapeProto { dim: vec![] };
        for &dim in dims {
            let mut dim_proto = onnx::tensor_shape_proto::Dimension {
                denotation: "".to_string(),
                ..Default::default()
            };
            if dim >= 0 {
                dim_proto.value = Some(onnx::tensor_shape_proto::dimension::Value::DimValue(dim));
            }
            shape_proto.dim.push(dim_proto);
        }

        let tensor_type = onnx::type_proto::Tensor {
            elem_type: self.onnx_elem_type_from_paddle(elem_type_str)?,
            shape: Some(shape_proto),
        };

        vi.r#type = Some(onnx::TypeProto {
            denotation: "".to_string(),
            value: Some(onnx::type_proto::Value::TensorType(tensor_type)),
        });
        Ok(vi)
    }

    pub fn build_value_info_for_id(
        &self,
        id: i64,
        name: String,
    ) -> anyhow::Result<onnx::ValueInfoProto> {
        let elem_type = self
            .state
            .tensor_types
            .get(&id)
            .map(String::as_str)
            .unwrap_or(paddle_tt::F32);
        let dims = self
            .state
            .tensor_shapes
            .get(&id)
            .cloned()
            .unwrap_or_default();
        self.build_value_info_from_meta(name, elem_type, &dims)
    }

    pub fn build_value_info(
        &mut self,
        op: &Value,
        is_input: bool,
    ) -> anyhow::Result<onnx::ValueInfoProto> {
        let default_name = if is_input {
            helper::op_out_id(op)
                .map(|id| format!("input_{}", id))
                .unwrap_or_else(|_| "input".to_string())
        } else {
            helper::op_input_ids(op)
                .first()
                .map(|id| format!("output_{}", id))
                .unwrap_or_else(|| "output".to_string())
        };
        let name = helper::attr(op, "name")
            .and_then(|d| d.as_str())
            .filter(|name| !name.is_empty())
            .map(str::to_owned)
            .unwrap_or(default_name);

        let mut vi = onnx::ValueInfoProto {
            name,
            ..Default::default()
        };

        let tt_d = op
            .get("O")
            .and_then(|o| o.as_array())
            .and_then(|o| o.first())
            .and_then(|o| o.get("TT"))
            .and_then(|tt| tt.get("D"))
            .and_then(|d| d.as_array());

        if let Some(tt) = tt_d
            && tt.len() >= 2
            && let Some(elem_type_str) = tt[0].get("#").and_then(|t| t.as_str())
        {
            let dims = tt[1]
                .as_array()
                .map(|dims| {
                    dims.iter()
                        .filter_map(|dim| dim.as_i64())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            vi = self.build_value_info_from_meta(vi.name.clone(), elem_type_str, &dims)?;
        }

        Ok(vi)
    }
}
