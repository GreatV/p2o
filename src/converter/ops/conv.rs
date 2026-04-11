use serde_json::Value;
use crate::proto::onnx;
use crate::helper;

impl super::super::Converter {
    pub fn op_pool2d(&mut self, op: &Value) -> anyhow::Result<()> {
        let mut pool_type = "MaxPool";
        if helper::attr(op, "pooling_type").and_then(|d| d.as_str()) == Some("avg") {
            pool_type = "AveragePool";
        }
        let mut onnx_node = onnx::NodeProto {
            op_type: pool_type.to_string(),
            ..Default::default()
        };
        
        if let Some(attrs) = op.get("A") {
            onnx_node.attribute = self.extract_attributes("pool2d", attrs);
        }

        let inputs = helper::op_input_ids(op);
        if !inputs.is_empty() {
            onnx_node.input.push(self.get_tensor_name(inputs[0])?);
            if inputs.len() > 1
                && let Some(ksize) = self.constants.get(&inputs[1]) {
                    onnx_node.attribute.push(onnx::AttributeProto {
                        name: "kernel_shape".to_string(),
                        r#type: crate::helper::at::INTS,
                        ints: ksize.iter().map(|&v| v as i64).collect(),
                        ..Default::default()
                    });
                }
        }

        if let Ok(out_id) = helper::op_out_id(op) {
            onnx_node.output.push(self.get_tensor_name(out_id)?);
        }
        self.onnx_graph.node.push(onnx_node);
        Ok(())
    }
}
