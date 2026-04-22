use crate::helper;
use crate::proto::onnx;
use serde_json::Value;

impl super::super::Converter {
    pub fn op_conv2d_transpose(&mut self, op: &Value) -> anyhow::Result<()> {
        let inputs = helper::op_input_ids(op);
        if inputs.len() < 2 {
            anyhow::bail!("conv2d_transpose missing input/filter tensors");
        }
        // Paddle's 3rd input is output_size, not ONNX bias. A 4th input is bias.
        if inputs.len() >= 4 {
            anyhow::bail!(
                "ConvTranspose with bias as 4th input is not supported; \
                 use a separate Add node for bias"
            );
        }

        let mut onnx_node = onnx::NodeProto {
            op_type: "ConvTranspose".to_string(),
            input: vec![
                self.get_tensor_name(inputs[0])?,
                self.get_tensor_name(inputs[1])?,
            ],
            ..Default::default()
        };
        if let Some(attrs) = op.get("A") {
            onnx_node.attribute = self.extract_attributes("conv2d_transpose", attrs);
        }
        if let Some(output_size) = inputs
            .get(2)
            .and_then(|id| self.state.constants.get(id))
            .map(|vals| vals.iter().map(|&v| v as i64).collect::<Vec<_>>())
        {
            onnx_node
                .attribute
                .retain(|attr| attr.name.as_str() != "output_shape");
            onnx_node
                .attribute
                .push(helper::attr_ints("output_shape", &output_size));
        }
        let out_id = helper::op_out_id(op)?;
        onnx_node.output.push(self.get_tensor_name(out_id)?);
        self.onnx_graph.node.push(onnx_node);
        Ok(())
    }

    pub fn op_pool2d(&mut self, op: &Value) -> anyhow::Result<()> {
        let is_avg = helper::attr(op, "pooling_type").and_then(|d| d.as_str()) == Some("avg");
        let is_adaptive = helper::attr(op, "adaptive")
            .and_then(|d| d.as_bool())
            .unwrap_or(false);
        let is_global = helper::attr(op, "global_pooling")
            .and_then(|d| d.as_bool())
            .unwrap_or(false);

        let mut pool_type = if is_avg { "AveragePool" } else { "MaxPool" };
        let mut onnx_node = onnx::NodeProto {
            op_type: pool_type.to_string(),
            ..Default::default()
        };

        if let Some(attrs) = op.get("A") {
            onnx_node.attribute = self.extract_attributes("pool2d", attrs);
        }
        if self.target_opset < 10 {
            if helper::attr(op, "ceil_mode")
                .and_then(|d| d.as_bool())
                .unwrap_or(false)
            {
                anyhow::bail!("pool2d ceil_mode requires opset >= 10");
            }
            onnx_node
                .attribute
                .retain(|attr| attr.name.as_str() != "ceil_mode");
        }

        let inputs = helper::op_input_ids(op);
        if !inputs.is_empty() {
            onnx_node.input.push(self.get_tensor_name(inputs[0])?);

            let output_size = inputs
                .get(1)
                .and_then(|id| self.state.constants.get(id))
                .map(|vals| vals.iter().map(|&v| v as i64).collect::<Vec<_>>());
            let has_same_auto_pad = onnx_node
                .attribute
                .iter()
                .any(|attr| attr.name == "auto_pad" && attr.s == b"SAME_UPPER");
            let use_global_pool =
                is_global || (matches!(output_size.as_deref(), Some([1, 1])) && is_adaptive);

            if use_global_pool {
                pool_type = if is_avg {
                    "GlobalAveragePool"
                } else {
                    "GlobalMaxPool"
                };
                onnx_node.op_type = pool_type.to_string();
                onnx_node.attribute.clear();
            } else if is_adaptive {
                anyhow::bail!(
                    "adaptive pool2d only supports global output_size=[1,1]; non-global adaptive pooling requires derived kernel/stride"
                );
            } else if let Some(ksize) = output_size {
                if has_same_auto_pad && ksize == [1, 1] {
                    anyhow::bail!(
                        "pool2d with padding_algorithm=SAME does not support output_size=[1,1] unless adaptive=true"
                    );
                }
                onnx_node
                    .attribute
                    .push(helper::attr_ints("kernel_shape", &ksize));
            }
        }
        if is_avg
            && !helper::attr(op, "exclusive")
                .and_then(|d| d.as_bool())
                .unwrap_or(true)
        {
            onnx_node
                .attribute
                .push(helper::attr_int("count_include_pad", 1));
        }

        let out_id = helper::op_out_id(op)?;
        onnx_node.output.push(self.get_tensor_name(out_id)?);
        self.onnx_graph.node.push(onnx_node);
        Ok(())
    }
}
