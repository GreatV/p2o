use crate::converter::FetchNameCache;
use crate::helper;
use crate::proto::onnx;
use serde_json::Value;
use std::collections::HashSet;

impl super::super::Converter {
    pub fn op_data(&mut self, op: &Value) -> anyhow::Result<()> {
        let vinfo = self.build_value_info(op, true)?;
        self.onnx_graph.input.push(vinfo.clone());
        let out_id = helper::op_out_id(op)?;
        self.state.id_to_name.insert(out_id, vinfo.name);
        Ok(())
    }

    pub fn op_fetch(&mut self, op: &Value) -> anyhow::Result<()> {
        let mut vinfo = self.build_value_info(op, false)?;
        if vinfo.r#type.is_none()
            && let Some(tt) = op
                .get("I")
                .and_then(|i| i.as_array())
                .and_then(|i| i.first())
                .and_then(|i| i.get("TT"))
                .and_then(|tt| tt.get("D"))
                .and_then(|d| d.as_array())
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
            vinfo = self.build_value_info_from_meta(vinfo.name, elem_type_str, &dims)?;
        }
        let inputs = helper::op_input_ids(op);
        if !inputs.is_empty() {
            let in_name = self.get_tensor_name(inputs[0])?;
            let node_count = self.onnx_graph.node.len();
            let output_count = self.onnx_graph.output.len();
            if let Some(cache) = self.fetch_name_cache.as_ref() {
                debug_assert!(cache.node_count <= node_count);
                debug_assert!(cache.output_count <= output_count);
            }
            // The converter only appends nodes/outputs; it never deletes them. Under
            // that append-only invariant, `(node_count, output_count)` is a stable cache
            // generation key for the existing-name set.
            if self.fetch_name_cache.as_ref().is_none_or(|cache| {
                cache.node_count != node_count || cache.output_count != output_count
            }) {
                let names = self
                    .onnx_graph
                    .output
                    .iter()
                    .map(|o| o.name.clone())
                    .chain(
                        self.onnx_graph
                            .node
                            .iter()
                            .flat_map(|node| node.output.iter().cloned()),
                    )
                    .collect::<HashSet<_>>();
                self.fetch_name_cache = Some(FetchNameCache {
                    node_count,
                    output_count,
                    names,
                });
            }
            let existing_names = &self.fetch_name_cache.as_ref().unwrap().names;
            let mut out_name = helper::attr(op, "name")
                .and_then(|d| d.as_str())
                .filter(|name| !name.is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| format!("fetch_{}", self.onnx_graph.output.len()));
            if existing_names.contains(out_name.as_str()) {
                let base = out_name.clone();
                let mut suffix = 1u32;
                loop {
                    out_name = format!("{}_{}", base, suffix);
                    if !existing_names.contains(&out_name) {
                        break;
                    }
                    suffix += 1;
                }
                log::warn!(
                    "fetch output name '{}' collides; renamed to '{}'",
                    base,
                    out_name
                );
            }
            vinfo.name = out_name.clone();
            self.onnx_graph.output.push(vinfo);
            if in_name != out_name {
                let node = onnx::NodeProto {
                    op_type: "Identity".to_string(),
                    input: vec![in_name],
                    output: vec![out_name.clone()],
                    ..Default::default()
                };
                self.onnx_graph.node.push(node);
            }
            if let Some(cache) = self.fetch_name_cache.as_mut() {
                cache.names.insert(out_name);
                cache.node_count = self.onnx_graph.node.len();
                cache.output_count = self.onnx_graph.output.len();
            }
        } else {
            anyhow::bail!("fetch op has no input; cannot produce graph output");
        }
        Ok(())
    }

    pub fn op_dropout(&mut self, op: &Value) -> anyhow::Result<()> {
        let mut onnx_node = onnx::NodeProto {
            op_type: "Identity".to_string(),
            ..Default::default()
        };
        let inputs = helper::op_input_ids(op);
        if !inputs.is_empty() {
            onnx_node.input.push(self.get_tensor_name(inputs[0])?);
        }
        let out_id = helper::op_out_id(op)?;
        onnx_node.output.push(self.get_tensor_name(out_id)?);
        self.onnx_graph.node.push(onnx_node);
        Ok(())
    }

    pub(crate) fn convert_generic_op(&mut self, op_type: &str, op: &Value) -> anyhow::Result<()> {
        let mut onnx_node = onnx::NodeProto {
            op_type: self.map_op_type(op_type)?,
            ..Default::default()
        };

        if let Some(attrs) = op.get("A") {
            onnx_node.attribute = self.extract_attributes(op_type, attrs);
        }

        let inputs = helper::op_input_ids(op);
        for input_id in inputs {
            onnx_node.input.push(self.get_tensor_name(input_id)?);
        }

        if let Some(outputs) = op.get("O").and_then(|o| o.as_array()) {
            for output in outputs {
                if let Some(id) = output.get("%").and_then(|id| id.as_i64()) {
                    onnx_node.output.push(self.get_tensor_name(id)?);
                }
            }
        }

        self.onnx_graph.node.push(onnx_node);
        Ok(())
    }
}
