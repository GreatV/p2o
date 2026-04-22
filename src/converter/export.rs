use anyhow::bail;
use prost::Message;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;

use crate::helper;
use crate::proto::onnx;

impl super::Converter {
    pub fn validate(&self) -> anyhow::Result<()> {
        let outer = HashSet::new();
        Self::validate_graph(&self.onnx_graph, &outer)
    }

    fn validate_graph(
        graph: &onnx::GraphProto,
        outer_defined: &HashSet<String>,
    ) -> anyhow::Result<()> {
        let mut defined: HashSet<String> = outer_defined.clone();
        for init in &graph.initializer {
            defined.insert(init.name.clone());
        }
        for inp in &graph.input {
            defined.insert(inp.name.clone());
        }
        let mut seen_node_outputs = HashSet::new();
        for node in &graph.node {
            for out in &node.output {
                if out.is_empty() {
                    continue;
                }
                if !seen_node_outputs.insert(out.clone()) {
                    bail!("tensor {} is produced by more than one node", out);
                }
                defined.insert(out.clone());
            }
        }
        for node in &graph.node {
            for inp in &node.input {
                if !inp.is_empty() && !defined.contains(inp.as_str()) {
                    bail!("node {} references undefined tensor {}", node.op_type, inp);
                }
            }
            for attr in &node.attribute {
                if let Some(subgraph) = &attr.g {
                    Self::validate_graph(subgraph, &defined)?;
                }
                for subgraph in &attr.graphs {
                    Self::validate_graph(subgraph, &defined)?;
                }
            }
        }
        for out in &graph.output {
            if !defined.contains(out.name.as_str()) {
                bail!("graph output {} is never produced by any node", out.name);
            }
        }
        Ok(())
    }

    fn collect_referenced_value_names(graph: &onnx::GraphProto) -> HashSet<String> {
        let mut used_names = HashSet::new();
        for node in &graph.node {
            for input in &node.input {
                if !input.is_empty() {
                    used_names.insert(input.clone());
                }
            }
            for attr in &node.attribute {
                if let Some(subgraph) = attr.g.as_ref() {
                    used_names.extend(Self::collect_referenced_value_names(subgraph));
                }
                for subgraph in &attr.graphs {
                    used_names.extend(Self::collect_referenced_value_names(subgraph));
                }
            }
        }
        used_names
    }

    fn materialize_initializer_only_outputs(graph: &mut onnx::GraphProto) {
        let produced_output_names = graph
            .node
            .iter()
            .flat_map(|node| node.output.iter())
            .filter(|name| !name.is_empty())
            .cloned()
            .collect::<HashSet<_>>();
        let output_names = graph
            .output
            .iter()
            .map(|output| output.name.clone())
            .collect::<HashSet<_>>();

        let mut remaining_initializers = Vec::with_capacity(graph.initializer.len());
        let mut constant_nodes = Vec::new();
        for initializer in graph.initializer.drain(..) {
            if output_names.contains(&initializer.name)
                && !produced_output_names.contains(&initializer.name)
            {
                constant_nodes.push(onnx::NodeProto {
                    op_type: "Constant".to_string(),
                    output: vec![initializer.name.clone()],
                    attribute: vec![helper::attr_tensor("value", initializer)],
                    ..Default::default()
                });
            } else {
                remaining_initializers.push(initializer);
            }
        }
        graph.initializer = remaining_initializers;
        graph.node.extend(constant_nodes);
    }

    pub(crate) fn sanitize_graph(graph: &mut onnx::GraphProto, prune_unused_initializers: bool) {
        for node in &mut graph.node {
            for attr in &mut node.attribute {
                if let Some(subgraph) = attr.g.as_mut() {
                    Self::sanitize_graph(subgraph, true);
                }
                for subgraph in &mut attr.graphs {
                    Self::sanitize_graph(subgraph, true);
                }
            }
        }

        let mut deduped_initializers = Vec::with_capacity(graph.initializer.len());
        let mut seen_initializer_names = HashSet::new();
        for initializer in graph.initializer.iter().rev() {
            if seen_initializer_names.insert(initializer.name.clone()) {
                deduped_initializers.push(initializer.clone());
            }
        }
        deduped_initializers.reverse();
        graph.initializer = deduped_initializers;

        let initializer_names = graph
            .initializer
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect::<HashSet<_>>();

        let mut deduped_inputs = Vec::with_capacity(graph.input.len());
        let mut seen_input_names = HashSet::new();
        for input in &graph.input {
            if initializer_names.contains(&input.name) {
                continue;
            }
            if seen_input_names.insert(input.name.clone()) {
                deduped_inputs.push(input.clone());
            }
        }
        graph.input = deduped_inputs;

        Self::materialize_initializer_only_outputs(graph);

        if prune_unused_initializers {
            let mut used_initializer_names = Self::collect_referenced_value_names(graph);
            used_initializer_names.extend(graph.output.iter().map(|output| output.name.clone()));
            graph
                .initializer
                .retain(|tensor| used_initializer_names.contains(&tensor.name));
        }
    }

    /// Finalizes and writes the ONNX model.
    ///
    /// Export intentionally sanitizes before validation: initializer-only graph
    /// outputs are materialized as Constant nodes, duplicate initializers/inputs
    /// are removed, and unused initializers are pruned before reference checks.
    pub fn export_onnx(&mut self, output_path: &str, opset_version: i64) -> anyhow::Result<()> {
        Self::sanitize_graph(&mut self.onnx_graph, true);
        self.validate()?;
        let graph = std::mem::take(&mut self.onnx_graph);
        let model = onnx::ModelProto {
            ir_version: 8,
            opset_import: vec![onnx::OperatorSetIdProto {
                domain: "".to_string(),
                version: opset_version,
            }],
            producer_name: env!("CARGO_PKG_NAME").to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            graph: Some(graph),
            ..Default::default()
        };

        let mut buf = Vec::new();
        model.encode(&mut buf)?;

        let mut file = File::create(output_path)?;
        file.write_all(&buf)?;
        log::info!("Saved ONNX model to: {}", output_path);
        Ok(())
    }
}
