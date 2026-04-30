use anyhow::bail;
use prost::Message;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;

use crate::helper;
use crate::proto::onnx;

const MAX_EXPORT_SUBGRAPH_DEPTH: usize = 128;

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

    fn ensure_export_depth(depth: usize) -> anyhow::Result<()> {
        // The root graph is depth 0, so 128 nested subgraph edges are allowed.
        if depth > MAX_EXPORT_SUBGRAPH_DEPTH {
            bail!(
                "maximum ONNX subgraph cleanup depth {} exceeded",
                MAX_EXPORT_SUBGRAPH_DEPTH
            );
        }
        Ok(())
    }

    fn sanitize_graph_checked(
        graph: &mut onnx::GraphProto,
        prune_unused_initializers: bool,
        depth: usize,
    ) -> anyhow::Result<()> {
        Self::ensure_export_depth(depth)?;
        for node in &mut graph.node {
            for attr in &mut node.attribute {
                if let Some(subgraph) = attr.g.as_mut() {
                    Self::sanitize_graph_checked(subgraph, true, depth + 1)?;
                }
                for subgraph in &mut attr.graphs {
                    Self::sanitize_graph_checked(subgraph, true, depth + 1)?;
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
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn sanitize_graph(graph: &mut onnx::GraphProto, prune_unused_initializers: bool) {
        Self::sanitize_graph_checked(graph, prune_unused_initializers, 0)
            .expect("sanitize_graph depth guard failed");
    }

    fn replace_node_inputs(graph: &mut onnx::GraphProto, replacements: &HashMap<String, String>) {
        for node in &mut graph.node {
            for input in &mut node.input {
                if let Some(replacement) = Self::resolve_replacement(input, replacements) {
                    *input = replacement;
                }
            }
            for attr in &mut node.attribute {
                if let Some(subgraph) = attr.g.as_mut() {
                    Self::replace_node_inputs(subgraph, replacements);
                }
                for subgraph in &mut attr.graphs {
                    Self::replace_node_inputs(subgraph, replacements);
                }
            }
        }
    }

    fn resolve_replacement(name: &str, replacements: &HashMap<String, String>) -> Option<String> {
        let mut current = replacements.get(name)?;
        let mut seen = HashSet::new();
        while let Some(next) = replacements.get(current) {
            if !seen.insert(current.clone()) {
                break;
            }
            current = next;
        }
        Some(current.clone())
    }

    fn graph_output_names(graph: &onnx::GraphProto) -> HashSet<String> {
        graph
            .output
            .iter()
            .map(|output| output.name.clone())
            .collect()
    }

    fn recursive_input_use_counts(graph: &onnx::GraphProto) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        Self::add_recursive_input_use_counts(graph, &mut counts);
        counts
    }

    fn add_recursive_input_use_counts(
        graph: &onnx::GraphProto,
        counts: &mut HashMap<String, usize>,
    ) {
        for node in &graph.node {
            for input in &node.input {
                if !input.is_empty() {
                    *counts.entry(input.clone()).or_default() += 1;
                }
            }
            for attr in &node.attribute {
                if let Some(subgraph) = attr.g.as_ref() {
                    Self::add_recursive_input_use_counts(subgraph, counts);
                }
                for subgraph in &attr.graphs {
                    Self::add_recursive_input_use_counts(subgraph, counts);
                }
            }
        }
    }

    fn cast_to_dtype(node: &onnx::NodeProto) -> Option<i64> {
        if node.op_type != "Cast" {
            return None;
        }
        node.attribute
            .iter()
            .find(|attr| attr.name == "to")
            .map(|attr| attr.i)
    }

    fn static_i64_initializer(graph: &onnx::GraphProto, name: &str) -> Option<Vec<i64>> {
        let tensor = graph
            .initializer
            .iter()
            .find(|tensor| tensor.name == name)?;
        if tensor.data_type != helper::dt::INT64 {
            return None;
        }
        if !tensor.int64_data.is_empty() {
            return Some(tensor.int64_data.clone());
        }
        if tensor.raw_data.len() % std::mem::size_of::<i64>() != 0 {
            return None;
        }
        Some(
            tensor
                .raw_data
                .chunks_exact(std::mem::size_of::<i64>())
                .map(|chunk| {
                    let mut bytes = [0_u8; std::mem::size_of::<i64>()];
                    bytes.copy_from_slice(chunk);
                    i64::from_le_bytes(bytes)
                })
                .collect(),
        )
    }

    fn static_axes(graph: &onnx::GraphProto, node: &onnx::NodeProto) -> Option<Vec<i64>> {
        if node.op_type != "Squeeze" && node.op_type != "Unsqueeze" {
            return None;
        }
        if let Some(attr) = node.attribute.iter().find(|attr| attr.name == "axes") {
            return Some(attr.ints.clone());
        }
        let axes_name = node.input.get(1)?;
        if axes_name.is_empty() {
            return None;
        }
        Self::static_i64_initializer(graph, axes_name)
    }

    fn same_static_axes(
        graph: &onnx::GraphProto,
        lhs: &onnx::NodeProto,
        rhs: &onnx::NodeProto,
    ) -> bool {
        let Some(mut lhs_axes) = Self::static_axes(graph, lhs) else {
            return false;
        };
        let Some(mut rhs_axes) = Self::static_axes(graph, rhs) else {
            return false;
        };
        if lhs_axes.iter().any(|axis| *axis < 0) || rhs_axes.iter().any(|axis| *axis < 0) {
            return false;
        }
        lhs_axes.sort_unstable();
        rhs_axes.sort_unstable();
        lhs_axes == rhs_axes
    }

    fn initializer_content_key(tensor: &onnx::TensorProto) -> Vec<u8> {
        let mut key_tensor = tensor.clone();
        key_tensor.name.clear();
        key_tensor.encode_to_vec()
    }

    fn canonicalize_graph_checked(
        graph: &mut onnx::GraphProto,
        depth: usize,
    ) -> anyhow::Result<()> {
        Self::ensure_export_depth(depth)?;
        for node in &mut graph.node {
            for attr in &mut node.attribute {
                if let Some(subgraph) = attr.g.as_mut() {
                    Self::canonicalize_graph_checked(subgraph, depth + 1)?;
                }
                for subgraph in &mut attr.graphs {
                    Self::canonicalize_graph_checked(subgraph, depth + 1)?;
                }
            }
        }

        loop {
            let changed = Self::remove_identity_nodes(graph)
                | Self::collapse_duplicate_casts(graph)
                | Self::remove_canceling_shape_adapters(graph);
            if !changed {
                break;
            }
        }
        Self::dedup_byte_identical_initializers(graph);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn canonicalize_graph(graph: &mut onnx::GraphProto) {
        Self::canonicalize_graph_checked(graph, 0).expect("canonicalize_graph depth guard failed");
    }

    fn remove_identity_nodes(graph: &mut onnx::GraphProto) -> bool {
        let graph_outputs = Self::graph_output_names(graph);
        let mut replacements = HashMap::new();
        let mut remove = vec![false; graph.node.len()];

        for (idx, node) in graph.node.iter().enumerate() {
            if node.op_type == "Identity"
                && node.input.len() == 1
                && node.output.len() == 1
                && !node.input[0].is_empty()
                && !node.output[0].is_empty()
                && !graph_outputs.contains(&node.output[0])
            {
                replacements.insert(node.output[0].clone(), node.input[0].clone());
                remove[idx] = true;
            }
        }

        if replacements.is_empty() {
            return false;
        }
        Self::replace_node_inputs(graph, &replacements);
        let mut idx = 0;
        graph.node.retain(|_| {
            let keep = !remove[idx];
            idx += 1;
            keep
        });
        true
    }

    fn collapse_duplicate_casts(graph: &mut onnx::GraphProto) -> bool {
        let graph_outputs = Self::graph_output_names(graph);
        let use_counts = Self::recursive_input_use_counts(graph);
        let mut producer_by_output = HashMap::new();
        for (idx, node) in graph.node.iter().enumerate() {
            for output in &node.output {
                if !output.is_empty() {
                    producer_by_output.insert(output.clone(), idx);
                }
            }
        }

        let mut remove = vec![false; graph.node.len()];
        let mut protected_consumers = HashSet::new();
        let mut rewrites = Vec::new();
        for (consumer_idx, consumer) in graph.node.iter().enumerate() {
            if consumer.op_type != "Cast" || consumer.input.len() != 1 || consumer.output.len() != 1
            {
                continue;
            }
            let temp = &consumer.input[0];
            if temp.is_empty()
                || graph_outputs.contains(temp)
                || use_counts.get(temp).copied().unwrap_or_default() != 1
            {
                continue;
            }
            let Some(&producer_idx) = producer_by_output.get(temp) else {
                continue;
            };
            if protected_consumers.contains(&producer_idx) {
                continue;
            }
            let producer = &graph.node[producer_idx];
            if producer.input.len() != 1 || producer.output.len() != 1 {
                continue;
            }
            let Some(producer_dtype) = Self::cast_to_dtype(producer) else {
                continue;
            };
            let Some(consumer_dtype) = Self::cast_to_dtype(consumer) else {
                continue;
            };
            if producer_dtype != consumer_dtype {
                continue;
            }
            remove[producer_idx] = true;
            protected_consumers.insert(consumer_idx);
            rewrites.push((consumer_idx, producer.input[0].clone()));
        }

        if rewrites.is_empty() {
            return false;
        }
        for (idx, input) in rewrites {
            graph.node[idx].input[0] = input;
        }
        let mut idx = 0;
        graph.node.retain(|_| {
            let keep = !remove[idx];
            idx += 1;
            keep
        });
        true
    }

    fn remove_canceling_shape_adapters(graph: &mut onnx::GraphProto) -> bool {
        let graph_outputs = Self::graph_output_names(graph);
        let use_counts = Self::recursive_input_use_counts(graph);
        let mut producer_by_output = HashMap::new();
        for (idx, node) in graph.node.iter().enumerate() {
            for output in &node.output {
                if !output.is_empty() {
                    producer_by_output.insert(output.clone(), idx);
                }
            }
        }

        let mut remove = vec![false; graph.node.len()];
        let mut replacements = HashMap::new();
        let mut output_identities = Vec::new();

        for (consumer_idx, consumer) in graph.node.iter().enumerate() {
            if consumer.input.is_empty() || consumer.output.len() != 1 {
                continue;
            }
            let temp = &consumer.input[0];
            if temp.is_empty()
                || graph_outputs.contains(temp)
                || use_counts.get(temp).copied().unwrap_or_default() != 1
            {
                continue;
            }
            let Some(&producer_idx) = producer_by_output.get(temp) else {
                continue;
            };
            if remove[producer_idx] {
                continue;
            }
            let producer = &graph.node[producer_idx];
            if producer.input.is_empty() || producer.output.len() != 1 {
                continue;
            }
            let is_canceling_pair = (producer.op_type == "Unsqueeze"
                && consumer.op_type == "Squeeze")
                || (producer.op_type == "Squeeze" && consumer.op_type == "Unsqueeze");
            if !is_canceling_pair || !Self::same_static_axes(graph, producer, consumer) {
                continue;
            }

            let source = producer.input[0].clone();
            let output = consumer.output[0].clone();
            remove[producer_idx] = true;
            if graph_outputs.contains(&output) {
                output_identities.push((consumer_idx, source));
            } else {
                remove[consumer_idx] = true;
                replacements.insert(output, source);
            }
        }

        if replacements.is_empty() && output_identities.is_empty() {
            return false;
        }
        Self::replace_node_inputs(graph, &replacements);
        for (idx, input) in output_identities {
            graph.node[idx].op_type = "Identity".to_string();
            graph.node[idx].input = vec![input];
            graph.node[idx].attribute.clear();
        }
        let mut idx = 0;
        graph.node.retain(|_| {
            let keep = !remove[idx];
            idx += 1;
            keep
        });
        true
    }

    fn dedup_byte_identical_initializers(graph: &mut onnx::GraphProto) -> bool {
        let graph_outputs = Self::graph_output_names(graph);
        let mut canonical_by_content: HashMap<Vec<u8>, String> = HashMap::new();
        let mut replacements = HashMap::new();
        let mut deduped = Vec::with_capacity(graph.initializer.len());

        for initializer in graph.initializer.drain(..) {
            if graph_outputs.contains(&initializer.name) {
                deduped.push(initializer);
                continue;
            }
            let key = Self::initializer_content_key(&initializer);
            if let Some(canonical_name) = canonical_by_content.get(&key) {
                replacements.insert(initializer.name, canonical_name.clone());
            } else {
                canonical_by_content.insert(key, initializer.name.clone());
                deduped.push(initializer);
            }
        }

        graph.initializer = deduped;
        if replacements.is_empty() {
            return false;
        }
        Self::replace_node_inputs(graph, &replacements);
        true
    }

    fn prune_dead_nodes_checked(graph: &mut onnx::GraphProto, depth: usize) -> anyhow::Result<()> {
        Self::ensure_export_depth(depth)?;
        for node in &mut graph.node {
            for attr in &mut node.attribute {
                if let Some(subgraph) = attr.g.as_mut() {
                    Self::prune_dead_nodes_checked(subgraph, depth + 1)?;
                }
                for subgraph in &mut attr.graphs {
                    Self::prune_dead_nodes_checked(subgraph, depth + 1)?;
                }
            }
        }
        while Self::prune_dead_nodes_once(graph) {}
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn prune_dead_nodes(graph: &mut onnx::GraphProto) {
        Self::prune_dead_nodes_checked(graph, 0).expect("prune_dead_nodes depth guard failed");
    }

    fn is_pruneable_helper_node(node: &onnx::NodeProto) -> bool {
        matches!(
            node.op_type.as_str(),
            "Shape" | "Cast" | "Unsqueeze" | "Squeeze" | "Concat" | "Identity"
        )
    }

    fn add_node_dependencies_to_live(node: &onnx::NodeProto, live: &mut HashSet<String>) {
        for input in &node.input {
            if !input.is_empty() {
                live.insert(input.clone());
            }
        }
        for attr in &node.attribute {
            if let Some(subgraph) = attr.g.as_ref() {
                live.extend(Self::collect_referenced_value_names(subgraph));
            }
            for subgraph in &attr.graphs {
                live.extend(Self::collect_referenced_value_names(subgraph));
            }
        }
    }

    fn prune_dead_nodes_once(graph: &mut onnx::GraphProto) -> bool {
        // This reverse walk expects topological order for transitive liveness.
        // Existing export may append non-pruneable Constants after consumers,
        // but helper-only pruning keeps those order violations from being
        // removed incorrectly.
        let mut live = graph
            .output
            .iter()
            .map(|output| output.name.clone())
            .collect::<HashSet<_>>();
        let mut keep = vec![true; graph.node.len()];

        for (idx, node) in graph.node.iter().enumerate().rev() {
            let output_is_live = node
                .output
                .iter()
                .any(|output| !output.is_empty() && live.contains(output));
            if output_is_live || !Self::is_pruneable_helper_node(node) {
                Self::add_node_dependencies_to_live(node, &mut live);
            } else {
                keep[idx] = false;
            }
        }

        if keep.iter().all(|keep| *keep) {
            return false;
        }
        let mut idx = 0;
        graph.node.retain(|_| {
            let retain = keep[idx];
            idx += 1;
            retain
        });
        true
    }

    fn prune_unused_initializers(graph: &mut onnx::GraphProto) {
        let mut used_initializer_names = Self::collect_referenced_value_names(graph);
        used_initializer_names.extend(graph.output.iter().map(|output| output.name.clone()));
        graph
            .initializer
            .retain(|tensor| used_initializer_names.contains(&tensor.name));
    }

    fn prune_unused_initializers_checked(
        graph: &mut onnx::GraphProto,
        depth: usize,
    ) -> anyhow::Result<()> {
        Self::ensure_export_depth(depth)?;
        for node in &mut graph.node {
            for attr in &mut node.attribute {
                if let Some(subgraph) = attr.g.as_mut() {
                    Self::prune_unused_initializers_checked(subgraph, depth + 1)?;
                }
                for subgraph in &mut attr.graphs {
                    Self::prune_unused_initializers_checked(subgraph, depth + 1)?;
                }
            }
        }
        Self::prune_unused_initializers(graph);
        Ok(())
    }

    pub(crate) fn prepare_graph_for_export(graph: &mut onnx::GraphProto) -> anyhow::Result<()> {
        Self::sanitize_graph_checked(graph, true, 0)?;
        Self::canonicalize_graph_checked(graph, 0)?;
        Self::prune_dead_nodes_checked(graph, 0)?;
        Self::prune_unused_initializers_checked(graph, 0)?;
        Ok(())
    }

    /// Finalizes and writes the ONNX model.
    ///
    /// Export intentionally runs narrow cleanup passes before validation:
    /// sanitization fixes structural initializer/input issues, canonicalization
    /// removes local post-lowering noise, and dead helper nodes are pruned before
    /// reference checks.
    pub fn export_onnx(&mut self, output_path: &str, opset_version: i64) -> anyhow::Result<()> {
        Self::prepare_graph_for_export(&mut self.onnx_graph)?;
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
