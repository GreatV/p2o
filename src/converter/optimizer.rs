use half::f16;
use std::collections::{HashMap, HashSet};

use crate::helper::{self, dt};
use crate::proto::onnx;

const MAX_OPTIMIZER_SUBGRAPH_DEPTH: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OptimizationLevel {
    None,
    #[default]
    Basic,
}

#[derive(Clone, Debug)]
pub(crate) struct OptimizerOptions {
    pub(crate) level: OptimizationLevel,
    pub(crate) target_opset: i64,
    pub(crate) max_iterations: usize,
    pub(crate) passes: OptimizerPasses,
    pub(crate) shapes: HashMap<String, Vec<i64>>,
}

impl Default for OptimizerOptions {
    fn default() -> Self {
        Self {
            level: OptimizationLevel::Basic,
            target_opset: crate::converter::DEFAULT_OPSET,
            max_iterations: 8,
            passes: OptimizerPasses::default(),
            shapes: HashMap::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct OptimizerPasses {
    pub(crate) flush_subnormal_initializers: bool,
    pub(crate) fuse_constant_reshape: bool,
    pub(crate) fuse_constant_unsqueeze: bool,
    pub(crate) fuse_constant_cast: bool,
    pub(crate) eliminate_arithmetic_identity: bool,
    pub(crate) optimize_transpose: bool,
    pub(crate) materialize_conv_kernel_shape: bool,
    pub(crate) fuse_conv_add_bias: bool,
    pub(crate) fuse_conv_batch_norm: bool,
    pub(crate) fuse_matmul_add_bias: bool,
}

impl Default for OptimizerPasses {
    fn default() -> Self {
        Self {
            // Off by default: zeroing subnormal float constants is a lossy
            // transform (an epsilon/denominator/threshold can change), so it
            // must be opted into rather than applied to a plain `--optimize basic`.
            flush_subnormal_initializers: false,
            fuse_constant_reshape: true,
            fuse_constant_unsqueeze: true,
            fuse_constant_cast: true,
            eliminate_arithmetic_identity: true,
            optimize_transpose: true,
            materialize_conv_kernel_shape: true,
            fuse_conv_add_bias: true,
            fuse_conv_batch_norm: true,
            fuse_matmul_add_bias: true,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OptimizationReport {
    pub iterations: usize,
    pub nodes_before: usize,
    pub nodes_after: usize,
    pub initializers_before: usize,
    pub initializers_after: usize,
    pub pass_changes: Vec<PassChange>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassChange {
    pub pass: &'static str,
    pub changes: usize,
}

impl OptimizationReport {
    fn merge(&mut self, child: OptimizationReport) {
        self.iterations = self.iterations.max(child.iterations);
        self.nodes_before += child.nodes_before;
        self.nodes_after += child.nodes_after;
        self.initializers_before += child.initializers_before;
        self.initializers_after += child.initializers_after;
        for change in child.pass_changes {
            self.record(change.pass, change.changes);
        }
    }

    fn record(&mut self, pass: &'static str, changes: usize) {
        if changes == 0 {
            return;
        }
        if let Some(existing) = self
            .pass_changes
            .iter_mut()
            .find(|change| change.pass == pass)
        {
            existing.changes += changes;
        } else {
            self.pass_changes.push(PassChange { pass, changes });
        }
    }

    pub fn changed(&self) -> bool {
        self.pass_changes.iter().any(|change| change.changes > 0)
    }
}

#[derive(Clone, Debug)]
struct GraphFacts {
    producers: HashMap<String, usize>,
    uses: HashMap<String, usize>,
    initializers: HashMap<String, usize>,
    graph_outputs: HashSet<String>,
    shapes: HashMap<String, Vec<i64>>,
    dtypes: HashMap<String, i32>,
}

#[derive(Clone, Copy, Debug)]
enum ConstLocation {
    Initializer(usize),
    ConstantNode(usize),
}

#[derive(Clone, Debug)]
struct ConstValue {
    location: ConstLocation,
    tensor: onnx::TensorProto,
}

pub(crate) struct GraphOptimizer;

impl GraphOptimizer {
    pub(crate) fn optimize_graph(
        graph: &mut onnx::GraphProto,
        options: &OptimizerOptions,
    ) -> anyhow::Result<OptimizationReport> {
        if options.level == OptimizationLevel::None {
            return Ok(OptimizationReport {
                nodes_before: graph.node.len(),
                nodes_after: graph.node.len(),
                initializers_before: graph.initializer.len(),
                initializers_after: graph.initializer.len(),
                ..Default::default()
            });
        }
        Self::optimize_graph_checked(graph, options, 0)
    }

    fn optimize_graph_checked(
        graph: &mut onnx::GraphProto,
        options: &OptimizerOptions,
        depth: usize,
    ) -> anyhow::Result<OptimizationReport> {
        if depth > MAX_OPTIMIZER_SUBGRAPH_DEPTH {
            anyhow::bail!(
                "maximum ONNX subgraph optimizer depth {} exceeded",
                MAX_OPTIMIZER_SUBGRAPH_DEPTH
            );
        }

        let mut report = OptimizationReport {
            nodes_before: graph.node.len(),
            initializers_before: graph.initializer.len(),
            ..Default::default()
        };

        let subgraph_options = Self::subgraph_options(options);
        for node in &mut graph.node {
            for attr in &mut node.attribute {
                if let Some(subgraph) = attr.g.as_mut() {
                    report.merge(Self::optimize_graph_checked(
                        subgraph,
                        &subgraph_options,
                        depth + 1,
                    )?);
                }
                for subgraph in &mut attr.graphs {
                    report.merge(Self::optimize_graph_checked(
                        subgraph,
                        &subgraph_options,
                        depth + 1,
                    )?);
                }
            }
        }

        for iteration in 0..options.max_iterations {
            let mut changed = false;
            if options.passes.flush_subnormal_initializers {
                changed |= Self::run_pass(
                    graph,
                    options,
                    &mut report,
                    "flush_subnormal_initializers",
                    Self::flush_subnormal_initializers,
                )?;
            }
            if options.passes.fuse_constant_reshape {
                changed |= Self::run_pass(
                    graph,
                    options,
                    &mut report,
                    "fuse_constant_reshape",
                    Self::fuse_constant_reshape,
                )?;
            }
            if options.passes.fuse_constant_unsqueeze {
                changed |= Self::run_pass(
                    graph,
                    options,
                    &mut report,
                    "fuse_constant_unsqueeze",
                    Self::fuse_constant_unsqueeze,
                )?;
            }
            if options.passes.fuse_constant_cast {
                changed |= Self::run_pass(
                    graph,
                    options,
                    &mut report,
                    "fuse_constant_cast",
                    Self::fuse_constant_cast,
                )?;
            }
            if options.passes.eliminate_arithmetic_identity {
                changed |= Self::run_pass(
                    graph,
                    options,
                    &mut report,
                    "eliminate_arithmetic_identity",
                    Self::eliminate_arithmetic_identity,
                )?;
            }
            if options.passes.optimize_transpose {
                changed |= Self::run_pass(
                    graph,
                    options,
                    &mut report,
                    "optimize_transpose",
                    Self::optimize_transpose,
                )?;
            }
            if options.passes.materialize_conv_kernel_shape {
                changed |= Self::run_pass(
                    graph,
                    options,
                    &mut report,
                    "materialize_conv_kernel_shape",
                    Self::materialize_conv_kernel_shape,
                )?;
            }
            if options.passes.fuse_conv_add_bias {
                changed |= Self::run_pass(
                    graph,
                    options,
                    &mut report,
                    "fuse_conv_add_bias",
                    Self::fuse_conv_add_bias,
                )?;
            }
            if options.passes.fuse_conv_batch_norm {
                changed |= Self::run_pass(
                    graph,
                    options,
                    &mut report,
                    "fuse_conv_batch_norm",
                    Self::fuse_conv_batch_norm,
                )?;
            }
            if options.passes.fuse_matmul_add_bias {
                changed |= Self::run_pass(
                    graph,
                    options,
                    &mut report,
                    "fuse_matmul_add_bias",
                    Self::fuse_matmul_add_bias,
                )?;
            }

            if !changed {
                report.iterations = iteration + 1;
                break;
            }
            report.iterations = iteration + 1;
        }

        // `merge` above already folded in each subgraph's before/after counts;
        // add this graph's own counts so the `*_after` totals stay consistent
        // with `*_before` rather than overwriting away the subgraph contributions.
        report.nodes_after += graph.node.len();
        report.initializers_after += graph.initializer.len();
        Ok(report)
    }

    fn subgraph_options(options: &OptimizerOptions) -> OptimizerOptions {
        let mut child = options.clone();
        child.shapes.clear();
        child
    }

    fn run_pass(
        graph: &mut onnx::GraphProto,
        options: &OptimizerOptions,
        report: &mut OptimizationReport,
        name: &'static str,
        pass: fn(&mut onnx::GraphProto, &OptimizerOptions) -> anyhow::Result<usize>,
    ) -> anyhow::Result<bool> {
        let changes = pass(graph, options)?;
        report.record(name, changes);
        Ok(changes > 0)
    }

    fn analyze(graph: &onnx::GraphProto, options: &OptimizerOptions) -> GraphFacts {
        let mut producers = HashMap::new();
        for (idx, node) in graph.node.iter().enumerate() {
            for output in &node.output {
                if !output.is_empty() {
                    producers.insert(output.clone(), idx);
                }
            }
        }

        let mut uses = HashMap::new();
        Self::add_input_uses(graph, &mut uses);

        let initializers = graph
            .initializer
            .iter()
            .enumerate()
            .map(|(idx, tensor)| (tensor.name.clone(), idx))
            .collect();
        let graph_outputs = graph
            .output
            .iter()
            .map(|output| output.name.clone())
            .collect::<HashSet<_>>();
        let mut shapes = options.shapes.clone();
        Self::add_value_info_shapes(&mut shapes, &graph.input);
        Self::add_value_info_shapes(&mut shapes, &graph.value_info);
        Self::add_value_info_shapes(&mut shapes, &graph.output);
        let mut dtypes = graph
            .initializer
            .iter()
            .map(|tensor| (tensor.name.clone(), tensor.data_type))
            .collect::<HashMap<_, _>>();
        Self::add_value_info_dtypes(&mut dtypes, &graph.input);
        Self::add_value_info_dtypes(&mut dtypes, &graph.value_info);
        Self::add_value_info_dtypes(&mut dtypes, &graph.output);

        GraphFacts {
            producers,
            uses,
            initializers,
            graph_outputs,
            shapes,
            dtypes,
        }
    }

    fn add_input_uses(graph: &onnx::GraphProto, uses: &mut HashMap<String, usize>) {
        for node in &graph.node {
            for input in &node.input {
                if !input.is_empty() {
                    *uses.entry(input.clone()).or_default() += 1;
                }
            }
            for attr in &node.attribute {
                if let Some(subgraph) = attr.g.as_ref() {
                    Self::add_input_uses(subgraph, uses);
                }
                for subgraph in &attr.graphs {
                    Self::add_input_uses(subgraph, uses);
                }
            }
        }
    }

    fn add_value_info_shapes(
        shapes: &mut HashMap<String, Vec<i64>>,
        value_infos: &[onnx::ValueInfoProto],
    ) {
        for value_info in value_infos {
            let Some(r#type) = value_info.r#type.as_ref() else {
                continue;
            };
            let Some(onnx::type_proto::Value::TensorType(tensor)) = r#type.value.as_ref() else {
                continue;
            };
            let Some(shape) = tensor.shape.as_ref() else {
                continue;
            };
            let dims = shape
                .dim
                .iter()
                .map(|dim| match dim.value.as_ref() {
                    Some(onnx::tensor_shape_proto::dimension::Value::DimValue(value)) => *value,
                    _ => -1,
                })
                .collect::<Vec<_>>();
            shapes.insert(value_info.name.clone(), dims);
        }
    }

    fn add_value_info_dtypes(
        dtypes: &mut HashMap<String, i32>,
        value_infos: &[onnx::ValueInfoProto],
    ) {
        for value_info in value_infos {
            let Some(r#type) = value_info.r#type.as_ref() else {
                continue;
            };
            let Some(onnx::type_proto::Value::TensorType(tensor)) = r#type.value.as_ref() else {
                continue;
            };
            dtypes.insert(value_info.name.clone(), tensor.elem_type);
        }
    }

    fn replace_inputs_recursive(graph: &mut onnx::GraphProto, from: &str, to: &str) {
        for node in &mut graph.node {
            for input in &mut node.input {
                if input == from {
                    *input = to.to_string();
                }
            }
            for attr in &mut node.attribute {
                if let Some(subgraph) = attr.g.as_mut()
                    && !Self::subgraph_shadows(subgraph, from)
                    && !Self::subgraph_shadows(subgraph, to)
                {
                    Self::replace_inputs_recursive(subgraph, from, to);
                }
                for subgraph in &mut attr.graphs {
                    if !Self::subgraph_shadows(subgraph, from)
                        && !Self::subgraph_shadows(subgraph, to)
                    {
                        Self::replace_inputs_recursive(subgraph, from, to);
                    }
                }
            }
        }
    }

    /// Whether `name` is rebound inside `graph` (as a formal input, initializer,
    /// formal output, or locally produced value). ONNX scoping makes such a definition shadow
    /// any outer value of the same name, so a parent-graph rewrite must not
    /// recurse into the subgraph for that name.
    fn subgraph_shadows(graph: &onnx::GraphProto, name: &str) -> bool {
        graph.input.iter().any(|value| value.name == name)
            || graph.output.iter().any(|value| value.name == name)
            || graph.initializer.iter().any(|tensor| tensor.name == name)
            || graph
                .node
                .iter()
                .any(|node| node.output.iter().any(|output| output == name))
    }

    fn graph_uses_outer_name(graph: &onnx::GraphProto, name: &str) -> bool {
        for node in &graph.node {
            if node.input.iter().any(|input| input == name) {
                return true;
            }
            for attr in &node.attribute {
                if let Some(subgraph) = attr.g.as_ref()
                    && !Self::subgraph_shadows(subgraph, name)
                    && Self::graph_uses_outer_name(subgraph, name)
                {
                    return true;
                }
                for subgraph in &attr.graphs {
                    if !Self::subgraph_shadows(subgraph, name)
                        && Self::graph_uses_outer_name(subgraph, name)
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn rewrite_is_safe(graph: &onnx::GraphProto, from: &str, to: &str) -> bool {
        if from == to {
            return false;
        }
        Self::rewrite_is_safe_in_subgraphs(graph, from, to)
    }

    fn rewrite_is_safe_in_subgraphs(graph: &onnx::GraphProto, from: &str, to: &str) -> bool {
        for node in &graph.node {
            for attr in &node.attribute {
                if let Some(subgraph) = attr.g.as_ref()
                    && !Self::subgraph_rewrite_is_safe(subgraph, from, to)
                {
                    return false;
                }
                for subgraph in &attr.graphs {
                    if !Self::subgraph_rewrite_is_safe(subgraph, from, to) {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn subgraph_rewrite_is_safe(graph: &onnx::GraphProto, from: &str, to: &str) -> bool {
        if Self::subgraph_shadows(graph, from) {
            return true;
        }
        if Self::subgraph_shadows(graph, to) && Self::graph_uses_outer_name(graph, from) {
            return false;
        }
        Self::rewrite_is_safe_in_subgraphs(graph, from, to)
    }

    fn apply_rewrites_recursive(
        graph: &mut onnx::GraphProto,
        rewrites: &[(String, String)],
    ) -> HashSet<String> {
        let rewrite_map = rewrites
            .iter()
            .cloned()
            .collect::<HashMap<String, String>>();
        let mut applied = HashSet::new();
        for (from, to) in rewrites {
            let resolved = Self::resolve_rewrite_target(to, &rewrite_map);
            if from != &resolved && Self::rewrite_is_safe(graph, from, &resolved) {
                Self::replace_inputs_recursive(graph, from, &resolved);
                applied.insert(from.clone());
            }
        }
        applied
    }

    fn mark_applied_rewrite_removals(
        remove: &mut [bool],
        rewrite_removals: &[(usize, String)],
        applied_rewrites: &HashSet<String>,
    ) {
        for (idx, output) in rewrite_removals {
            if applied_rewrites.contains(output) {
                remove[*idx] = true;
            }
        }
    }

    fn resolve_rewrite_target(name: &str, rewrite_map: &HashMap<String, String>) -> String {
        let mut current = name;
        let mut seen = HashSet::new();
        while seen.insert(current.to_string()) {
            let Some(next) = rewrite_map.get(current) else {
                return current.to_string();
            };
            current = next;
        }
        name.to_string()
    }

    fn remove_nodes(graph: &mut onnx::GraphProto, remove: &[bool]) {
        let mut idx = 0;
        graph.node.retain(|_| {
            let keep = !remove[idx];
            idx += 1;
            keep
        });
    }

    fn attr_int(node: &onnx::NodeProto, name: &str) -> Option<i64> {
        node.attribute
            .iter()
            .find(|attr| attr.name == name)
            .map(|attr| attr.i)
    }

    fn flush_subnormal_initializers(
        graph: &mut onnx::GraphProto,
        _options: &OptimizerOptions,
    ) -> anyhow::Result<usize> {
        let mut changed = 0;
        for tensor in &mut graph.initializer {
            if tensor.data_type != dt::FLOAT {
                continue;
            }

            let mut tensor_changed = false;
            for value in &mut tensor.float_data {
                if *value != 0.0 && value.abs() < f32::MIN_POSITIVE {
                    *value = 0.0;
                    tensor_changed = true;
                }
            }

            if !tensor.raw_data.is_empty() {
                if tensor.raw_data.len() % std::mem::size_of::<f32>() != 0 {
                    continue;
                }
                for chunk in tensor.raw_data.chunks_exact_mut(std::mem::size_of::<f32>()) {
                    let mut bytes = [0_u8; std::mem::size_of::<f32>()];
                    bytes.copy_from_slice(chunk);
                    let value = f32::from_le_bytes(bytes);
                    if value != 0.0 && value.abs() < f32::MIN_POSITIVE {
                        chunk.copy_from_slice(&0.0_f32.to_le_bytes());
                        tensor_changed = true;
                    }
                }
            }

            changed += usize::from(tensor_changed);
        }
        Ok(changed)
    }

    fn attr_ints(node: &onnx::NodeProto, name: &str) -> Option<Vec<i64>> {
        node.attribute
            .iter()
            .find(|attr| attr.name == name)
            .map(|attr| attr.ints.clone())
    }

    fn set_attr_ints(node: &mut onnx::NodeProto, name: &str, values: Vec<i64>) {
        if let Some(attr) = node.attribute.iter_mut().find(|attr| attr.name == name) {
            attr.ints = values;
            return;
        }
        node.attribute.push(helper::attr_ints(name, &values));
    }

    fn constant_tensor_from_node(node: &onnx::NodeProto) -> Option<onnx::TensorProto> {
        if node.op_type != "Constant" {
            return None;
        }
        node.attribute
            .iter()
            .find(|attr| attr.name == "value")
            .and_then(|attr| attr.t.clone())
    }

    fn set_constant_tensor_on_node(node: &mut onnx::NodeProto, tensor: onnx::TensorProto) {
        if let Some(attr) = node.attribute.iter_mut().find(|attr| attr.name == "value") {
            attr.t = Some(tensor);
            return;
        }
        node.attribute.push(helper::attr_tensor("value", tensor));
    }

    fn const_value(graph: &onnx::GraphProto, facts: &GraphFacts, name: &str) -> Option<ConstValue> {
        if let Some(&idx) = facts.initializers.get(name) {
            return Some(ConstValue {
                location: ConstLocation::Initializer(idx),
                tensor: graph.initializer[idx].clone(),
            });
        }
        let &idx = facts.producers.get(name)?;
        let tensor = Self::constant_tensor_from_node(&graph.node[idx])?;
        Some(ConstValue {
            location: ConstLocation::ConstantNode(idx),
            tensor,
        })
    }

    fn set_const_value(graph: &mut onnx::GraphProto, value: ConstValue) {
        match value.location {
            ConstLocation::Initializer(idx) => {
                graph.initializer[idx] = value.tensor;
            }
            ConstLocation::ConstantNode(idx) => {
                Self::set_constant_tensor_on_node(&mut graph.node[idx], value.tensor);
            }
        }
    }

    fn single_use(facts: &GraphFacts, name: &str) -> bool {
        // A value exported as a graph output counts as an extra use: folding it
        // in place or rewiring its sole consumer would change what the graph
        // exposes, so such values are never treated as single-use.
        !facts.graph_outputs.contains(name)
            && facts.uses.get(name).copied().unwrap_or_default() == 1
    }

    fn static_i64_tensor(tensor: &onnx::TensorProto) -> Option<Vec<i64>> {
        match tensor.data_type {
            dt::INT64 => {
                if !tensor.int64_data.is_empty() {
                    return Some(tensor.int64_data.clone());
                }
                if !tensor
                    .raw_data
                    .len()
                    .is_multiple_of(std::mem::size_of::<i64>())
                {
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
            dt::INT32 => {
                if !tensor.int32_data.is_empty() {
                    return Some(
                        tensor
                            .int32_data
                            .iter()
                            .map(|&value| value as i64)
                            .collect(),
                    );
                }
                if !tensor
                    .raw_data
                    .len()
                    .is_multiple_of(std::mem::size_of::<i32>())
                {
                    return None;
                }
                Some(
                    tensor
                        .raw_data
                        .chunks_exact(std::mem::size_of::<i32>())
                        .map(|chunk| {
                            let mut bytes = [0_u8; std::mem::size_of::<i32>()];
                            bytes.copy_from_slice(chunk);
                            i32::from_le_bytes(bytes) as i64
                        })
                        .collect(),
                )
            }
            _ => None,
        }
    }

    fn static_i64_const(
        graph: &onnx::GraphProto,
        facts: &GraphFacts,
        name: &str,
    ) -> Option<Vec<i64>> {
        let value = Self::const_value(graph, facts, name)?;
        Self::static_i64_tensor(&value.tensor)
    }

    fn tensor_shape(graph: &onnx::GraphProto, facts: &GraphFacts, name: &str) -> Option<Vec<i64>> {
        facts
            .shapes
            .get(name)
            .cloned()
            .or_else(|| Self::const_value(graph, facts, name).map(|value| value.tensor.dims))
    }

    fn tensor_dtype(graph: &onnx::GraphProto, facts: &GraphFacts, name: &str) -> Option<i32> {
        facts
            .dtypes
            .get(name)
            .copied()
            .or_else(|| Self::const_value(graph, facts, name).map(|value| value.tensor.data_type))
    }

    fn gemm_supported_dtype(dtype: i32, target_opset: i64) -> bool {
        matches!(dtype, dt::FLOAT16 | dt::FLOAT | dt::DOUBLE)
            || (target_opset >= 13 && dtype == dt::BFLOAT16)
    }

    fn is_floating_dtype(dtype: i32) -> bool {
        matches!(dtype, dt::FLOAT | dt::FLOAT16 | dt::DOUBLE | dt::BFLOAT16)
    }

    fn static_axes(
        graph: &onnx::GraphProto,
        facts: &GraphFacts,
        node: &onnx::NodeProto,
    ) -> Option<Vec<i64>> {
        if let Some(axes) = Self::attr_ints(node, "axes") {
            return Some(axes);
        }
        let axes = node.input.get(1)?;
        if axes.is_empty() {
            return None;
        }
        Self::static_i64_const(graph, facts, axes)
    }

    fn reshape_dims(input_dims: &[i64], target: &[i64], allowzero: i64) -> Option<Vec<i64>> {
        let mut output = target.to_vec();
        for (idx, dim) in output.iter_mut().enumerate() {
            if *dim == 0 && allowzero == 0 {
                *dim = *input_dims.get(idx)?;
            }
        }
        let unknowns = output.iter().filter(|&&dim| dim == -1).count();
        if unknowns > 1 {
            return None;
        }
        if unknowns == 1 {
            let input_numel = Self::static_numel(input_dims)?;
            let known_numel =
                output
                    .iter()
                    .filter(|&&dim| dim != -1)
                    .try_fold(
                        1_i64,
                        |acc, &dim| {
                            if dim < 0 { None } else { acc.checked_mul(dim) }
                        },
                    )?;
            if known_numel == 0 || input_numel % known_numel != 0 {
                return None;
            }
            let inferred = input_numel / known_numel;
            let idx = output.iter().position(|&dim| dim == -1)?;
            output[idx] = inferred;
        }
        Some(output)
    }

    fn static_numel(dims: &[i64]) -> Option<i64> {
        dims.iter().try_fold(
            1_i64,
            |acc, &dim| {
                if dim < 0 { None } else { acc.checked_mul(dim) }
            },
        )
    }

    fn fuse_constant_reshape(
        graph: &mut onnx::GraphProto,
        options: &OptimizerOptions,
    ) -> anyhow::Result<usize> {
        let facts = Self::analyze(graph, options);
        let mut remove = vec![false; graph.node.len()];
        let mut rewrites = Vec::new();
        let mut rewrite_removals = Vec::new();
        let mut updates = Vec::new();
        let mut new_initializers = Vec::new();

        for (idx, node) in graph.node.iter().enumerate() {
            if node.op_type != "Reshape" || node.input.is_empty() || node.output.len() != 1 {
                continue;
            }
            let output = &node.output[0];
            if facts.graph_outputs.contains(output) {
                continue;
            }
            let input_name = &node.input[0];
            if input_name.is_empty() {
                continue;
            }
            let Some(mut value) = Self::const_value(graph, &facts, input_name) else {
                continue;
            };
            let Some(target_shape) = Self::attr_ints(node, "shape").or_else(|| {
                node.input
                    .get(1)
                    .and_then(|name| Self::static_i64_const(graph, &facts, name))
            }) else {
                continue;
            };
            let allowzero = Self::attr_int(node, "allowzero").unwrap_or(0);
            let Some(new_dims) = Self::reshape_dims(&value.tensor.dims, &target_shape, allowzero)
            else {
                continue;
            };
            let rewrite_is_safe = Self::rewrite_is_safe(graph, output, input_name);
            value.tensor.dims = new_dims;
            if Self::single_use(&facts, input_name) {
                if !rewrite_is_safe {
                    continue;
                }
                updates.push(value);
                rewrites.push((output.clone(), input_name.clone()));
                rewrite_removals.push((idx, output.clone()));
            } else if matches!(value.location, ConstLocation::Initializer(_)) {
                value.tensor.name = output.clone();
                new_initializers.push(value.tensor);
                remove[idx] = true;
            } else {
                continue;
            }
        }

        // Shared-initializer clones also remove a Reshape node but are not in
        // `rewrites`; count them so the pass reports a change and the fixpoint
        // loop keeps iterating instead of stopping early.
        let cloned = new_initializers.len();
        for update in updates {
            Self::set_const_value(graph, update);
        }
        graph.initializer.extend(new_initializers);
        let applied = Self::apply_rewrites_recursive(graph, &rewrites);
        Self::mark_applied_rewrite_removals(&mut remove, &rewrite_removals, &applied);
        Self::remove_nodes(graph, &remove);
        Ok(applied.len() + cloned)
    }

    fn unsqueeze_dims(input_dims: &[i64], axes: &[i64]) -> Option<Vec<i64>> {
        let output_rank = input_dims.len() + axes.len();
        let mut normalized = Vec::with_capacity(axes.len());
        for &axis in axes {
            let axis = if axis < 0 {
                axis + output_rank as i64
            } else {
                axis
            };
            if axis < 0 || axis as usize >= output_rank {
                return None;
            }
            normalized.push(axis as usize);
        }
        normalized.sort_unstable();
        normalized.dedup();
        if normalized.len() != axes.len() {
            return None;
        }
        let mut output = input_dims.to_vec();
        for axis in normalized {
            output.insert(axis, 1);
        }
        Some(output)
    }

    fn fuse_constant_unsqueeze(
        graph: &mut onnx::GraphProto,
        options: &OptimizerOptions,
    ) -> anyhow::Result<usize> {
        let facts = Self::analyze(graph, options);
        let mut remove = vec![false; graph.node.len()];
        let mut rewrites = Vec::new();
        let mut rewrite_removals = Vec::new();
        let mut updates = Vec::new();

        for (idx, node) in graph.node.iter().enumerate() {
            if node.op_type != "Unsqueeze" || node.input.is_empty() || node.output.len() != 1 {
                continue;
            }
            let output = &node.output[0];
            if facts.graph_outputs.contains(output) {
                continue;
            }
            let input_name = &node.input[0];
            if input_name.is_empty() || !Self::single_use(&facts, input_name) {
                continue;
            }
            let Some(mut value) = Self::const_value(graph, &facts, input_name) else {
                continue;
            };
            let Some(axes) = Self::static_axes(graph, &facts, node) else {
                continue;
            };
            let Some(new_dims) = Self::unsqueeze_dims(&value.tensor.dims, &axes) else {
                continue;
            };
            if !Self::rewrite_is_safe(graph, output, input_name) {
                continue;
            }
            value.tensor.dims = new_dims;
            updates.push(value);
            rewrites.push((output.clone(), input_name.clone()));
            rewrite_removals.push((idx, output.clone()));
        }

        for update in updates {
            Self::set_const_value(graph, update);
        }
        let applied = Self::apply_rewrites_recursive(graph, &rewrites);
        Self::mark_applied_rewrite_removals(&mut remove, &rewrite_removals, &applied);
        Self::remove_nodes(graph, &remove);
        Ok(applied.len())
    }

    fn tensor_to_f64(tensor: &onnx::TensorProto) -> Option<Vec<f64>> {
        match tensor.data_type {
            dt::FLOAT => {
                if !tensor.float_data.is_empty() {
                    return Some(
                        tensor
                            .float_data
                            .iter()
                            .map(|&value| value as f64)
                            .collect(),
                    );
                }
                Self::raw_chunks::<4>(&tensor.raw_data).map(|chunks| {
                    chunks
                        .into_iter()
                        .map(|bytes| f32::from_le_bytes(bytes) as f64)
                        .collect()
                })
            }
            dt::DOUBLE => {
                if !tensor.double_data.is_empty() {
                    return Some(tensor.double_data.clone());
                }
                Self::raw_chunks::<8>(&tensor.raw_data)
                    .map(|chunks| chunks.into_iter().map(f64::from_le_bytes).collect())
            }
            dt::FLOAT16 => Self::raw_chunks::<2>(&tensor.raw_data).map(|chunks| {
                chunks
                    .into_iter()
                    .map(|bytes| f16::from_bits(u16::from_le_bytes(bytes)).to_f32() as f64)
                    .collect()
            }),
            dt::INT64 => Self::static_i64_tensor(tensor)
                .map(|values| values.into_iter().map(|value| value as f64).collect()),
            dt::INT32 => {
                if !tensor.int32_data.is_empty() {
                    return Some(
                        tensor
                            .int32_data
                            .iter()
                            .map(|&value| value as f64)
                            .collect(),
                    );
                }
                Self::raw_chunks::<4>(&tensor.raw_data).map(|chunks| {
                    chunks
                        .into_iter()
                        .map(|bytes| i32::from_le_bytes(bytes) as f64)
                        .collect()
                })
            }
            dt::BOOL => {
                // ONNX stores BOOL values in int32_data when raw_data is empty
                // (same field family as INT8/16/32), so check it first, matching
                // the INT32 arm above.
                if !tensor.int32_data.is_empty() {
                    return Some(
                        tensor
                            .int32_data
                            .iter()
                            .map(|&value| f64::from(value != 0))
                            .collect(),
                    );
                }
                Some(
                    tensor
                        .raw_data
                        .iter()
                        .map(|&value| f64::from(value != 0))
                        .collect(),
                )
            }
            _ => None,
        }
    }

    fn raw_chunks<const N: usize>(raw_data: &[u8]) -> Option<Vec<[u8; N]>> {
        if !raw_data.len().is_multiple_of(N) {
            return None;
        }
        raw_data
            .chunks_exact(N)
            .map(|chunk| chunk.try_into().ok())
            .collect()
    }

    fn encode_f64_values(values: &[f64], data_type: i32) -> Option<Vec<u8>> {
        let mut raw = Vec::new();
        for &value in values {
            match data_type {
                dt::BOOL => {
                    if value.is_nan() {
                        return None;
                    }
                    raw.push(u8::from(value != 0.0));
                }
                dt::FLOAT16 => {
                    raw.extend_from_slice(&f16::from_f32(value as f32).to_bits().to_le_bytes());
                }
                dt::FLOAT => raw.extend_from_slice(&(value as f32).to_le_bytes()),
                dt::DOUBLE => raw.extend_from_slice(&value.to_le_bytes()),
                dt::INT32 => {
                    if !Self::is_integral_in_range(value, i32::MIN as f64, i32::MAX as f64) {
                        return None;
                    }
                    raw.extend_from_slice(&(value as i32).to_le_bytes());
                }
                dt::INT64 => {
                    if !Self::is_integral_in_range(value, i64::MIN as f64, i64::MAX as f64) {
                        return None;
                    }
                    raw.extend_from_slice(&(value as i64).to_le_bytes());
                }
                _ => return None,
            }
        }
        Some(raw)
    }

    fn is_integral_in_range(value: f64, min: f64, max: f64) -> bool {
        value.is_finite() && value.trunc() == value && value >= min && value <= max
    }

    fn fuse_constant_cast(
        graph: &mut onnx::GraphProto,
        options: &OptimizerOptions,
    ) -> anyhow::Result<usize> {
        let facts = Self::analyze(graph, options);
        let mut remove = vec![false; graph.node.len()];
        let mut rewrites = Vec::new();
        let mut rewrite_removals = Vec::new();
        let mut updates = Vec::new();

        for (idx, node) in graph.node.iter().enumerate() {
            if node.op_type != "Cast" || node.input.len() != 1 || node.output.len() != 1 {
                continue;
            }
            let output = &node.output[0];
            if facts.graph_outputs.contains(output) {
                continue;
            }
            let input_name = &node.input[0];
            if input_name.is_empty() || !Self::single_use(&facts, input_name) {
                continue;
            }
            let Some(target_dtype) = Self::attr_int(node, "to").map(|value| value as i32) else {
                continue;
            };
            let Some(mut value) = Self::const_value(graph, &facts, input_name) else {
                continue;
            };
            if value.tensor.data_type != target_dtype {
                let Some(values) = Self::tensor_to_f64(&value.tensor) else {
                    continue;
                };
                let Some(raw_data) = Self::encode_f64_values(&values, target_dtype) else {
                    continue;
                };
                value.tensor.data_type = target_dtype;
                value.tensor.raw_data = raw_data;
                value.tensor.float_data.clear();
                value.tensor.int32_data.clear();
                value.tensor.int64_data.clear();
                value.tensor.double_data.clear();
                value.tensor.uint64_data.clear();
            }
            if !Self::rewrite_is_safe(graph, output, input_name) {
                continue;
            }
            updates.push(value);
            rewrites.push((output.clone(), input_name.clone()));
            rewrite_removals.push((idx, output.clone()));
        }

        for update in updates {
            Self::set_const_value(graph, update);
        }
        let applied = Self::apply_rewrites_recursive(graph, &rewrites);
        Self::mark_applied_rewrite_removals(&mut remove, &rewrite_removals, &applied);
        Self::remove_nodes(graph, &remove);
        Ok(applied.len())
    }

    fn const_is_scalar_or_single_value(tensor: &onnx::TensorProto, expected: f64) -> bool {
        let numel = if tensor.dims.is_empty() {
            1
        } else {
            tensor.dims.iter().product::<i64>()
        };
        if numel != 1 {
            return false;
        }
        let Some(values) = Self::tensor_to_f64(tensor) else {
            return false;
        };
        values.len() == 1 && (values[0] - expected).abs() == 0.0
    }

    /// True when every element of the constant is negative zero (`-0.0`). Adding
    /// `-0.0` is the genuine IEEE-754 additive identity, so such a node can be
    /// folded for floating types while `x + (+0.0)` cannot.
    fn const_is_negative_zero(tensor: &onnx::TensorProto) -> bool {
        match Self::tensor_to_f64(tensor) {
            Some(values) => {
                !values.is_empty() && values.iter().all(|&v| v == 0.0 && v.is_sign_negative())
            }
            None => false,
        }
    }

    /// A single-element identity const (all dims == 1) is only safe to drop when
    /// broadcasting it into the passthrough does not raise the output rank, i.e.
    /// `rank(const) <= rank(passthrough)`. A rank-0 scalar never expands; for any
    /// higher rank the passthrough rank must be known, otherwise we keep the node.
    fn identity_preserves_shape(
        graph: &onnx::GraphProto,
        facts: &GraphFacts,
        const_tensor: &onnx::TensorProto,
        passthrough_name: &str,
    ) -> bool {
        let const_rank = const_tensor.dims.len();
        if const_rank == 0 {
            return true;
        }
        match Self::tensor_shape(graph, facts, passthrough_name) {
            Some(shape) => const_rank <= shape.len(),
            None => false,
        }
    }

    fn eliminate_arithmetic_identity(
        graph: &mut onnx::GraphProto,
        options: &OptimizerOptions,
    ) -> anyhow::Result<usize> {
        let facts = Self::analyze(graph, options);
        let mut remove = vec![false; graph.node.len()];
        let mut rewrites = Vec::new();
        let mut rewrite_removals = Vec::new();

        for (idx, node) in graph.node.iter().enumerate() {
            if !matches!(node.op_type.as_str(), "Add" | "Mul")
                || node.input.len() != 2
                || node.output.len() != 1
            {
                continue;
            }
            let output = &node.output[0];
            if facts.graph_outputs.contains(output) {
                continue;
            }
            let expected = if node.op_type == "Add" { 0.0 } else { 1.0 };
            for const_pos in 0..=1 {
                let const_name = &node.input[const_pos];
                let passthrough_name = &node.input[1 - const_pos];
                let Some(value) = Self::const_value(graph, &facts, const_name) else {
                    continue;
                };
                // IEEE-754's additive identity is -0.0, not +0.0: `x + (-0.0) == x` for
                // every x, but `x + (+0.0)` flushes a -0.0 operand to +0.0 (a downstream
                // `Reciprocal` would then turn -inf into +inf). So for floating types only
                // fold the additive identity when the constant is negative zero; integer
                // Add and `x * 1` (all dtypes) are unaffected.
                if node.op_type == "Add"
                    && Self::is_floating_dtype(value.tensor.data_type)
                    && !Self::const_is_negative_zero(&value.tensor)
                {
                    continue;
                }
                if Self::const_is_scalar_or_single_value(&value.tensor, expected)
                    && Self::identity_preserves_shape(
                        graph,
                        &facts,
                        &value.tensor,
                        passthrough_name,
                    )
                    && Self::rewrite_is_safe(graph, output, passthrough_name)
                {
                    rewrites.push((output.clone(), passthrough_name.clone()));
                    rewrite_removals.push((idx, output.clone()));
                    break;
                }
            }
        }

        let applied = Self::apply_rewrites_recursive(graph, &rewrites);
        Self::mark_applied_rewrite_removals(&mut remove, &rewrite_removals, &applied);
        Self::remove_nodes(graph, &remove);
        Ok(applied.len())
    }

    fn transpose_perm(node: &onnx::NodeProto) -> Option<Vec<i64>> {
        Self::attr_ints(node, "perm")
    }

    fn is_identity_perm(perm: &[i64]) -> bool {
        perm.iter()
            .enumerate()
            .all(|(idx, &axis)| axis == idx as i64)
    }

    fn optimize_transpose(
        graph: &mut onnx::GraphProto,
        options: &OptimizerOptions,
    ) -> anyhow::Result<usize> {
        let facts = Self::analyze(graph, options);
        let mut remove = vec![false; graph.node.len()];
        let mut rewrites = Vec::new();
        let mut rewrite_removals = Vec::new();

        for (idx, node) in graph.node.iter().enumerate() {
            if node.op_type != "Transpose" || node.input.len() != 1 || node.output.len() != 1 {
                continue;
            }
            let output = &node.output[0];
            if facts.graph_outputs.contains(output) {
                continue;
            }
            if let Some(perm) = Self::transpose_perm(node)
                && Self::is_identity_perm(&perm)
                && Self::rewrite_is_safe(graph, output, &node.input[0])
            {
                rewrites.push((output.clone(), node.input[0].clone()));
                rewrite_removals.push((idx, output.clone()));
            }
        }

        let applied = Self::apply_rewrites_recursive(graph, &rewrites);
        Self::mark_applied_rewrite_removals(&mut remove, &rewrite_removals, &applied);
        Self::remove_nodes(graph, &remove);
        let mut changes = applied.len();
        if changes > 0 {
            return Ok(changes);
        }

        let facts = Self::analyze(graph, options);
        let mut remove = vec![false; graph.node.len()];
        for consumer_idx in 0..graph.node.len() {
            let consumer = &graph.node[consumer_idx];
            if consumer.op_type != "Transpose"
                || consumer.input.len() != 1
                || consumer.output.len() != 1
                || facts.graph_outputs.contains(&consumer.output[0])
            {
                continue;
            }
            let temp = &consumer.input[0];
            if facts.graph_outputs.contains(temp) || !Self::single_use(&facts, temp) {
                continue;
            }
            let Some(&producer_idx) = facts.producers.get(temp) else {
                continue;
            };
            let producer = &graph.node[producer_idx];
            if producer.op_type != "Transpose" || producer.input.len() != 1 {
                continue;
            }
            let (Some(producer_perm), Some(consumer_perm)) = (
                Self::transpose_perm(producer),
                Self::transpose_perm(consumer),
            ) else {
                continue;
            };
            if producer_perm.len() != consumer_perm.len() {
                continue;
            }
            let Some(composed) = consumer_perm
                .iter()
                .map(|&axis| producer_perm.get(axis as usize).copied())
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            let producer_input = producer.input[0].clone();
            if Self::is_identity_perm(&composed) {
                let output = graph.node[consumer_idx].output[0].clone();
                if !Self::rewrite_is_safe(graph, &output, &producer_input) {
                    continue;
                }
                Self::replace_inputs_recursive(graph, &output, &producer_input);
                remove[producer_idx] = true;
                remove[consumer_idx] = true;
            } else {
                graph.node[consumer_idx].input[0] = producer_input;
                Self::set_attr_ints(&mut graph.node[consumer_idx], "perm", composed);
                remove[producer_idx] = true;
            }
            changes += 1;
            break;
        }
        Self::remove_nodes(graph, &remove);
        Ok(changes)
    }

    fn tensor_to_f32(tensor: &onnx::TensorProto) -> Option<Vec<f32>> {
        if tensor.data_type != dt::FLOAT {
            return None;
        }
        if !tensor.float_data.is_empty() {
            return Some(tensor.float_data.clone());
        }
        Self::raw_chunks::<4>(&tensor.raw_data).map(|chunks| {
            chunks
                .into_iter()
                .map(f32::from_le_bytes)
                .collect::<Vec<_>>()
        })
    }

    fn write_f32_tensor(tensor: &mut onnx::TensorProto, values: &[f32]) {
        tensor.data_type = dt::FLOAT;
        tensor.raw_data.clear();
        tensor.float_data.clear();
        tensor.int32_data.clear();
        tensor.int64_data.clear();
        tensor.double_data.clear();
        tensor.uint64_data.clear();
        for &value in values {
            tensor.raw_data.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn make_f32_initializer(name: String, dims: Vec<i64>, values: &[f32]) -> onnx::TensorProto {
        let mut tensor = onnx::TensorProto {
            name,
            dims,
            data_type: dt::FLOAT,
            ..Default::default()
        };
        Self::write_f32_tensor(&mut tensor, values);
        tensor
    }

    fn unique_initializer_name(graph: &onnx::GraphProto, base: &str) -> String {
        let mut used = graph
            .initializer
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect::<HashSet<_>>();
        used.extend(
            graph
                .node
                .iter()
                .flat_map(|node| node.output.iter().filter(|name| !name.is_empty()).cloned()),
        );
        // Also avoid colliding with graph boundary tensors so a fresh
        // initializer can never shadow an existing input/output/value_info.
        used.extend(
            graph
                .input
                .iter()
                .chain(graph.output.iter())
                .chain(graph.value_info.iter())
                .map(|value| value.name.clone()),
        );
        if !used.contains(base) {
            return base.to_string();
        }
        for idx in 0.. {
            let candidate = format!("{base}_{idx}");
            if !used.contains(&candidate) {
                return candidate;
            }
        }
        unreachable!()
    }

    fn has_attr(node: &onnx::NodeProto, name: &str) -> bool {
        node.attribute.iter().any(|attr| attr.name == name)
    }

    fn materialize_conv_kernel_shape(
        graph: &mut onnx::GraphProto,
        options: &OptimizerOptions,
    ) -> anyhow::Result<usize> {
        let facts = Self::analyze(graph, options);
        let mut changes = 0;
        for node in &mut graph.node {
            if !matches!(node.op_type.as_str(), "Conv" | "ConvTranspose")
                || node.input.len() < 2
                || Self::has_attr(node, "kernel_shape")
            {
                continue;
            }
            let Some(weight_idx) = facts.initializers.get(&node.input[1]).copied() else {
                continue;
            };
            let weight_dims = &graph.initializer[weight_idx].dims;
            if weight_dims.len() < 3 {
                continue;
            }
            let kernel_shape = weight_dims[2..].to_vec();
            if kernel_shape.iter().any(|dim| *dim <= 0) {
                continue;
            }
            node.attribute
                .push(helper::attr_ints("kernel_shape", &kernel_shape));
            changes += 1;
        }
        Ok(changes)
    }

    fn fuse_conv_add_bias(
        graph: &mut onnx::GraphProto,
        options: &OptimizerOptions,
    ) -> anyhow::Result<usize> {
        let mut changes = 0;
        loop {
            let changed = Self::fuse_conv_add_bias_once(graph, options)?;
            if changed == 0 {
                break;
            }
            changes += changed;
        }
        Ok(changes)
    }

    fn fuse_conv_add_bias_once(
        graph: &mut onnx::GraphProto,
        options: &OptimizerOptions,
    ) -> anyhow::Result<usize> {
        let facts = Self::analyze(graph, options);
        for add_idx in 0..graph.node.len() {
            if graph.node[add_idx].op_type != "Add"
                || graph.node[add_idx].input.len() != 2
                || graph.node[add_idx].output.len() != 1
            {
                continue;
            }
            let add_inputs = graph.node[add_idx].input.clone();
            let add_output = graph.node[add_idx].output[0].clone();
            let mut conv_input_pos = None;
            for (pos, add_input) in add_inputs.iter().enumerate() {
                let Some(&producer_idx) = facts.producers.get(add_input) else {
                    continue;
                };
                if graph.node[producer_idx].op_type == "Conv" {
                    conv_input_pos = Some((producer_idx, pos));
                    break;
                }
            }
            let Some((conv_idx, conv_pos)) = conv_input_pos else {
                continue;
            };
            let conv_output = graph.node[conv_idx]
                .output
                .first()
                .cloned()
                .unwrap_or_default();
            if facts.graph_outputs.contains(&conv_output) || !Self::single_use(&facts, &conv_output)
            {
                continue;
            }
            if graph.node[conv_idx].input.len() != 2 || graph.node[conv_idx].output.len() != 1 {
                continue;
            }
            let bias_name = add_inputs[1 - conv_pos].clone();
            let Some(mut bias) = Self::const_value(graph, &facts, &bias_name) else {
                continue;
            };
            let weight_name = &graph.node[conv_idx].input[1];
            let Some(weight) = Self::const_value(graph, &facts, weight_name) else {
                continue;
            };
            let out_channels = match weight.tensor.dims.as_slice() {
                [channels, ..] if *channels > 0 => *channels,
                _ => facts
                    .shapes
                    .get(&conv_output)
                    .and_then(|shape| shape.get(1))
                    .copied()
                    .filter(|channels| *channels > 0)
                    .unwrap_or_default(),
            };
            if out_channels <= 0 {
                continue;
            }
            let bias_dims = bias.tensor.dims.clone();
            let conv_output_rank = facts.shapes.get(&conv_output).map(|shape| shape.len());
            let can_use_bias = if bias_dims == [out_channels] {
                // A 1-D [C] addend only aligns to the channel axis when the conv
                // output is rank-2 ([N, C]); for the usual rank-3+ outputs ONNX
                // broadcasts [C] to the last (spatial) axis, so treating it as a
                // channel bias would change semantics. Require a known rank-2 shape.
                conv_output_rank == Some(2)
            } else if bias_dims == [1, out_channels, 1, 1] && Self::single_use(&facts, &bias_name) {
                bias.tensor.dims = vec![out_channels];
                true
            } else {
                false
            };
            if !can_use_bias {
                continue;
            }

            Self::set_const_value(graph, bias);
            graph.node[conv_idx].input.push(bias_name);
            graph.node[conv_idx].output[0] = add_output;
            let mut remove = vec![false; graph.node.len()];
            remove[add_idx] = true;
            Self::remove_nodes(graph, &remove);
            return Ok(1);
        }
        Ok(0)
    }

    fn fuse_conv_batch_norm(
        graph: &mut onnx::GraphProto,
        options: &OptimizerOptions,
    ) -> anyhow::Result<usize> {
        let mut changes = 0;
        loop {
            let changed = Self::fuse_conv_batch_norm_once(graph, options)?;
            if changed == 0 {
                break;
            }
            changes += changed;
        }
        Ok(changes)
    }

    fn fuse_conv_batch_norm_once(
        graph: &mut onnx::GraphProto,
        options: &OptimizerOptions,
    ) -> anyhow::Result<usize> {
        let facts = Self::analyze(graph, options);
        for bn_idx in 0..graph.node.len() {
            let bn = &graph.node[bn_idx];
            if bn.op_type != "BatchNormalization" || bn.input.len() < 5 || bn.output.len() != 1 {
                continue;
            }
            let Some(&conv_idx) = facts.producers.get(&bn.input[0]) else {
                continue;
            };
            if graph.node[conv_idx].op_type != "Conv"
                || graph.node[conv_idx].input.len() < 2
                || graph.node[conv_idx].input.len() > 3
                || graph.node[conv_idx].output.len() != 1
            {
                continue;
            }
            let conv_output = graph.node[conv_idx].output[0].clone();
            if facts.graph_outputs.contains(&conv_output) || !Self::single_use(&facts, &conv_output)
            {
                continue;
            }
            let weight_name = graph.node[conv_idx].input[1].clone();
            let Some(weight_idx) = facts.initializers.get(&weight_name).copied() else {
                continue;
            };
            if !Self::single_use(&facts, &weight_name) {
                continue;
            }
            let weight_tensor = graph.initializer[weight_idx].clone();
            let Some(mut weight_values) = Self::tensor_to_f32(&weight_tensor) else {
                continue;
            };
            let channels = match weight_tensor.dims.as_slice() {
                [channels, ..] if *channels > 0 => *channels as usize,
                _ => continue,
            };
            if channels == 0 || weight_values.len() % channels != 0 {
                continue;
            }

            let Some(scale) = Self::const_value(graph, &facts, &bn.input[1])
                .and_then(|value| Self::tensor_to_f32(&value.tensor))
            else {
                continue;
            };
            let Some(beta) = Self::const_value(graph, &facts, &bn.input[2])
                .and_then(|value| Self::tensor_to_f32(&value.tensor))
            else {
                continue;
            };
            let Some(mean) = Self::const_value(graph, &facts, &bn.input[3])
                .and_then(|value| Self::tensor_to_f32(&value.tensor))
            else {
                continue;
            };
            let Some(var) = Self::const_value(graph, &facts, &bn.input[4])
                .and_then(|value| Self::tensor_to_f32(&value.tensor))
            else {
                continue;
            };
            if scale.len() != channels
                || beta.len() != channels
                || mean.len() != channels
                || var.len() != channels
            {
                continue;
            }

            let epsilon = Self::attr_float(&graph.node[bn_idx], "epsilon").unwrap_or(1.0e-5);
            let elements_per_channel = weight_values.len() / channels;
            let mut folded_bias = vec![0.0_f32; channels];
            if graph.node[conv_idx].input.len() == 3 {
                let bias_name = graph.node[conv_idx].input[2].clone();
                let Some(bias_idx) = facts.initializers.get(&bias_name).copied() else {
                    continue;
                };
                if !Self::single_use(&facts, &bias_name) {
                    continue;
                }
                let Some(values) = Self::tensor_to_f32(&graph.initializer[bias_idx]) else {
                    continue;
                };
                if values.len() != channels {
                    continue;
                }
                folded_bias = values;
            }

            for channel in 0..channels {
                let factor = scale[channel] / (var[channel] + epsilon).sqrt();
                for value in &mut weight_values
                    [channel * elements_per_channel..(channel + 1) * elements_per_channel]
                {
                    *value *= factor;
                }
                folded_bias[channel] =
                    (folded_bias[channel] - mean[channel]) * factor + beta[channel];
            }

            Self::write_f32_tensor(&mut graph.initializer[weight_idx], &weight_values);
            if graph.node[conv_idx].input.len() == 3 {
                let bias_name = graph.node[conv_idx].input[2].clone();
                let bias_idx = facts.initializers[&bias_name];
                graph.initializer[bias_idx].dims = vec![channels as i64];
                Self::write_f32_tensor(&mut graph.initializer[bias_idx], &folded_bias);
            } else {
                let bias_name =
                    Self::unique_initializer_name(graph, &format!("{}_bn_bias", conv_output));
                graph.initializer.push(Self::make_f32_initializer(
                    bias_name.clone(),
                    vec![channels as i64],
                    &folded_bias,
                ));
                graph.node[conv_idx].input.push(bias_name);
            }
            graph.node[conv_idx].output[0] = graph.node[bn_idx].output[0].clone();
            let mut remove = vec![false; graph.node.len()];
            remove[bn_idx] = true;
            Self::remove_nodes(graph, &remove);
            return Ok(1);
        }
        Ok(0)
    }

    fn attr_float(node: &onnx::NodeProto, name: &str) -> Option<f32> {
        node.attribute
            .iter()
            .find(|attr| attr.name == name)
            .map(|attr| attr.f)
    }

    /// Whether `src` is unidirectionally broadcastable to `target` (ONNX rules):
    /// rank(src) <= rank(target), and each right-aligned src dim is either 1 or
    /// equal to the target dim. Unknown dims (< 0) on either side are only safe
    /// when the src dim is 1 (which broadcasts regardless).
    fn unidir_broadcastable_to(src: &[i64], target: &[i64]) -> bool {
        if src.len() > target.len() {
            return false;
        }
        let offset = target.len() - src.len();
        for (idx, &src_dim) in src.iter().enumerate() {
            if src_dim == 1 {
                continue;
            }
            let target_dim = target[offset + idx];
            if src_dim < 0 || target_dim != src_dim {
                return false;
            }
        }
        true
    }

    fn fuse_matmul_add_bias(
        graph: &mut onnx::GraphProto,
        options: &OptimizerOptions,
    ) -> anyhow::Result<usize> {
        let mut changes = 0;
        loop {
            let changed = Self::fuse_matmul_add_bias_once(graph, options)?;
            if changed == 0 {
                break;
            }
            changes += changed;
        }
        Ok(changes)
    }

    fn fuse_matmul_add_bias_once(
        graph: &mut onnx::GraphProto,
        options: &OptimizerOptions,
    ) -> anyhow::Result<usize> {
        let facts = Self::analyze(graph, options);
        for add_idx in 0..graph.node.len() {
            if graph.node[add_idx].op_type != "Add"
                || graph.node[add_idx].input.len() != 2
                || graph.node[add_idx].output.len() != 1
            {
                continue;
            }
            let add_inputs = graph.node[add_idx].input.clone();
            let add_output = graph.node[add_idx].output[0].clone();
            let mut matmul_input_pos = None;
            for (pos, add_input) in add_inputs.iter().enumerate() {
                let Some(&producer_idx) = facts.producers.get(add_input) else {
                    continue;
                };
                if graph.node[producer_idx].op_type == "MatMul" {
                    matmul_input_pos = Some((producer_idx, pos));
                    break;
                }
            }
            let Some((matmul_idx, matmul_pos)) = matmul_input_pos else {
                continue;
            };
            let matmul_output = graph.node[matmul_idx]
                .output
                .first()
                .cloned()
                .unwrap_or_default();
            if facts.graph_outputs.contains(&matmul_output)
                || !Self::single_use(&facts, &matmul_output)
                || graph.node[matmul_idx].input.len() != 2
                || graph.node[matmul_idx].output.len() != 1
            {
                continue;
            }
            let matmul_inputs = graph.node[matmul_idx].input.clone();
            let Some(lhs_shape) = Self::tensor_shape(graph, &facts, &matmul_inputs[0]) else {
                continue;
            };
            let Some(rhs_shape) = Self::tensor_shape(graph, &facts, &matmul_inputs[1]) else {
                continue;
            };
            if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
                continue;
            }
            let bias_name = add_inputs[1 - matmul_pos].clone();
            let Some(bias) = Self::const_value(graph, &facts, &bias_name) else {
                continue;
            };
            let Some(lhs_dtype) = Self::tensor_dtype(graph, &facts, &matmul_inputs[0]) else {
                continue;
            };
            let Some(rhs_dtype) = Self::tensor_dtype(graph, &facts, &matmul_inputs[1]) else {
                continue;
            };
            if !Self::gemm_supported_dtype(lhs_dtype, options.target_opset)
                || lhs_dtype != rhs_dtype
                || bias.tensor.data_type != lhs_dtype
            {
                continue;
            }
            // ONNX Gemm requires C to be unidirectionally broadcastable to (M, N)
            // and always produces a rank-2 (M, N) result, whereas the original Add
            // uses bidirectional broadcasting. Only fuse when the bias cannot expand
            // the rank/shape beyond (M, N); otherwise the Gemm would change output.
            let gemm_shape = [lhs_shape[0], rhs_shape[1]];
            if !Self::unidir_broadcastable_to(&bias.tensor.dims, &gemm_shape) {
                continue;
            }
            graph.node[matmul_idx].op_type = "Gemm".to_string();
            graph.node[matmul_idx].input.push(bias_name);
            graph.node[matmul_idx].output[0] = add_output;
            graph.node[matmul_idx]
                .attribute
                .push(helper::attr_float("alpha", 1.0));
            graph.node[matmul_idx]
                .attribute
                .push(helper::attr_float("beta", 1.0));
            graph.node[matmul_idx]
                .attribute
                .push(helper::attr_int("transA", 0));
            graph.node[matmul_idx]
                .attribute
                .push(helper::attr_int("transB", 0));
            let mut remove = vec![false; graph.node.len()];
            remove[add_idx] = true;
            Self::remove_nodes(graph, &remove);
            return Ok(1);
        }
        Ok(0)
    }

    #[cfg(test)]
    pub(crate) fn optimize_for_test(graph: &mut onnx::GraphProto) -> OptimizationReport {
        let options = OptimizerOptions::default();
        Self::optimize_graph(graph, &options).expect("optimizer failed")
    }
}

impl super::Converter {
    pub(crate) fn optimizer_options(&self) -> OptimizerOptions {
        let mut shapes = HashMap::new();
        for (id, dims) in self.state.tensor_shapes.iter() {
            let name = self
                .state
                .id_to_name
                .get(id)
                .cloned()
                .unwrap_or_else(|| format!("tensor_{id}"));
            shapes.insert(name, dims.clone());
        }
        OptimizerOptions {
            level: self.optimization_level,
            target_opset: self.target_opset,
            shapes,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GraphOptimizer;
    use crate::helper::{self, dt};
    use crate::proto::onnx;

    fn f32_tensor(name: &str, dims: Vec<i64>, values: &[f32]) -> onnx::TensorProto {
        let mut tensor = onnx::TensorProto {
            name: name.to_string(),
            dims,
            data_type: dt::FLOAT,
            ..Default::default()
        };
        for &value in values {
            tensor.raw_data.extend_from_slice(&value.to_le_bytes());
        }
        tensor
    }

    fn i64_tensor(name: &str, values: &[i64]) -> onnx::TensorProto {
        let mut tensor = onnx::TensorProto {
            name: name.to_string(),
            dims: vec![values.len() as i64],
            data_type: dt::INT64,
            ..Default::default()
        };
        for &value in values {
            tensor.raw_data.extend_from_slice(&value.to_le_bytes());
        }
        tensor
    }

    fn i64_tensor_with_dims(name: &str, dims: Vec<i64>, values: &[i64]) -> onnx::TensorProto {
        let mut tensor = onnx::TensorProto {
            name: name.to_string(),
            dims,
            data_type: dt::INT64,
            ..Default::default()
        };
        for &value in values {
            tensor.raw_data.extend_from_slice(&value.to_le_bytes());
        }
        tensor
    }

    fn bf16_tensor(name: &str, dims: Vec<i64>) -> onnx::TensorProto {
        let elements = dims.iter().product::<i64>().max(0) as usize;
        onnx::TensorProto {
            name: name.to_string(),
            dims,
            data_type: dt::BFLOAT16,
            raw_data: vec![0; elements * 2],
            ..Default::default()
        }
    }

    fn node(op_type: &str, inputs: &[&str], outputs: &[&str]) -> onnx::NodeProto {
        onnx::NodeProto {
            op_type: op_type.to_string(),
            input: inputs.iter().map(|name| name.to_string()).collect(),
            output: outputs.iter().map(|name| name.to_string()).collect(),
            ..Default::default()
        }
    }

    fn graph_output(name: &str) -> onnx::ValueInfoProto {
        onnx::ValueInfoProto {
            name: name.to_string(),
            ..Default::default()
        }
    }

    fn value_info(name: &str, dims: &[i64]) -> onnx::ValueInfoProto {
        value_info_with_dtype(name, dims, dt::FLOAT)
    }

    fn value_info_with_dtype(name: &str, dims: &[i64], elem_type: i32) -> onnx::ValueInfoProto {
        onnx::ValueInfoProto {
            name: name.to_string(),
            r#type: Some(onnx::TypeProto {
                value: Some(onnx::type_proto::Value::TensorType(
                    onnx::type_proto::Tensor {
                        elem_type,
                        shape: Some(onnx::TensorShapeProto {
                            dim: dims
                                .iter()
                                .map(|&dim| onnx::tensor_shape_proto::Dimension {
                                    value: Some(
                                        onnx::tensor_shape_proto::dimension::Value::DimValue(dim),
                                    ),
                                    ..Default::default()
                                })
                                .collect(),
                        }),
                    },
                )),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn raw_f32_values(tensor: &onnx::TensorProto) -> Vec<f32> {
        tensor
            .raw_data
            .chunks_exact(4)
            .map(|chunk| {
                let mut bytes = [0_u8; 4];
                bytes.copy_from_slice(chunk);
                f32::from_le_bytes(bytes)
            })
            .collect()
    }

    #[test]
    fn optimizer_flushes_f32_subnormal_initializers() {
        let mut raw = Vec::new();
        for value in [f32::MIN_POSITIVE / 2.0, -f32::MIN_POSITIVE / 4.0, 1.0] {
            raw.extend_from_slice(&value.to_le_bytes());
        }
        let mut graph = onnx::GraphProto {
            initializer: vec![
                onnx::TensorProto {
                    name: "raw".to_string(),
                    dims: vec![3],
                    data_type: dt::FLOAT,
                    raw_data: raw,
                    ..Default::default()
                },
                onnx::TensorProto {
                    name: "float_data".to_string(),
                    dims: vec![2],
                    data_type: dt::FLOAT,
                    float_data: vec![f32::MIN_POSITIVE / 2.0, 2.0],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        // The pass is opt-in, so enable it explicitly for this test.
        let mut options = super::OptimizerOptions::default();
        options.passes.flush_subnormal_initializers = true;
        let report =
            GraphOptimizer::optimize_graph(&mut graph, &options).expect("optimizer failed");

        assert!(
            report
                .pass_changes
                .iter()
                .any(|change| change.pass == "flush_subnormal_initializers")
        );
        assert_eq!(raw_f32_values(&graph.initializer[0]), vec![0.0, 0.0, 1.0]);
        assert_eq!(graph.initializer[1].float_data, vec![0.0, 2.0]);
    }

    #[test]
    fn optimizer_fuses_constant_reshape() {
        let mut graph = onnx::GraphProto {
            initializer: vec![
                f32_tensor("data", vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                i64_tensor("shape", &[3, 2]),
            ],
            node: vec![
                node("Reshape", &["data", "shape"], &["reshaped"]),
                node("Relu", &["reshaped"], &["out"]),
            ],
            output: vec![graph_output("out")],
            ..Default::default()
        };

        let report = GraphOptimizer::optimize_for_test(&mut graph);

        assert!(report.changed());
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Relu");
        assert_eq!(graph.node[0].input, vec!["data"]);
        assert_eq!(graph.initializer[0].dims, vec![3, 2]);
    }

    #[test]
    fn optimizer_fuses_shared_initializer_reshape_by_cloning() {
        let mut graph = onnx::GraphProto {
            initializer: vec![
                f32_tensor("bias", vec![2], &[1.0, 2.0]),
                i64_tensor("shape", &[1, -1, 1, 1]),
            ],
            node: vec![
                node("Reshape", &["bias", "shape"], &["bias4d_a"]),
                node("Reshape", &["bias", "shape"], &["bias4d_b"]),
                node("Add", &["x", "bias4d_a"], &["a"]),
                node("Add", &["a", "bias4d_b"], &["out"]),
            ],
            output: vec![graph_output("out")],
            ..Default::default()
        };

        GraphOptimizer::optimize_for_test(&mut graph);

        assert!(!graph.node.iter().any(|node| node.op_type == "Reshape"));
        assert!(
            graph
                .initializer
                .iter()
                .any(|tensor| tensor.name == "bias" && tensor.dims == [2])
        );
        assert!(
            graph
                .initializer
                .iter()
                .any(|tensor| tensor.name == "bias4d_a" && tensor.dims == [1, 2, 1, 1])
        );
        assert!(
            graph
                .initializer
                .iter()
                .any(|tensor| tensor.name == "bias4d_b" && tensor.dims == [1, 2, 1, 1])
        );
    }

    #[test]
    fn optimizer_fuses_constant_unsqueeze_and_cast() {
        let mut cast = node("Cast", &["expanded"], &["casted"]);
        cast.attribute
            .push(helper::attr_int("to", dt::DOUBLE as i64));
        let mut graph = onnx::GraphProto {
            initializer: vec![
                f32_tensor("data", vec![2], &[1.0, 2.0]),
                i64_tensor("axes", &[0]),
            ],
            node: vec![
                node("Unsqueeze", &["data", "axes"], &["expanded"]),
                cast,
                node("Relu", &["casted"], &["out"]),
            ],
            output: vec![graph_output("out")],
            ..Default::default()
        };

        GraphOptimizer::optimize_for_test(&mut graph);

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].input, vec!["data"]);
        assert_eq!(graph.initializer[0].dims, vec![1, 2]);
        assert_eq!(graph.initializer[0].data_type, dt::DOUBLE);
        assert_eq!(graph.initializer[0].raw_data.len(), 16);
    }

    #[test]
    fn optimizer_eliminates_arithmetic_identity_and_transposes() {
        let mut t1 = node("Transpose", &["x"], &["t1"]);
        t1.attribute.push(helper::attr_ints("perm", &[1, 0]));
        let mut t2 = node("Transpose", &["t1"], &["t2"]);
        t2.attribute.push(helper::attr_ints("perm", &[1, 0]));
        let mut identity_transpose = node("Transpose", &["t2"], &["t3"]);
        identity_transpose
            .attribute
            .push(helper::attr_ints("perm", &[0, 1]));
        let mut graph = onnx::GraphProto {
            initializer: vec![i64_tensor_with_dims("zero", vec![], &[0])],
            node: vec![
                t1,
                t2,
                identity_transpose,
                node("Add", &["t3", "zero"], &["out"]),
                node("Relu", &["out"], &["final"]),
            ],
            output: vec![graph_output("final")],
            ..Default::default()
        };

        GraphOptimizer::optimize_for_test(&mut graph);

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Relu");
        assert_eq!(graph.node[0].input, vec!["x"]);
    }

    #[test]
    fn optimizer_resolves_chained_identity_rewrites() {
        let mut graph = onnx::GraphProto {
            initializer: vec![
                i64_tensor_with_dims("zero", vec![], &[0]),
                f32_tensor("one", vec![], &[1.0]),
            ],
            node: vec![
                node("Add", &["x", "zero"], &["add_out"]),
                node("Mul", &["add_out", "one"], &["mul_out"]),
                node("Relu", &["mul_out"], &["final"]),
            ],
            output: vec![graph_output("final")],
            ..Default::default()
        };

        GraphOptimizer::optimize_for_test(&mut graph);

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Relu");
        assert_eq!(graph.node[0].input, vec!["x"]);
    }

    #[test]
    fn optimizer_keeps_producer_when_resolved_rewrite_is_unsafe_for_subgraph() {
        let then_graph = onnx::GraphProto {
            input: vec![graph_output("x")],
            node: vec![node("Identity", &["b"], &["then_out"])],
            output: vec![graph_output("then_out")],
            ..Default::default()
        };
        let mut if_node = node("If", &["cond"], &["if_out"]);
        if_node.attribute.push(onnx::AttributeProto {
            name: "then_branch".to_string(),
            g: Some(then_graph),
            ..Default::default()
        });
        let mut graph = onnx::GraphProto {
            initializer: vec![i64_tensor_with_dims("zero", vec![], &[0])],
            node: vec![
                node("Add", &["x", "zero"], &["a"]),
                node("Add", &["a", "zero"], &["b"]),
                if_node,
            ],
            output: vec![graph_output("if_out")],
            ..Default::default()
        };

        GraphOptimizer::optimize_for_test(&mut graph);

        assert!(
            graph
                .node
                .iter()
                .any(|node| node.op_type == "Add" && node.output == vec!["b"])
        );
        let then_graph = graph
            .node
            .iter()
            .find(|node| node.op_type == "If")
            .unwrap()
            .attribute[0]
            .g
            .as_ref()
            .unwrap();
        assert_eq!(then_graph.node[0].input, vec!["b"]);
    }

    #[test]
    fn optimizer_skips_transpose_pair_when_subgraph_shadows_replacement() {
        let mut t1 = node("Transpose", &["x"], &["t1"]);
        t1.attribute.push(helper::attr_ints("perm", &[1, 0]));
        let mut t2 = node("Transpose", &["t1"], &["out"]);
        t2.attribute.push(helper::attr_ints("perm", &[1, 0]));
        let then_graph = onnx::GraphProto {
            input: vec![graph_output("x")],
            node: vec![node("Identity", &["out"], &["then_out"])],
            output: vec![graph_output("then_out")],
            ..Default::default()
        };
        let mut if_node = node("If", &["cond"], &["if_out"]);
        if_node.attribute.push(onnx::AttributeProto {
            name: "then_branch".to_string(),
            g: Some(then_graph),
            ..Default::default()
        });
        let mut graph = onnx::GraphProto {
            node: vec![t1, t2, if_node],
            output: vec![graph_output("if_out")],
            ..Default::default()
        };

        GraphOptimizer::optimize_for_test(&mut graph);

        assert!(
            graph
                .node
                .iter()
                .any(|node| node.op_type == "Transpose" && node.output == vec!["out"])
        );
        let then_graph = graph
            .node
            .iter()
            .find(|node| node.op_type == "If")
            .unwrap()
            .attribute[0]
            .g
            .as_ref()
            .unwrap();
        assert_eq!(then_graph.node[0].input, vec!["out"]);
    }

    #[test]
    fn optimizer_skips_rewrite_when_subgraph_shadows_target() {
        let then_graph = onnx::GraphProto {
            input: vec![graph_output("x")],
            node: vec![node("Identity", &["add_out"], &["then_out"])],
            output: vec![graph_output("then_out")],
            ..Default::default()
        };
        let mut if_node = node("If", &["cond"], &["if_out"]);
        if_node.attribute.push(onnx::AttributeProto {
            name: "then_branch".to_string(),
            g: Some(then_graph),
            ..Default::default()
        });
        let mut graph = onnx::GraphProto {
            initializer: vec![i64_tensor_with_dims("zero", vec![], &[0])],
            node: vec![node("Add", &["x", "zero"], &["add_out"]), if_node],
            output: vec![graph_output("if_out")],
            ..Default::default()
        };

        GraphOptimizer::optimize_for_test(&mut graph);

        assert_eq!(graph.node[0].op_type, "Add");
        let then_graph = graph.node[1].attribute[0].g.as_ref().unwrap();
        assert_eq!(then_graph.node[0].input, vec!["add_out"]);
    }

    #[test]
    fn optimizer_does_not_use_parent_shapes_inside_subgraphs() {
        let then_graph = onnx::GraphProto {
            initializer: vec![i64_tensor("zero", &[0])],
            input: vec![graph_output("x")],
            node: vec![
                node("Add", &["x", "zero"], &["hidden"]),
                node("Relu", &["hidden"], &["then_out"]),
            ],
            output: vec![graph_output("then_out")],
            ..Default::default()
        };
        let mut if_node = node("If", &["cond"], &["if_out"]);
        if_node.attribute.push(onnx::AttributeProto {
            name: "then_branch".to_string(),
            g: Some(then_graph),
            ..Default::default()
        });
        let mut graph = onnx::GraphProto {
            node: vec![if_node],
            output: vec![graph_output("if_out")],
            ..Default::default()
        };
        let mut options = super::OptimizerOptions::default();
        options.shapes.insert("x".to_string(), vec![1]);

        GraphOptimizer::optimize_graph(&mut graph, &options).expect("optimizer failed");

        let then_graph = graph.node[0].attribute[0].g.as_ref().unwrap();
        assert!(
            then_graph
                .node
                .iter()
                .any(|node| node.op_type == "Add" && node.output == vec!["hidden"])
        );
    }

    #[test]
    fn optimizer_fuses_conv_add_bias() {
        let mut graph = onnx::GraphProto {
            initializer: vec![
                f32_tensor("weight", vec![2, 1, 1, 1], &[1.0, 2.0]),
                f32_tensor("bias4d", vec![1, 2, 1, 1], &[0.5, -1.0]),
            ],
            node: vec![
                node("Conv", &["x", "weight"], &["conv_out"]),
                node("Add", &["conv_out", "bias4d"], &["out"]),
            ],
            output: vec![graph_output("out")],
            ..Default::default()
        };

        GraphOptimizer::optimize_for_test(&mut graph);

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Conv");
        assert_eq!(graph.node[0].input, vec!["x", "weight", "bias4d"]);
        assert_eq!(graph.node[0].output, vec!["out"]);
        assert_eq!(graph.initializer[1].dims, vec![2]);
    }

    #[test]
    fn optimizer_materializes_conv_kernel_shape() {
        let mut graph = onnx::GraphProto {
            initializer: vec![f32_tensor(
                "weight",
                vec![8, 3, 3, 5],
                &[0.0; 8 * 3 * 3 * 5],
            )],
            node: vec![node("Conv", &["x", "weight"], &["out"])],
            output: vec![graph_output("out")],
            ..Default::default()
        };

        GraphOptimizer::optimize_for_test(&mut graph);

        let kernel_shape = graph.node[0]
            .attribute
            .iter()
            .find(|attr| attr.name == "kernel_shape")
            .unwrap();
        assert_eq!(kernel_shape.ints, vec![3, 5]);
    }

    #[test]
    fn optimizer_folds_conv_batch_norm() {
        let mut bn = node(
            "BatchNormalization",
            &["conv_out", "scale", "beta", "mean", "var"],
            &["out"],
        );
        bn.attribute.push(helper::attr_float("epsilon", 0.0));
        let mut graph = onnx::GraphProto {
            initializer: vec![
                f32_tensor("weight", vec![2, 1, 1, 1], &[2.0, 4.0]),
                f32_tensor("scale", vec![2], &[2.0, 4.0]),
                f32_tensor("beta", vec![2], &[0.5, -1.0]),
                f32_tensor("mean", vec![2], &[1.0, 2.0]),
                f32_tensor("var", vec![2], &[1.0, 4.0]),
            ],
            node: vec![node("Conv", &["x", "weight"], &["conv_out"]), bn],
            output: vec![graph_output("out")],
            ..Default::default()
        };

        GraphOptimizer::optimize_for_test(&mut graph);

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Conv");
        assert_eq!(graph.node[0].output, vec!["out"]);
        assert_eq!(graph.node[0].input.len(), 3);
        assert_eq!(raw_f32_values(&graph.initializer[0]), vec![4.0, 8.0]);
        let bias_name = &graph.node[0].input[2];
        let bias = graph
            .initializer
            .iter()
            .find(|tensor| &tensor.name == bias_name)
            .unwrap();
        assert_eq!(raw_f32_values(bias), vec![-1.5, -5.0]);
    }

    #[test]
    fn optimizer_fuses_matmul_add_to_gemm() {
        let mut graph = onnx::GraphProto {
            initializer: vec![f32_tensor("bias", vec![3], &[1.0, 2.0, 3.0])],
            input: vec![value_info("x", &[2, 4]), value_info("w", &[4, 3])],
            node: vec![
                node("MatMul", &["x", "w"], &["mm"]),
                node("Add", &["mm", "bias"], &["out"]),
            ],
            output: vec![graph_output("out")],
            ..Default::default()
        };

        GraphOptimizer::optimize_for_test(&mut graph);

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Gemm");
        assert_eq!(graph.node[0].input, vec!["x", "w", "bias"]);
        assert_eq!(graph.node[0].output, vec!["out"]);
    }

    #[test]
    fn optimizer_does_not_fuse_integer_matmul_add_to_gemm() {
        let mut graph = onnx::GraphProto {
            initializer: vec![i64_tensor_with_dims("bias", vec![3], &[1, 2, 3])],
            input: vec![
                value_info_with_dtype("x", &[2, 4], dt::INT64),
                value_info_with_dtype("w", &[4, 3], dt::INT64),
            ],
            node: vec![
                node("MatMul", &["x", "w"], &["mm"]),
                node("Add", &["mm", "bias"], &["out"]),
            ],
            output: vec![graph_output("out")],
            ..Default::default()
        };

        GraphOptimizer::optimize_for_test(&mut graph);

        assert_eq!(graph.node.len(), 2);
        assert_eq!(graph.node[0].op_type, "MatMul");
        assert_eq!(graph.node[1].op_type, "Add");
    }

    #[test]
    fn optimizer_does_not_fuse_bfloat16_matmul_add_to_gemm_before_opset_13() {
        let mut graph = onnx::GraphProto {
            initializer: vec![bf16_tensor("bias", vec![3])],
            input: vec![
                value_info_with_dtype("x", &[2, 4], dt::BFLOAT16),
                value_info_with_dtype("w", &[4, 3], dt::BFLOAT16),
            ],
            node: vec![
                node("MatMul", &["x", "w"], &["mm"]),
                node("Add", &["mm", "bias"], &["out"]),
            ],
            output: vec![graph_output("out")],
            ..Default::default()
        };
        let options = super::OptimizerOptions {
            target_opset: 12,
            ..Default::default()
        };

        GraphOptimizer::optimize_graph(&mut graph, &options).expect("optimizer failed");

        assert_eq!(graph.node.len(), 2);
        assert_eq!(graph.node[0].op_type, "MatMul");
        assert_eq!(graph.node[1].op_type, "Add");
    }

    #[test]
    fn optimizer_fuses_bfloat16_matmul_add_to_gemm_at_opset_13() {
        let mut graph = onnx::GraphProto {
            initializer: vec![bf16_tensor("bias", vec![3])],
            input: vec![
                value_info_with_dtype("x", &[2, 4], dt::BFLOAT16),
                value_info_with_dtype("w", &[4, 3], dt::BFLOAT16),
            ],
            node: vec![
                node("MatMul", &["x", "w"], &["mm"]),
                node("Add", &["mm", "bias"], &["out"]),
            ],
            output: vec![graph_output("out")],
            ..Default::default()
        };
        let options = super::OptimizerOptions {
            target_opset: 13,
            ..Default::default()
        };

        GraphOptimizer::optimize_graph(&mut graph, &options).expect("optimizer failed");

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Gemm");
    }

    #[test]
    fn optimizer_does_not_fuse_matmul_add_with_rank_expanding_bias() {
        // Bias [2, 1, 3] makes the original Add broadcast to rank-3 [2, 2, 3];
        // a Gemm could only emit rank-2 [2, 3], so fusion must be declined.
        let mut graph = onnx::GraphProto {
            initializer: vec![f32_tensor("bias", vec![2, 1, 3], &[0.0; 6])],
            input: vec![value_info("x", &[2, 4]), value_info("w", &[4, 3])],
            node: vec![
                node("MatMul", &["x", "w"], &["mm"]),
                node("Add", &["mm", "bias"], &["out"]),
            ],
            output: vec![graph_output("out")],
            ..Default::default()
        };

        GraphOptimizer::optimize_for_test(&mut graph);

        assert_eq!(graph.node.len(), 2);
        assert_eq!(graph.node[0].op_type, "MatMul");
        assert_eq!(graph.node[1].op_type, "Add");
    }

    #[test]
    fn optimizer_keeps_identity_add_when_const_rank_exceeds_passthrough() {
        // x is rank-1 [4]; a numel-1 const of rank 2 [1, 1] broadcasts the Add
        // output to rank-2 [1, 4], so the Add must not be eliminated.
        let mut graph = onnx::GraphProto {
            initializer: vec![i64_tensor_with_dims("zero", vec![1, 1], &[0])],
            input: vec![value_info("x", &[4])],
            node: vec![
                node("Add", &["x", "zero"], &["out"]),
                node("Relu", &["out"], &["final"]),
            ],
            output: vec![graph_output("final")],
            ..Default::default()
        };

        GraphOptimizer::optimize_for_test(&mut graph);

        assert!(graph.node.iter().any(|node| node.op_type == "Add"));
    }

    #[test]
    fn optimizer_keeps_float_add_positive_zero_to_preserve_signed_zero() {
        // `x + (+0.0)` is not semantics-preserving for floats: `-0.0 + 0.0` is `+0.0`,
        // but forwarding `x` keeps `-0.0`, which a downstream Reciprocal turns into
        // `-inf`. The float Add-positive-zero identity must be left intact.
        let mut graph = onnx::GraphProto {
            initializer: vec![f32_tensor("zero", vec![], &[0.0])],
            input: vec![value_info("x", &[4])],
            node: vec![
                node("Add", &["x", "zero"], &["out"]),
                node("Reciprocal", &["out"], &["final"]),
            ],
            output: vec![graph_output("final")],
            ..Default::default()
        };

        GraphOptimizer::optimize_for_test(&mut graph);

        assert!(
            graph
                .node
                .iter()
                .any(|node| node.op_type == "Add" && node.output == vec!["out"])
        );
    }

    #[test]
    fn optimizer_eliminates_float_add_negative_zero() {
        // `-0.0` is the genuine IEEE-754 additive identity: `x + (-0.0) == x` for all
        // x, so folding it away is always safe even for floating types.
        let mut graph = onnx::GraphProto {
            initializer: vec![f32_tensor("neg_zero", vec![], &[-0.0])],
            input: vec![value_info("x", &[4])],
            node: vec![
                node("Add", &["x", "neg_zero"], &["out"]),
                node("Reciprocal", &["out"], &["final"]),
            ],
            output: vec![graph_output("final")],
            ..Default::default()
        };

        GraphOptimizer::optimize_for_test(&mut graph);

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Reciprocal");
        assert_eq!(graph.node[0].input, vec!["x"]);
    }

    #[test]
    fn optimizer_still_eliminates_float_mul_one() {
        // The signed-zero guard only applies to Add; `x * 1.0` stays foldable for
        // floating types.
        let mut graph = onnx::GraphProto {
            initializer: vec![f32_tensor("one", vec![], &[1.0])],
            input: vec![value_info("x", &[4])],
            node: vec![
                node("Mul", &["x", "one"], &["out"]),
                node("Relu", &["out"], &["final"]),
            ],
            output: vec![graph_output("final")],
            ..Default::default()
        };

        GraphOptimizer::optimize_for_test(&mut graph);

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Relu");
        assert_eq!(graph.node[0].input, vec!["x"]);
    }

    #[test]
    fn optimizer_does_not_fuse_conv_add_1d_bias_on_spatial_output() {
        // conv_out is rank-4 [1, 2, 3, 3]; a 1-D [2] addend broadcasts to the
        // width axis under ONNX rules, not the channel axis, so it must not be
        // fused as a Conv channel bias.
        let mut graph = onnx::GraphProto {
            initializer: vec![
                f32_tensor("weight", vec![2, 1, 1, 1], &[1.0, 2.0]),
                f32_tensor("bias", vec![2], &[0.5, -1.0]),
            ],
            node: vec![
                node("Conv", &["x", "weight"], &["conv_out"]),
                node("Add", &["conv_out", "bias"], &["out"]),
            ],
            value_info: vec![value_info("conv_out", &[1, 2, 3, 3])],
            output: vec![graph_output("out")],
            ..Default::default()
        };

        GraphOptimizer::optimize_for_test(&mut graph);

        assert!(graph.node.iter().any(|node| node.op_type == "Add"));
        assert_eq!(
            graph
                .node
                .iter()
                .find(|n| n.op_type == "Conv")
                .unwrap()
                .input
                .len(),
            2
        );
    }

    #[test]
    fn optimizer_does_not_fuse_batched_matmul_add_to_gemm() {
        let mut graph = onnx::GraphProto {
            initializer: vec![f32_tensor("bias", vec![3], &[1.0, 2.0, 3.0])],
            input: vec![value_info("x", &[1, 2, 4]), value_info("w", &[4, 3])],
            node: vec![
                node("MatMul", &["x", "w"], &["mm"]),
                node("Add", &["mm", "bias"], &["out"]),
            ],
            output: vec![graph_output("out")],
            ..Default::default()
        };

        GraphOptimizer::optimize_for_test(&mut graph);

        assert_eq!(graph.node.len(), 2);
        assert_eq!(graph.node[0].op_type, "MatMul");
        assert_eq!(graph.node[1].op_type, "Add");
    }
}
