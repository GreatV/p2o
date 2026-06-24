use anyhow::bail;
use half::f16;
use serde_json::Value;
use std::borrow::Borrow;
use std::collections::{HashMap, HashSet, hash_map::Entry};
use std::fs::File;
use std::hash::Hash;
use std::ops::Deref;
use std::sync::Arc;

use crate::helper::{self, dt, paddle_op};
use crate::proto::onnx;

mod attr;
mod export;
mod ops;
mod optimizer;
mod value_info;
mod weights;

pub use optimizer::{OptimizationLevel, OptimizationReport};

#[cfg(test)]
mod tests;

pub const DEFAULT_OPSET: i64 = 17;
const MAX_MODEL_JSON_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone, Default)]
pub(crate) struct ParamMeta {
    pub(crate) onnx_dtype: Option<i32>,
    pub(crate) dims: Vec<i64>,
}

/// Copy-on-write metadata map shared across sub-converters.
///
/// Cloning this wrapper is cheap until the first mutation. That makes it a good
/// fit for pass1/pass2 metadata that subgraphs mostly read, but the first write
/// in a child scope still clones the entire map.
#[derive(Clone, Default)]
pub(crate) struct CloneOnWriteMap<K, V> {
    inner: Arc<HashMap<K, V>>,
}

impl<K, V> Deref for CloneOnWriteMap<K, V> {
    type Target = HashMap<K, V>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<K, V> From<HashMap<K, V>> for CloneOnWriteMap<K, V> {
    fn from(value: HashMap<K, V>) -> Self {
        Self {
            inner: Arc::new(value),
        }
    }
}

impl<K, V> CloneOnWriteMap<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    pub(crate) fn insert(&mut self, key: K, value: V) -> Option<V> {
        Arc::make_mut(&mut self.inner).insert(key, value)
    }

    #[allow(dead_code)]
    pub(crate) fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        Arc::make_mut(&mut self.inner).get_mut(key)
    }

    #[allow(dead_code)]
    pub(crate) fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        Arc::make_mut(&mut self.inner).remove(key)
    }

    #[allow(dead_code)]
    pub(crate) fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        Arc::make_mut(&mut self.inner).entry(key)
    }
}

/// Copy-on-write metadata vector shared across sub-converters.
#[derive(Clone, Default)]
pub(crate) struct CloneOnWriteVec<T> {
    inner: Arc<Vec<T>>,
}

impl<T> Deref for CloneOnWriteVec<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> From<Vec<T>> for CloneOnWriteVec<T> {
    fn from(value: Vec<T>) -> Self {
        Self {
            inner: Arc::new(value),
        }
    }
}

impl<T> CloneOnWriteVec<T>
where
    T: Clone,
{
    pub(crate) fn push(&mut self, value: T) {
        Arc::make_mut(&mut self.inner).push(value);
    }

    #[allow(dead_code)]
    pub(crate) fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        Arc::make_mut(&mut self.inner).get_mut(index)
    }

    #[allow(dead_code)]
    pub(crate) fn pop(&mut self) -> Option<T> {
        Arc::make_mut(&mut self.inner).pop()
    }
}

#[derive(Clone, Default)]
pub struct ConverterState {
    pub(crate) id_to_name: CloneOnWriteMap<i64, String>,
    pub(crate) param_names: CloneOnWriteVec<String>,
    pub(crate) param_meta: CloneOnWriteMap<String, ParamMeta>,
    pub(crate) combines: CloneOnWriteMap<i64, Vec<i64>>,
    pub(crate) stack_parts: CloneOnWriteMap<i64, Vec<i64>>,
    pub(crate) splits: CloneOnWriteMap<i64, Vec<i64>>,
    pub(crate) constants: CloneOnWriteMap<i64, Vec<f64>>,
    pub(crate) tensor_shapes: CloneOnWriteMap<i64, Vec<i64>>,
    pub(crate) tensor_types: CloneOnWriteMap<i64, String>,
}

pub struct Converter {
    pub(crate) onnx_graph: onnx::GraphProto,
    pub(crate) state: ConverterState,
    pub(crate) target_opset: i64,
    pub(crate) optimization_level: OptimizationLevel,
    pub(crate) strict: bool,
    pub(crate) warned_multinomial_degraded: bool,
    pub(crate) fetch_name_cache: Option<FetchNameCache>,
}

pub(crate) struct FetchNameCache {
    pub(crate) node_count: usize,
    pub(crate) output_count: usize,
    pub(crate) names: HashSet<String>,
}

impl Default for Converter {
    fn default() -> Self {
        Self::new()
    }
}

impl Converter {
    pub fn new() -> Self {
        Self {
            onnx_graph: onnx::GraphProto {
                node: vec![],
                name: "Paddle2ONNX_Model".to_string(),
                initializer: vec![],
                input: vec![],
                output: vec![],
                value_info: vec![],
                ..Default::default()
            },
            state: ConverterState::default(),
            target_opset: DEFAULT_OPSET,
            optimization_level: OptimizationLevel::Basic,
            strict: false,
            warned_multinomial_degraded: false,
            fetch_name_cache: None,
        }
    }

    pub fn set_target_opset(&mut self, opset: i64) {
        self.target_opset = opset;
    }

    pub fn set_optimization_level(&mut self, level: OptimizationLevel) {
        self.optimization_level = level;
    }

    pub fn set_strict(&mut self, strict: bool) {
        self.strict = strict;
    }

    #[allow(dead_code)]
    pub fn graph(&self) -> &onnx::GraphProto {
        &self.onnx_graph
    }

    #[allow(dead_code)]
    pub fn target_opset(&self) -> i64 {
        self.target_opset
    }

    pub fn require_opset(&self, min_opset: i64, op_name: &str) -> anyhow::Result<()> {
        if self.target_opset < min_opset {
            bail!(
                "{} requires opset >= {}, but target opset is {}",
                op_name,
                min_opset,
                self.target_opset
            );
        }
        Ok(())
    }

    pub fn get_tensor_name(&self, id: i64) -> anyhow::Result<String> {
        if id == 0 {
            return Ok("".to_string());
        }
        if let Some(name) = self.state.id_to_name.get(&id) {
            return Ok(name.clone());
        }
        if id < 0 {
            bail!("Undefined value id {}", id);
        }
        Ok(format!("tensor_{}", id))
    }

    fn record_tensor_meta(&mut self, value: &Value) {
        if let Some(id) = value
            .get("%")
            .and_then(|id| id.as_i64())
            .or_else(|| value.get("#").and_then(|id| id.as_i64()))
            && let Some(tt) = value
                .get("TT")
                .and_then(|tt| tt.get("D"))
                .and_then(|d| d.as_array())
        {
            if let Some(type_name) = tt.first().and_then(|t| t.get("#")).and_then(|t| t.as_str()) {
                self.state.tensor_types.insert(id, type_name.to_string());
            }
            if let Some(shape) = tt.get(1).and_then(|shape| shape.as_array()) {
                self.state.tensor_shapes.insert(
                    id,
                    shape.iter().map(|dim| dim.as_i64().unwrap_or(-1)).collect(),
                );
            }
        }
    }

    pub(crate) fn collect_pass1_from_ops(&mut self, ops: &[Value]) -> anyhow::Result<()> {
        let constant_producers = Self::constant_producer_positions(ops);
        for (op_index, op) in ops.iter().enumerate() {
            self.collect_pass1_from_op(op, op_index, &constant_producers)?;
        }
        Ok(())
    }

    pub fn sub_converter(&self) -> Self {
        Self {
            onnx_graph: onnx::GraphProto {
                node: vec![],
                name: "Paddle2ONNX_Subgraph".to_string(),
                initializer: vec![],
                input: vec![],
                output: vec![],
                value_info: vec![],
                ..Default::default()
            },
            state: self.state.clone(),
            target_opset: self.target_opset,
            optimization_level: self.optimization_level,
            strict: self.strict,
            warned_multinomial_degraded: false,
            fetch_name_cache: None,
        }
    }

    fn record_pass1_outputs(&mut self, op: &Value) {
        if let Some(outputs) = op.get("O") {
            if let Some(outputs) = outputs.as_array() {
                for output in outputs {
                    self.record_tensor_meta(output);
                }
            } else if outputs.is_object() {
                self.record_tensor_meta(outputs);
            }
        }
    }

    fn collect_pass1_from_op(
        &mut self,
        op: &Value,
        op_index: usize,
        constant_producers: &HashMap<i64, usize>,
    ) -> anyhow::Result<()> {
        self.record_pass1_outputs(op);

        if let Some(op_type) = helper::op_type(op) {
            self.collect_pass1_metadata_op(op_type, op);
            if let Some(folder) = Self::pass1_constant_folder(op_type) {
                folder(self, op, op_index, constant_producers)?;
            }
        }

        self.collect_pass1_nested_regions(op)
    }

    fn collect_pass1_metadata_op(&mut self, op_type: &str, op: &Value) {
        match op_type {
            "p" => self.collect_pass1_param(op),
            "0.combine" => {
                let inputs = helper::op_input_ids(op);
                if let Ok(out_id) = helper::op_out_id(op) {
                    self.state.combines.insert(out_id, inputs);
                }
            }
            "0.split" => {
                let inputs = helper::op_input_ids(op);
                if let Some(&input_id) = inputs.first() {
                    let out_ids = op
                        .get("O")
                        .and_then(|o| o.as_array())
                        .map(|outputs| {
                            outputs
                                .iter()
                                .filter_map(|output| output.get("%").and_then(|id| id.as_i64()))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    self.state.splits.insert(input_id, out_ids);
                }
            }
            _ => {}
        }
    }

    fn with_constant_scope<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        let outer_constants = self.state.constants.clone();
        let result = f(self);
        self.state.constants = outer_constants;
        result
    }

    fn collect_pass1_from_block(&mut self, block: &Value) -> anyhow::Result<()> {
        if let Some(args) = block.get("args").and_then(|a| a.as_array()) {
            for arg in args {
                self.record_tensor_meta(arg);
            }
        }
        if let Some(block_ops) = block.get("ops").and_then(|o| o.as_array()) {
            self.with_constant_scope(|this| this.collect_pass1_from_ops(block_ops))?;
        }
        Ok(())
    }

    fn collect_pass1_param(&mut self, op: &Value) {
        let name = op
            .get("A")
            .and_then(|a| a.as_array())
            .and_then(|a| a.iter().find_map(|x| x.as_str()))
            .unwrap_or("unknown_param");

        if let Ok(id) = helper::op_out_id(op) {
            self.state.id_to_name.insert(id, name.to_string());
            self.state.param_names.push(name.to_string());
            self.state.param_meta.insert(
                name.to_string(),
                ParamMeta {
                    onnx_dtype: self
                        .state
                        .tensor_types
                        .get(&id)
                        .and_then(|dtype| helper::paddle_elem_type_to_onnx(dtype)),
                    dims: self
                        .state
                        .tensor_shapes
                        .get(&id)
                        .cloned()
                        .unwrap_or_default(),
                },
            );
        }
    }

    fn collect_pass1_nested_regions(&mut self, op: &Value) -> anyhow::Result<()> {
        if let Some(regions) = op.get("regions").and_then(|r| r.as_array()) {
            for region in regions {
                if let Some(blocks) = region.get("blocks").and_then(|b| b.as_array()) {
                    for block in blocks {
                        self.collect_pass1_from_block(block)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn constant_producer_positions(ops: &[Value]) -> HashMap<i64, usize> {
        let mut producers = HashMap::new();
        for (index, op) in ops.iter().enumerate() {
            if let Some(op_type) = helper::op_type(op)
                && Self::pass1_constant_folder(op_type).is_some()
                && let Ok(out_id) = helper::op_out_id(op)
            {
                producers.insert(out_id, index);
            }
        }
        producers
    }

    fn pass1_constant_folder(op_type: &str) -> Option<Pass1ConstantFolder> {
        PASS1_CONSTANT_FOLDERS
            .iter()
            .find_map(|(candidate, folder)| (*candidate == op_type).then_some(*folder))
    }

    fn fold_literal_constant(
        &mut self,
        op: &Value,
        _: usize,
        _: &HashMap<i64, usize>,
    ) -> anyhow::Result<()> {
        let Ok(out_id) = helper::op_out_id(op) else {
            return Ok(());
        };
        let op_type = helper::op_type(op).unwrap_or_default();
        let mut vals = Vec::new();
        if op_type == paddle_op::ASSIGN_VALUE {
            if let Some(values) = helper::attr(op, "values").and_then(|d| d.as_array()) {
                for item in values {
                    if let Some(v) = helper::value_as_f64(item) {
                        vals.push(v);
                    }
                }
            }
        } else if let Some(v) = helper::attr_f64(op, "value") {
            vals.push(v);
        } else if let Some(values) = helper::attr(op, "value").and_then(|d| d.as_array()) {
            for item in values {
                if let Some(v) = helper::value_as_f64(item) {
                    vals.push(v);
                }
            }
        }
        self.state.constants.insert(out_id, vals);
        Ok(())
    }

    fn fold_cast_constant(
        &mut self,
        op: &Value,
        op_index: usize,
        constant_producers: &HashMap<i64, usize>,
    ) -> anyhow::Result<()> {
        let Ok(out_id) = helper::op_out_id(op) else {
            return Ok(());
        };
        let inputs = helper::op_input_ids(op);
        if let Some(&input_id) = inputs.first() {
            if let Some(values) = self.state.constants.get(&input_id).cloned() {
                let target_dtype = helper::attr(op, "dtype").and_then(|d| d.as_str());
                let converted = Self::cast_constant_values(&values, target_dtype)?;
                self.state.constants.insert(out_id, converted);
            } else {
                Self::ensure_constant_not_defined_later(
                    input_id,
                    op_index,
                    constant_producers,
                    helper::op_type(op).unwrap_or_default(),
                )?;
            }
        }
        Ok(())
    }

    fn fold_value_preserving_shape_constant(
        &mut self,
        op: &Value,
        op_index: usize,
        constant_producers: &HashMap<i64, usize>,
    ) -> anyhow::Result<()> {
        let Ok(out_id) = helper::op_out_id(op) else {
            return Ok(());
        };
        let inputs = helper::op_input_ids(op);
        if let Some(&input_id) = inputs.first() {
            if let Some(values) = self.state.constants.get(&input_id).cloned() {
                self.state.constants.insert(out_id, values);
            } else {
                Self::ensure_constant_not_defined_later(
                    input_id,
                    op_index,
                    constant_producers,
                    helper::op_type(op).unwrap_or_default(),
                )?;
            }
        }
        Ok(())
    }

    fn fold_scale_constant(
        &mut self,
        op: &Value,
        op_index: usize,
        constant_producers: &HashMap<i64, usize>,
    ) -> anyhow::Result<()> {
        let Ok(out_id) = helper::op_out_id(op) else {
            return Ok(());
        };
        let inputs = helper::op_input_ids(op);
        if let Some(&input_id) = inputs.first() {
            if let Some(values) = self.state.constants.get(&input_id).cloned() {
                // Only fold when scale/bias come from attrs (not input tensors).
                if inputs.len() == 1 {
                    let scale = helper::attr_f64(op, "scale").unwrap_or(1.0);
                    let bias = helper::attr_f64(op, "bias").unwrap_or(0.0);
                    let bias_after_scale = helper::attr(op, "bias_after_scale")
                        .and_then(|d| d.as_bool())
                        .unwrap_or(true);
                    let folded: Vec<f64> = if bias_after_scale {
                        values.iter().map(|&v| v * scale + bias).collect()
                    } else {
                        values.iter().map(|&v| (v + bias) * scale).collect()
                    };
                    self.state.constants.insert(out_id, folded);
                }
            } else {
                Self::ensure_constant_not_defined_later(
                    input_id,
                    op_index,
                    constant_producers,
                    helper::op_type(op).unwrap_or_default(),
                )?;
            }
        }
        Ok(())
    }

    fn ensure_constant_not_defined_later(
        input_id: i64,
        consumer_index: usize,
        constant_producers: &HashMap<i64, usize>,
        consumer_op_type: &str,
    ) -> anyhow::Result<()> {
        if let Some(&producer_index) = constant_producers.get(&input_id)
            && producer_index > consumer_index
        {
            bail!(
                "constant folding for {} saw input {} before its constant producer at op index {}",
                consumer_op_type,
                input_id,
                producer_index
            );
        }
        Ok(())
    }

    fn cast_constant_values(
        values: &[f64],
        target_dtype: Option<&str>,
    ) -> anyhow::Result<Vec<f64>> {
        let Some(dtype) = target_dtype else {
            return Ok(values.to_vec());
        };
        let ensure_integer =
            |value: f64, dtype_name: &str, min: f64, max: f64| -> anyhow::Result<f64> {
                if !value.is_finite() {
                    bail!(
                        "cannot cast non-finite constant {} to {}",
                        value,
                        dtype_name
                    );
                }
                if value < min || value > max {
                    bail!("constant {} is out of range for {}", value, dtype_name);
                }
                if value.trunc() != value {
                    bail!(
                        "cannot cast non-integer constant {} to {} without losing precision",
                        value,
                        dtype_name
                    );
                }
                Ok(value)
            };

        Ok(match dtype {
            "bool" => values
                .iter()
                .map(|&v| {
                    if v.is_nan() {
                        bail!("cannot cast NaN constant to bool");
                    }
                    Ok(if v != 0.0 { 1.0 } else { 0.0 })
                })
                .collect::<anyhow::Result<Vec<_>>>()?,
            "int8" => values
                .iter()
                .map(|&v| ensure_integer(v, "int8", i8::MIN as f64, i8::MAX as f64))
                .collect::<anyhow::Result<Vec<_>>>()?,
            "uint8" => values
                .iter()
                .map(|&v| ensure_integer(v, "uint8", u8::MIN as f64, u8::MAX as f64))
                .collect::<anyhow::Result<Vec<_>>>()?,
            "int16" => values
                .iter()
                .map(|&v| ensure_integer(v, "int16", i16::MIN as f64, i16::MAX as f64))
                .collect::<anyhow::Result<Vec<_>>>()?,
            "int32" => values
                .iter()
                .map(|&v| ensure_integer(v, "int32", i32::MIN as f64, i32::MAX as f64))
                .collect::<anyhow::Result<Vec<_>>>()?,
            "int64" => values
                .iter()
                .map(|&v| ensure_integer(v, "int64", i64::MIN as f64, i64::MAX as f64))
                .collect::<anyhow::Result<Vec<_>>>()?,
            // float targets: no narrowing needed for constant propagation
            _ => values.to_vec(),
        })
    }

    fn vector_i64(name: String, values: &[i64]) -> onnx::TensorProto {
        let mut tensor = onnx::TensorProto {
            name,
            dims: vec![values.len() as i64],
            data_type: dt::INT64,
            ..Default::default()
        };
        for &value in values {
            tensor.raw_data.extend_from_slice(&value.to_le_bytes());
        }
        tensor
    }

    pub(crate) fn maybe_onnx_dtype_for_tensor_id(&self, id: i64) -> anyhow::Result<Option<i32>> {
        self.state
            .tensor_types
            .get(&id)
            .map(|dtype| self.onnx_elem_type_from_paddle(dtype))
            .transpose()
    }

    pub(crate) fn onnx_dtype_for_tensor_id(&self, id: i64) -> anyhow::Result<i32> {
        self.maybe_onnx_dtype_for_tensor_id(id)?
            .ok_or_else(|| anyhow::anyhow!("missing dtype metadata for tensor {}", id))
    }

    pub(crate) fn encode_scalar_f64_as_raw_data(
        &self,
        value: f64,
        data_type: i32,
    ) -> anyhow::Result<Vec<u8>> {
        let ensure_integer = |dtype_name: &str, min: f64, max: f64| -> anyhow::Result<()> {
            if !value.is_finite() {
                bail!("cannot encode non-finite value {} as {}", value, dtype_name);
            }
            if value.trunc() != value {
                bail!(
                    "cannot encode non-integer value {} as {} without losing precision",
                    value,
                    dtype_name
                );
            }
            if value < min || value > max {
                bail!("value {} is out of range for {}", value, dtype_name);
            }
            Ok(())
        };

        Ok(match data_type {
            dt::BOOL => {
                if value.is_nan() {
                    bail!("cannot encode NaN as bool");
                }
                vec![u8::from(value != 0.0)]
            }
            dt::FLOAT16 => {
                let narrowed = value as f32;
                if value.is_finite() && !narrowed.is_finite() {
                    bail!(
                        "cannot encode out-of-range finite value {} as float16",
                        value
                    );
                }
                let narrowed = f16::from_f32(narrowed);
                if value.is_finite() && !narrowed.is_finite() {
                    bail!("value {} is out of range for float16", value);
                }
                narrowed.to_bits().to_le_bytes().to_vec()
            }
            dt::FLOAT => {
                let narrowed = value as f32;
                if value.is_finite() && !narrowed.is_finite() {
                    bail!(
                        "cannot encode out-of-range finite value {} as float32",
                        value
                    );
                }
                narrowed.to_le_bytes().to_vec()
            }
            dt::DOUBLE => value.to_le_bytes().to_vec(),
            dt::INT8 => {
                ensure_integer("int8", i8::MIN as f64, i8::MAX as f64)?;
                (value as i8).to_le_bytes().to_vec()
            }
            dt::UINT8 => {
                ensure_integer("uint8", u8::MIN as f64, u8::MAX as f64)?;
                (value as u8).to_le_bytes().to_vec()
            }
            dt::INT16 => {
                ensure_integer("int16", i16::MIN as f64, i16::MAX as f64)?;
                (value as i16).to_le_bytes().to_vec()
            }
            dt::INT32 => {
                ensure_integer("int32", i32::MIN as f64, i32::MAX as f64)?;
                (value as i32).to_le_bytes().to_vec()
            }
            dt::INT64 => {
                ensure_integer("int64", i64::MIN as f64, i64::MAX as f64)?;
                (value as i64).to_le_bytes().to_vec()
            }
            _ => bail!(
                "unsupported initializer dtype {}",
                helper::onnx_dtype_name(data_type)
            ),
        })
    }

    pub(crate) fn push_numeric_initializer(
        &mut self,
        name: String,
        dims: Vec<i64>,
        data_type: i32,
        values: &[f64],
    ) -> anyhow::Result<()> {
        let mut tensor = onnx::TensorProto {
            name,
            dims,
            data_type,
            ..Default::default()
        };
        for &value in values {
            tensor
                .raw_data
                .extend_from_slice(&self.encode_scalar_f64_as_raw_data(value, data_type)?);
        }
        self.onnx_graph.initializer.push(tensor);
        Ok(())
    }

    fn push_i64_initializer(&mut self, name: String, dims: Vec<i64>, values: &[i64]) {
        // Empty dims means scalar only for single values; multi-value callers get a 1-D tensor.
        let tensor_dims = if dims.is_empty() && values.len() != 1 {
            vec![values.len() as i64]
        } else {
            dims
        };
        let mut tensor = onnx::TensorProto {
            name,
            dims: tensor_dims,
            data_type: dt::INT64,
            ..Default::default()
        };
        for &value in values {
            tensor.raw_data.extend_from_slice(&value.to_le_bytes());
        }
        self.onnx_graph.initializer.push(tensor);
    }

    fn push_f32_initializer(&mut self, name: String, dims: Vec<i64>, values: &[f32]) {
        // Empty dims means scalar only for single values; multi-value callers get a 1-D tensor.
        let tensor_dims = if dims.is_empty() && values.len() != 1 {
            vec![values.len() as i64]
        } else {
            dims
        };
        let mut tensor = onnx::TensorProto {
            name,
            dims: tensor_dims,
            data_type: dt::FLOAT,
            ..Default::default()
        };
        for &value in values {
            tensor.raw_data.extend_from_slice(&value.to_le_bytes());
        }
        self.onnx_graph.initializer.push(tensor);
    }

    pub(crate) fn add_cast_node(&mut self, input: String, output: String, to: i32) {
        let mut cast = onnx::NodeProto {
            op_type: "Cast".to_string(),
            input: vec![input],
            output: vec![output],
            ..Default::default()
        };
        cast.attribute.push(helper::attr_int("to", to as i64));
        self.onnx_graph.node.push(cast);
    }

    pub(crate) fn add_binary_node(
        &mut self,
        op_type: &str,
        lhs: String,
        rhs: String,
        output: String,
    ) {
        self.onnx_graph.node.push(onnx::NodeProto {
            op_type: op_type.to_string(),
            input: vec![lhs, rhs],
            output: vec![output],
            ..Default::default()
        });
    }

    pub fn add_unsqueeze_node(
        &mut self,
        input: String,
        output: String,
        axes: &[i64],
        axes_name: String,
    ) {
        let mut node = onnx::NodeProto {
            op_type: "Unsqueeze".to_string(),
            input: vec![input],
            output: vec![output],
            ..Default::default()
        };
        if self.target_opset >= 13 {
            self.onnx_graph
                .initializer
                .push(Self::vector_i64(axes_name.clone(), axes));
            node.input.push(axes_name);
        } else {
            node.attribute.push(helper::attr_ints("axes", axes));
        }
        self.onnx_graph.node.push(node);
    }

    pub fn add_squeeze_node(
        &mut self,
        input: String,
        output: String,
        axes: Option<&[i64]>,
        axes_name: Option<String>,
    ) {
        let mut node = onnx::NodeProto {
            op_type: "Squeeze".to_string(),
            input: vec![input],
            output: vec![output],
            ..Default::default()
        };
        if let Some(axes) = axes {
            if self.target_opset >= 13 {
                let name = axes_name.unwrap_or_else(|| "squeeze_axes".to_string());
                self.onnx_graph
                    .initializer
                    .push(Self::vector_i64(name.clone(), axes));
                node.input.push(name);
            } else {
                node.attribute.push(helper::attr_ints("axes", axes));
            }
        }
        self.onnx_graph.node.push(node);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_slice_node(
        &mut self,
        input: String,
        output: String,
        starts: &[i64],
        ends: &[i64],
        axes: Option<&[i64]>,
        steps: Option<&[i64]>,
        name_prefix: &str,
    ) -> anyhow::Result<()> {
        if starts.len() != ends.len() {
            bail!(
                "Slice starts/ends length mismatch: {} vs {}",
                starts.len(),
                ends.len()
            );
        }
        if let Some(axes) = axes
            && axes.len() != starts.len()
        {
            bail!(
                "Slice axes length {} does not match starts length {}",
                axes.len(),
                starts.len()
            );
        }
        if let Some(steps) = steps
            && steps.len() != starts.len()
        {
            bail!(
                "Slice steps length {} does not match starts length {}",
                steps.len(),
                starts.len()
            );
        }

        if self.target_opset < 10 {
            if let Some(steps) = steps
                && steps.iter().any(|&step| step != 1)
            {
                bail!("Slice with steps != 1 requires opset >= 10");
            }

            let mut node = onnx::NodeProto {
                op_type: "Slice".to_string(),
                input: vec![input],
                output: vec![output],
                ..Default::default()
            };
            node.attribute.push(helper::attr_ints("starts", starts));
            node.attribute.push(helper::attr_ints("ends", ends));
            if let Some(axes) = axes {
                node.attribute.push(helper::attr_ints("axes", axes));
            }
            self.onnx_graph.node.push(node);
            return Ok(());
        }

        let starts_name = format!("{}_starts", name_prefix);
        self.onnx_graph
            .initializer
            .push(Self::vector_i64(starts_name.clone(), starts));
        let ends_name = format!("{}_ends", name_prefix);
        self.onnx_graph
            .initializer
            .push(Self::vector_i64(ends_name.clone(), ends));

        let mut node = onnx::NodeProto {
            op_type: "Slice".to_string(),
            input: vec![input, starts_name, ends_name],
            output: vec![output],
            ..Default::default()
        };

        if let Some(axes) = axes {
            let axes_name = format!("{}_axes", name_prefix);
            self.onnx_graph
                .initializer
                .push(Self::vector_i64(axes_name.clone(), axes));
            node.input.push(axes_name);
        }

        if let Some(steps) = steps {
            if axes.is_none() {
                let axes_name = format!("{}_axes", name_prefix);
                let inferred_axes = (0..steps.len() as i64).collect::<Vec<_>>();
                self.onnx_graph
                    .initializer
                    .push(Self::vector_i64(axes_name.clone(), &inferred_axes));
                node.input.push(axes_name);
            }
            let steps_name = format!("{}_steps", name_prefix);
            self.onnx_graph
                .initializer
                .push(Self::vector_i64(steps_name.clone(), steps));
            node.input.push(steps_name);
        }

        self.onnx_graph.node.push(node);
        Ok(())
    }

    pub fn add_reduce_node(
        &mut self,
        op_type: &str,
        input: String,
        output: String,
        axes: Option<&[i64]>,
        keepdims: i64,
        name_prefix: &str,
    ) {
        let mut node = onnx::NodeProto {
            op_type: op_type.to_string(),
            input: vec![input],
            output: vec![output],
            ..Default::default()
        };

        if let Some(axes) = axes {
            if self.target_opset >= 18 {
                let axes_name = format!("{}_axes", name_prefix);
                self.onnx_graph
                    .initializer
                    .push(Self::vector_i64(axes_name.clone(), axes));
                node.input.push(axes_name);
            } else {
                node.attribute.push(helper::attr_ints("axes", axes));
            }
        }

        node.attribute.push(helper::attr_int("keepdims", keepdims));
        self.onnx_graph.node.push(node);
    }

    pub fn load_paddle_model(&mut self, json_path: &str) -> anyhow::Result<()> {
        log::info!("Loading PaddlePaddle JSON model: {}", json_path);
        let file = File::open(json_path)?;
        let file_len = file.metadata()?.len();
        if file_len > MAX_MODEL_JSON_BYTES {
            bail!(
                "model JSON is too large: {} bytes exceeds soft limit {} bytes",
                file_len,
                MAX_MODEL_JSON_BYTES
            );
        }
        let data: Value = serde_json::from_reader(file)?;

        let ops = data
            .pointer("/program/regions/0/blocks/0/ops")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Could not find ops array in JSON under /program/regions/0/blocks/0/ops"
                )
            })?;

        // Pass 1: Extract metadata, parameters, constants, and combines recursively.
        self.collect_pass1_from_ops(ops)?;

        // Paddle's save_combine writes vars in sorted-by-name order,
        // see python/paddle/static/io.py::_serialize_persistables.
        Arc::make_mut(&mut self.state.param_names.inner).sort_unstable();
        log::info!("Found {} parameters.", self.state.param_names.len());

        // Pass 2: Convert computation operations
        for op in ops {
            if let Some(op_type) = helper::op_type(op) {
                self.process_pass2_op(op_type, op)?;
            }
        }

        Ok(())
    }
}

type Pass1ConstantFolder =
    fn(&mut Converter, &Value, usize, &HashMap<i64, usize>) -> anyhow::Result<()>;

const PASS1_CONSTANT_FOLDERS: &[(&str, Pass1ConstantFolder)] = &[
    (paddle_op::FULL, Converter::fold_literal_constant),
    (paddle_op::FULL_INT_ARRAY, Converter::fold_literal_constant),
    (paddle_op::ASSIGN_VALUE, Converter::fold_literal_constant),
    (paddle_op::CAST, Converter::fold_cast_constant),
    (
        paddle_op::SQUEEZE,
        Converter::fold_value_preserving_shape_constant,
    ),
    (
        paddle_op::SQUEEZE_INPLACE,
        Converter::fold_value_preserving_shape_constant,
    ),
    (
        paddle_op::UNSQUEEZE,
        Converter::fold_value_preserving_shape_constant,
    ),
    (
        paddle_op::UNSQUEEZE_INPLACE,
        Converter::fold_value_preserving_shape_constant,
    ),
    (
        paddle_op::RESHAPE,
        Converter::fold_value_preserving_shape_constant,
    ),
    (
        paddle_op::FLATTEN,
        Converter::fold_value_preserving_shape_constant,
    ),
    (paddle_op::SCALE, Converter::fold_scale_constant),
];
