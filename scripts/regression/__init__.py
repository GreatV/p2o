from .compare import compare_outputs
from .inventory import (
    build_inventory,
    collect_model_inventory,
    extract_struct_name,
    get_model_ops,
    load_model_ir,
)
from .io import (
    infer_image_hw,
    infer_input_info,
    make_generic_input,
    make_input_tensors,
    materialize_input_spec,
    numpy_dtype_for_tensor_type,
    resolve_generic_shape,
)
from .manifest import ModelSpec, load_manifest, load_target_catalog, select_model_names
from .runtime import (
    REPO_ROOT,
    convert_model,
    ensure_converter_binary,
    load_semantic_baseline,
    run_diff_suite,
    run_onnx,
    run_paddle,
)

__all__ = [
    "REPO_ROOT",
    "ModelSpec",
    "build_inventory",
    "collect_model_inventory",
    "compare_outputs",
    "convert_model",
    "ensure_converter_binary",
    "extract_struct_name",
    "get_model_ops",
    "infer_image_hw",
    "infer_input_info",
    "load_manifest",
    "load_model_ir",
    "load_semantic_baseline",
    "load_target_catalog",
    "make_generic_input",
    "make_input_tensors",
    "materialize_input_spec",
    "numpy_dtype_for_tensor_type",
    "resolve_generic_shape",
    "run_diff_suite",
    "run_onnx",
    "run_paddle",
    "select_model_names",
]
