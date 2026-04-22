from __future__ import annotations

from typing import Any

from .inventory import get_model_ops, load_model_ir
from .manifest import ModelSpec


def infer_input_info(model_spec: ModelSpec) -> dict[str, dict[str, Any]]:
    data = load_model_ir(model_spec.model_dir / "inference.json")
    info: dict[str, dict[str, Any]] = {}
    for op in get_model_ops(data):
        if op["#"] != "1.data":
            continue
        attrs = {
            attr["N"]: attr["AT"]["D"]
            for attr in op.get("A", [])
            if isinstance(attr, dict) and isinstance(attr.get("AT"), dict)
        }
        output = op["O"][0]
        tensor_type = output["TT"]["D"][0]["#"]
        shape = output["TT"]["D"][1]
        info[attrs["name"]] = {"type": tensor_type, "shape": shape}
    return info


def numpy_dtype_for_tensor_type(tensor_type: str) -> Any:
    import numpy as np

    mapping = {
        "0.t_f32": np.float32,
        "0.t_f64": np.float64,
        "0.t_i32": np.int32,
        "0.t_i64": np.int64,
        "0.t_ui8": np.uint8,
        "0.t_bool": np.bool_,
    }
    return np.dtype(mapping.get(tensor_type, np.float32))


def materialize_input_spec(
    name: str,
    spec: dict[str, Any],
    inferred_dtype,
    rng,
):
    import numpy as np

    if not isinstance(spec, dict):
        raise ValueError(f"sample_inputs['{name}'] must be an object")

    dtype = np.dtype(spec.get("dtype", inferred_dtype))
    if "value" in spec:
        return np.asarray(spec["value"], dtype=dtype)

    shape = spec.get("shape")
    if not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
        raise ValueError(f"sample_inputs['{name}'] must define either 'value' or int-list 'shape'")

    mode = spec.get("mode")
    if mode is None:
        if np.issubdtype(dtype, np.floating):
            mode = "randn"
        elif np.issubdtype(dtype, np.bool_):
            mode = "zeros"
        else:
            mode = "ones"

    if mode == "randn":
        return rng.standard_normal(tuple(shape), dtype=np.float32).astype(dtype, copy=False)
    if mode == "zeros":
        return np.zeros(tuple(shape), dtype=dtype)
    if mode == "ones":
        return np.ones(tuple(shape), dtype=dtype)
    raise ValueError(f"Unsupported sample_inputs['{name}'].mode: {mode}")


def resolve_generic_shape(name: str, shape: list[int], input_kind: str | None) -> tuple[int, ...]:
    if not shape:
        return ()

    resolved: list[int] = []
    rank = len(shape)
    for idx, dim in enumerate(shape):
        if dim != -1:
            resolved.append(dim)
            continue

        if idx == 0:
            resolved.append(1)
            continue

        if input_kind == "rec" and rank == 4 and idx == 2:
            resolved.append(48)
            continue
        if input_kind == "rec" and rank == 4 and idx == 3:
            resolved.append(320)
            continue
        if input_kind == "det" and rank == 4 and idx in (2, 3):
            resolved.append(640)
            continue
        if input_kind == "layout" and rank == 4 and idx in (2, 3):
            resolved.append(800)
            continue

        if name == "im_shape" and rank == 2 and idx == 1:
            resolved.append(2)
            continue
        if name == "scale_factor" and rank == 2 and idx == 1:
            resolved.append(2)
            continue

        if rank == 4 and idx in (2, 3):
            resolved.append(224)
            continue

        resolved.append(1)
    return tuple(resolved)


def make_generic_input(
    name: str,
    tensor_info: dict[str, Any],
    rng,
    input_kind: str | None,
    image_hw: tuple[int, int] | None,
) -> Any:
    import numpy as np

    dtype = numpy_dtype_for_tensor_type(tensor_info["type"])
    shape = list(tensor_info["shape"])
    resolved_shape = resolve_generic_shape(name, shape, input_kind)

    if name == "im_shape":
        height, width = image_hw or (resolved_shape[-2] if len(resolved_shape) >= 2 else 224, resolved_shape[-1] if len(resolved_shape) >= 1 else 224)
        batch = resolved_shape[0] if resolved_shape else 1
        return np.asarray([[float(height), float(width)]] * batch, dtype=dtype)

    if name == "scale_factor":
        batch = resolved_shape[0] if resolved_shape else 1
        return np.asarray([[1.0, 1.0]] * batch, dtype=dtype)

    if np.issubdtype(dtype, np.floating):
        if not resolved_shape:
            return np.asarray(rng.standard_normal(), dtype=dtype)
        return rng.standard_normal(resolved_shape, dtype=np.float32).astype(dtype, copy=False)

    if np.issubdtype(dtype, np.bool_):
        return np.zeros(resolved_shape, dtype=dtype) if resolved_shape else np.asarray(False, dtype=dtype)

    return np.ones(resolved_shape, dtype=dtype) if resolved_shape else np.asarray(1, dtype=dtype)


def infer_image_hw(
    model_spec: ModelSpec,
    input_info: dict[str, dict[str, Any]],
    inputs: dict[str, Any],
) -> tuple[int, int] | None:
    for name in ("image", "x"):
        if name in inputs and inputs[name].ndim >= 4:
            return int(inputs[name].shape[-2]), int(inputs[name].shape[-1])
    for name in ("image", "x"):
        tensor_info = input_info.get(name)
        if tensor_info is None:
            continue
        resolved = resolve_generic_shape(name, list(tensor_info["shape"]), model_spec.input_kind)
        if len(resolved) >= 4:
            return int(resolved[-2]), int(resolved[-1])
    return None


def make_input_tensors(model_spec: ModelSpec, input_info: dict[str, dict[str, Any]], seed: int) -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    inputs: dict[str, Any] = {}
    sample_inputs = model_spec.sample_inputs or {}

    for name, tensor_info in input_info.items():
        inferred_dtype = numpy_dtype_for_tensor_type(tensor_info["type"])
        if name in sample_inputs:
            inputs[name] = materialize_input_spec(name, sample_inputs[name], inferred_dtype, rng)

    image_hw = infer_image_hw(model_spec, input_info, inputs)

    for name, tensor_info in input_info.items():
        if name in inputs:
            continue
        inputs[name] = make_generic_input(name, tensor_info, rng, model_spec.input_kind, image_hw)
    return inputs
