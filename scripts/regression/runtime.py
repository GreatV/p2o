from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .compare import compare_outputs
from .inventory import build_inventory
from .io import infer_input_info, make_input_tensors
from .manifest import ModelSpec


REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_converter_binary() -> Path:
    subprocess.run(["cargo", "build", "--quiet"], cwd=REPO_ROOT, check=True)
    return REPO_ROOT / "target" / "debug" / "p2o"


def convert_model(binary: Path, model_spec: ModelSpec, output_path: Path, opset: int) -> None:
    subprocess.run(
        [
            str(binary),
            str(model_spec.model_dir / "inference.json"),
            str(model_spec.model_dir / "inference.pdiparams"),
            str(output_path),
            "--opset",
            str(opset),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def run_paddle(
    model_spec: ModelSpec,
    inputs: dict[str, Any],
    device: str,
    seed: int = 20260412,
) -> list[Any]:
    import paddle

    paddle.seed(seed)

    config = paddle.inference.Config(
        str(model_spec.model_dir / "inference.json"),
        str(model_spec.model_dir / "inference.pdiparams"),
    )
    if device == "cpu":
        config.disable_gpu()
        config.disable_onednn()
    elif device == "gpu":
        if not paddle.device.is_compiled_with_cuda():
            raise RuntimeError("Paddle is not compiled with CUDA support")
        config.enable_use_gpu(1024, 0)
    else:
        raise ValueError(f"Unsupported Paddle device: {device}")
    predictor = paddle.inference.create_predictor(config)
    for name in predictor.get_input_names():
        predictor.get_input_handle(name).copy_from_cpu(inputs[name])
    predictor.run()
    return [predictor.get_output_handle(name).copy_to_cpu() for name in predictor.get_output_names()]


def run_onnx(model_path: Path, inputs: dict[str, Any], provider: str) -> list[Any]:
    import onnxruntime as ort

    available_providers = ort.get_available_providers()
    if provider not in available_providers:
        available = ", ".join(available_providers)
        raise RuntimeError(f"ONNX Runtime provider '{provider}' is unavailable. Available providers: {available}")
    session = ort.InferenceSession(str(model_path), providers=[provider])
    feed = {inp.name: inputs[inp.name] for inp in session.get_inputs()}
    output_names = [output.name for output in session.get_outputs()]
    return session.run(output_names, feed)


def load_semantic_baseline(path: Path) -> list[Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as payload:
        names = sorted(
            payload.files,
            key=lambda name: int(name.split("_", 1)[1]) if name.startswith("output_") else -1,
        )
        if not names:
            raise ValueError(f"Semantic baseline '{path}' does not contain any outputs")
        outputs = []
        for index, name in enumerate(names):
            expected = f"output_{index}"
            if name != expected:
                raise ValueError(
                    f"Semantic baseline '{path}' has unexpected key '{name}', expected '{expected}'"
                )
            outputs.append(payload[name])
        return outputs


def run_diff_suite(
    model_specs: dict[str, ModelSpec],
    model_names: list[str],
    operator_targets: dict[str, dict[str, str]],
    subgraph_targets: list[dict[str, str]],
    opsets: list[int] | None,
    rtol: float,
    atol: float,
    seed: int,
    keep_onnx: bool,
    paddle_device: str,
    onnx_provider: str,
) -> dict[str, Any]:
    binary = ensure_converter_binary()
    inventory = build_inventory(model_specs, model_names, operator_targets, subgraph_targets)
    reports = []

    tmp_dir = REPO_ROOT / ".tmp_regression"
    tmp_dir.mkdir(exist_ok=True)

    for model_name in model_names:
        model_spec = model_specs[model_name]
        input_info = infer_input_info(model_spec)
        inputs = make_input_tensors(model_spec, input_info, seed)
        model_inventory = next(item for item in inventory["models"] if item["model"] == model_name)
        target_opsets = opsets if opsets is not None else list(model_spec.default_opsets)

        for opset in target_opsets:
            onnx_path = tmp_dir / f"{model_name}_opset{opset}.onnx"
            effective_rtol = model_spec.diff_rtol if model_spec.diff_rtol is not None else rtol
            effective_atol = model_spec.diff_atol if model_spec.diff_atol is not None else atol
            report_item: dict[str, Any] = {
                "model": model_name,
                "model_dir": str(model_spec.model_dir),
                "opset": opset,
                "onnx_path": str(onnx_path),
                "rtol": effective_rtol,
                "atol": effective_atol,
                "paddle_device": paddle_device,
                "onnx_provider": onnx_provider,
                "covered_operator_targets": model_inventory["operator_targets"],
                "covered_subgraph_targets": [item["label"] for item in model_inventory["subgraph_targets"]],
            }
            try:
                convert_model(binary, model_spec, onnx_path, opset)
                onnx_outputs = run_onnx(onnx_path, inputs, onnx_provider)
                if model_spec.semantic_baseline is not None:
                    reference_outputs = load_semantic_baseline(model_spec.semantic_baseline)
                    report_item["reference_kind"] = "semantic_baseline"
                    report_item["reference_path"] = str(model_spec.semantic_baseline)
                else:
                    reference_outputs = run_paddle(model_spec, inputs, paddle_device, seed)
                    report_item["reference_kind"] = "paddle"
                report_item["diff"] = compare_outputs(
                    reference_outputs,
                    onnx_outputs,
                    rtol=effective_rtol,
                    atol=effective_atol,
                    compare_config=model_spec.compare_config,
                )
                report_item["status"] = "ok"
            except Exception as exc:
                report_item["status"] = "error"
                report_item["error"] = f"{type(exc).__name__}: {exc}"
                report_item["diff"] = {"passed": False, "outputs": [], "output_count_match": False}
            reports.append(report_item)
            if not keep_onnx:
                onnx_path.unlink(missing_ok=True)

    return {
        "rtol": rtol,
        "atol": atol,
        "seed": seed,
        "results": reports,
        "passed": all(item["status"] == "ok" and item["diff"]["passed"] for item in reports),
    }
