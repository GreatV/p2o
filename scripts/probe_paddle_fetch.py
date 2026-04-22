#!/usr/bin/env python3
"""Probe internal Paddle tensors by temporarily retargeting fetch ops.

This is a maintained diagnostic helper for regression triage, not a one-off
spike. It reuses regression_suite input generation/run helpers so CPU/GPU or
Paddle/ONNX discrepancies can be localized to selected tensor ids.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from regression import infer_input_info, load_manifest, make_input_tensors, run_paddle


REPO_ROOT = Path(__file__).resolve().parents[1]


def tensor_meta_by_id(ops: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    meta: dict[int, dict[str, Any]] = {}
    for op in ops:
        outputs = op.get("O", [])
        if isinstance(outputs, dict):
            outputs = [outputs]
        for output in outputs:
            if isinstance(output, dict) and "%" in output:
                meta[int(output["%"])] = output
    return meta


def rewrite_fetch_model(model_dir: Path, tensor_ids: list[int], out_dir: Path) -> None:
    model_json = model_dir / "inference.json"
    model_params = model_dir / "inference.pdiparams"
    data = json.loads(model_json.read_text())
    ops = data["program"]["regions"][0]["blocks"][0]["ops"]
    fetch_indices = [idx for idx, op in enumerate(ops) if op.get("#") == "1.fetch"]
    if len(tensor_ids) > len(fetch_indices):
        raise ValueError(f"Requested {len(tensor_ids)} tensors, but model only has {len(fetch_indices)} fetch ops")

    tensor_meta = tensor_meta_by_id(ops)
    for fetch_idx, tensor_id in zip(fetch_indices, tensor_ids):
        if tensor_id not in tensor_meta:
            raise ValueError(f"Tensor %{tensor_id} not found in inference.json")
        fetch_op = ops[fetch_idx]
        fetch_op["I"] = [{"%": tensor_id}]
        fetch_output = fetch_op["O"]
        if isinstance(fetch_output, list) and fetch_output and isinstance(fetch_output[0], dict):
            fetch_output[0]["TT"] = tensor_meta[tensor_id]["TT"]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inference.json").write_text(json.dumps(data, separators=(",", ":")))
    target = out_dir / "inference.pdiparams"
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(model_params)


def summarize_diff(reference, candidate) -> dict[str, Any]:
    import numpy as np

    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    if ref.shape != cand.shape:
        return {
            "shape_match": False,
            "reference_shape": list(ref.shape),
            "candidate_shape": list(cand.shape),
            "reference_dtype": str(ref.dtype),
            "candidate_dtype": str(cand.dtype),
        }

    if np.issubdtype(ref.dtype, np.integer) or np.issubdtype(ref.dtype, np.bool_):
        diff = ref.astype(np.int64, copy=False) - cand.astype(np.int64, copy=False)
    else:
        diff = ref.astype(np.float64, copy=False) - cand.astype(np.float64, copy=False)
    abs_diff = np.abs(diff)
    ref_abs_mean = float(np.mean(np.abs(ref.astype(np.float64, copy=False)))) if ref.size else 0.0
    return {
        "shape_match": True,
        "reference_shape": list(ref.shape),
        "candidate_shape": list(cand.shape),
        "reference_dtype": str(ref.dtype),
        "candidate_dtype": str(cand.dtype),
        "max_abs_diff": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "mean_abs_diff": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "reference_mean_abs": ref_abs_mean,
        "relative_mean_abs_diff": float(np.mean(abs_diff) / (ref_abs_mean + 1e-12)) if abs_diff.size else 0.0,
        "reference_nan_count": int(np.isnan(ref).sum()) if np.issubdtype(ref.dtype, np.floating) else 0,
        "candidate_nan_count": int(np.isnan(cand).sum()) if np.issubdtype(cand.dtype, np.floating) else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe internal Paddle tensors by rewriting fetch ops.")
    parser.add_argument("--manifest", type=Path, default=Path("scripts/regression_models.json"))
    parser.add_argument("--model", default="ppdoclayoutv3")
    parser.add_argument("--seed", type=int, default=20260412)
    parser.add_argument("--devices", nargs=2, default=["cpu", "gpu"])
    parser.add_argument("--tensors", nargs="+", type=int, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    specs = load_manifest(args.manifest)
    if args.model not in specs:
        raise ValueError(f"Unknown model '{args.model}'")
    model_spec = specs[args.model]

    input_info = infer_input_info(model_spec)
    inputs = make_input_tensors(model_spec, input_info, args.seed)

    with tempfile.TemporaryDirectory(prefix=f"{args.model}_probe_", dir=REPO_ROOT / ".tmp_regression") as tmp_raw:
        tmp_dir = Path(tmp_raw)
        rewrite_fetch_model(model_spec.model_dir, args.tensors, tmp_dir)
        probe_spec = replace(model_spec, model_dir=tmp_dir)

        outputs_by_device: dict[str, list[Any]] = {}
        for device in args.devices:
            outputs_by_device[device] = run_paddle(probe_spec, inputs, device)

    reference_name, candidate_name = args.devices
    results = []
    for index, tensor_id in enumerate(args.tensors):
        results.append(
            {
                "tensor_id": tensor_id,
                reference_name: {
                    "shape": list(outputs_by_device[reference_name][index].shape),
                    "dtype": str(outputs_by_device[reference_name][index].dtype),
                },
                candidate_name: {
                    "shape": list(outputs_by_device[candidate_name][index].shape),
                    "dtype": str(outputs_by_device[candidate_name][index].dtype),
                },
                "diff": summarize_diff(
                    outputs_by_device[reference_name][index],
                    outputs_by_device[candidate_name][index],
                ),
            }
        )

    report = {
        "model": args.model,
        "seed": args.seed,
        "devices": args.devices,
        "results": results,
    }
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
