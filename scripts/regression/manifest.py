from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TARGETS_CONFIG = Path(__file__).resolve().parents[1] / "regression_targets.json"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_dir: Path
    default_opsets: tuple[int, ...]
    input_kind: str | None = None
    sample_inputs: dict[str, dict[str, Any]] | None = None
    diff_rtol: float | None = None
    diff_atol: float | None = None
    compare_config: dict[str, Any] | None = None
    semantic_baseline: Path | None = None


def load_target_catalog() -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    payload = json.loads(TARGETS_CONFIG.read_text())

    operator_targets = payload.get("operator_targets")
    if not isinstance(operator_targets, dict) or not operator_targets:
        raise ValueError(f"Target catalog {TARGETS_CONFIG} must contain a non-empty 'operator_targets' object")

    subgraph_targets = payload.get("subgraph_targets")
    if not isinstance(subgraph_targets, list) or not subgraph_targets:
        raise ValueError(f"Target catalog {TARGETS_CONFIG} must contain a non-empty 'subgraph_targets' list")

    return operator_targets, subgraph_targets


def load_manifest(manifest_path: Path) -> dict[str, ModelSpec]:
    payload = json.loads(manifest_path.read_text())
    model_items = payload.get("models")
    if not isinstance(model_items, list) or not model_items:
        raise ValueError(f"Manifest {manifest_path} must contain a non-empty 'models' list")

    specs: dict[str, ModelSpec] = {}
    for item in model_items:
        if not isinstance(item, dict):
            raise ValueError(f"Manifest {manifest_path} contains a non-object model entry")
        name = item.get("name")
        model_dir_raw = item.get("model_dir")
        if not isinstance(name, str) or not name:
            raise ValueError("Each model entry must define a non-empty string 'name'")
        if not isinstance(model_dir_raw, str) or not model_dir_raw:
            raise ValueError(f"Model '{name}' must define a non-empty string 'model_dir'")
        model_dir = Path(model_dir_raw)
        if not model_dir.is_absolute():
            model_dir = (manifest_path.parent / model_dir).resolve()

        default_opsets_raw = item.get("default_opsets", [17])
        if not isinstance(default_opsets_raw, list) or not default_opsets_raw or not all(isinstance(v, int) for v in default_opsets_raw):
            raise ValueError(f"Model '{name}' must define 'default_opsets' as a non-empty int list")

        input_kind = item.get("input_kind")
        if input_kind is not None and not isinstance(input_kind, str):
            raise ValueError(f"Model '{name}' has invalid 'input_kind'")

        sample_inputs = item.get("sample_inputs")
        if sample_inputs is not None and not isinstance(sample_inputs, dict):
            raise ValueError(f"Model '{name}' has invalid 'sample_inputs'")

        compare_config = item.get("compare_config")
        if compare_config is not None and not isinstance(compare_config, dict):
            raise ValueError(f"Model '{name}' has invalid 'compare_config'")

        semantic_baseline_raw = item.get("semantic_baseline")
        semantic_baseline: Path | None = None
        if semantic_baseline_raw is not None:
            if not isinstance(semantic_baseline_raw, str) or not semantic_baseline_raw:
                raise ValueError(f"Model '{name}' has invalid 'semantic_baseline'")
            semantic_baseline = Path(semantic_baseline_raw)
            if not semantic_baseline.is_absolute():
                semantic_baseline = (manifest_path.parent / semantic_baseline).resolve()

        diff_tolerance = item.get("diff_tolerance")
        diff_rtol: float | None = None
        diff_atol: float | None = None
        if diff_tolerance is not None:
            if not isinstance(diff_tolerance, dict):
                raise ValueError(f"Model '{name}' has invalid 'diff_tolerance'")
            diff_rtol_raw = diff_tolerance.get("rtol")
            diff_atol_raw = diff_tolerance.get("atol")
            if diff_rtol_raw is not None:
                if not isinstance(diff_rtol_raw, (int, float)):
                    raise ValueError(f"Model '{name}' has invalid diff_tolerance.rtol")
                diff_rtol = float(diff_rtol_raw)
            if diff_atol_raw is not None:
                if not isinstance(diff_atol_raw, (int, float)):
                    raise ValueError(f"Model '{name}' has invalid diff_tolerance.atol")
                diff_atol = float(diff_atol_raw)

        specs[name] = ModelSpec(
            name=name,
            model_dir=model_dir,
            default_opsets=tuple(default_opsets_raw),
            input_kind=input_kind,
            sample_inputs=sample_inputs,
            diff_rtol=diff_rtol,
            diff_atol=diff_atol,
            compare_config=compare_config,
            semantic_baseline=semantic_baseline,
        )
    return specs


def select_model_names(model_specs: dict[str, ModelSpec], requested: list[str] | None) -> list[str]:
    if not requested:
        return list(model_specs)
    unknown = [name for name in requested if name not in model_specs]
    if unknown:
        available = ", ".join(sorted(model_specs))
        raise ValueError(f"Unknown models: {', '.join(unknown)}. Available: {available}")
    return requested
