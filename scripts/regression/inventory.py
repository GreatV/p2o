from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .manifest import ModelSpec


def load_model_ir(model_json: Path) -> dict[str, Any]:
    return json.loads(model_json.read_text())


def get_model_ops(data: dict[str, Any]) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []

    def walk_region(region: dict[str, Any]) -> None:
        for block in region.get("blocks", []):
            for op in block.get("ops", []):
                if not isinstance(op, dict) or "#" not in op:
                    continue
                collected.append(op)
                for sub_region in op.get("regions", []):
                    if isinstance(sub_region, dict):
                        walk_region(sub_region)

    walk_region(data["program"]["regions"][0])
    return collected


def extract_struct_name(op: dict[str, Any]) -> str | None:
    attrs = op.get("A")
    if not isinstance(attrs, list):
        return None
    for attr in attrs:
        if isinstance(attr, dict) and attr.get("N") == "struct_name":
            at = attr.get("AT")
            if isinstance(at, dict):
                value = at.get("D")
                if isinstance(value, str) and value:
                    return value
    return None


def collect_model_inventory(model_spec: ModelSpec, subgraph_targets: list[dict[str, str]]) -> dict[str, Any]:
    data = load_model_ir(model_spec.model_dir / "inference.json")
    ops = get_model_ops(data)
    op_types = sorted({op["#"] for op in ops})
    struct_names = sorted({name for op in ops if (name := extract_struct_name(op))})
    subgraphs: list[dict[str, Any]] = []
    for target in subgraph_targets:
        matched_ops = [op for op in ops if target["match"] in (extract_struct_name(op) or "")]
        if not matched_ops:
            continue
        subgraphs.append(
            {
                "id": target["id"],
                "label": target["label"],
                "match": target["match"],
                "op_types": sorted({op["#"] for op in matched_ops}),
                "op_count": len(matched_ops),
            }
        )
    return {
        "model": model_spec.name,
        "model_dir": str(model_spec.model_dir),
        "operator_targets": op_types,
        "struct_names": struct_names,
        "subgraph_targets": subgraphs,
    }


def build_inventory(
    model_specs: dict[str, ModelSpec],
    model_names: list[str],
    operator_targets: dict[str, dict[str, str]],
    subgraph_targets: list[dict[str, str]],
) -> dict[str, Any]:
    per_model = [collect_model_inventory(model_specs[name], subgraph_targets) for name in model_names]
    covered_ops = sorted({op for model in per_model for op in model["operator_targets"]})

    operator_target_items = []
    for op_name in sorted(operator_targets):
        meta = operator_targets[op_name]
        used_by = [model["model"] for model in per_model if op_name in model["operator_targets"]]
        operator_target_items.append(
            {
                "op_name": op_name,
                "kind": meta["kind"],
                "implementation": meta["impl"],
                "note": meta["note"],
                "covered_by_models": used_by,
                "currently_covered": bool(used_by),
            }
        )

    subgraph_target_items = []
    for target in subgraph_targets:
        matched_models = []
        for model in per_model:
            for subgraph in model["subgraph_targets"]:
                if subgraph["id"] == target["id"]:
                    matched_models.append(
                        {
                            "model": model["model"],
                            "op_count": subgraph["op_count"],
                            "op_types": subgraph["op_types"],
                        }
                    )
        subgraph_target_items.append(
            {
                "id": target["id"],
                "label": target["label"],
                "match": target["match"],
                "covered_by_models": matched_models,
                "currently_covered": bool(matched_models),
            }
        )

    return {
        "models": per_model,
        "operator_targets": operator_target_items,
        "subgraph_targets": subgraph_target_items,
        "summary": {
            "selected_models": model_names,
            "implemented_operator_target_count": len(operator_targets),
            "covered_operator_target_count": len(covered_ops),
            "implemented_subgraph_target_count": len(subgraph_targets),
            "covered_subgraph_target_count": sum(1 for item in subgraph_target_items if item["currently_covered"]),
        },
    }
