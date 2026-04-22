#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from regression.manifest import load_manifest, load_target_catalog, select_model_names
from regression.inventory import build_inventory
from regression.runtime import run_diff_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory regression targets and run Paddle vs ONNX diff tests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser("inventory", help="Inventory operator and subgraph targets.")
    inventory_parser.add_argument("--manifest", type=Path, required=True, help="JSON manifest describing regression models.")
    inventory_parser.add_argument("--models", nargs="*", help="Subset of model names from the manifest.")
    inventory_parser.add_argument("--json-out", type=Path)

    diff_parser = subparsers.add_parser("diff", help="Run Paddle vs ONNX diff tests for selected models.")
    diff_parser.add_argument("--manifest", type=Path, required=True, help="JSON manifest describing regression models.")
    diff_parser.add_argument("--models", nargs="*", help="Subset of model names from the manifest.")
    diff_parser.add_argument("--opsets", nargs="*", type=int)
    diff_parser.add_argument("--rtol", type=float, default=1e-3)
    diff_parser.add_argument("--atol", type=float, default=1e-4)
    diff_parser.add_argument("--seed", type=int, default=20260412)
    diff_parser.add_argument("--json-out", type=Path)
    diff_parser.add_argument("--keep-onnx", action="store_true")
    diff_parser.add_argument("--paddle-device", choices=["cpu", "gpu"], default="cpu")
    diff_parser.add_argument("--onnx-provider", default="CPUExecutionProvider")
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> int:
    args = parse_args()
    model_specs = load_manifest(args.manifest.resolve())
    model_names = select_model_names(model_specs, args.models)
    operator_targets, subgraph_targets = load_target_catalog()

    if args.command == "inventory":
        inventory = build_inventory(model_specs, model_names, operator_targets, subgraph_targets)
        if args.json_out:
            write_json(args.json_out, inventory)
        print(json.dumps(inventory, indent=2, ensure_ascii=False))
        return 0

    if args.command == "diff":
        report = run_diff_suite(
            model_specs=model_specs,
            model_names=model_names,
            operator_targets=operator_targets,
            subgraph_targets=subgraph_targets,
            opsets=args.opsets,
            rtol=args.rtol,
            atol=args.atol,
            seed=args.seed,
            keep_onnx=args.keep_onnx,
            paddle_device=args.paddle_device,
            onnx_provider=args.onnx_provider,
        )
        if args.json_out:
            write_json(args.json_out, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["passed"] else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
