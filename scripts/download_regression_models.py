#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


DEFAULT_BASE_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0"
)
REQUIRED_FILES = ("inference.json", "inference.pdiparams")


def load_manifest(manifest_path: Path) -> list[tuple[str, Path]]:
    payload = json.loads(manifest_path.read_text())
    model_items = payload.get("models")
    if not isinstance(model_items, list) or not model_items:
        raise ValueError(f"Manifest {manifest_path} must contain a non-empty 'models' list")

    models: list[tuple[str, Path]] = []
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
        models.append((name, model_dir))
    return models


def select_models(
    models: list[tuple[str, Path]],
    requested: list[str] | None,
) -> list[tuple[str, Path]]:
    if not requested:
        return models

    available = {name for name, _ in models}
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown models: {', '.join(unknown)}. Available: {', '.join(sorted(available))}"
        )
    requested_set = set(requested)
    return [(name, model_dir) for name, model_dir in models if name in requested_set]


def has_required_files(model_dir: Path) -> bool:
    return model_dir.is_dir() and all((model_dir / required).is_file() for required in REQUIRED_FILES)


def download_archive(url: str, archive_path: Path) -> None:
    with urllib.request.urlopen(url) as response, archive_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def safe_extract_all(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()
            if destination.resolve() not in member_path.parents and member_path != destination.resolve():
                raise ValueError(f"Refusing to extract path outside destination: {member.name}")
        archive.extractall(destination)


def find_extracted_root(extract_dir: Path, expected_name: str) -> Path:
    expected = extract_dir / expected_name
    if expected.is_dir():
        return expected

    candidates = [path for path in extract_dir.rglob("*") if path.is_dir() and has_required_files(path)]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Could not locate extracted model directory '{expected_name}' under {extract_dir}"
    )


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def ensure_model(model_name: str, model_dir: Path, base_url: str, dry_run: bool) -> None:
    archive_name = f"{model_dir.name}.tar"
    archive_url = f"{base_url.rstrip('/')}/{archive_name}"

    if has_required_files(model_dir):
        print(f"[skip] {model_name}: {model_dir}")
        return

    if dry_run:
        print(f"[plan] {model_name}: {archive_url} -> {model_dir}")
        return

    print(f"[download] {model_name}: {archive_url}")
    model_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".download_{model_dir.name}_",
        dir=model_dir.parent,
    ) as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        archive_path = temp_dir / archive_name
        extract_dir = temp_dir / "extract"
        extract_dir.mkdir()

        download_archive(archive_url, archive_path)
        safe_extract_all(archive_path, extract_dir)
        extracted_root = find_extracted_root(extract_dir, model_dir.name)

        if model_dir.exists():
            remove_existing(model_dir)
        shutil.move(str(extracted_root), str(model_dir))

    if not has_required_files(model_dir):
        raise FileNotFoundError(
            f"Downloaded model '{model_name}' is missing required files under {model_dir}"
        )
    print(f"[ready] {model_name}: {model_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract regression inference models referenced by a manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON manifest describing regression models.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL used to fetch <model_dir_name>.tar archives.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional subset of model names from the manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved downloads without fetching archives.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    models = load_manifest(manifest_path)
    selected = select_models(models, args.models)
    for model_name, model_dir in selected:
        ensure_model(model_name, model_dir, args.base_url, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
