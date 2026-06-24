#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_BASE_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0"
)
REQUIRED_FILES = ("inference.json", "inference.pdiparams")
OPTIONAL_FILES = ("inference.yml",)
# Bound every network read so a hung connection can't stall CI indefinitely.
DOWNLOAD_TIMEOUT = 30


class ModelDownload:
    def __init__(
        self,
        name: str,
        model_dir: Path,
        hf_repo: str | None = None,
        hf_revision: str = "main",
    ):
        self.name = name
        self.model_dir = model_dir
        self.hf_repo = hf_repo
        self.hf_revision = hf_revision


def load_manifest(manifest_path: Path) -> list[ModelDownload]:
    payload = json.loads(manifest_path.read_text())
    model_items = payload.get("models")
    if not isinstance(model_items, list) or not model_items:
        raise ValueError(f"Manifest {manifest_path} must contain a non-empty 'models' list")

    models: list[ModelDownload] = []
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
        hf_repo = item.get("hf_repo")
        if hf_repo is not None and (not isinstance(hf_repo, str) or not hf_repo):
            raise ValueError(f"Model '{name}' has invalid 'hf_repo'")
        hf_revision = item.get("hf_revision", "main")
        if not isinstance(hf_revision, str) or not hf_revision:
            raise ValueError(f"Model '{name}' has invalid 'hf_revision'")
        models.append(ModelDownload(name, model_dir, hf_repo, hf_revision))
    return models


def select_models(
    models: list[ModelDownload],
    requested: list[str] | None,
) -> list[ModelDownload]:
    if not requested:
        return models

    available = {model.name for model in models}
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown models: {', '.join(unknown)}. Available: {', '.join(sorted(available))}"
        )
    requested_set = set(requested)
    return [model for model in models if model.name in requested_set]


def has_required_files(model_dir: Path) -> bool:
    return model_dir.is_dir() and all((model_dir / required).is_file() for required in REQUIRED_FILES)


def download_archive(url: str, archive_path: Path) -> None:
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response, archive_path.open(
        "wb"
    ) as handle:
        shutil.copyfileobj(response, handle)


def download_file(url: str, output_path: Path) -> None:
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response, output_path.open(
        "wb"
    ) as handle:
        shutil.copyfileobj(response, handle)


def safe_extract_all(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise ValueError(
                    f"Refusing to extract unsupported tar member type: {member.name}"
                )
            if not (member.isdir() or member.isfile()):
                raise ValueError(
                    f"Refusing to extract unsupported tar member type: {member.name}"
                )

            member_path = (destination / member.name).resolve()
            if member_path != destination and destination not in member_path.parents:
                raise ValueError(
                    f"Refusing to extract path outside destination: {member.name}"
                )

            if member.isdir():
                member_path.mkdir(parents=True, exist_ok=True)
                continue

            member_path.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"Could not read file from archive: {member.name}")
            with extracted, member_path.open("wb") as handle:
                shutil.copyfileobj(extracted, handle)
            if member.mode:
                member_path.chmod(member.mode & 0o777)


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


def hf_file_url(repo: str, filename: str, revision: str = "main") -> str:
    quoted_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo.split("/"))
    quoted_revision = urllib.parse.quote(revision, safe="")
    quoted_filename = urllib.parse.quote(filename, safe="/")
    return f"https://huggingface.co/{quoted_repo}/resolve/{quoted_revision}/{quoted_filename}"


def ensure_hf_model(model: ModelDownload, dry_run: bool) -> None:
    if has_required_files(model.model_dir):
        print(f"[skip] {model.name}: {model.model_dir}")
        return

    assert model.hf_repo is not None
    if dry_run:
        for filename in REQUIRED_FILES + OPTIONAL_FILES:
            url = hf_file_url(model.hf_repo, filename, model.hf_revision)
            print(f"[plan] {model.name}: {url} -> {model.model_dir / filename}")
        return

    print(f"[download] {model.name}: hf://{model.hf_repo}@{model.hf_revision}")
    model.model_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".download_{model.model_dir.name}_",
        dir=model.model_dir.parent,
    ) as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        download_dir = temp_dir / "model"
        download_dir.mkdir()
        for filename in REQUIRED_FILES:
            download_file(
                hf_file_url(model.hf_repo, filename, model.hf_revision),
                download_dir / filename,
            )
        for filename in OPTIONAL_FILES:
            try:
                download_file(
                    hf_file_url(model.hf_repo, filename, model.hf_revision),
                    download_dir / filename,
                )
            except Exception as exc:
                print(f"[warn] {model.name}: optional {filename} not downloaded ({exc})")

        if model.model_dir.exists():
            remove_existing(model.model_dir)
        shutil.move(str(download_dir), str(model.model_dir))

    if not has_required_files(model.model_dir):
        raise FileNotFoundError(
            f"Downloaded model '{model.name}' is missing required files under {model.model_dir}"
        )
    print(f"[ready] {model.name}: {model.model_dir}")


def ensure_tar_model(model: ModelDownload, base_url: str, dry_run: bool) -> None:
    archive_name = f"{model.model_dir.name}.tar"
    archive_url = f"{base_url.rstrip('/')}/{archive_name}"

    if has_required_files(model.model_dir):
        print(f"[skip] {model.name}: {model.model_dir}")
        return

    if dry_run:
        print(f"[plan] {model.name}: {archive_url} -> {model.model_dir}")
        return

    print(f"[download] {model.name}: {archive_url}")
    model.model_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".download_{model.model_dir.name}_",
        dir=model.model_dir.parent,
    ) as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        archive_path = temp_dir / archive_name
        extract_dir = temp_dir / "extract"
        extract_dir.mkdir()

        download_archive(archive_url, archive_path)
        safe_extract_all(archive_path, extract_dir)
        extracted_root = find_extracted_root(extract_dir, model.model_dir.name)

        if model.model_dir.exists():
            remove_existing(model.model_dir)
        shutil.move(str(extracted_root), str(model.model_dir))

    if not has_required_files(model.model_dir):
        raise FileNotFoundError(
            f"Downloaded model '{model.name}' is missing required files under {model.model_dir}"
        )
    print(f"[ready] {model.name}: {model.model_dir}")


def ensure_model(model: ModelDownload, base_url: str, dry_run: bool) -> None:
    if model.hf_repo is not None:
        ensure_hf_model(model, dry_run)
    else:
        ensure_tar_model(model, base_url, dry_run)


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
    for model in selected:
        ensure_model(model, args.base_url, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
