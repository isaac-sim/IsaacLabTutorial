"""Integrity checks for the retained reproducibility artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from so101_vial_place.reset.dataset import load_reset_dataset

REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST = REPOSITORY / "checkpoints" / "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_records(value: Any):
    if isinstance(value, dict):
        path = value.get("path")
        digest = value.get("sha256")
        if isinstance(path, str) and path.startswith("checkpoints/") and isinstance(digest, str):
            yield path, digest
        for child in value.values():
            yield from _artifact_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _artifact_records(child)


@pytest.mark.parametrize("record_name", ["reset_dataset", "canonical_bridge_reset_dataset"])
def test_manifest_reset_dataset_matches_bundled_artifact(record_name):
    manifest = json.loads(MANIFEST.read_text())
    record = manifest[record_name]
    path = REPOSITORY / record["path"]
    artifact = load_reset_dataset(path)

    assert artifact["row_count"] == record["rows"]
    assert artifact["schema_version"] == record["schema_version"]
    assert artifact["content_sha256"] == record["content_sha256"]
    assert _sha256(path) == record["file_sha256"]


def test_manifest_retained_artifact_hashes_match_files():
    manifest = json.loads(MANIFEST.read_text())
    records = set(_artifact_records(manifest))

    assert records
    for relative_path, expected_digest in records:
        path = REPOSITORY / relative_path
        assert path.is_file(), relative_path
        assert _sha256(path) == expected_digest, relative_path


def test_manifest_isaaclab_commit_is_pinned_in_lockfile():
    manifest = json.loads(MANIFEST.read_text())
    commit = manifest["isaaclab"]["resolved_commit"]
    lockfile = (REPOSITORY / "uv.lock").read_text()

    assert len(commit) == 40
    assert f"#{commit}" in lockfile
