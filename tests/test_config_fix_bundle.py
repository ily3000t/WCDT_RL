from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from safe_rl.accvp.contracts.schema import file_sha256
from safe_rl.pipeline.export_config_fix_bundle import standalone_payload
from safe_rl.ppo_replicates import plain
from safe_rl.utils.config import REPO_ROOT, STANDALONE_CONFIG_KEY


FROZEN = (
    REPO_ROOT
    / "safe_rl/config/config_fix/accvp_vnext_selector4_hybrid_v4_frozen"
)
def _yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    assert isinstance(payload, Mapping)
    return dict(payload)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, Mapping)
    return dict(payload)


def test_frozen_config_fix_snapshots_are_exact_and_have_no_extends() -> None:
    manifest = _json(FROZEN / "equivalence_manifest.json")
    assert manifest["mode"] == "standalone_exact_snapshot"
    assert len(manifest["entries"]) == 10
    for record in manifest["entries"]:
        source = REPO_ROOT / record["source"]
        snapshot = FROZEN / record["name"]
        assert file_sha256(source) == record["source_sha256"]
        assert file_sha256(snapshot) == record["snapshot_sha256"]
        assert plain(_yaml(snapshot)) == plain(standalone_payload(source))
        assert _yaml(snapshot)[STANDALONE_CONFIG_KEY] is True
        assert "extends" not in _yaml(snapshot)
    for path in FROZEN.glob("*.yaml"):
        assert "extends" not in _yaml(path)
