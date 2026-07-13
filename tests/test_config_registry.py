from __future__ import annotations

import json
from pathlib import Path

import yaml

from safe_rl.utils.config import load_config
from safe_rl.pipeline.run_full_pipeline import _pipeline_profile_config_path


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "safe_rl" / "config"


def _registry() -> dict:
    return yaml.safe_load((CONFIG_ROOT / "registry.yaml").read_text(encoding="utf-8"))


def test_config_registry_resolves_every_explicit_entry_and_canonical_vnext_set():
    registry = _registry()
    assert registry["schema_version"] == 1
    assert registry["artifact_kind"] == "safe_rl_config_registry_v1"
    configs = registry["configs"]
    paths = [str(record["path"]) for record in configs.values()]
    assert len(paths) == len(set(paths))
    assert all((CONFIG_ROOT / path).is_file() for path in paths)
    canonical = {name for name, record in configs.items() if record["status"] == "canonical"}
    assert canonical == {
        "accvp_vnext_pilot",
        "accvp_vnext_oracle_regression",
        "accvp_vnext_formal",
        "accvp_vnext_train",
        "ppo_accvp_candidate_table_vnext_dev",
        "ppo_accvp_candidate_table_vnext_full",
    }
    assert all(configs[name]["protocol"] == "accvp-vnext-correctness-v1" for name in canonical)


def test_config_registry_archive_families_cover_all_archived_configs():
    registry = _registry()
    covered: set[Path] = set()
    for record in registry["families"].values():
        assert record["status"] == "diagnostic_only"
        matches = set(CONFIG_ROOT.glob(str(record["path_glob"])))
        assert matches
        covered.update(matches)
    public_archive = {
        path
        for root in (CONFIG_ROOT / "archive", CONFIG_ROOT / "examples" / "legacy")
        for path in root.rglob("*")
        if path.suffix in {".yaml", ".json"}
    }
    assert covered == public_archive


def test_all_public_yaml_configs_parse_and_supported_overlays_load():
    public_roots = (
        CONFIG_ROOT / "active",
        CONFIG_ROOT / "baselines",
        CONFIG_ROOT / "examples",
        CONFIG_ROOT / "archive",
    )
    yaml_paths = sorted(path for root in public_roots for path in root.rglob("*.yaml"))
    assert len(yaml_paths) == 63
    for path in yaml_paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict), path
    json_paths = sorted(path for root in public_roots for path in root.rglob("*.json"))
    assert len(json_paths) == 1
    assert isinstance(json.loads(json_paths[0].read_text(encoding="utf-8")), dict)
    for root in (CONFIG_ROOT / "active", CONFIG_ROOT / "baselines"):
        for path in root.rglob("*.yaml"):
            cfg = load_config(path)
            assert cfg.run


def test_advanced_is_retired_and_local_yaml_is_gitignored_by_policy():
    advanced = CONFIG_ROOT / "advanced"
    assert not advanced.exists() or not any(advanced.iterdir())
    registry = _registry()
    assert registry["local"]["tracked"] is False


def test_pipeline_profile_internal_paths_match_registry():
    configs = _registry()["configs"]
    assert _pipeline_profile_config_path("smoke") == (
        CONFIG_ROOT / configs["pipeline_smoke_fast"]["path"]
    )
    assert _pipeline_profile_config_path("performance") == (
        CONFIG_ROOT / configs["pipeline_performance"]["path"]
    )
