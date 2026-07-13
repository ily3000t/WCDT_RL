from __future__ import annotations

import json
from pathlib import Path

import torch

from safe_rl.pipeline import stage1_collect_accvp_jobs as collection_pipeline
from safe_rl.risk.risk_module import RiskModule, RiskModuleWrapper
from safe_rl.utils.config import load_config


def test_risk_checkpoint_uses_restricted_weights_only_loader(tmp_path: Path, monkeypatch) -> None:
    cfg = load_config()
    model = RiskModule(
        explicit_dim=int(cfg.risk_module.explicit_feature_dim),
        latent_dim=int(cfg.risk_module.latent_dim),
        action_embedding_dim=int(cfg.risk_module.action_embedding_dim),
        hidden_dim=int(cfg.risk_module.hidden_dim),
        risk_type_count=int(cfg.risk_module.risk_type_count),
    )
    checkpoint = tmp_path / "risk.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "safety_metric_version": str(cfg.risk_module.safety_metric_version),
            "vehicle_state_ordering_version": str(cfg.scenario.vehicle_state_ordering_version),
        },
        checkpoint,
    )
    original_load = torch.load
    seen: dict[str, object] = {}

    def tracked_load(*args, **kwargs):
        seen.update(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    RiskModuleWrapper(cfg, checkpoint=str(checkpoint))
    assert seen["map_location"] == "cpu"
    assert seen["weights_only"] is True


def test_collection_shard_logs_lifecycle_and_counts(tmp_path: Path, monkeypatch, capsys) -> None:
    cfg = load_config("safe_rl/config/active/accvp_vnext/formal.yaml")
    cfg.run["output_root"] = str(tmp_path)
    payload = {
        "name": "unit_shard_s000",
        "collection_source": "unit",
        "root_policy": "mixed",
        "root_filter": "all",
        "root_budget": 2,
        "workers": 1,
        "episode_seeds": [101, 102],
    }
    output = collection_pipeline._shard_dir(cfg, payload["name"])

    def fake_collect(*_args, **_kwargs):
        manifests = output / "manifests"
        manifests.mkdir(parents=True)
        (manifests / "dataset_manifest.json").write_text(
            json.dumps({"collected_roots": 2, "complete_roots": 2, "failed_branches": 0}),
            encoding="utf-8",
        )
        return output

    monkeypatch.setattr(collection_pipeline, "collect", fake_collect)
    collection_pipeline._run_collection_shard(cfg, payload, shard_index=3, shard_total=10)
    stdout = capsys.readouterr().out
    assert "SHARD_START index=3/10" in stdout
    assert "seed_range=101..102" in stdout
    assert "SHARD_END index=3/10" in stdout
    assert "collected_roots=2 complete_roots=2 failed_branches=0" in stdout
