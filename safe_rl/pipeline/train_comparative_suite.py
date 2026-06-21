from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from safe_rl.pipeline import stage2_train_prediction_risk, stage3_train_ppo
from safe_rl.prediction.actor_selector import actor_selection_config_hash
from safe_rl.sim.action_space import ACTIONS
from safe_rl.utils.config import REPO_ROOT, clone_with_overrides, load_config
from safe_rl.utils.stage1_dataset import (
    STAGE1_BUFFER_SCHEMA_VERSION,
    open_stage1_dataset,
    sha256_file,
)


def _run_dir(base_run_id: str) -> Path:
    return REPO_ROOT / "safe_rl_output" / "runs" / base_run_id


def _require_schema9(path: Path) -> None:
    with open_stage1_dataset(path) as data:
        version = int(data.manifest.get("stage1_buffer_schema_version", 0))
        required = {
            "trajectory_vehicle_id_table",
            "trajectory_agent_vehicle_id_index",
            "trajectory_selector_selected_count",
        }
        missing = sorted(required - set(data.files))
    if version < STAGE1_BUFFER_SCHEMA_VERSION or missing:
        raise ValueError(
            "Comparative WcDT v1 training requires a schema9 Stage1 buffer with selector row IDs; "
            f"found schema={version}, missing={missing}."
        )


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)
    return path


def _provenance(base: Path, *, stage1: Path, risk: Path, v3: Path) -> dict[str, Any]:
    cfg = load_config()
    snapshot_manifest = base / "scenario_snapshot" / "manifest.json"
    canonical = lambda value: hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "base_run": str(base),
        "stage1_manifest": str(stage1 / "manifest.json"),
        "stage1_manifest_sha256": sha256_file(stage1 / "manifest.json"),
        "risk_checkpoint": str(risk),
        "risk_checkpoint_sha256": sha256_file(risk),
        "wcdt_v3_checkpoint": str(v3),
        "wcdt_v3_checkpoint_sha256": sha256_file(v3),
        "scenario_snapshot_manifest": str(snapshot_manifest) if snapshot_manifest.exists() else None,
        "scenario_snapshot_manifest_sha256": (
            sha256_file(snapshot_manifest) if snapshot_manifest.exists() else None
        ),
        "actor_selection_config_hash": actor_selection_config_hash(cfg),
        "route_projection_config_sha256": canonical(dict(cfg.prediction.get("route_projection", {}) or {})),
        "safety_metric_version": str(cfg.risk_module.get("safety_metric_version", "")),
        "reward_profile": "merge_timing_forecast",
        "action_space_sha256": canonical([action.__dict__ for action in ACTIONS]),
    }


def _forecast_settings(source: str, v1_checkpoint: Path, v3_checkpoint: Path) -> dict[str, Any]:
    if source == "ppo":
        return {
            "forecast_features": {"enabled": False, "use_for_ppo_observation": False},
            "rl": {"use_wcdt_forecast_features": False},
        }
    settings: dict[str, Any] = {
        "forecast_features": {
            "enabled": True,
            "use_for_ppo_observation": True,
            "source": source,
            "allow_heuristic_fallback": False,
        },
        "rl": {"use_wcdt_forecast_features": True},
    }
    if source == "wcdt":
        settings["forecast_features"]["checkpoint"] = str(v1_checkpoint)
    elif source == "wcdt_v3":
        settings["forecast_features"]["checkpoint"] = str(v3_checkpoint)
    return settings


def run(
    *,
    base_run_id: str,
    experiment_id: str = "wcdt_v1_rule_comparison",
    training_seeds: list[int] | None = None,
    ppo_timesteps: int = 20_000,
    stage5_episodes: int = 20,
    upstream_root: str | Path | None = None,
    upstream_commit: str = "6baa2330fc3f620863d358b5d7f36323b4bfccae",
    allowed_differences: list[str] | None = None,
    formal: bool = False,
) -> Path:
    seeds = training_seeds or [101]
    if int(stage5_episodes) <= 0:
        raise ValueError("stage5_episodes must be positive.")
    if formal and (sorted(seeds) != [101, 202, 303] or int(stage5_episodes) != 50):
        raise ValueError("Formal comparative runs require training seeds 101,202,303 and 50 scenario seeds.")
    if formal and upstream_root is None:
        raise ValueError("Formal comparative runs require --upstream-root to generate a source diff manifest.")
    base = _run_dir(base_run_id)
    stage1 = base / "stage1" / "risk_probe_buffer"
    risk = base / "stage2" / "risk_module.pt"
    v3 = base / "stage2" / "wcdt_v3_predictor.pt"
    for path in (stage1, risk, v3):
        if not path.exists():
            raise FileNotFoundError(f"Comparative base artifact is missing: {path}")
    _require_schema9(stage1)

    comparative_root = base / "comparative_eval"
    experiment_root = comparative_root / experiment_id
    manifests = experiment_root / "manifests"
    manifests.mkdir(parents=True, exist_ok=False)
    provenance = _provenance(base, stage1=stage1, risk=risk, v3=v3)
    if upstream_root is not None:
        from safe_rl.tools.audit_wcdt_upstream import run as audit_upstream

        source_diff = manifests / "source_diff_manifest.json"
        audit = audit_upstream(
            upstream_root=Path(upstream_root),
            output=source_diff,
            upstream_commit=upstream_commit,
            allowed_differences=set(allowed_differences or []) or None,
        )
        provenance["source_diff_manifest"] = str(source_diff)
        provenance["source_fidelity"] = str(audit["source_fidelity"])
    else:
        provenance["source_fidelity"] = "unverified"
    if formal and provenance["source_fidelity"] != "verified":
        raise ValueError("Formal comparative runs require a verified source_diff_manifest.json.")
    (manifests / "input_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Predictor-only Stage2 run: uses the immutable base buffer and never trains or
    # overwrites a Risk Module in the comparison namespace.
    v1_cfg = load_config()
    v1_cfg.run["output_root"] = str(comparative_root)
    v1_cfg.run["run_id"] = experiment_id
    v1_cfg.stage2["input_stage1"] = str(stage1)
    v1_cfg.stage2["train_risk_module"] = False
    v1_cfg.stage2["risk_checkpoint_reference"] = str(risk)
    v1_cfg.prediction["wcdt_v1_train_enabled"] = True
    v1_cfg.prediction["wcdt_v2_train_enabled"] = False
    v1_cfg.prediction["wcdt_v3_train_enabled"] = False
    v1_cfg.prediction["wcdt_v1_max_agents"] = 6
    stage2_train_prediction_risk.run(v1_cfg)
    v1_checkpoint = experiment_root / "stage2" / "wcdt_predictor.pt"
    if not v1_checkpoint.exists():
        raise FileNotFoundError(f"WcDT v1 checkpoint was not produced: {v1_checkpoint}")
    provenance["wcdt_v1_checkpoint"] = str(v1_checkpoint)
    provenance["wcdt_v1_checkpoint_sha256"] = sha256_file(v1_checkpoint)
    (manifests / "input_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    model_paths: dict[str, dict[int, Path]] = {}
    for source in ("ppo", "constant_velocity", "wcdt", "wcdt_v3"):
        model_paths[source] = {}
        for seed in seeds:
            policy_root = experiment_root / "policies" / source
            cfg = load_config()
            cfg.run["output_root"] = str(policy_root)
            cfg.run["run_id"] = f"seed_{seed}"
            cfg.run["seed"] = int(seed)
            cfg.rl["total_timesteps"] = int(ppo_timesteps)
            cfg.rl["reward_profile"] = "merge_timing_forecast"
            cfg.rl["shield_guided_reward"] = {"risk_checkpoint": str(risk)}
            cfg.shield["forecast_aware_candidate_ranking_mode"] = "off"
            cfg.shield["forecast_task_shadow_enabled"] = False
            cfg.shield["task_backstop_enabled"] = False
            for section, values in _forecast_settings(source, v1_checkpoint, v3_checkpoint).items():
                cfg[section].update(values)
            stage3_train_ppo.run(cfg)
            path = policy_root / f"seed_{seed}" / "stage3" / str(cfg.stage3.model_name)
            model_paths[source][seed] = path

    groups: list[dict[str, Any]] = [
        {
            "name": "rule_gap_acceptance",
            "policy_type": "rule_gap_acceptance",
            "forecast_features": False,
            "shield": False,
            "comparative": {
                "method": "rule_gap_acceptance",
                "training_seed": None,
                "evaluation_variant": "policy",
            },
        }
    ]
    source_names = {"ppo": "ppo", "constant_velocity": "cv", "wcdt": "wcdt_v1_adapted", "wcdt_v3": "wcdt_v3"}
    for source, display in source_names.items():
        for seed in seeds:
            group = {
                "name": f"{display}_seed_{seed}",
                "policy_type": "sb3_ppo",
                "forecast_features": source != "ppo",
                "shield": False,
                "model_path": str(model_paths[source][seed]),
                "comparative": {
                    "method": display,
                    "training_seed": int(seed),
                    "evaluation_variant": "policy",
                },
            }
            if source != "ppo":
                group["forecast_source"] = source
                if source == "wcdt":
                    group["forecast_checkpoint"] = str(v1_checkpoint)
                elif source == "wcdt_v3":
                    group["forecast_checkpoint"] = str(v3)
            groups.append(group)
            shielded = dict(group)
            shielded["name"] = f"{display}_shield_seed_{seed}"
            shielded["shield"] = True
            shielded["comparative"] = {
                "method": display,
                "training_seed": int(seed),
                "evaluation_variant": "shield",
            }
            groups.append(shielded)
    payload = {
        "run": {"output_root": str(comparative_root), "run_id": experiment_id},
        "stage5": {
            "risk_checkpoint": str(risk),
            "default_model_path": str(model_paths["ppo"][seeds[0]]),
            "episodes_per_group": int(stage5_episodes),
            "seeds": list(range(1, int(stage5_episodes) + 1)),
            "groups": groups,
        },
    }
    _write_yaml(experiment_root / "configs" / "stage5_comparative_groups.yaml", payload)
    return experiment_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Train isolated WcDT v1/CV/v3 comparison policies.")
    parser.add_argument("--base-run-id", required=True)
    parser.add_argument("--experiment-id", default="wcdt_v1_rule_comparison")
    parser.add_argument("--training-seeds", default="101")
    parser.add_argument("--ppo-timesteps", type=int, default=20_000)
    parser.add_argument("--stage5-episodes", type=int, default=20)
    parser.add_argument("--upstream-root")
    parser.add_argument("--upstream-commit", default="6baa2330fc3f620863d358b5d7f36323b4bfccae")
    parser.add_argument("--allowed-difference", action="append", default=[])
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    run(
        base_run_id=str(args.base_run_id),
        experiment_id=str(args.experiment_id),
        training_seeds=[int(value) for value in str(args.training_seeds).split(",") if value.strip()],
        ppo_timesteps=int(args.ppo_timesteps),
        stage5_episodes=int(args.stage5_episodes),
        upstream_root=args.upstream_root,
        upstream_commit=str(args.upstream_commit),
        allowed_differences=list(args.allowed_difference),
        formal=bool(args.formal),
    )


if __name__ == "__main__":
    main()
