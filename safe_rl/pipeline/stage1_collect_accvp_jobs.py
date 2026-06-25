from __future__ import annotations

from pathlib import Path

from safe_rl.pipeline.common import load_stage_config, parse_config_arg
from safe_rl.accvp.protocol import counterfactual_data_contract, data_contract_hash, effective_activation_distance
from safe_rl.accvp.schema import file_sha256, read_json
from safe_rl.stage1_counterfactual.collector import collect
from safe_rl.utils.config import REPO_ROOT, clone_with_overrides


_REWARD_RISK_PROFILES = {"shield_guided_forecast", "merge_timing_forecast"}


def materialise_collection_job(cfg, job) -> tuple[object, dict]:
    """Apply a job-local policy/config without mutating the parent experiment."""

    payload = dict(job)
    if not payload.get("name") or not payload.get("root_policy") or not payload.get("root_filter"):
        raise ValueError("each ACCVP collection job requires name, root_policy, and root_filter")
    overrides = dict(payload.get("config_overrides", {}) or {})
    accvp_overrides = dict(overrides.get("accvp", {}) or {})
    counterfactual_overrides = dict(accvp_overrides.get("counterfactual", {}) or {})
    for name in ("root_budget", "roots_per_episode_limit", "workers", "shard_roots", "output_name"):
        if name in payload:
            counterfactual_overrides[name] = payload[name]
    checkpoint = payload.get("root_policy_checkpoint")
    if checkpoint:
        policy_checkpoints = dict(counterfactual_overrides.get("policy_checkpoints", {}) or {})
        policy_checkpoints[str(payload["root_policy"])] = str(checkpoint)
        counterfactual_overrides["policy_checkpoints"] = policy_checkpoints
    accvp_overrides["counterfactual"] = counterfactual_overrides
    overrides["accvp"] = accvp_overrides

    # Collection jobs may intentionally switch the root policy to the
    # merge-timing reward profile.  That environment construction needs the
    # same frozen Risk Module artifact used by the counterfactual data
    # contract; without this, the job fails before the root-policy rollout
    # starts.  This does not change the ACCVP branch semantics or the
    # deployment Shield: it only supplies the already-declared reward model
    # dependency for the root collector environment.
    rl_overrides = dict(overrides.get("rl", {}) or {})
    reward_profile = str(rl_overrides.get("reward_profile", cfg.rl.get("reward_profile", "default")))
    if reward_profile in _REWARD_RISK_PROFILES:
        reward_cfg = dict(cfg.rl.get("shield_guided_reward", {}) or {})
        reward_cfg.update(dict(rl_overrides.get("shield_guided_reward", {}) or {}))
        if not reward_cfg.get("risk_checkpoint"):
            risk_checkpoint = counterfactual_overrides.get(
                "risk_checkpoint", cfg.accvp.counterfactual.get("risk_checkpoint")
            )
            if not risk_checkpoint:
                raise FileNotFoundError(
                    f"collection job {payload['name']} uses reward_profile={reward_profile} "
                    "but no accvp.counterfactual.risk_checkpoint is configured"
                )
            reward_cfg["risk_checkpoint"] = str(risk_checkpoint)
        rl_overrides["shield_guided_reward"] = reward_cfg
        overrides["rl"] = rl_overrides
    return clone_with_overrides(cfg, overrides), payload


def existing_complete_shard(cfg, collection_id: str) -> Path | None:
    """Return a completed immutable shard path so failed job batches can resume."""

    run_id = cfg.run.get("run_id")
    if not run_id:
        return None
    output_root = Path(cfg.run.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    shard = (
        output_root
        / str(run_id)
        / "stage1_counterfactual"
        / str(cfg.accvp.counterfactual.output_name)
        / "shards"
        / str(collection_id)
    )
    if (shard / "manifests" / "dataset_manifest.json").exists():
        return shard
    if shard.exists():
        raise FileExistsError(
            f"counterfactual shard directory exists but is incomplete: {shard}; "
            "delete or move it before resuming collection"
        )
    return None


def validate_required_pilot(cfg) -> None:
    """Reject formal ACCVP collection unless the matching 240 m pilot passed."""

    report_path = cfg.accvp.counterfactual.get("required_pilot_report")
    phase = str(cfg.accvp.counterfactual.get("collection_phase", "ad_hoc"))
    if phase not in {"ad_hoc", "pilot", "formal"}:
        raise ValueError("counterfactual.collection_phase must be ad_hoc, pilot, or formal")
    if phase == "formal" and not report_path:
        raise FileNotFoundError("formal ACCVP collection requires counterfactual.required_pilot_report")
    if not report_path:
        return
    report = read_json(report_path)
    if str(report.get("pilot_state", "")) != "pass":
        raise ValueError("formal ACCVP collection requires a pilot report with pilot_state='pass'")
    activation = effective_activation_distance(cfg)
    if abs(float(report.get("accvp_activation_distance_m", -1.0)) - activation) > 1.0e-9:
        raise ValueError("pilot report activation window does not match formal ACCVP collection")
    risk_checkpoint = cfg.accvp.counterfactual.get("risk_checkpoint")
    if not risk_checkpoint:
        raise FileNotFoundError("formal ACCVP collection requires counterfactual.risk_checkpoint")
    expected_contract = counterfactual_data_contract(cfg, f"risk_checkpoint:{file_sha256(risk_checkpoint)}")
    if str(report.get("data_contract_hash", "")) != data_contract_hash(expected_contract):
        raise ValueError("pilot report data contract does not match formal ACCVP collection")


def main() -> None:
    args = parse_config_arg("Collect configured immutable ACCVP counterfactual shards")
    cfg = load_stage_config(args)
    jobs = list(cfg.accvp.counterfactual.get("collection_jobs", []))
    if not jobs:
        raise ValueError("accvp.counterfactual.collection_jobs must not be empty")
    validate_required_pilot(cfg)
    for job in jobs:
        job_cfg, payload = materialise_collection_job(cfg, job)
        name = str(payload["name"])
        existing = existing_complete_shard(job_cfg, name)
        if existing is not None:
            print(f"[stage1_counterfactual] skip existing complete shard collection_id={name} dataset={existing}")
            continue
        collect(
            job_cfg,
            root_policy=str(payload["root_policy"]),
            root_filter=str(payload["root_filter"]),
            episode_seeds=payload.get("episode_seeds"),
            episodes=payload.get("episodes"),
            collection_id=name,
            collection_source=str(payload.get("collection_source", name)),
            collection_job=payload,
        )


if __name__ == "__main__":
    main()
