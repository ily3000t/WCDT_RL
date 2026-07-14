from __future__ import annotations

import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from safe_rl.pipeline.common import load_stage_config, parse_config_arg
from safe_rl.accvp.contracts.protocol import counterfactual_data_contract, data_contract_hash, effective_activation_distance
from safe_rl.accvp.contracts.schema import file_sha256, read_json, write_json_atomic
from safe_rl.stage1_counterfactual.collector import collect
from safe_rl.utils.config import REPO_ROOT, clone_with_overrides
from safe_rl.utils.progress import stage_log
from safe_rl.utils.sumo_installation import resolve_sumo_installation, sumo_installation_from_config


_REWARD_RISK_PROFILES = {"shield_guided_forecast", "merge_timing_forecast"}


def _cfg_with_sumo_installation_fingerprint(cfg):
    """Mirror the collector's scenario fingerprint mutation for preflight checks."""

    candidate = clone_with_overrides(cfg, {})
    installation = (
        sumo_installation_from_config(candidate.scenario)
        if candidate.scenario.get("sumo_installation_fingerprint")
        else resolve_sumo_installation(candidate.scenario)
    )
    candidate.scenario["sumo_binary"] = installation.sumo_binary
    candidate.scenario["sumo_gui_binary"] = installation.sumo_gui_binary
    candidate.scenario["netconvert_binary"] = installation.netconvert_binary
    candidate.scenario["sumo_tools_directory"] = installation.tools_directory
    candidate.scenario["sumo_home"] = installation.sumo_home
    candidate.scenario["sumo_version"] = installation.sumo_version
    candidate.scenario["sumo_installation_fingerprint"] = installation.to_dict()
    return candidate


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


def split_collection_job(cfg, payload: dict) -> list[dict]:
    """Expand a large collection job into bounded immutable sub-shards."""

    budget = int(payload.get("root_budget", cfg.accvp.counterfactual.get("root_budget", 0)) or 0)
    shard_roots = int(payload.get("shard_roots", cfg.accvp.counterfactual.get("shard_roots", 0)) or 0)
    if budget <= 0 or shard_roots <= 0 or budget <= shard_roots:
        return [dict(payload)]

    episodes_per_shard = int(payload.get("episodes", cfg.stage1.episodes))
    if episodes_per_shard <= 0:
        raise ValueError("collection job splitting requires a positive stage1.episodes or job episodes")
    if payload.get("episode_seeds"):
        seeds = [int(value) for value in payload["episode_seeds"]]
    else:
        seeds = None

    subjobs: list[dict] = []
    count = int(math.ceil(float(budget) / float(shard_roots)))
    for index in range(count):
        remaining = budget - index * shard_roots
        current_budget = min(shard_roots, remaining)
        subjob = dict(payload)
        subjob["parent_collection_id"] = str(payload["name"])
        subjob["name"] = f"{payload['name']}_s{index:03d}"
        subjob["root_budget"] = current_budget
        if seeds is None:
            seed_base = int(cfg.run.seed) + index * episodes_per_shard
            subjob["episode_seeds"] = [seed_base + offset for offset in range(episodes_per_shard)]
        else:
            start = index * episodes_per_shard
            stop = start + episodes_per_shard
            subjob["episode_seeds"] = seeds[start:stop]
            if not subjob["episode_seeds"]:
                raise ValueError(f"not enough explicit episode_seeds to split collection job {payload['name']}")
        subjobs.append(subjob)
    return subjobs


def _shard_dir(cfg, collection_id: str) -> Path:
    run_id = cfg.run.get("run_id")
    if not run_id:
        raise ValueError("immutable ACCVP collection requires run.run_id")
    output_root = Path(cfg.run.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    return (
        output_root
        / str(run_id)
        / "stage1_counterfactual"
        / str(cfg.accvp.counterfactual.output_name)
        / "shards"
        / str(collection_id)
    )


def existing_complete_shard(cfg, collection_id: str, *, fail_on_incomplete: bool = True) -> Path | None:
    """Return a completed immutable shard path so failed job batches can resume."""

    if not cfg.run.get("run_id"):
        return None
    shard = _shard_dir(cfg, collection_id)
    dataset_manifest = shard / "manifests" / "dataset_manifest.json"
    if dataset_manifest.exists():
        manifest = read_json(dataset_manifest)
        expected_schema = int(
            cfg.accvp.get(
                "schema_version",
                manifest.get("counterfactual_schema_version", -1),
            )
        )
        complete = (
            int(manifest.get("counterfactual_schema_version", -1)) == expected_schema
            and int(manifest.get("failed_branches", 1)) == 0
            and int(manifest.get("complete_roots", -1))
            == int(manifest.get("collected_roots", -2))
        )
        if complete:
            return shard
    if shard.exists():
        if not fail_on_incomplete:
            return None
        policy = str(
            cfg.accvp.counterfactual.get("incomplete_shard_policy", "error")
        ).strip().lower()
        if policy not in {"error", "quarantine"}:
            raise ValueError(
                "accvp.counterfactual.incomplete_shard_policy must be error or quarantine"
            )
        if policy == "quarantine":
            quarantine_root = shard.parent / "_failed_attempts"
            quarantine_root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            # Keep the quarantine leaf short enough for Windows path limits;
            # collection_id remains recorded in the audit JSON.
            target = quarantine_root / stamp
            target = Path(shutil.move(str(shard), str(target)))
            write_json_atomic(
                target / "quarantine_record.json",
                {
                    "artifact_kind": "counterfactual_incomplete_shard_quarantine_v1",
                    "schema_version": 1,
                    "collection_id": str(collection_id),
                    "original_path": str(shard.resolve()),
                    "quarantined_path": str(target.resolve()),
                    "quarantined_at_utc": datetime.now(timezone.utc).isoformat(),
                    "reason": "incomplete_or_invalid_dataset_manifest",
                },
            )
            print(
                "[stage1_counterfactual] quarantined incomplete shard "
                f"collection_id={collection_id} from={shard} to={target}"
            )
            return None
        raise FileExistsError(
            f"counterfactual shard directory exists but is incomplete: {shard}; "
            "delete or move it before resuming collection"
        )
    return None


def _run_collection_shard(
    job_cfg,
    payload: dict,
    *,
    shard_index: int,
    shard_total: int,
) -> Path:
    """Run one immutable shard with flushed, machine-readable lifecycle logs."""

    name = str(payload["name"])
    seeds = [int(value) for value in payload.get("episode_seeds", []) or []]
    root_budget = int(
        payload.get("root_budget", job_cfg.accvp.counterfactual.get("root_budget", 0)) or 0
    )
    workers = int(payload.get("workers", job_cfg.accvp.counterfactual.get("workers", 1)) or 1)
    output = _shard_dir(job_cfg, name)
    source = str(payload.get("collection_source", name))
    policy = str(payload["root_policy"])
    root_filter = str(payload["root_filter"])
    seed_range = "none" if not seeds else f"{seeds[0]}..{seeds[-1]}"
    started_at = datetime.now(timezone.utc).isoformat()
    stage_log(
        "stage1_counterfactual",
        "SHARD_START "
        f"index={shard_index}/{shard_total} collection_id={name} source={source} "
        f"root_policy={policy} root_filter={root_filter} root_budget={root_budget or 'unbounded'} "
        f"workers={workers} seed_count={len(seeds)} seed_range={seed_range} "
        f"started_at_utc={started_at} output={output}",
    )
    started = perf_counter()
    try:
        dataset = collect(
            job_cfg,
            root_policy=policy,
            root_filter=root_filter,
            episode_seeds=payload.get("episode_seeds"),
            episodes=payload.get("episodes"),
            collection_id=name,
            collection_source=source,
            collection_job=payload,
        )
    except BaseException as exc:
        stage_log(
            "stage1_counterfactual",
            "SHARD_FAILED "
            f"index={shard_index}/{shard_total} collection_id={name} "
            f"elapsed_s={perf_counter() - started:.3f} error_type={type(exc).__name__} "
            f"error={str(exc)!r} output={output}",
        )
        raise
    manifest = read_json(Path(dataset) / "manifests" / "dataset_manifest.json")
    stage_log(
        "stage1_counterfactual",
        "SHARD_END "
        f"index={shard_index}/{shard_total} collection_id={name} "
        f"elapsed_s={perf_counter() - started:.3f} collected_roots={manifest.get('collected_roots')} "
        f"complete_roots={manifest.get('complete_roots')} failed_branches={manifest.get('failed_branches')} "
        f"output={dataset}",
    )
    return Path(dataset)


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
    risk_fingerprint = f"risk_checkpoint:{file_sha256(risk_checkpoint)}"
    expected_hashes = {data_contract_hash(counterfactual_data_contract(cfg, risk_fingerprint))}
    try:
        sumo_cfg = _cfg_with_sumo_installation_fingerprint(cfg)
        expected_hashes.add(data_contract_hash(counterfactual_data_contract(sumo_cfg, risk_fingerprint)))
    except Exception:
        # If the raw config hash already matches, validation should not require
        # a SUMO installation.  If it does not match, the final error below
        # still blocks formal collection.
        pass
    if str(report.get("data_contract_hash", "")) not in expected_hashes:
        raise ValueError("pilot report data contract does not match formal ACCVP collection")


def main() -> None:
    args = parse_config_arg("Collect configured immutable ACCVP counterfactual shards")
    cfg = load_stage_config(args)
    jobs = list(cfg.accvp.counterfactual.get("collection_jobs", []))
    if not jobs:
        raise ValueError("accvp.counterfactual.collection_jobs must not be empty")
    validate_required_pilot(cfg)
    expanded_jobs: list[tuple[object, dict, list[dict]]] = []
    shard_total = 0
    for job in jobs:
        base_cfg, base_payload = materialise_collection_job(cfg, job)
        subjobs = split_collection_job(cfg, base_payload)
        expanded_jobs.append((base_cfg, base_payload, subjobs))
        shard_total += len(subjobs)
    stage_log(
        "stage1_counterfactual",
        f"BATCH_START run_id={cfg.run.run_id} jobs={len(jobs)} shards={shard_total} "
        f"collection_phase={cfg.accvp.counterfactual.get('collection_phase', 'ad_hoc')}",
    )
    shard_index = 0
    executed = 0
    skipped = 0
    for base_cfg, base_payload, subjobs in expanded_jobs:
        base_name = str(base_payload["name"])
        existing = existing_complete_shard(base_cfg, base_name, fail_on_incomplete=(len(subjobs) == 1))
        if existing is not None:
            first_index = shard_index + 1
            shard_index += len(subjobs)
            skipped += len(subjobs)
            stage_log(
                "stage1_counterfactual",
                f"SHARD_SKIP index={first_index}-{shard_index}/{shard_total} "
                f"collection_id={base_name} reason=existing_complete dataset={existing}",
            )
            continue
        for subjob in subjobs:
            shard_index += 1
            job_cfg, payload = materialise_collection_job(cfg, subjob)
            name = str(payload["name"])
            existing = existing_complete_shard(job_cfg, name)
            if existing is not None:
                skipped += 1
                stage_log(
                    "stage1_counterfactual",
                    f"SHARD_SKIP index={shard_index}/{shard_total} collection_id={name} "
                    f"reason=existing_complete dataset={existing}",
                )
                continue
            _run_collection_shard(
                job_cfg,
                payload,
                shard_index=shard_index,
                shard_total=shard_total,
            )
            executed += 1
    stage_log(
        "stage1_counterfactual",
        f"BATCH_END run_id={cfg.run.run_id} shards={shard_total} executed={executed} skipped={skipped}",
    )


if __name__ == "__main__":
    main()
