from __future__ import annotations

import concurrent.futures
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from safe_rl.accvp.branch_worker import run_branch_job
from safe_rl.accvp.root_context import capture_root_context, synchronise_root_state
from safe_rl.accvp.schema import canonical_json, file_sha256, stable_hash, write_json_atomic
from safe_rl.accvp.shards import assert_new_shard, immutable_shard_dir
from safe_rl.accvp.snapshot_store import CounterfactualSnapshotStore
from safe_rl.risk.merge_local import is_candidate_legal
from safe_rl.risk.risk_module import RiskModuleWrapper
from safe_rl.shield.safety_shield import SafetyShield
from safe_rl.sim.action_space import ACTIONS
from safe_rl.utils.config import REPO_ROOT, prepare_run_dir
from safe_rl.utils.progress import stage_log


def _deadline_bin(context: dict[str, Any], deadline_distance: float) -> str:
    local = context.get("merge_local")
    if local is None or not bool(local.ego_on_auxiliary):
        return "not_auxiliary"
    distance = float(local.merge_distance)
    if distance <= 0.0:
        return "past_taper"
    return "deadline" if distance <= float(deadline_distance) else "pre_deadline"


class _RootPolicy:
    """Frozen root-policy adapter; filtering is intentionally outside this class."""

    def __init__(self, cfg: Any, policy_name: str):
        self.cfg = cfg
        self.policy_name = policy_name
        self.model = None
        self.rule = None
        if policy_name in {"ppo", "merge_timing"}:
            checkpoint = cfg.accvp.counterfactual.policy_checkpoints.get(policy_name)
            if not checkpoint:
                raise FileNotFoundError(
                    f"counterfactual.policy_checkpoints.{policy_name} is required for root_policy={policy_name!r}"
                )
            from safe_rl.rl.ppo import _training_device, load_ppo

            self.model = load_ppo(checkpoint, device=_training_device(cfg))
        elif policy_name == "rule":
            from safe_rl.baselines import RuleGapAcceptancePolicy

            self.rule = RuleGapAcceptancePolicy(cfg)
        elif policy_name != "mixed":
            raise ValueError(f"unsupported counterfactual root_policy={policy_name!r}")

    def select(self, env: Any, observation: Any, rng: np.random.Generator) -> int:
        if self.policy_name == "mixed":
            from safe_rl.risk.stage1_sampling import select_stage1_action

            action, _mode = select_stage1_action(self.cfg, rng, env.get_risk_context())
            return int(action)
        if self.model is not None:
            action, _state = self.model.predict(observation, deterministic=True)
            return int(action)
        if self.rule is not None:
            return int(self.rule.act(env.get_rule_control_context()).action)
        raise RuntimeError(f"root policy has no selector: {self.policy_name!r}")


def _root_filter_matches(root_filter: str, deadline_bin: str) -> bool:
    if root_filter == "all":
        return True
    if root_filter == "deadline":
        return deadline_bin == "deadline"
    raise ValueError(f"unsupported counterfactual root_filter={root_filter!r}; expected 'all' or 'deadline'")


def _cache_dir(cfg: Any, output_name: str, counterfactual: Any) -> Path:
    configured = counterfactual.get("cache_root") or cfg.run.get("cache_root")
    if configured:
        root = Path(str(configured))
        if not root.is_absolute():
            root = REPO_ROOT / root
    else:
        # Keep transient SUMO snapshots in the output tree, never beside source
        # files or inside the durable counterfactual dataset.  The normal
        # output root is ``safe_rl_output/runs``; use its parent so cache
        # cleanup remains separate from individual run artifacts.
        output_root = Path(str(cfg.run.output_root))
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root
        root = (output_root.parent if output_root.name.lower() == "runs" else output_root) / ".cache"
    return root / str(cfg.run.run_id) / "stage1_counterfactual" / str(output_name)


def _drain_one(
    pending: dict[Any, tuple[str, int]],
    store: CounterfactualSnapshotStore,
    root_rows: list[dict[str, Any]],
    branch_rows: list[dict[str, Any]],
) -> None:
    future = next(concurrent.futures.as_completed(pending))
    root_id, action_id = pending.pop(future)
    try:
        result = future.result()
    except Exception as exc:  # pragma: no cover - broken worker process
        result = {"ok": False, "root_id": root_id, "action_id": action_id, "error": f"worker_process_error:{exc}"}
    if bool(result.get("ok", False)):
        row = dict(result["row"])
        store.write_branch(row)
        branch_rows.append(row)
    else:
        store.mark_branch_failed(root_id, action_id, str(result.get("error", "worker_failed")))
        branch_rows.append(dict(result))
    if store.finalise_root_if_complete(root_id):
        next(row for row in root_rows if row["root_id"] == root_id)["complete"] = True


def _seed_schedule(cfg: Any, episode_seeds: Iterable[int] | None, episodes: int | None) -> list[int]:
    configured = list(episode_seeds) if episode_seeds is not None else list(cfg.accvp.counterfactual.get("episode_seeds") or [])
    if configured:
        return [int(value) for value in configured]
    count = int(episodes if episodes is not None else cfg.stage1.episodes)
    return [int(cfg.run.seed) + index for index in range(count)]


class _SecondaryRiskEvaluator:
    """Frozen Risk Module snapshot used by offline ACCVP selection diagnostics."""

    def __init__(self, cfg: Any):
        configured = cfg.accvp.counterfactual.get("risk_checkpoint")
        self.checkpoint = Path(str(configured)).resolve() if configured else None
        if self.checkpoint is not None and not self.checkpoint.exists():
            raise FileNotFoundError(f"counterfactual Risk Module checkpoint does not exist: {self.checkpoint}")
        self.fingerprint = (
            f"risk_checkpoint:{file_sha256(self.checkpoint)}" if self.checkpoint is not None else "heuristic:risk_module_v1"
        )
        self.shield = SafetyShield(cfg, RiskModuleWrapper(cfg, checkpoint=str(self.checkpoint) if self.checkpoint else None))
        self.shield.enabled = True

    def score(self, context: dict[str, Any], legal_ids: list[int]) -> dict[str, dict[str, Any]]:
        by_index = {action.index: action for action in ACTIONS}
        result: dict[str, dict[str, Any]] = {}
        for action_id in legal_ids:
            check = self.shield.evaluate_candidate(by_index[int(action_id)], context)
            result[str(action_id)] = {
                "candidate_legal": bool(check["candidate_legal"]),
                "risk_score": float(check["risk_score"]),
                "risk_uncertainty": float(check["risk_uncertainty"]),
                "secondary_safety_pass": bool(check["safety_pass"]),
                "veto_reason": str(check.get("veto_reason", "")),
            }
        return result


def _collection_id(
    counterfactual: Any,
    policy_name: str,
    filter_name: str,
    seed_schedule: list[int],
    explicit: str | None,
) -> str:
    configured = explicit or counterfactual.get("collection_id")
    if configured:
        return str(configured)
    digest = stable_hash(
        {
            "root_policy": policy_name,
            "root_filter": filter_name,
            "episode_seeds": seed_schedule,
            "root_budget": int(counterfactual.root_budget),
        }
    )[:12]
    return f"{policy_name}_{filter_name}_{digest}"


def collect(
    cfg: Any,
    *,
    root_policy: str | None = None,
    root_filter: str | None = None,
    episode_seeds: Iterable[int] | None = None,
    episodes: int | None = None,
    root_source: str | None = None,
    collection_id: str | None = None,
) -> Path:
    """Collect bounded roots without ever loading state on the root connection.

    ``root_source`` remains a compatibility alias for ``root_policy``. New
    callers must select policy, state filter and episode seeds independently.
    """

    from safe_rl.pipeline.common import make_env

    counterfactual = cfg.accvp.counterfactual
    policy_name = str(root_policy or root_source or counterfactual.get("root_policy", "mixed"))
    filter_name = str(root_filter or counterfactual.get("root_filter", "all"))
    # Compatibility with the former overloaded source name.
    if policy_name == "deadline_hard":
        policy_name, filter_name = "mixed", "deadline"
    root_budget = max(1, int(counterfactual.root_budget))
    roots_per_episode = max(1, int(counterfactual.roots_per_episode_limit))
    workers = max(1, int(counterfactual.workers))
    output_name = str(counterfactual.output_name)
    stage_dir = prepare_run_dir(cfg, "stage1_counterfactual")
    seed_schedule = _seed_schedule(cfg, episode_seeds, episodes)
    shard_id = _collection_id(counterfactual, policy_name, filter_name, seed_schedule, collection_id)
    shard_dir = assert_new_shard(immutable_shard_dir(stage_dir, output_name, shard_id))
    store = CounterfactualSnapshotStore(
        shard_dir,
        cache_dir=_cache_dir(cfg, output_name, counterfactual) / shard_id,
    )
    branch_tensor_dir = store.branches_dir / "tensors"
    branch_tensor_dir.mkdir(parents=True, exist_ok=True)
    stage_log(
        "stage1_counterfactual",
        f"collection_id={shard_id} root_policy={policy_name} root_filter={filter_name} root_budget={root_budget} workers={workers} seeds={len(seed_schedule)}",
    )
    cfg.accvp["_counterfactual_collection_enabled"] = True
    env = make_env(cfg, seed=int(seed_schedule[0] if seed_schedule else cfg.run.seed), shield_enabled=False)
    policy = _RootPolicy(cfg, policy_name)
    secondary_risk = _SecondaryRiskEvaluator(cfg)
    root_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    pending: dict[Any, tuple[str, int]] = {}
    collected = 0
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for episode_index, episode_seed in enumerate(seed_schedule):
                if collected >= root_budget:
                    break
                rng = np.random.default_rng(np.random.SeedSequence([int(episode_seed), episode_index]))
                observation, _info = env.reset(seed=int(episode_seed))
                terminated = truncated = False
                roots_this_episode = 0
                while not (terminated or truncated):
                    context = env.get_risk_context()
                    deadline_bin = _deadline_bin(context, float(cfg.accvp.deadline_distance))
                    raw_action = policy.select(env, observation, rng)
                    raw_action_legal = bool(
                        next((is_candidate_legal(action, context) for action in ACTIONS if action.index == raw_action), False)
                    )
                    collect_this = (
                        int(env._decision_index) > 0
                        and roots_this_episode < roots_per_episode
                        and collected < root_budget
                        and _root_filter_matches(filter_name, deadline_bin)
                    )
                    if collect_this:
                        # Align root metadata to the live state before saveState.
                        synchronise_root_state(env)
                        context = env.get_risk_context()
                        legal_ids = [int(action.index) for action in ACTIONS if is_candidate_legal(action, context)]
                        if legal_ids:
                            # The raw action may have been chosen from a stale
                            # subscription view. Oracle semantics must use the
                            # same synchronised state as the snapshot.
                            raw_action_legal = int(raw_action) in set(legal_ids)
                            secondary_scores = secondary_risk.score(context, legal_ids)
                            root_id = f"seed{int(env.seed_value)}_decision{int(env._decision_index)}_{uuid.uuid4().hex[:12]}"
                            snapshot_path = store.save_snapshot_from_root(env, root_id)
                            root = capture_root_context(
                                env,
                                root_id=root_id,
                                root_policy=policy_name,
                                root_filter=filter_name,
                                raw_action_id=raw_action,
                                raw_action_legal=raw_action_legal,
                                traffic_profile=str(context.get("curriculum_profile", "unknown")),
                                deadline_bin=deadline_bin,
                                snapshot_path=snapshot_path,
                            )
                            root.metadata["secondary_risk"] = secondary_scores
                            root.metadata["risk_model_fingerprint"] = secondary_risk.fingerprint
                            root.metadata["collection_id"] = shard_id
                            metadata_path, tensor_path = store.write_root(root, legal_ids)
                            root_rows.append(
                                {
                                    "root_id": root_id,
                                    "root_episode_id": str(root.metadata["root_episode_id"]),
                                    "episode_seed": int(env.seed_value),
                                    "root_source": policy_name,
                                    "root_policy": policy_name,
                                    "root_filter": filter_name,
                                    "raw_action_id": int(raw_action),
                                    "raw_action_legal": bool(raw_action_legal),
                                    "traffic_profile": str(context.get("curriculum_profile", "unknown")),
                                    "deadline_bin": deadline_bin,
                                    "scenario_config_hash": str(root.metadata["scenario_config_hash"]),
                                    "config_hash": str(root.metadata["config_hash"]),
                                    "action_execution_profile": str(root.metadata["action_execution_profile"]),
                                    "candidate_plan_profile": str(root.metadata["candidate_plan_profile"]),
                                    "risk_model_fingerprint": secondary_risk.fingerprint,
                                    "collection_id": shard_id,
                                    "metadata_path": str(metadata_path),
                                    "tensor_path": str(tensor_path),
                                    "expected_action_ids": legal_ids,
                                    "complete": False,
                                }
                            )
                            for action_id in legal_ids:
                                job = {
                                    "config": dict(cfg),
                                    "root_id": root_id,
                                    "root_metadata_path": str(metadata_path),
                                    "root_tensor_path": str(tensor_path),
                                    "action_id": action_id,
                                    "output_dir": str(branch_tensor_dir),
                                }
                                pending[pool.submit(run_branch_job, job)] = (root_id, action_id)
                            collected += 1
                            roots_this_episode += 1
                            while len(pending) >= workers * 4:
                                _drain_one(pending, store, root_rows, branch_rows)
                    observation, _reward, terminated, truncated, _info = env.step(raw_action)
            while pending:
                _drain_one(pending, store, root_rows, branch_rows)
    finally:
        env.close()
    manifests = store.manifest_dir
    with (manifests / "roots.jsonl").open("w", encoding="utf-8") as handle:
        for row in root_rows:
            handle.write(canonical_json(row) + "\n")
    with (manifests / "branches.jsonl").open("w", encoding="utf-8") as handle:
        for row in branch_rows:
            handle.write(canonical_json(row) + "\n")
    scenario_config_hash = stable_hash(dict(cfg.scenario))
    dataset_manifest = {
        "artifact_kind": "counterfactual_shard_v1",
        "counterfactual_schema_version": 1,
        "collection_id": shard_id,
        "root_policy": policy_name,
        "root_filter": filter_name,
        "episode_seeds": seed_schedule,
        "root_budget": root_budget,
        "collected_roots": collected,
        "complete_roots": sum(1 for row in root_rows if row.get("complete")),
        "failed_branches": sum(1 for row in branch_rows if row.get("branch_status") != "completed"),
        "collection_jobs": list(counterfactual.get("collection_jobs", [])),
        "cache_dir": str(store.cache_dir),
        "config_hash": stable_hash(dict(cfg)),
        "scenario_config_hash": scenario_config_hash,
        "action_execution_profile": str(cfg.scenario.get("action_execution_profile", "current_v1")),
        "candidate_plan_profile": str(cfg.accvp.candidate_plan_profile),
        "risk_model_fingerprint": secondary_risk.fingerprint,
        "branch_status_counts": dict(Counter(str(row.get("branch_status", "failed")) for row in branch_rows)),
    }
    write_json_atomic(manifests / "dataset_manifest.json", dataset_manifest)
    stage_log("stage1_counterfactual", f"dataset={store.output_dir} complete_roots={dataset_manifest['complete_roots']}")
    return store.output_dir
