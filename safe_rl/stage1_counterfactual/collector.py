from __future__ import annotations

import concurrent.futures
import json
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from safe_rl.accvp.branch_worker import run_branch_job
from safe_rl.accvp.root_context import capture_root_context, synchronise_root_state
from safe_rl.accvp.schema import canonical_json, stable_hash
from safe_rl.accvp.snapshot_store import CounterfactualSnapshotStore
from safe_rl.risk.merge_local import is_candidate_legal
from safe_rl.sim.action_space import ACTIONS
from safe_rl.utils.config import prepare_run_dir
from safe_rl.utils.progress import stage_log


def _deadline_bin(context: dict[str, Any], deadline_distance: float) -> str:
    local = context.get("merge_local")
    if local is None or not bool(local.ego_on_auxiliary):
        return "not_auxiliary"
    distance = float(local.merge_distance)
    if distance <= 0.0:
        return "past_taper"
    if distance <= float(deadline_distance):
        return "deadline"
    return "pre_deadline"


class _RootPolicy:
    def __init__(self, cfg: Any, source: str):
        self.cfg = cfg
        self.source = source
        self.model = None
        self.rule = None
        if source in {"ppo", "merge_timing"}:
            checkpoint = cfg.accvp.counterfactual.policy_checkpoints.get(source)
            if not checkpoint:
                raise FileNotFoundError(
                    f"counterfactual.policy_checkpoints.{source} is required for root_source={source!r}"
                )
            from safe_rl.rl.ppo import _training_device, load_ppo

            self.model = load_ppo(checkpoint, device=_training_device(cfg))
        elif source == "rule":
            from safe_rl.baselines import RuleGapAcceptancePolicy

            self.rule = RuleGapAcceptancePolicy(cfg)

    def select(self, env: Any, observation: Any, rng: Any) -> int:
        if self.source in {"mixed", "deadline_hard"}:
            from safe_rl.risk.stage1_sampling import select_stage1_action

            action, _mode = select_stage1_action(self.cfg, rng, env.get_risk_context())
            return int(action)
        if self.model is not None:
            action, _state = self.model.predict(observation, deterministic=True)
            return int(action)
        if self.rule is not None:
            return int(self.rule.act(env.get_rule_control_context()).action)
        raise RuntimeError(f"unsupported counterfactual root source {self.source!r}")


def _select_root_action(cfg: Any, env: Any, observation: Any, rng: Any, source: str, policy: _RootPolicy) -> int:
    return policy.select(env, observation, rng)


def _drain_one(pending: dict[Any, tuple[str, int]], store: CounterfactualSnapshotStore, root_rows: list[dict], branch_rows: list[dict]) -> None:
    future = next(concurrent.futures.as_completed(pending))
    root_id, action_id = pending.pop(future)
    result = future.result()
    if bool(result.get("ok", False)):
        row = dict(result["row"])
        store.write_branch(row)
        branch_rows.append(row)
    else:
        store.mark_branch_failed(root_id, action_id, str(result.get("error", "worker_failed")))
        branch_rows.append(dict(result))
    if store.finalise_root_if_complete(root_id):
        for root_row in root_rows:
            if root_row["root_id"] == root_id:
                root_row["complete"] = True
                break


def collect(cfg: Any, *, root_source: str = "mixed", episodes: int | None = None) -> Path:
    """Collect bounded roots while branch processes run independently of SUMO root collection."""

    from safe_rl.pipeline.common import make_env

    counterfactual = cfg.accvp.counterfactual
    root_budget = max(1, int(counterfactual.root_budget))
    roots_per_episode = max(1, int(counterfactual.roots_per_episode_limit))
    workers = max(1, int(counterfactual.workers))
    stage_dir = prepare_run_dir(cfg, "stage1_counterfactual")
    store = CounterfactualSnapshotStore(stage_dir / str(counterfactual.output_name))
    branch_tensor_dir = store.branches_dir / "tensors"
    branch_tensor_dir.mkdir(parents=True, exist_ok=True)
    root_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    pending: dict[Any, tuple[str, int]] = {}
    source = str(root_source)
    episode_count = int(episodes if episodes is not None else cfg.stage1.episodes)
    stage_log("stage1_counterfactual", f"source={source} root_budget={root_budget} workers={workers}")
    cfg.accvp["_counterfactual_collection_enabled"] = True
    env = make_env(cfg, seed=int(cfg.run.seed), shield_enabled=False)
    policy = _RootPolicy(cfg, source)
    collected = 0
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for episode in range(episode_count):
                if collected >= root_budget:
                    break
                rng = __import__("numpy").random.default_rng(__import__("numpy").random.SeedSequence([int(cfg.run.seed), episode]))
                observation, _info = env.reset(seed=int(cfg.run.seed) + episode)
                terminated = truncated = False
                roots_this_episode = 0
                while not (terminated or truncated):
                    context = env.get_risk_context()
                    local = context.get("merge_local")
                    deadline_bin = _deadline_bin(context, float(cfg.accvp.deadline_distance))
                    # The reset-time curriculum moveTo command is not faithfully
                    # represented by SUMO saveState until one normal control step
                    # has completed. Never branch at decision zero.
                    collect_this = (
                        int(env._decision_index) > 0
                        and roots_this_episode < roots_per_episode
                        and collected < root_budget
                    )
                    if source == "deadline_hard":
                        collect_this = collect_this and deadline_bin == "deadline"
                    if collect_this:
                        # Synchronise with raw TraCI getters before saveState so the
                        # Python root metadata and SUMO snapshot represent one state.
                        synchronise_root_state(env)
                        root_id = f"seed{int(env.seed_value)}_decision{int(env._decision_index)}_{uuid.uuid4().hex[:12]}"
                        snapshot_path = store.save_snapshot_from_root(env, root_id)
                        root = capture_root_context(
                            env,
                            root_id=root_id,
                            root_source=source,
                            traffic_profile=str(context.get("curriculum_profile", "unknown")),
                            deadline_bin=deadline_bin,
                            snapshot_path=snapshot_path,
                        )
                        legal_ids = [int(action.index) for action in ACTIONS if is_candidate_legal(action, context)]
                        if legal_ids:
                            metadata_path, tensor_path = store.write_root(root, legal_ids)
                            root_rows.append(
                                {
                                    "root_id": root_id,
                                    "episode_seed": int(env.seed_value),
                                    "root_source": source,
                                    "traffic_profile": str(context.get("curriculum_profile", "unknown")),
                                    "deadline_bin": deadline_bin,
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
                                future = pool.submit(run_branch_job, job)
                                pending[future] = (root_id, action_id)
                            collected += 1
                            roots_this_episode += 1
                            while len(pending) >= workers * 4:
                                _drain_one(pending, store, root_rows, branch_rows)
                    action = _select_root_action(cfg, env, observation, rng, source, policy)
                    observation, _reward, terminated, truncated, _info = env.step(action)
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
    dataset_manifest = {
        "counterfactual_schema_version": 1,
        "root_source": source,
        "root_budget": root_budget,
        "collected_roots": collected,
        "complete_roots": sum(1 for row in root_rows if row.get("complete")),
        "failed_branches": sum(1 for row in branch_rows if not row.get("branch_status") == "completed"),
        "root_strata": list(counterfactual.root_strata),
        "config_hash": stable_hash(dict(cfg)),
        "branch_status_counts": dict(Counter(str(row.get("branch_status", "failed")) for row in branch_rows)),
    }
    with (manifests / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        handle.write(canonical_json(dataset_manifest))
    stage_log("stage1_counterfactual", f"dataset={store.output_dir} complete_roots={dataset_manifest['complete_roots']}")
    return store.output_dir
