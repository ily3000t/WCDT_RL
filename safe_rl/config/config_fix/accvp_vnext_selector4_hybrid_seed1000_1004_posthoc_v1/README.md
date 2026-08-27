# Selector-v4 hybrid post-hoc seed amendment v1

Optimizer cohort: `1000, 1001, 1002, 1003, 1004`.

Intentional change: optimizer seed `1005` is replaced by `1000` after the
original Stage5 comparison was observed. The reason recorded by the requester
is that the `1005` policy may represent a local optimum or a scenario where the
WcDT policy is more suitable, while Candidate performed better across most of
the observed cells.

This directory is self-contained at the experiment-config level:

- every runtime-loaded YAML is a standalone snapshot with no `extends`;
- the optimizer seed ledger and revocation manifest live in this directory;
- workflow paths point only to configs in this directory;
- downstream PPO, runtime, Stage5, and holdout outputs use independent paths;
- upstream data/model configs are byte-identical to the frozen v4 snapshot and
  therefore continue to bind the already completed Selector/data/ACCVP stages.

The amended protocol is
`accvp-vnext-correctness-v4-selector4-hybrid-posthoc-seed-amendment-v1`.
It is not the original preregistered confirmatory protocol.

## Phase-to-config map

| Pipeline phase | Configuration |
|---|---|
| selector contract audit | `selector_audit.yaml` |
| pilot collection, merge, validation | `pilot.yaml` |
| oracle collection and merge | `oracle_regression.yaml` |
| pilot latency smoke training | `pilot_latency_smoke_train.yaml` |
| pilot latency smoke runtime | `pilot_latency_smoke_runtime.yaml` |
| formal collection, merge, validation | `formal.yaml` |
| ACCVP training/calibration/operating point | `train.yaml` |
| scorer preflight and Candidate PPO | `ppo_candidate_table_full.yaml` |
| WcDT baseline PPO | `baseline_ppo_wcdt_v3_reward_v2.yaml` |
| PPO factorial method matrix | `ppo_ablation_matrix.yaml` |
| Stage5 evaluation protocol | `evaluation_protocol.yaml` |
| all orchestration, gates, paths, seeds | `workflow.yaml` |

`amendment_manifest.json` records hashes, the phase mapping, unchanged upstream
snapshots, and every class of intentional difference from the frozen bundle.
