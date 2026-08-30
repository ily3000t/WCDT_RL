# ACCVP main-method table v1

This standalone `config_fix` bundle freezes the six-method main table without
editing or extending the earlier Candidate-factorial configuration tree.

The four forecast-attribution PPO methods share Reward-v2, no commitment,
100,000 requested timesteps, five optimizer seeds, a 1,024-transition rollout
budget and the same checkpoint-selection cohort. ACCVP final is a complete
method comparison and is not interpreted as a single predictor ablation.

Formal legacy reuse is conditional. A checkpoint with a different executor
contract (`1 x 1024` versus `4 x 256`) is admitted only after a source-specific
acceleration equivalence report proves identical policy state and checkpoint
selection. The stopped WcDT-v3 seed-1000 run is not a complete manifest and is
never discovered as reusable evidence.

The experiment produces three independent conclusions:

1. `method_effect`: blocking-exact closed loop, Safety Shield disabled;
2. `system_effect`: the same frozen policy with the same Safety Shield enabled;
3. `deployment_runtime_only`: runtime feasibility, never a method-effect gate.

WcDT-v1 remains labelled `adapted` until its pinned upstream source-diff audit
passes. WcDT-v2 is intentionally reserved for the prediction/backbone ablation
and does not add five PPO replicates to this main table.

## Frozen method interpretation

- `Rule-based Gap Acceptance (IDM-style longitudinal control)` is a
  deterministic rule baseline.  It is not described as a complete IDM policy.
- No-Forecast, Constant Velocity, WcDT-v1 adapted and WcDT-v3 share Reward-v2,
  have no policy commitment and differ only in the forecast observation source.
- `ACCVP final` is Reward-v3.1 plus commitment.  Its comparison with WcDT-v3 is
  the preregistered complete-method comparison, not a predictor-only ablation.
- WcDT-v3 and ACCVP use the same frozen WcDT-v3 checkpoint as backbone/warm
  start where applicable.  Existing model bytes are reused only after their
  manifest, config, checkpoint and Stage3 report hashes pass the audit.

## Short development benchmark (run before formal PPO)

This benchmark is development-only and uses optimizer seed `93001`.  It does
not consume any of the five formal optimizer seeds.  Start with WcDT-v3 because
that is the only legacy baseline whose `1 x 1024` rows may need cross-executor
reuse authorization:

```powershell
python -m safe_rl.pipeline.main_method_acceleration prepare --protocol safe_rl/config/config_fix/accvp_main_method_table_v1/protocol.yaml --output-root safe_rl_output/runs/accvp_main_method_table_v1/acceleration --methods wcdt_reward_v2

python -m safe_rl.pipeline.main_method_acceleration run --plan safe_rl_output/runs/accvp_main_method_table_v1/acceleration/acceleration_plan.json
```

The report compares `1 x 1024`, `2 x 512` and `4 x 256`, as well as serial
versus four-worker checkpoint selection.  Throughput alone is insufficient:
reuse across executor contracts is authorized only if policy state, optimizer
state, checkpoint selection, reward/observation contracts and actual timestep
budget are all exactly equal.  A failed equivalence result is evidence to train
the WcDT-v3 cohort under the target `4 x 256` contract, not a reason to relax the
test.

## WcDT-v1 adapted predictor

Use an official WcDT checkout at the pinned commit.  `--prepare-only` performs
the source audit without training; omit it to train the adapter.  Risk is bound
to the existing schema-9 checkpoint and is not retrained.

```powershell
python -m safe_rl.pipeline.train_main_method_wcdt_v1 --config safe_rl/config/config_fix/accvp_main_method_table_v1/wcdt_v1_predictor.yaml --upstream-root E:\path\to\official\WcDT --upstream-commit 6baa2330fc3f620863d358b5d7f36323b4bfccae --prepare-only

python -m safe_rl.pipeline.train_main_method_wcdt_v1 --config safe_rl/config/config_fix/accvp_main_method_table_v1/wcdt_v1_predictor.yaml --upstream-root E:\path\to\official\WcDT --upstream-commit 6baa2330fc3f620863d358b5d7f36323b4bfccae
```

## Formal PPO: prepare first, then explicitly run

Preparation is read/audit/materialisation work.  It prints, per method and
seed, whether the row will be reused or trained.  It never scans arbitrary
partial run directories.  In particular, the stopped WcDT-v3 seed-1000 run is
not eligible.  Pass the acceleration report only after the development
benchmark completed:

```powershell
python -m safe_rl.pipeline.main_method_ppo_suite prepare --protocol safe_rl/config/config_fix/accvp_main_method_table_v1/protocol.yaml --output-root safe_rl_output/runs/accvp_main_method_table_v1/ppo_suite --acceleration-equivalence safe_rl_output/runs/accvp_main_method_table_v1/acceleration/acceleration_equivalence_report.json

python -m safe_rl.pipeline.main_method_ppo_suite run --plan safe_rl_output/runs/accvp_main_method_table_v1/ppo_suite/ppo_suite_plan.json
```

Expected reuse under an identical target executor is all five ACCVP-final
checkpoints.  Legacy WcDT-v3 rows are reused only if the exact-equivalence
benchmark explicitly authorizes `1 x 1024 -> 4 x 256`; otherwise its five rows
are scheduled for formal training.  No-Forecast, Constant Velocity and
WcDT-v1 adapted each require five PPO rows.

## Independent Stage5 reports

Both modes evaluate the same frozen checkpoint bytes and simulator seeds.
The primary WcDT-v3/ACCVP-final comparison uses 300 seeds; the other learned
method comparisons use the registered 100-seed secondary scope.  Episode
caches share the 100-seed prefix with the 300-seed primary evaluation.

```powershell
python -m safe_rl.pipeline.main_method_stage5 prepare --protocol safe_rl/config/config_fix/accvp_main_method_table_v1/protocol.yaml --suite-manifest safe_rl_output/runs/accvp_main_method_table_v1/ppo_suite/ppo_suite_manifest.json --mode method_effect --output-root safe_rl_output/runs/accvp_main_method_table_v1/stage5

python -m safe_rl.pipeline.main_method_stage5 run --request safe_rl_output/runs/accvp_main_method_table_v1/stage5/method_effect_request.json

python -m safe_rl.pipeline.main_method_stage5 prepare --protocol safe_rl/config/config_fix/accvp_main_method_table_v1/protocol.yaml --suite-manifest safe_rl_output/runs/accvp_main_method_table_v1/ppo_suite/ppo_suite_manifest.json --mode system_effect --output-root safe_rl_output/runs/accvp_main_method_table_v1/stage5

python -m safe_rl.pipeline.main_method_stage5 run --request safe_rl_output/runs/accvp_main_method_table_v1/stage5/system_effect_request.json
```

`method_effect` disables Safety Shield, including for ACCVP observation, under
an explicit strict-protocol exception.  `system_effect` enables the same Risk
Shield for every learned method.  Neither report reads or applies a deployment
runtime gate.

## Independent deployment-runtime report

The request binds exactly the new `1000-1004` PPO suite.  The earlier passing
ACCVP runtime report used `1001-1005` and is therefore recorded as historical,
not silently reused.  ACCVP final keeps its specialized per-decision latency,
timeout, stale/fail-closed and coverage gate.  Other methods report descriptive
closed-loop workload latency and throughput with the same Risk Shield; they do
not inherit ACCVP-specific thresholds.

```powershell
python -m safe_rl.pipeline.main_method_runtime prepare --protocol safe_rl/config/config_fix/accvp_main_method_table_v1/protocol.yaml --suite-manifest safe_rl_output/runs/accvp_main_method_table_v1/ppo_suite/ppo_suite_manifest.json --output-root safe_rl_output/runs/accvp_main_method_table_v1/deployment_runtime

python -m safe_rl.pipeline.main_method_runtime run --request safe_rl_output/runs/accvp_main_method_table_v1/deployment_runtime/deployment_runtime_request.json --resume
```

The three formal conclusions are intentionally separate: a method-effect
regression must not be hidden by Shield, a system-effect result must not be
attributed solely to the predictor, and a deployment-runtime failure must not
rewrite blocking-exact policy observations or invalidate an otherwise valid
closed-loop method comparison.
