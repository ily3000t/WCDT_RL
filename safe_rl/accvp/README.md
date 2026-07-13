# ACCVP VNext / ACV-Shield-240 execution order

## Compatibility boundary

VNext requires counterfactual schema `3`, data contract `accvp_240_v2`,
`selected_indices_v2`, `model_input_fingerprint_v3`, conditional entry-time
labels, and `accvp_loss_v2`. All schema-2 datasets, calibrations, operating
points, Risk-secondary profiles, checkpoints, and final-test reports are
`diagnostic_only`: they may be inspected for regression history but may not be
merged into VNext data, used for calibration, selected as an operating point,
or loaded as a deployable artifact. VNext collection must write new immutable
shards; do not overwrite or relabel old output directories.

The default configuration keeps both `accvp.enabled` and
`accvp.observation.enabled` false. Enabling Candidate Table observation is an
explicit experiment decision and does not make ACCVP a safety authority.

## Non-bypassable VNext gates

1. Every model row must have one authoritative `actor_row_id` resolved through
   WcDT `selected_indices`. Root metadata, root tensor, every branch manifest,
   and every branch tensor must share the same mapping hash.
2. Splits are connected components over `root_episode_id`, the model-visible
   observation fingerprint, and the versioned
   `(scenario_route_hash, traffic_profile, episode_seed)` key. All three
   cross-split overlap counters in `split_provenance.json` must be zero.
3. Entry time is conditional on observed successful entry. Failure and
   horizon-censored branches never receive a synthetic entry-time regression
   target.
4. Deployable training requires at least three unique, explicitly seeded
   ensemble members. Each member draws one fixed bootstrap multiset of split
   components; epochs only reshuffle that fixed replicate. Repeated
   `(model_input_fingerprint, action)` rows retain all outcomes but contribute
   total training weight one. A one-member checkpoint is shadow-only.
5. Threshold selection uses development/calibration/operating-point data only.
   Formal calibration first collapses duplicate fingerprint-action outcomes,
   then forms one bounded mean per split component and score bin; one-sided
   Hoeffding bounds therefore do not count correlated episode rows as
   independent Wilson trials. Final test is diagnostic evaluation of a frozen
   profile, never a selector.
6. Candidate Table PPO may start only after the schema-3 artifact passes the
   policy-free scorer preflight with `risk_gated_candidate_table_v3_bounded_stale`,
   `bounded_last_valid_v2`, at least 1,000 activation-window decisions and at
   least 30 distinct episode seeds. A stale table is reusable for at most one
   decision, 0.5 seconds, and the configured context-delta bounds; otherwise
   fail closed. Risk-secondary must use the audited vectorized geometry
   backend.
7. VNext files use the `accvp_vnext_schema3_*` prefix and are joined by a
   bundle-v2 manifest. A sealed candidate is shadow-only. Formal
   `viability_branch`/`viability_lite` runtime is permitted only after one-shot
   holdout promotion to `holdout_evaluated_go`; NO-GO and revoked bundles are
   rejected before model deserialization.

Canonical overlays are:

- `safe_rl/config/active/accvp_vnext/pilot.yaml`
- `safe_rl/config/active/accvp_vnext/formal.yaml`
- `safe_rl/config/active/accvp_vnext/oracle_regression.yaml`
- `safe_rl/config/active/accvp_vnext/train.yaml`
- `safe_rl/config/active/accvp_vnext/ppo_candidate_table_dev.yaml`
- `safe_rl/config/active/accvp_vnext/ppo_candidate_table_full.yaml`

Run the development PPO overlay before the full overlay. Neither overlay is a
formal comparison result by itself; the frozen evaluation protocol supplies
the confirmatory seed ledger and report gates.

## Collection and training sequence

1. Run the bounded mechanics smoke:

   ```powershell
   python -m safe_rl.pipeline.stage1_counterfactual --config safe_rl/config/active/smoke/accvp_snapshot_smoke.yaml --root-source mixed
   ```

   The collector never calls `loadState()` on its root TraCI connection. Each
   branch is a separate process and SUMO connection. Completed roots delete
   their snapshot only after every legal-action branch has passed schema and
   checksum validation.

2. Run the schema-3 240 m pilot collection. `accvp.activation_distance` is the
   ACV-Shield window only; it does not modify the physical taper, PPO reward,
   Task Backstop, or `current_v1` action execution. The VNext overlay has a new
   run ID, output name and transient cache root. Do not substitute the legacy
   `accvp_240_pilot.yaml`: it is schema 2 and its completed shards are eligible
   for resume under the old directory.

   ```powershell
   python -m safe_rl.pipeline.stage1_collect_accvp_jobs --config safe_rl/config/active/accvp_vnext/pilot.yaml
   ```

   Merge the five model-eligible pilot sources into a new schema-3 pilot
   dataset. Seed 2/5 is collected separately under the oracle-regression
   overlay with `oracle_only=true`; never merge those roots into the pilot or
   any model split. Validate the pilot against the separately generated oracle
   report:

   ```powershell
   python -m safe_rl.pipeline.stage1_merge_counterfactual --config safe_rl/config/active/accvp_vnext/pilot.yaml --shard <pilot-shard-a> --shard <pilot-shard-b> --output safe_rl_output/runs/accvp_vnext_pilot_dataset
   python -m safe_rl.pipeline.stage1_collect_accvp_jobs --config safe_rl/config/active/accvp_vnext/oracle_regression.yaml
   python -m safe_rl.pipeline.stage1_merge_counterfactual --config safe_rl/config/active/accvp_vnext/oracle_regression.yaml --shard <oracle-shard> --output safe_rl_output/runs/accvp_vnext_oracle_regression_dataset
   python -m safe_rl.pipeline.accvp_oracle_smoke --dataset safe_rl_output/runs/accvp_vnext_oracle_regression_dataset --seeds 2 5 --root-policy merge_timing --output safe_rl_output/runs/accvp_vnext_oracle_regression/oracle_report.json
   python -m safe_rl.pipeline.stage1_validate_accvp_pilot --config safe_rl/config/active/accvp_vnext/pilot.yaml --dataset safe_rl_output/runs/accvp_vnext_pilot_dataset --oracle-report safe_rl_output/runs/accvp_vnext_oracle_regression/oracle_report.json --output safe_rl_output/runs/accvp_vnext_pilot/pilot_report.json
   ```

   A pass requires 90% source coverage, 99% branch success, 70% observed
   viability labels in the activation window, and a `go` seed-2/5 oracle. The
   merger allows source-specific PPO observation configs, but rejects data
   contract mismatches (scenario/route, profiles, actor layout, horizons,
   events, activation distance, and frozen Risk Module). Temporary SUMO states
   remain below the overlay's dedicated
   `safe_rl_output/.cache/accvp_vnext_pilot/` root.

3. Only after the pilot passes, collect the new 5,000-root schema-3 frozen
   formal pool. The formal overlay refuses to start without the matching
   `accvp_vnext_pilot/pilot_report.json`; its run, output and cache paths are
   distinct from both the pilot and all legacy `accvp_240_*` artifacts.

   ```powershell
   python -m safe_rl.pipeline.stage1_collect_accvp_jobs --config safe_rl/config/active/accvp_vnext/formal.yaml
   python -m safe_rl.pipeline.stage1_merge_counterfactual --config safe_rl/config/active/accvp_vnext/formal.yaml --shard <formal-shard-a> --shard <formal-shard-b> --output safe_rl_output/runs/accvp_vnext_formal_dataset
   ```

   Before training, run actor-mapping and fingerprint/scenario-seed leakage
   audits on the merged formal pool and bind the independent seed-2/5 oracle
   regression report to its provenance. The training report and formal dataset
   must have identical manifest, root, branch, contract, Risk Module, and
   activation-window provenance. Oracle state is one of
   `insufficient_coverage`, `no_safe_viable_alternative` or `go`; do not train
   on either non-go state.

4. Set `accvp.dataset_dir`, `accvp.oracle_report`, `accvp.activation_distance: 240.0`,
   `accvp.risk_checkpoint` and `accvp.warm_start.checkpoint` to the frozen
   artifacts, then train:

   ```powershell
   python -m safe_rl.pipeline.stage2_train_accvp --config safe_rl/config/active/accvp_vnext/train.yaml
   ```

   The trainer rejects non-`go` or provenance-mismatched oracle reports and
   writes a generation-aware checkpoint, calibration bundle, held-out
   operating point, exact per-member/per-epoch training history and a sealed
   candidate bundle. It does **not** open the final-test split. Runtime resolves
   predictor, calibration, operating point and training history from the
   manifest-relative bundle entries and rechecks both entry and top-level
   hashes.

5. Before training the first 159D PPO, run a policy-free scorer preflight with
   the rule controller. It executes the real SUMO state path and Candidate
   Table/Risk scorer, but deliberately skips PPO observation-shape validation.
   After Candidate Table PPO training, run the separate policy runtime
   preflight and benchmark with the frozen 159D checkpoint. All formal runs
   reject a duplicate or shorter-than-30 seed schedule and refuse to overwrite
   an existing report.

   ```powershell
   python -m safe_rl.pipeline.accvp_runtime_benchmark --config <vnext-config> --policy-type rule_gap_acceptance --seeds <30-or-more-distinct-seeds> --backend vectorized --output <scorer-preflight.json>

   # Run these only after the 159D PPO checkpoint exists.
   python -m safe_rl.pipeline.accvp_observation_preflight --config <vnext-config> --policy-model <ppo.zip> --seeds <30-or-more-distinct-seeds> --output <preflight.json>
   python -m safe_rl.pipeline.accvp_runtime_benchmark --config <vnext-config> --policy-type sb3_ppo --policy-model <ppo.zip> --seeds <30-or-more-distinct-seeds> --backend vectorized --output <runtime-benchmark.json>
   python -m safe_rl.pipeline.accvp_runtime_fault_audit --output <fault-audit.json>
   ```

   A scorer-preflight report can authorize PPO training, but it cannot promote
   a controller. Final holdout accepts only the post-training
   `benchmark_scope=policy_runtime`, `policy_type=sb3_ppo` report whose policy
   checkpoint is on the Stage5 candidate side.

   `accvp_runtime_fault_audit` deterministically exercises timeout, exception,
   NaN, wrong/missing/duplicate/unexpected rows and worker-crash API semantics
   through fresh, bounded-stale, hard-default, recovery and cross-episode
   states. Its current scope is explicitly synthetic and soft-deadline: it does
   not claim OS-process preemption or hard real-time fault tolerance. The full
   table runtime likewise declares `soft_realtime_post_return_v1` until a
   separately parity- and latency-audited process supervisor is implemented.

6. Run paired Stage5 comparisons for at least five frozen PPO optimizer seeds.
   Each optimizer replicate must use the identical simulator-seed ledger and
   carry explicit left/right checkpoint SHA-256 lineage. Formal aggregation
   defaults to a non-lowerable five-training-seed minimum. Aggregate the
   complete balanced optimizer-seed by simulator-seed matrix with the crossed
   bootstrap entry point; it rejects duplicate training seeds, missing cells,
   unequal simulator schedules, reused checkpoints and non-strict lineage.

   ```powershell
   python -m safe_rl.pipeline.stage5_replicated_aggregate --manifest <replicated-request.json> --output <replicated-report.json>
   ```

   Start from
   `safe_rl/config/examples/vnext/stage5_replicated_aggregate_vnext.example.json`.
   A formal request must bind the candidate bundle, candidate side, runtime
   contract and one source acceptance key. Every source report must pass that
   acceptance profile; empty metric output is rejected.

   Binary replicated inference reports a crossed-bootstrap risk-difference
   interval and deliberately does not pool correlated cells into an exact
   McNemar test. Shadow must retain the exact raw/Shield action sequence before
   viability mode is enabled.

7. The training bundle carries the full safety/viability gate operating-point
   schema and is not interchangeable with the task-only lite controller. If a
   lite confirmatory evaluation is planned, derive a separate sealed VNext
   lite bundle on the operating-point split. The derivation reuses and hashes
   the source predictor, calibration and training history, replaces only the
   operating point, records the source candidate fingerprint, and writes
   `artifact_variant=viability_lite_task_v1`; it never creates a legacy-v1
   alias.
   Duplicate model-equivalent decisions use outcome-mass aggregation v2:
   safety events remain fractional, merge success is conditioned on total
   observed mass, and repairable capture is paired by source root rather than
   majority-voted after collapse.

   ```powershell
   python -m safe_rl.pipeline.accvp_tune_viability_lite --config safe_rl/config/active/accvp_vnext/train.yaml --output-dir safe_rl_output/runs/accvp_vnext_lite/accvp
   ```

8. Open the sealed test cohort exactly once per evaluation protocol only after the runtime gate and PPO
   ablation are frozen. The holdout claim freezes the bundle manifest, split
   manifest, predictor, calibration, operating point, dataset manifest and
   training history. Lite mode requires the derived lite manifest and rejects
   a full-gate manifest before model loading. The resulting validated bundle
   is promoted to GO or NO-GO; test data never feeds threshold selection.

   ```powershell
   python -m safe_rl.pipeline.accvp_final_holdout_eval --config safe_rl/config/active/accvp_vnext/train.yaml --artifact-manifest safe_rl_output/runs/accvp_vnext_lite/accvp/accvp_vnext_schema3_lite_candidate_manifest.json --runtime-benchmark <passed-vectorized-runtime-report.json> --stage5-replicated-report <formal-five-seed-stage5-report.json> --output-dir safe_rl_output/runs/accvp_vnext_lite/holdout --mode lite
   ```

   VNext holdout refuses to open unless both reports have valid internal
   fingerprints, the runtime report passes the strict 30-seed/1,000-decision
   vectorized gate and binds this bundle (or its source full bundle), and the
   Stage5 report is a balanced formal crossed-bootstrap matrix with at least
   five distinct optimizer seeds under the same protocol ID. Stage5 must pass
   its candidate-promotion gate, bind this bundle (or its full source bundle),
   and prove the runtime-benchmarked policy is on the declared candidate side.
   Lite GO also requires the preregistered minimum decision, seed, component,
   replacement, observed and repairable masses in the VNext config. Both
   reports are frozen into the atomic holdout claim and the promoted manifest.

   Running `--mode full` against the original
   `artifact_variant=full_candidate_gate_v1` bundle remains diagnostic and
   returns NO-GO for controller deployment by construction.
