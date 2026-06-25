# ACCVP-v1 / ACV-Shield-240 execution order

1. Run the bounded mechanics smoke:

   ```powershell
   python -m safe_rl.pipeline.stage1_counterfactual --config safe_rl/config/advanced/accvp_snapshot_smoke.yaml --root-source mixed
   ```

   The collector never calls `loadState()` on its root TraCI connection. Each
   branch is a separate process and SUMO connection. Completed roots delete
   their snapshot only after every legal-action branch has passed schema and
   checksum validation.

2. Run the 240 m pilot collection. `accvp.activation_distance` is the
   ACV-Shield window only; it does not modify the physical taper, PPO reward,
   Task Backstop, or `current_v1` action execution.

   ```powershell
   python -m safe_rl.pipeline.stage1_collect_accvp_jobs --config safe_rl/config/advanced/accvp_240_pilot.yaml
   ```

   Merge the six immutable pilot shards into a new formal dataset, run the
   seed-2/5 oracle on that exact merged manifest, and validate the pilot:

   ```powershell
   python -m safe_rl.pipeline.stage1_merge_counterfactual --config safe_rl/config/advanced/accvp_240_pilot.yaml --shard <shard-a> --shard <shard-b> --output <pilot-dataset>
   python -m safe_rl.pipeline.accvp_oracle_smoke --dataset <pilot-dataset> --seeds 2 5 --root-policy merge_timing --output <pilot-oracle.json>
   python -m safe_rl.pipeline.stage1_validate_accvp_pilot --config safe_rl/config/advanced/accvp_240_pilot.yaml --dataset <pilot-dataset> --oracle-report <pilot-oracle.json> --output safe_rl_output/runs/accvp_240_pilot/pilot_report.json
   ```

   A pass requires 90% source coverage, 99% branch success, 70% observed
   viability labels in the activation window, and a `go` seed-2/5 oracle. The
   merger allows source-specific PPO observation configs, but rejects data
   contract mismatches (scenario/route, profiles, actor layout, horizons,
   events, activation distance, and frozen Risk Module). Temporary SUMO states
   remain below `safe_rl_output/.cache/<run_id>/stage1_counterfactual/`.

3. Only after the pilot passes, collect the frozen 5,000-root formal pool with
   `safe_rl/config/advanced/accvp_240_formal.yaml`, merge it, and rerun the
   seed-2/5 oracle on the final merged dataset. The training report and the
   formal dataset must have identical manifest, root, branch, contract, Risk
   Module, and activation-window provenance.

   ```powershell
   python -m safe_rl.pipeline.accvp_oracle_smoke --dataset <formal-dataset> --seeds 2 5 --root-policy merge_timing --output <formal-oracle.json>
   ```

   Oracle state is one of `insufficient_coverage`,
   `no_safe_viable_alternative` or `go`. Only `go` proves that a raw frozen PPO
   action is infeasible while another legal candidate is safety-safe and
   observed to merge before taper. Do not train on either non-go state.

4. Set `accvp.dataset_dir`, `accvp.oracle_report`, `accvp.activation_distance: 240.0`,
   `accvp.risk_checkpoint` and `accvp.warm_start.checkpoint` to the frozen
   artifacts, then train:

   ```powershell
   python -m safe_rl.pipeline.stage2_train_accvp --config <accvp-train-overlay.yaml>
   ```

   The trainer rejects non-`go` or provenance-mismatched oracle reports and
   writes a checkpoint, calibration bundle, held-out operating point, final
   diagnostics and an artifact manifest. Runtime requires the matched artifact
   manifest, calibration bundle, Risk Module checkpoint and operating point.

5. Run the fixed-seed Stage5 groups from
   `safe_rl/config/advanced/stage5_accvp_240.example.yaml`. Shadow must retain
   the exact raw/Shield action sequence before viability mode is enabled.
