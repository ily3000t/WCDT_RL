# ACCVP-v1 execution order

1. Run the bounded mechanics smoke:

   ```powershell
   python -m safe_rl.pipeline.stage1_counterfactual --config safe_rl/config/advanced/accvp_snapshot_smoke.yaml --root-source mixed
   ```

   The collector never calls `loadState()` on its root TraCI connection. Each
   branch is a separate process and SUMO connection. Completed roots delete
   their snapshot only after every legal-action branch has passed schema and
   checksum validation.

2. Build immutable root shards with frozen policy and Risk Module checkpoints.
   `collection_jobs` defines the mixed/PPO/merge-timing/rule/deadline jobs;
   each job has its own non-overwritable `collection_id`. The required failure
   query is explicit:

   ```powershell
   python -m safe_rl.pipeline.stage1_counterfactual --config <formal.yaml> --root-policy merge_timing --root-filter deadline --episode-seeds 2 5
   ```

   Then merge the completed shards into one new formal dataset:

   ```powershell
   python -m safe_rl.pipeline.stage1_merge_counterfactual --config <formal.yaml> --shard <shard-a> --shard <shard-b> --output <formal-dataset>
   ```

   The merger rejects duplicate roots and profile/config/Risk Module hash
   mismatches. Temporary SUMO states are written below `run.cache_root`, not the
   counterfactual dataset directory. With the default `cache_root: null`, they
   are stored under `safe_rl_output/.cache/<run_id>/stage1_counterfactual/`;
   they are not written to the repository root or treated as experiment data.

3. Before training, run the seed-2/5 oracle on a dataset containing their
   deadline roots:

   ```powershell
   python -m safe_rl.pipeline.accvp_oracle_smoke --dataset <counterfactual-dataset> --seeds 2 5
   ```

   Oracle state is one of `insufficient_coverage`,
   `no_safe_viable_alternative` or `go`. Only `go` proves that a raw frozen PPO
   action is infeasible while another legal candidate is safety-safe and
   observed to merge before taper. Do not train on either non-go state.

4. Set `accvp.dataset_dir`, `accvp.oracle_report`,
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
   `safe_rl/config/advanced/stage5_accvp_v1.example.yaml`. Shadow must retain
   the exact raw/Shield action sequence before viability mode is enabled.
