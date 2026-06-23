# ACCVP-v1 execution order

1. Run the bounded mechanics smoke:

   ```powershell
   python -m safe_rl.pipeline.stage1_counterfactual --config safe_rl/config/advanced/accvp_snapshot_smoke.yaml --root-source mixed
   ```

   The collector never calls `loadState()` on its root TraCI connection. Each
   branch is a separate process and SUMO connection. Completed roots delete
   their snapshot only after every legal-action branch has passed schema and
   checksum validation.

2. Build the formal root pool with frozen checkpoints supplied under
   `accvp.counterfactual.policy_checkpoints`. Policy, state filter and seeds
   are independent. The required failure query is therefore explicit:

   ```powershell
   python -m safe_rl.pipeline.stage1_counterfactual --config <formal.yaml> --root-policy merge_timing --root-filter deadline --episode-seeds 2 5
   ```

   Run bounded pools for `mixed`, `ppo`, `merge_timing` and `rule`; preserve
   root policy, filter, raw action, traffic profile and deadline bin in every
   manifest. Temporary SUMO states are written below `run.cache_root`, not the
   counterfactual dataset directory.

3. Before training, run the seed-2/5 oracle on a dataset containing their
   deadline roots:

   ```powershell
   python -m safe_rl.pipeline.accvp_oracle_smoke --dataset <counterfactual-dataset> --seeds 2 5
   ```

   Oracle state is one of `insufficient_coverage`,
   `no_safe_viable_alternative` or `go`. Only `go` proves that a raw frozen PPO
   action is infeasible while another legal candidate is safety-safe and
   observed to merge before taper. Do not train on either non-go state.

4. Set `accvp.dataset_dir` and `accvp.warm_start.checkpoint` to the frozen
   WcDT-v3 ensemble, then train:

   ```powershell
   python -m safe_rl.pipeline.stage2_train_accvp --config <accvp-train-overlay.yaml>
   ```

   The trainer writes a checkpoint, calibration bundle, and held-out
   operating-point bundle. `viability_branch` refuses to start without the
   operating-point bundle.

5. Run the fixed-seed Stage5 groups from
   `safe_rl/config/advanced/stage5_accvp_v1.example.yaml`. Shadow must retain
   the exact raw/Shield action sequence before viability mode is enabled.
