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
   `accvp.counterfactual.policy_checkpoints`. Run separate, bounded collections
   for `mixed`, `ppo`, `merge_timing`, `rule`, and `deadline_hard`; preserve
   `root_source`, traffic profile and deadline bins in every manifest.

3. Before training, run the seed-2/5 oracle on a dataset containing their
   deadline roots:

   ```powershell
   python -m safe_rl.pipeline.accvp_oracle_smoke --dataset <counterfactual-dataset> --seeds 2 5
   ```

   `go_for_training=false` means no observed safe viable action exists for at
   least one required failure seed. In that case do not increase predictor
   capacity; investigate policy merge timing or scenario feasibility first.

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
