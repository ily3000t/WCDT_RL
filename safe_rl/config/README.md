# Configuration layout

`registry.yaml` is the authoritative index. Do not infer formal status from a
filename or from whether a YAML file can be loaded.

- `active/accvp_vnext/`: canonical VNext entry configurations and the method-only PPO ablation matrix.
- `active/pipeline/` and `active/smoke/`: maintained operational utilities.
- `baselines/`: maintained comparison arms, grouped by model family.
- `examples/`: templates; they are not frozen runnable experiments.
- `archive/`: diagnostic-only historical configurations. They may reproduce an
  old result but cannot train/calibrate/select/promote a VNext artifact.
- `local/`: ignored machine-specific copies. Never cite these as formal
  evidence.

The former `advanced/` flat directory has been retired. Historical basenames
are preserved under their archive family, while canonical VNext files use
short role names (`pilot.yaml`, `formal.yaml`, and so on).

Run a canonical configuration with its registry path, for example:

```powershell
python -m safe_rl.pipeline.stage1_collect_accvp_jobs --config safe_rl/config/active/accvp_vnext/pilot.yaml
```

Do not move existing `safe_rl_output/runs` directories merely to mirror this
layout. Run reports and manifests contain path/hash lineage; moving them can
invalidate provenance. Retention or deletion should be a separate, manifest-
driven operation after reproducing required evidence.

## VNext replicated evidence workflow

`run.seed` is the simulator episode-schedule seed. PPO optimizer replication
uses `rl.optimizer_seed`; changing `run.seed` would change both optimization
and traffic realizations and is therefore forbidden for a formal replicate.
The canonical Candidate Table template executes Reward-v2 with policy-side
commitment disabled. Reward-v3.1 and commitment are explicit, separately
named variants in `active/accvp_vnext/ppo_ablation_matrix.yaml`.

Inspect the complete artifact-gated workflow without starting a long run:

```powershell
python -m safe_rl.pipeline.run_accvp_vnext_pipeline
```

The coordinator reports the first incomplete phase and its exact command. It
executes at most one phase per invocation when `--execute-next` is supplied;
it will never open the sealed final holdout unless
`--allow-final-holdout` is also explicitly supplied. Generated replicate and
Stage5 configs live under `safe_rl_output/runs`, not in the canonical config
tree.
