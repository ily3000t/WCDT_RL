# ACCVP configuration snapshots

This directory keeps experiment configuration snapshots separate from the
historical `active/`, `baselines/`, and `examples/` trees. The historical files
are not moved or edited, so their Git revisions continue to reproduce the
original runs.

## Bundles

- `accvp_vnext_selector4_hybrid_v4_frozen/` is an exact standalone snapshot of
  the original Selector-v4 hybrid workflow. Its generated YAML files contain
  no `extends` key. `equivalence_manifest.json` binds every source file,
  snapshot file, and resolved parameter hash.
- `accvp_vnext_selector4_hybrid_seed1000_1004_posthoc_v1/` is the explicitly
  post-hoc optimizer-seed amendment. It replaces optimizer seed `1005` with
  `1000`, uses a new protocol/output identity, and preserves all upstream
  Selector, collection, ACCVP-training, and latency-smoke snapshots byte for
  byte.

The amended cohort is exploratory evidence. It must not be presented as the
original preregistered confirmatory experiment because seed `1005` was removed
after its Stage5 outcome was observed.

## Amended workflow entry point

The coordinator default points at the amended workflow. The explicit form is:

```powershell
python -m safe_rl.pipeline.run_accvp_vnext_pipeline `
  --workflow-config safe_rl/config/config_fix/accvp_vnext_selector4_hybrid_seed1000_1004_posthoc_v1/workflow.yaml `
  --run-until stage5_replicates_and_aggregate
```

The first incomplete amended phase is expected to be Candidate PPO training.
The new protocol/output identity prevents an old `1001-1005` factorial manifest
from being accepted as the revised `1000-1004` cohort.
