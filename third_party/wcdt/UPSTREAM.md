# WcDT Provenance

`wcdt_v1_adapted` uses the published WcDT architecture as a SUMO-compatible
prediction baseline. It is not a reproduction of the Waymo traffic-scene
generation protocol.

- Upstream: https://github.com/yangchen1997/WcDT
- Pinned commit: `6baa2330fc3f620863d358b5d7f36323b4bfccae`
- License: Apache-2.0
- Core files retained: `net_works/diffusion.py`, `scene_encoder.py`,
  `traj_decoder.py`, `transformer.py`, `common/waymo_dataset.py`, and the
  original multimodal loss contract.

SAFE_RL adds only SUMO adaptation layers: selector-v2 vehicle-ID row alignment,
route-aware trajectory projection, OBB evaluation, fixed 30-step horizon
evaluation, and mode-wise scalar feature aggregation. `net_works/back_bone.py`
must be source-diffed against this commit before a comparative result is marked
as source-faithful. A diffusion-enabled variant, if introduced, must use a
separate experiment name and cannot replace this baseline.
