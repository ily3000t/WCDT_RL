# Step2.5：ACV-Shield-lite v3 Targeted Benchmark Evidence

## Summary

本实验固定 reward-v2 PPO、Risk/Safety Shield 和 lite-v3 artifact，不重新调阈值、不重新训练 ACCVP predictor，也不扩大自然 seeds。目标是验证：

```text
Reward-v2 PPO -> Risk/Safety Shield -> ACV-Shield-lite v3
```

在 targeted task-infeasible states 中，ACV-Shield-lite v3 是否能低频执行 Risk-audited left action change，并在不引入 safety regression 的情况下改善合流时机。

本结果只作为 targeted benchmark evidence，不主张自然分布统计显著性。

## Protocol

- Run: `stage5_reward_v2_accvp_lite_targeted_benchmark_v3`
- Targeted seeds: `[2, 3, 4, 7, 9, 13, 17, 19]`
- Baseline: `wcdt_v3_merge_timing_reward_v2_prediction_shield`
- Candidate: `wcdt_v3_merge_timing_reward_v2_prediction_shield_accvp_lite_v3`
- ACCVP role: task viability only
- Safety authority: Risk Module / Safety Shield
- ACCVP safety head: logging only, not hard gate
- Lite-v3 profile: `secondary_safety_profile=audited_merge_left_v1`, `max_secondary_risk_score=0.05`

## Main Table

| Metric | Reward-v2 PPO | Reward-v2 PPO + Shield | Reward-v2 PPO + Shield + ACV-Shield-lite v3 |
|---|---:|---:|---:|
| terminal_success_rate | 1.0000 | 1.0000 | 1.0000 |
| timely_merge_success_rate | 1.0000 | 1.0000 | 1.0000 |
| first_target_lane_entry_distance_to_taper_p50_m | 146.55 | 146.55 | 162.17 |
| first_target_lane_entry_distance_to_taper_mean_m | 143.54 | 143.54 | 154.47 |
| deadline_opportunity_capture_rate | 0.3077 | 0.3077 | 0.3696 |
| late_merge_request_rate | 0.0000 | 0.0000 | 0.0000 |
| taper_miss_rate | 0.0000 | 0.0000 | 0.0000 |
| proxy_collision_rate | 0.0000 | 0.0000 | 0.0000 |
| safety_violation_rate | 0.0000 | 0.0000 | 0.0000 |
| fallback_rate | 0.0000 | 0.0000 | 0.0000 |
| ACCVP true action-change count | 0 | 0 | 8 |
| ACCVP same-action confirm count | 0 | 0 | 4 |
| ACCVP p95 latency_s | 0.0000 | 0.0000 | 0.0265 |

## Replacement Case Evidence

The replacement case table contains one row per true action-change, not same-action confirm. All 8 action-changes selected action `8` (`left_accelerate`) from raw/shield action `5` (`keep_accelerate`).

Observed pattern:

```text
raw/shield action = keep_accelerate
ACV-Shield-lite selected = left_accelerate
p_merge_before_taper(selected) > p_merge_before_taper(raw)
selected action passes audited merge-left Risk profile
episode finishes with merge_success
proxy_collision = false
safety_violation = false
```

The case table is saved at:

```text
safe_rl_output/runs/accvp_240_lite_v3/accvp/targeted_benchmark/accvp_lite_v3_replacement_case_table.csv
safe_rl_output/runs/accvp_240_lite_v3/accvp/targeted_benchmark/accvp_lite_v3_replacement_case_table.json
```

## Gates

| Gate | Result |
|---|---:|
| engineering_gate_pass | true |
| safety_gate_pass | true |
| task_gate_pass | true |
| latency_gate_pass | true |
| performance_benefit_claim_allowed | true |

Task improvements responsible for passing the task gate:

```text
first_target_lane_entry_distance_to_taper_p50 increased:
  146.55 m -> 162.17 m

deadline_opportunity_capture_rate increased:
  0.3077 -> 0.3696
```

Late merge request rate did not decrease because it was already `0.0` in the Shield baseline.

## Archived Plotting Data

For plotting and future ablation/comparison reporting, the key outputs were copied into:

```text
safe_rl_output/experiment_archive/accvp_step2_5_targeted_benchmark/
  comparison/
  ablation/
```

Use `comparison/main_table.csv` for the main bar/table plot and `comparison/replacement_case_table.csv` for case-level visualization.
