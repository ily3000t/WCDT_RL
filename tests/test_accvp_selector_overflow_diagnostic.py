from __future__ import annotations

from safe_rl.pipeline.diagnose_accvp_selector_overflow import diagnose_report
from safe_rl.utils.config import load_config


def _actor(
    vehicle_id: str,
    *,
    role: str = "auxiliary_local",
    lane: int = 2,
    conflict: bool = False,
    reasons: list[str] | None = None,
) -> dict:
    return {
        "vehicle_id": vehicle_id,
        "role": role,
        "edge_id": "main_aux",
        "lane_index": lane,
        "gap_m": 20.0,
        "candidate_conflict_eligible": conflict,
        "nearest_candidate_conflict": False,
        "trigger_reasons": reasons or ["current_gap", "merge_local"],
    }


def test_lane_aware_shadow_removes_only_unprotected_edge_wide_local_actors():
    cfg = load_config(
        "safe_rl/config/active/accvp_vnext_selector3/ppo_candidate_table_full.yaml"
    )
    actors = [
        _actor("front", role="target_front", lane=1),
        _actor("rear", role="target_rear", lane=1),
        _actor("conflict", conflict=True),
        _actor("lowest", reasons=["lowest_ttc"]),
        _actor("aux_lane", lane=0),
        _actor("edge_wide_1", lane=2),
        _actor("edge_wide_2", lane=3),
        _actor("edge_wide_3", lane=2),
        _actor("edge_wide_4", lane=3),
    ]
    report = {
        "metrics": {"accvp_table_critical_actor_overflow_count": 1},
        "episodes": [
            {
                "episode_seed": 55027,
                "accvp_table_critical_actor_overflow_examples": [
                    {
                        "episode_seed": 55027,
                        "optimizer_seed": 1005,
                        "decision_index": 44,
                        "taper_distance_m": 90.0,
                        "capacity": 8,
                        "critical_count": 9,
                        "critical_actors": actors,
                        "dropped_critical_ids": ["edge_wide_4"],
                    }
                ],
            }
        ],
    }

    result = diagnose_report(report, cfg, capacities=[6, 8])

    assert result["reconstruction_complete"] is True
    assert result["protected_coverage_complete"] is True
    assert result["lane_aware_critical_count_histogram"] == {"5": 1}
    assert result["overflow_example_count_by_capacity"] == {"6": 0, "8": 0}
    assert result["supports_edge_wide_local_overclassification"] is True


def test_lane_aware_shadow_keeps_true_capacity_pressure_blocked():
    cfg = load_config(
        "safe_rl/config/active/accvp_vnext_selector3/ppo_candidate_table_full.yaml"
    )
    actors = [
        _actor(f"conflict_{index}", conflict=True)
        for index in range(9)
    ]
    report = {
        "metrics": {"accvp_table_critical_actor_overflow_count": 1},
        "episodes": [
            {
                "episode_seed": 55027,
                "accvp_table_critical_actor_overflow_examples": [
                    {
                        "optimizer_seed": 1005,
                        "decision_index": 50,
                        "taper_distance_m": 80.0,
                        "capacity": 8,
                        "critical_count": 9,
                        "critical_actors": actors,
                        "dropped_critical_ids": ["conflict_8"],
                    }
                ],
            }
        ],
    }

    result = diagnose_report(report, cfg, capacities=[8])

    assert result["protected_coverage_complete"] is True
    assert result["overflow_example_count_by_capacity"] == {"8": 1}
    assert result["supports_edge_wide_local_overclassification"] is False
    assert result["true_capacity_pressure_remains_at_capacity8"] is True
