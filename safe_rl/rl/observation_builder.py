from __future__ import annotations


BASE_OBSERVATION_COMPONENTS = (
    "ego_state",
    "top_k_neighbor_relative_states",
    "merge_geometry",
)

FORECAST_OBSERVATION_COMPONENTS = (
    "forecast_min_distance",
    "forecast_min_ttc",
    "forecast_max_drac",
    "forecast_collision_probability",
    "forecast_uncertainty",
    "forecast_merge_gap",
    "forecast_nearest_vehicle_future_dx",
    "forecast_nearest_vehicle_future_dy",
    "forecast_risk_top1",
    "forecast_risk_top2",
    "forecast_risk_top3",
)
