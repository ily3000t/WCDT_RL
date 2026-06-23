from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from safe_rl.accvp.model import ACCVPPredictor, accvp_loss, model_kwargs_from_config, warm_start_scene_encoder
from safe_rl.prediction.wcdt_v3_predictor import WcDTV3TemporalInteractionPredictor
from safe_rl.utils.config import load_config


def _inputs(cfg, candidates: int = 3):
    actors = int(cfg.accvp.actor_count)
    history = int(cfg.scenario.history_steps)
    response = int(cfg.accvp.response_horizon_steps)
    plan = int(cfg.accvp.candidate_plan_horizon_steps)
    return {
        "history_features": torch.randn(1, actors, history, 10),
        "history_valid_mask": torch.ones(1, actors, history),
        "history_lane_ids": torch.ones(1, actors, history, dtype=torch.long),
        "history_edge_role_ids": torch.ones(1, actors, history, dtype=torch.long),
        "role_ids": torch.ones(1, actors, dtype=torch.long),
        "lane_ids": torch.ones(1, actors, dtype=torch.long),
        "edge_role_ids": torch.ones(1, actors, dtype=torch.long),
        "actor_mask": torch.ones(1, actors),
        "candidate_plan": torch.randn(candidates, plan, 5),
        "candidate_action_ids": torch.tensor([0, 4, 7][:candidates], dtype=torch.long),
        "response": response,
    }


def test_batch_candidates_reuse_one_scene_encoding_and_match_single_calls():
    cfg = load_config()
    model = ACCVPPredictor(**model_kwargs_from_config(cfg)).eval()
    values = _inputs(cfg)
    with torch.no_grad():
        scene = model.encode_scene(
            values["history_features"],
            values["history_valid_mask"],
            values["history_lane_ids"],
            values["history_edge_role_ids"],
            values["role_ids"],
            values["lane_ids"],
            values["edge_role_ids"],
            values["actor_mask"],
        )
        batch = model.forward_from_scene(
            scene.expand(3, -1, -1),
            values["actor_mask"].expand(3, -1),
            values["candidate_plan"],
            values["candidate_action_ids"],
        )
        singles = [
            model.forward_from_scene(
                scene,
                values["actor_mask"],
                values["candidate_plan"][index : index + 1],
                values["candidate_action_ids"][index : index + 1],
            )["event_logits"]
            for index in range(3)
        ]
    assert model.scene_encode_calls == 1
    assert batch["actor_response"].shape == (3, int(cfg.accvp.actor_count), values["response"], 5)
    assert torch.allclose(batch["event_logits"], torch.cat(singles, dim=0), atol=1.0e-6, rtol=1.0e-6)


def test_v3_encoder_warm_start_leaves_accvp_heads_independent():
    cfg = load_config()
    source = WcDTV3TemporalInteractionPredictor(
        history_steps=int(cfg.scenario.history_steps),
        horizon_steps=int(cfg.accvp.response_horizon_steps),
        hidden_dim=int(cfg.prediction.wcdt_v3_hidden_dim),
        temporal_layers=int(cfg.prediction.wcdt_v3_temporal_layers),
        actor_attention_layers=int(cfg.prediction.wcdt_v3_actor_attention_layers),
        num_heads=int(cfg.prediction.wcdt_v3_num_heads),
        dropout=float(cfg.prediction.wcdt_v3_dropout),
    )
    target = ACCVPPredictor(**model_kwargs_from_config(cfg))
    warm_start_scene_encoder(target, source.state_dict())
    assert torch.allclose(target.scene.history_projection.weight, source.history_projection.weight)
    assert target.response_decoder[-1].out_features == int(cfg.accvp.response_horizon_steps) * 5


def test_loss_supports_masked_censored_viability_and_quantile_heads():
    cfg = load_config()
    model = ACCVPPredictor(**model_kwargs_from_config(cfg))
    values = _inputs(cfg, candidates=3)
    output = model(
        values["history_features"].expand(3, -1, -1, -1),
        values["history_valid_mask"].expand(3, -1, -1),
        values["history_lane_ids"].expand(3, -1, -1),
        values["history_edge_role_ids"].expand(3, -1, -1),
        values["role_ids"].expand(3, -1),
        values["lane_ids"].expand(3, -1),
        values["edge_role_ids"].expand(3, -1),
        values["actor_mask"].expand(3, -1),
        values["candidate_plan"],
        values["candidate_action_ids"],
    )
    batch = {
        "actor_response": torch.randn_like(output["actor_response"]),
        "actor_response_mask": torch.ones(3, int(cfg.accvp.actor_count), int(cfg.accvp.response_horizon_steps)),
        "event_targets": torch.randint(0, 2, (3, 4), dtype=torch.float32),
        "event_mask": torch.tensor([[1.0, 1.0, 1.0, 0.0]] * 3),
        "geometry_targets": torch.rand(3, 5),
        "geometry_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0]] * 3),
        "candidate_plan": values["candidate_plan"],
    }
    loss, parts = accvp_loss(output, batch, {"event_positive_weights": [2.0, 2.0, 2.0, 2.0]})
    assert torch.isfinite(loss)
    assert {"trajectory", "events", "geometry", "ordering", "smoothness"}.issubset(parts)
