from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.pipeline.common import make_env, stage_file, write_report
from safe_rl.utils.config import clone_with_overrides, load_config
from safe_rl.utils.progress import stage_log


VALID_SOURCES = ("wcdt_v2", "wcdt_v3")


def _predictor(cfg: Any, source: str, checkpoint: Path):
    if source == "wcdt_v2":
        from safe_rl.prediction.wcdt_v2_predictor import WcDTV2Predictor

        return WcDTV2Predictor(cfg, checkpoint)
    if source == "wcdt_v3":
        from safe_rl.prediction.wcdt_v3_predictor import WcDTV3Predictor

        return WcDTV3Predictor(cfg, checkpoint)
    raise ValueError(f"unsupported source: {source}")


def _synchronize(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def benchmark(run_id: str, sources: list[str], devices: list[str], steps: int) -> Path:
    cfg = load_config()
    cfg.run["run_id"] = run_id
    env = make_env(cfg, seed=int(cfg.run.seed), shield_enabled=False)
    try:
        env.reset(seed=int(cfg.run.seed))
        for _ in range(max(1, int(cfg.scenario.history_steps))):
            _obs, _reward, terminated, truncated, _info = env.step(4)
            if terminated or truncated:
                env.reset(seed=int(cfg.run.seed))
        context = env.get_risk_context()
    finally:
        env.close()

    rows: list[dict[str, Any]] = []
    for source in sources:
        if source not in VALID_SOURCES:
            raise ValueError(f"sources must be selected from {VALID_SOURCES}; got {source!r}")
        checkpoint = stage_file(cfg, "stage2", f"{source}_predictor.pt")
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing predictor checkpoint: {checkpoint}")
        for device in devices:
            local_cfg = clone_with_overrides(cfg, {"training": {"forecast_runtime_device": device}})
            predictor = _predictor(local_cfg, source, checkpoint)
            predictor.predict(context)
            durations = []
            for _ in range(int(steps)):
                _synchronize(device)
                started = time.perf_counter()
                predictor.predict(context)
                _synchronize(device)
                durations.append((time.perf_counter() - started) * 1000.0)
            values = np.asarray(durations, dtype=np.float64)
            row = {
                "source": source,
                "device": device,
                "steps": int(values.size),
                "latency_ms_mean": float(np.mean(values)),
                "latency_ms_p50": float(np.percentile(values, 50)),
                "latency_ms_p95": float(np.percentile(values, 95)),
            }
            rows.append(row)
            stage_log("benchmark", str(row))
    output = stage_file(cfg, "stage2", "forecast_runtime_benchmark.json")
    write_report(output, {"run_id": run_id, "results": rows})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark WcDT runtime predictor latency on existing checkpoints.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sources", default="wcdt_v2,wcdt_v3")
    parser.add_argument("--devices", default="cpu,cuda")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    output = benchmark(
        args.run_id,
        [item.strip() for item in args.sources.split(",") if item.strip()],
        [item.strip() for item in args.devices.split(",") if item.strip()],
        int(args.steps),
    )
    stage_log("benchmark", f"report={output}")


if __name__ == "__main__":
    main()
