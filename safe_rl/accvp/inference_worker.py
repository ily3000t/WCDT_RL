"""Persistent process-owned ACCVP inference with a bounded caller wait."""

from __future__ import annotations

import concurrent.futures
import multiprocessing
from pathlib import Path
from typing import Any


_SCORER: Any | None = None


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _config_dict(value: Any) -> Any:
    from safe_rl.utils.config import ConfigDict

    if isinstance(value, dict):
        return ConfigDict({key: _config_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_config_dict(item) for item in value]
    return value


def _initialise(config_payload: dict[str, Any], checkpoint: str) -> None:
    global _SCORER
    from safe_rl.accvp.runtime import ACCVPRuntimePredictor

    _SCORER = ACCVPRuntimePredictor(_config_dict(config_payload), checkpoint, use_inference_worker=False)


def _score(prepared: dict[str, Any]) -> list[dict[str, Any]]:
    if _SCORER is None:  # pragma: no cover - process initialisation contract
        raise RuntimeError("ACCVP inference worker was not initialised")
    return _SCORER.score_prepared(prepared)


def _ping() -> bool:
    if _SCORER is None:  # pragma: no cover - process initialisation contract
        raise RuntimeError("ACCVP inference worker was not initialised")
    return True


class PersistentACCVPInferenceWorker:
    """One model-owning process. A timed-out process is discarded before reuse."""

    def __init__(self, config: Any, checkpoint: str | Path):
        self._payload = _plain(dict(config))
        self._checkpoint = str(Path(checkpoint).resolve())
        self._executor: concurrent.futures.ProcessPoolExecutor | None = None

    def _ensure_executor(self) -> concurrent.futures.ProcessPoolExecutor:
        if self._executor is None:
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialise,
                initargs=(self._payload, self._checkpoint),
            )
        return self._executor

    def start(self, timeout_s: float = 15.0) -> None:
        future = self._ensure_executor().submit(_ping)
        try:
            future.result(timeout=max(0.001, float(timeout_s)))
        except concurrent.futures.TimeoutError as exc:
            self._restart_after_timeout()
            raise TimeoutError("ACCVP inference worker startup exceeded the budget") from exc

    def score(self, prepared: dict[str, Any], timeout_s: float) -> list[dict[str, Any]]:
        future = self._ensure_executor().submit(_score, prepared)
        try:
            return future.result(timeout=max(0.001, float(timeout_s)))
        except concurrent.futures.TimeoutError as exc:
            self._restart_after_timeout()
            raise TimeoutError("ACCVP inference worker exceeded the control budget") from exc

    def _restart_after_timeout(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
