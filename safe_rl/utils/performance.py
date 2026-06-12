from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator


class PerformanceTracker:
    """Low-overhead cumulative wall-clock timing for pipeline hot paths."""

    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self._seconds: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self._seconds[str(name)] += time.perf_counter() - started
            self._counts[str(name)] += 1

    def add_time(self, name: str, seconds: float, count: int = 1) -> None:
        self._seconds[str(name)] += max(0.0, float(seconds))
        self._counts[str(name)] += max(0, int(count))

    def increment(self, name: str, count: int = 1) -> None:
        self._counts[str(name)] += int(count)

    def summary(
        self,
        *,
        steps: int | None = None,
        episodes: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        wall_time = max(0.0, time.perf_counter() - self.started_at)
        payload: dict[str, Any] = {
            "wall_time": wall_time,
            **{name: float(value) for name, value in sorted(self._seconds.items())},
            "operation_counts": {name: int(value) for name, value in sorted(self._counts.items())},
        }
        if steps is not None:
            payload["steps_per_second"] = float(steps / wall_time) if wall_time > 0.0 else 0.0
        if episodes is not None:
            payload["episodes_per_hour"] = float(episodes * 3600.0 / wall_time) if wall_time > 0.0 else 0.0
        if extra:
            payload.update(extra)
        return payload
