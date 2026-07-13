from __future__ import annotations

import copy
import inspect
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

from safe_rl.evaluation_protocol import stable_hash


PREDICTOR_STAGE = "predictor.score_candidates"
RISK_STAGE = "risk.predict_many"
SUPPORTED_STAGES = frozenset({PREDICTOR_STAGE, RISK_STAGE})
SUPPORTED_FAULTS = frozenset(
    {
        "timeout",
        "exception",
        "nan",
        "wrong_shape",
        "missing_action",
        "duplicate_action",
        "unexpected_action",
        "worker_crash",
    }
)


class InjectedFaultError(RuntimeError):
    """Base exception for an intentional diagnostic-only fault."""


class InjectedWorkerCrash(InjectedFaultError):
    """Simulated worker-boundary crash; this does not terminate an OS process."""


@dataclass(frozen=True, order=True)
class FaultEvent:
    episode_seed: int
    decision_index: int
    stage: str
    fault: str
    delay_s: float = 0.0

    def __post_init__(self) -> None:
        stage = str(self.stage).strip()
        fault = str(self.fault).strip().lower()
        delay_s = float(self.delay_s)
        if stage not in SUPPORTED_STAGES:
            raise ValueError(f"unsupported fault-injection stage {stage!r}")
        if fault not in SUPPORTED_FAULTS:
            raise ValueError(f"unsupported fault-injection fault {fault!r}")
        if delay_s < 0.0:
            raise ValueError("fault-injection delay_s must be non-negative")
        object.__setattr__(self, "episode_seed", int(self.episode_seed))
        object.__setattr__(self, "decision_index", int(self.decision_index))
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "fault", fault)
        object.__setattr__(self, "delay_s", delay_s)

    @property
    def key(self) -> tuple[int, int, str]:
        return self.episode_seed, self.decision_index, self.stage

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_seed": int(self.episode_seed),
            "decision_index": int(self.decision_index),
            "stage": str(self.stage),
            "fault": str(self.fault),
            "delay_s": float(self.delay_s),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FaultEvent":
        return cls(
            episode_seed=int(value["episode_seed"]),
            decision_index=int(value["decision_index"]),
            stage=str(value["stage"]),
            fault=str(value["fault"]),
            delay_s=float(value.get("delay_s", 0.0)),
        )


class FaultSchedule:
    """An explicit deterministic lookup keyed by episode, decision, and stage.

    Probability-based or RNG-driven injection is intentionally unsupported.
    Formal training/evaluation code must not construct a non-empty schedule.
    """

    def __init__(self, events: Iterable[FaultEvent | Mapping[str, Any]] = ()) -> None:
        normalised = [
            event if isinstance(event, FaultEvent) else FaultEvent.from_mapping(event)
            for event in events
        ]
        by_key: dict[tuple[int, int, str], FaultEvent] = {}
        for event in normalised:
            if event.key in by_key:
                raise ValueError(f"duplicate fault-injection schedule key {event.key}")
            by_key[event.key] = event
        self._by_key = by_key
        self._events = tuple(sorted(normalised))

    def event_for(self, context: Mapping[str, Any], stage: str) -> FaultEvent | None:
        key = (
            int(context.get("episode_seed", -1)),
            int(context.get("decision_index", -1)),
            str(stage),
        )
        return self._by_key.get(key)

    @property
    def events(self) -> tuple[FaultEvent, ...]:
        return self._events

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "deterministic_fault_schedule_v1",
            "random_injection_supported": False,
            "events": [event.to_dict() for event in self._events],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FaultSchedule":
        forbidden = {
            "probability",
            "random_probability",
            "random_seed",
            "fault_rate",
        }
        present = sorted(key for key in forbidden if key in payload)
        if present:
            raise ValueError(
                "random fault injection is not supported; use explicit deterministic events: "
                f"{present}"
            )
        kind = str(payload.get("artifact_kind", "deterministic_fault_schedule_v1"))
        if kind != "deterministic_fault_schedule_v1":
            raise ValueError("unsupported fault schedule artifact_kind")
        return cls(payload.get("events", []) or [])


def assert_fault_injection_allowed(
    schedule: FaultSchedule,
    *,
    formal_pipeline: bool,
    random_injection: bool = False,
) -> None:
    if bool(random_injection):
        raise ValueError("random fault injection is prohibited")
    if bool(formal_pipeline) and schedule.events:
        raise ValueError("fault injection must be disabled in formal training/evaluation pipelines")


class _FaultInjectingBase:
    def __init__(
        self,
        delegate: Any,
        schedule: FaultSchedule,
        *,
        stage: str,
        soft_deadline_s: float | None,
    ) -> None:
        if stage not in SUPPORTED_STAGES:
            raise ValueError(f"unsupported wrapper stage {stage!r}")
        if soft_deadline_s is not None and float(soft_deadline_s) <= 0.0:
            raise ValueError("soft_deadline_s must be positive when configured")
        self.delegate = delegate
        self.schedule = schedule
        self.stage = stage
        self.soft_deadline_s = None if soft_deadline_s is None else float(soft_deadline_s)
        self.trigger_records: list[dict[str, Any]] = []

    def _event(self, context: Mapping[str, Any]) -> FaultEvent | None:
        event = self.schedule.event_for(context, self.stage)
        if event is not None:
            self.trigger_records.append(
                {
                    **event.to_dict(),
                    "injection_scope": "diagnostic_api_wrapper",
                    "simulated_worker_crash": event.fault == "worker_crash",
                }
            )
            if event.delay_s > 0.0:
                time.sleep(event.delay_s)
        return event

    @staticmethod
    def _raise_immediate(event: FaultEvent | None) -> None:
        if event is None:
            return
        if event.fault == "timeout":
            raise TimeoutError("injected deterministic timeout")
        if event.fault == "exception":
            raise InjectedFaultError("injected deterministic exception")
        if event.fault == "worker_crash":
            raise InjectedWorkerCrash("injected simulated worker crash")

    def _check_soft_deadline(self, started: float) -> None:
        if self.soft_deadline_s is None:
            return
        if time.perf_counter() - started > self.soft_deadline_s:
            # This is a cooperative, post-return soft deadline. It cannot
            # preempt a hung in-process delegate and is not a hard real-time
            # guarantee.
            raise TimeoutError("fault wrapper soft deadline exceeded")


class FaultInjectingPredictor(_FaultInjectingBase):
    """Wrap the predictor API without exposing its optional fast-path methods."""

    def __init__(
        self,
        delegate: Any,
        schedule: FaultSchedule,
        *,
        soft_deadline_s: float | None = None,
    ) -> None:
        super().__init__(
            delegate,
            schedule,
            stage=PREDICTOR_STAGE,
            soft_deadline_s=soft_deadline_s,
        )

    def score_candidates(self, context: Mapping[str, Any], actions: Sequence[Any], *, timeout_s=None):
        started = time.perf_counter()
        event = self._event(context)
        self._raise_immediate(event)
        if hasattr(self.delegate, "score_candidates"):
            scorer = self.delegate.score_candidates
            if "timeout_s" in inspect.signature(scorer).parameters:
                rows = scorer(context, actions, timeout_s=timeout_s)
            else:
                rows = scorer(context, actions)
        elif hasattr(self.delegate, "prepare_candidates") and hasattr(self.delegate, "score_prepared"):
            rows = self.delegate.score_prepared(self.delegate.prepare_candidates(context, actions))
        else:
            raise AttributeError("predictor delegate has no supported scoring API")
        self._check_soft_deadline(started)
        return self._mutate(rows, event, actions)

    @staticmethod
    def _mutate(rows: Any, event: FaultEvent | None, actions: Sequence[Any]) -> Any:
        if event is None:
            return rows
        if event.fault == "wrong_shape":
            return {"scores": rows}
        values = [dict(row) for row in list(rows)]
        if event.fault == "nan":
            if values:
                values[0]["p_merge_before_taper"] = float("nan")
        elif event.fault == "missing_action":
            values = values[:-1]
        elif event.fault == "duplicate_action":
            if values:
                values.append(dict(values[0]))
        elif event.fault == "unexpected_action":
            if values:
                expected = [int(getattr(action, "index", -1)) for action in actions]
                values[0]["action_id"] = max(expected or [-1]) + 1
        return values


class FaultInjectingRiskModel(_FaultInjectingBase):
    """Wrap positional Risk ``predict_many`` results for diagnostic faults."""

    def __init__(
        self,
        delegate: Any,
        schedule: FaultSchedule,
        *,
        soft_deadline_s: float | None = None,
    ) -> None:
        super().__init__(
            delegate,
            schedule,
            stage=RISK_STAGE,
            soft_deadline_s=soft_deadline_s,
        )

    def predict_many(self, actions: Sequence[Any], context: Mapping[str, Any]):
        started = time.perf_counter()
        event = self._event(context)
        self._raise_immediate(event)
        rows = self.delegate.predict_many(actions, context)
        self._check_soft_deadline(started)
        return self._mutate(rows, event)

    def predict(self, action: Any, context: Mapping[str, Any]):
        rows = self.predict_many([action], context)
        if len(rows) != 1:
            raise ValueError("fault-injecting Risk single prediction did not return one row")
        return rows[0]

    @staticmethod
    def _mutate(rows: Any, event: FaultEvent | None) -> Any:
        if event is None:
            return rows
        if event.fault == "wrong_shape":
            return {"predictions": rows}
        values = [copy.copy(row) for row in list(rows)]
        if event.fault == "nan":
            if values:
                source = values[0]
                values[0] = SimpleNamespace(
                    risk_score=float("nan"),
                    risk_uncertainty=float(getattr(source, "risk_uncertainty", 0.0)),
                )
        elif event.fault == "missing_action":
            values = values[:-1]
        elif event.fault in {"duplicate_action", "unexpected_action"}:
            if values:
                # Risk batches are positional and carry no action ID. An extra
                # result is the observable duplicate/unexpected-action fault at
                # this API boundary.
                values.append(copy.copy(values[0]))
        return values
