from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

from safe_rl.accvp.contracts.schema import read_json
from safe_rl.pipeline import stage5_run_replicates
from safe_rl.pipeline.stage5_factorial_aggregate import (
    load_factorial_request,
    run as aggregate_factorial,
    validate_child_report,
)
from safe_rl.utils.config import REPO_ROOT


def _resolve(path: str | Path, *, relative_to: Path | None = None) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value.resolve()
    return ((relative_to or REPO_ROOT) / value).resolve()


def _request_path(source: str | Path) -> Path:
    path = _resolve(source)
    return path / "factorial_request.json" if path.is_dir() else path


def _run_child(
    *,
    generated_dir: Path,
    aggregate_output: Path,
    resume: bool,
) -> None:
    parameters = inspect.signature(stage5_run_replicates.run).parameters
    if "resume" in parameters:
        stage5_run_replicates.run(
            generated_dir=generated_dir,
            aggregate_output=aggregate_output,
            resume=resume,
        )
        return
    # Compatibility with the pre-factorial helper is safe only for a fresh
    # comparison.  A partial comparison must never be restarted by silently
    # invoking the old overwrite-refusing path.
    request = read_json(generated_dir / "replicated_request.json")
    completed = [
        _resolve(row.get("stage5_report", ""), relative_to=generated_dir)
        for row in request.get("replicates", []) or []
        if _resolve(row.get("stage5_report", ""), relative_to=generated_dir).is_file()
    ]
    if completed:
        raise RuntimeError(
            "the installed stage5_run_replicates helper lacks resume support, but this "
            f"comparison already contains {len(completed)} source report(s)"
        )
    stage5_run_replicates.run(
        generated_dir=generated_dir,
        aggregate_output=aggregate_output,
    )


def run(
    *,
    request: str | Path,
    output: str | Path | None = None,
    resume: bool = True,
) -> Path:
    request_path = _request_path(request)
    source, payload = load_factorial_request(request_path)
    rows = list(payload.get("comparisons", []) or [])
    produced: list[dict[str, Any]] = []
    for index, comparison in enumerate(rows, start=1):
        comparison_id = str(comparison.get("comparison_id", ""))
        aggregate_output = _resolve(
            comparison.get("aggregate_report", ""),
            relative_to=source.parent,
        )
        print(
            f"[stage5_factorial] comparison_start index={index}/{len(rows)} "
            f"comparison_id={comparison_id}",
            flush=True,
        )
        if aggregate_output.is_file():
            if not resume:
                raise FileExistsError(
                    f"formal Stage5 comparison report already exists: {aggregate_output}"
                )
            validate_child_report(comparison, request_path=source)
            state = "validated_existing"
        else:
            generated_dir = _resolve(
                comparison.get("generated_dir", ""),
                relative_to=source.parent,
            )
            _run_child(
                generated_dir=generated_dir,
                aggregate_output=aggregate_output,
                resume=resume,
            )
            validate_child_report(comparison, request_path=source)
            state = "completed"
        produced.append(
            {
                "comparison_id": comparison_id,
                "report": str(aggregate_output),
                "state": state,
            }
        )
        print(
            f"[stage5_factorial] comparison_end index={index}/{len(rows)} "
            f"comparison_id={comparison_id} state={state}",
            flush=True,
        )

    output_path = (
        _resolve(output)
        if output is not None
        else source.parent / "factorial_report.json"
    )
    result = aggregate_factorial(source, output_path)
    print(
        f"[stage5_factorial] complete comparisons={len(produced)} report={result}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute and safely resume the formal Stage5 factorial"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--request")
    source.add_argument("--generated-dir")
    parser.add_argument("--output")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    path = run(
        request=args.request or args.generated_dir,
        output=args.output,
        resume=not args.no_resume,
    )
    print(f"stage5_factorial_report={path}")


if __name__ == "__main__":
    main()
