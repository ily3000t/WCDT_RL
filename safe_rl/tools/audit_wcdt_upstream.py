from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

from safe_rl.utils.config import REPO_ROOT
from safe_rl.utils.stage1_dataset import sha256_file


CORE_FILES = (
    "net_works/diffusion.py",
    "net_works/scene_encoder.py",
    "net_works/traj_decoder.py",
    "net_works/transformer.py",
    "net_works/back_bone.py",
    "common/waymo_dataset.py",
    "tasks/train_model_task.py",
    "config.yaml",
)


def _diff_summary(local: Path, upstream: Path) -> dict:
    if not local.is_file() or not upstream.is_file():
        return {"available": False, "changed_line_count": None, "unified_diff_sha256": None}
    local_lines = local.read_text(encoding="utf-8", errors="replace").splitlines()
    upstream_lines = upstream.read_text(encoding="utf-8", errors="replace").splitlines()
    diff = list(
        difflib.unified_diff(
            upstream_lines,
            local_lines,
            fromfile="upstream",
            tofile="local",
            lineterm="",
        )
    )
    payload = "\n".join(diff).encode("utf-8")
    return {
        "available": True,
        "changed_line_count": sum(
            1 for line in diff if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        ),
        "unified_diff_sha256": __import__("hashlib").sha256(payload).hexdigest(),
    }


def run(
    *,
    upstream_root: Path,
    output: Path,
    upstream_commit: str,
    allowed_differences: set[str] | None = None,
) -> dict:
    allowed = set(allowed_differences or {"net_works/back_bone.py"})
    rows = []
    for relative in CORE_FILES:
        local = REPO_ROOT / relative
        upstream = upstream_root / relative
        local_hash = sha256_file(local) if local.is_file() else None
        upstream_hash = sha256_file(upstream) if upstream.is_file() else None
        rows.append(
            {
                "path": relative,
                "local_exists": local_hash is not None,
                "upstream_exists": upstream_hash is not None,
                "local_sha256": local_hash,
                "upstream_sha256": upstream_hash,
                "matches": bool(local_hash and local_hash == upstream_hash),
            }
        )
    unexpected = [
        item["path"]
        for item in rows
        if not item["matches"] and item["path"] not in allowed
    ]
    report = {
        "upstream_url": "https://github.com/yangchen1997/WcDT",
        "upstream_commit": upstream_commit,
        "upstream_root": str(upstream_root.resolve()),
        "core_files": rows,
        "all_core_files_match": all(item["matches"] for item in rows),
        "allowed_differences": sorted(allowed),
        "unexpected_differences": unexpected,
        "backbone_diff": _diff_summary(
            REPO_ROOT / "net_works/back_bone.py",
            upstream_root / "net_works/back_bone.py",
        ),
        "diffusion_configuration": {
            "local_file": "net_works/diffusion.py",
            "local_sha256": sha256_file(REPO_ROOT / "net_works/diffusion.py")
            if (REPO_ROOT / "net_works/diffusion.py").is_file()
            else None,
        },
        "source_fidelity": "verified" if not unexpected else "unverified",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local WcDT core files against an official checkout.")
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--upstream-commit", default="6baa2330fc3f620863d358b5d7f36323b4bfccae")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allowed-difference", action="append", default=[])
    args = parser.parse_args()
    report = run(
        upstream_root=Path(args.upstream_root),
        output=Path(args.output),
        upstream_commit=str(args.upstream_commit),
        allowed_differences=set(args.allowed_difference) or None,
    )
    print(json.dumps({"all_core_files_match": report["all_core_files_match"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
