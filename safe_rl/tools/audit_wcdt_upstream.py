from __future__ import annotations

import argparse
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


def run(*, upstream_root: Path, output: Path, upstream_commit: str) -> dict:
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
    report = {
        "upstream_url": "https://github.com/yangchen1997/WcDT",
        "upstream_commit": upstream_commit,
        "upstream_root": str(upstream_root.resolve()),
        "core_files": rows,
        "all_core_files_match": all(item["matches"] for item in rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local WcDT core files against an official checkout.")
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--upstream-commit", default="6baa2330fc3f620863d358b5d7f36323b4bfccae")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(
        upstream_root=Path(args.upstream_root),
        output=Path(args.output),
        upstream_commit=str(args.upstream_commit),
    )
    print(json.dumps({"all_core_files_match": report["all_core_files_match"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
