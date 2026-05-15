from __future__ import annotations

from pathlib import Path
from typing import Iterable, TypeVar


T = TypeVar("T")


def stage_log(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)


def progress_iter(iterable: Iterable[T], *, desc: str, total: int | None = None) -> Iterable[T]:
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        return iterable
    return tqdm(iterable, desc=desc, total=total)


class TensorboardLogger:
    """Small optional TensorBoard wrapper.

    It prefers torch's SummaryWriter because torch is already required for Stage2/3.
    When torch is unavailable, calls become no-ops and the pipeline continues.
    """

    def __init__(self, log_dir: str | Path, enabled: bool = True):
        self.log_dir = Path(log_dir)
        self.writer = None
        if not enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:
            stage_log("tensorboard", f"未启用：当前环境无法导入 SummaryWriter，目标目录 {self.log_dir}")
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        stage_log("tensorboard", f"写入目录：{self.log_dir}")

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, float(value), int(step))

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
