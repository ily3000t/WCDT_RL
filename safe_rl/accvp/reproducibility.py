from __future__ import annotations

import os
import platform
import sys
from typing import Any

import numpy as np


_TORCH_PROCESS_BASELINES: dict[int, dict[str, Any]] = {}


def _process_baseline(torch: Any) -> dict[str, Any]:
    key = id(torch)
    if key not in _TORCH_PROCESS_BASELINES:
        cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
        cuda_matmul = getattr(
            getattr(getattr(torch, "backends", None), "cuda", None),
            "matmul",
            None,
        )
        _TORCH_PROCESS_BASELINES[key] = {
            "torch_threads": int(torch.get_num_threads()),
            "cublas_workspace_config_present": "CUBLAS_WORKSPACE_CONFIG" in os.environ,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
            "cudnn_deterministic": (
                None if cudnn is None else bool(cudnn.deterministic)
            ),
            "cudnn_benchmark": None if cudnn is None else bool(cudnn.benchmark),
            "cudnn_allow_tf32": (
                None
                if cudnn is None or not hasattr(cudnn, "allow_tf32")
                else bool(cudnn.allow_tf32)
            ),
            "cuda_matmul_allow_tf32": (
                None
                if cuda_matmul is None or not hasattr(cuda_matmul, "allow_tf32")
                else bool(cuda_matmul.allow_tf32)
            ),
        }
    return dict(_TORCH_PROCESS_BASELINES[key])


def configure_deterministic_training(
    torch: Any,
    *,
    enabled: bool,
    torch_threads: int | None = None,
) -> dict[str, Any]:
    # ``use_deterministic_algorithms`` is process-global. Set it explicitly in
    # both directions so a preceding formal run cannot silently force a later
    # shadow run into a different profile than requested.
    if torch_threads is not None and int(torch_threads) <= 0:
        raise ValueError("deterministic torch thread count must be positive")
    baseline = _process_baseline(torch)
    cuda_available = bool(torch.cuda.is_available())
    is_initialized = getattr(torch.cuda, "is_initialized", None)
    cuda_initialized_before = bool(
        cuda_available and callable(is_initialized) and is_initialized()
    )
    if (
        enabled
        and cuda_initialized_before
        and str(baseline.get("cublas_workspace_config", "")) != ":4096:8"
    ):
        raise RuntimeError(
            "formal deterministic training cannot establish the CUBLAS contract "
            "after CUDA has already been initialized"
        )
    torch.use_deterministic_algorithms(bool(enabled))
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    cuda_matmul = getattr(
        getattr(getattr(torch, "backends", None), "cuda", None),
        "matmul",
        None,
    )
    if enabled:
        # This must be present before the first CUDA BLAS operation. Keeping it
        # here also makes the requested contract explicit in the training log.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if cudnn is not None:
            cudnn.deterministic = True
            cudnn.benchmark = False
            if hasattr(cudnn, "allow_tf32"):
                cudnn.allow_tf32 = False
        if cuda_matmul is not None and hasattr(cuda_matmul, "allow_tf32"):
            cuda_matmul.allow_tf32 = False
    else:
        if bool(baseline["cublas_workspace_config_present"]):
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(
                baseline["cublas_workspace_config"]
            )
        else:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        if cudnn is not None:
            cudnn.deterministic = bool(baseline["cudnn_deterministic"])
            cudnn.benchmark = bool(baseline["cudnn_benchmark"])
            if hasattr(cudnn, "allow_tf32") and baseline["cudnn_allow_tf32"] is not None:
                cudnn.allow_tf32 = bool(baseline["cudnn_allow_tf32"])
        if (
            cuda_matmul is not None
            and hasattr(cuda_matmul, "allow_tf32")
            and baseline["cuda_matmul_allow_tf32"] is not None
        ):
            cuda_matmul.allow_tf32 = bool(baseline["cuda_matmul_allow_tf32"])
    configured_threads = (
        int(baseline["torch_threads"])
        if torch_threads is None and not enabled
        else (None if torch_threads is None else int(torch_threads))
    )
    if configured_threads is not None:
        torch.set_num_threads(configured_threads)
    deterministic_algorithms = bool(torch.are_deterministic_algorithms_enabled())
    payload: dict[str, Any] = {
        "level": (
            "same_host_same_software_deterministic"
            if enabled and deterministic_algorithms
            else "seeded_best_effort"
        ),
        "enabled": bool(enabled),
        "deterministic_algorithms": deterministic_algorithms,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "torch_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
        "cuda_available": cuda_available,
        "cuda_initialized_before_configuration": cuda_initialized_before,
        "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "process_baseline": baseline,
    }
    if cudnn is not None:
        payload.update(
            {
                "cudnn_version": cudnn.version(),
                "cudnn_deterministic": bool(cudnn.deterministic),
                "cudnn_benchmark": bool(cudnn.benchmark),
                "cudnn_allow_tf32": (
                    None
                    if not hasattr(cudnn, "allow_tf32")
                    else bool(cudnn.allow_tf32)
                ),
            }
        )
    payload["cuda_matmul_allow_tf32"] = (
        None
        if cuda_matmul is None or not hasattr(cuda_matmul, "allow_tf32")
        else bool(cuda_matmul.allow_tf32)
    )
    if cuda_available:
        payload["cuda_devices"] = [
            str(torch.cuda.get_device_name(index))
            for index in range(int(torch.cuda.device_count()))
        ]
    return payload
