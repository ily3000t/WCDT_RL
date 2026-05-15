from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised only in minimal environments
    gym = None

    class _Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = low
            self.high = high
            self.shape = tuple(shape)
            self.dtype = dtype

        def sample(self):
            return np.zeros(self.shape, dtype=self.dtype)

    class _Discrete:
        def __init__(self, n: int):
            self.n = int(n)

        def sample(self):
            return 0

    class _Spaces:
        Box = _Box
        Discrete = _Discrete

    class _Env:
        metadata: dict = {}

    class _Gym:
        Env = _Env

    spaces = _Spaces()
    gym = _Gym()
