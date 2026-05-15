#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: __init__.py.py
@Author: YangChen
@Date: 2023/12/21
"""
from utils.map_utils import MapUtil
from utils.math_utils import MathUtil
from utils.visualize_utils import VisualizeUtil

MapUtil = MapUtil
MathUtil = MathUtil
VisualizeUtil = VisualizeUtil

__all__ = ["MapUtil", "DataUtil", "MathUtil", "VisualizeUtil"]


def __getattr__(name):
    if name == "DataUtil":
        try:
            from utils.data_utils import DataUtil
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("waymo_open_dataset"):
                raise ModuleNotFoundError(
                    "DataUtil requires the optional Waymo dependency "
                    "`waymo_open_dataset`. SAFE_RL SUMO stages do not need this "
                    "dependency; install it only when running Waymo preprocessing tasks."
                ) from exc
            raise
        return DataUtil
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
