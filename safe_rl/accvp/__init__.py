"""Action-Conditioned Counterfactual Viability Planning (ACCVP-v1).

This package is deliberately independent from the legacy Stage1/schema9 and
forecast-feature paths.  Importing it must not require SUMO or torch until a
collector or model is explicitly constructed.
"""

from safe_rl.accvp.schema import COUNTERFACTUAL_SCHEMA_VERSION

__all__ = ["COUNTERFACTUAL_SCHEMA_VERSION"]
