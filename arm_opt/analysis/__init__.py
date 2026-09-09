"""Analysis and evaluation package."""

from arm_opt.analysis.metrics import TrajectoryMetrics, compute_metrics
from arm_opt.analysis.reality_check import (
    RealityCheckResult,
    perform_reality_check,
)

__all__ = [
    "TrajectoryMetrics",
    "compute_metrics",
    "RealityCheckResult",
    "perform_reality_check",
]

