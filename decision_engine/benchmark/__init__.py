# decision_engine/benchmark/__init__.py
"""
Standardized Benchmarking Suite for TAMEV.
Provides metrics, report generators, and runner for local and remote models.
"""

from decision_engine.benchmark.metrics import (
    compute_classification_and_calibration,
    compute_latency_percentiles,
    compute_permutation_drift,
)
from decision_engine.benchmark.runner import BenchmarkRunner
from decision_engine.benchmark.scorecard import (
    generate_json_report,
    generate_markdown_scorecard,
)

__all__ = [
    "BenchmarkRunner",
    "compute_classification_and_calibration",
    "compute_latency_percentiles",
    "compute_permutation_drift",
    "generate_json_report",
    "generate_markdown_scorecard",
]
