# decision_engine/benchmark/scorecard.py
"""
Scorecard and Report Generators for TAMEV Benchmarks.
Formats benchmark outcomes into human-readable Markdown tables and machine-readable JSON.
"""

import json
from pathlib import Path
from typing import Any


def generate_json_report(report_data: dict[str, Any], output_path: str | Path) -> Path:
    """Saves benchmark results to a formatted JSON report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    return path


def generate_markdown_scorecard(report_data: dict[str, Any], output_path: str | Path) -> Path:
    """Generates an executive-ready Markdown scorecard."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_name = report_data.get("model_name", "TAMEV Model")
    clean = report_data.get("clean_metrics", {})
    perm = report_data.get("permutation_metrics", {})
    lat = report_data.get("latency_metrics", {})
    tasks = report_data.get("task_breakdown", {})

    top1 = clean.get("top1_accuracy", 0.0)
    top3 = clean.get("top3_accuracy", 0.0)
    ece = clean.get("ece", 0.0)
    brier = clean.get("brier_score", 0.0)
    n = clean.get("n", 0)

    p50 = lat.get("p50", 0.0)
    p95 = lat.get("p95", 0.0)
    mean_lat = lat.get("mean", 0.0)
    qps = lat.get("qps", 0.0)

    mean_drift = perm.get("mean_max_drift", 0.0)
    flip_rate = perm.get("flip_rate", 0.0)

    # Production gate checks
    gate_drift = "PASS" if mean_drift < 1e-6 else "FAIL"
    gate_lat = "PASS" if p95 <= 10.0 else "WARN"
    gate_ece = "PASS" if ece <= 0.05 else "WARN"

    lines = [
        f"# Benchmark Scorecard: {model_name}",
        "",
        "## Executive Summary",
        "",
        "| Gate / Metric | Value | Threshold / SLA | Status |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Top-1 Accuracy** | **{top1 * 100:.2f}%** | Higher is better | INFO |",
        f"| **Top-3 Accuracy** | {top3 * 100:.2f}% | Higher is better | INFO |",
        f"| **Calibration (ECE)** | {ece:.4f} | $\\le 0.05$ | **{gate_ece}** |",
        f"| **Brier Score** | {brier:.4f} | Lower is better | INFO |",
        f"| **Permutation Drift** | {mean_drift:.8f} | Exact $0.00000000$ | **{gate_drift}** |",
        f"| **Decision Flip Rate** | {flip_rate * 100:.2f}% | 0.00% | **{gate_drift}** |",
        f"| **Latency p50** | {p50:.2f} ms | $\\le 6.0$ ms | INFO |",
        f"| **Latency p95** | {p95:.2f} ms | $\\le 10.0$ ms | **{gate_lat}** |",
        f"| **Throughput (QPS)** | {qps} qps | $\\ge 100$ qps | INFO |",
        f"| **Test Samples** | {n} | - | INFO |",
        "",
        "## Clean Performance & Calibration",
        "",
        "| Metric | Result |",
        "| :--- | :--- |",
        f"| Sample Count (n) | {n} |",
        f"| Top-1 Accuracy | {top1 * 100:.2f}% ({top1:.4f}) |",
        f"| Top-3 Accuracy | {top3 * 100:.2f}% ({top3:.4f}) |",
        f"| Expected Calibration Error (ECE) | {ece:.4f} |",
        f"| Brier Score | {brier:.4f} |",
        "",
        "## Latency & Hardware Profile",
        "",
        "| Metric | Duration / Throughput |",
        "| :--- | :--- |",
        f"| Median Latency (p50) | {p50:.2f} ms |",
        f"| 95th Percentile (p95) | {p95:.2f} ms |",
        f"| Mean Latency | {mean_lat:.2f} ms |",
        f"| Estimated QPS | {qps} req/s |",
        "",
    ]

    if tasks:
        lines.extend(
            [
                "## Domain / Task Breakdown",
                "",
                "| Task / Domain | Samples (n) | Top-1 Accuracy | Top-3 Accuracy | ECE |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for task_name, t_metrics in tasks.items():
            t_acc = t_metrics.get("top1_accuracy", 0.0)
            t_top3 = t_metrics.get("top3_accuracy", 0.0)
            t_ece = t_metrics.get("ece", 0.0)
            t_n = t_metrics.get("n", 0)
            lines.append(
                f"| {task_name} | {t_n} | {t_acc * 100:.2f}% | {t_top3 * 100:.2f}% | {t_ece:.4f} |"
            )
        lines.append("")

    content = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return path
