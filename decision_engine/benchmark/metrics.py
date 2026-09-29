# decision_engine/benchmark/metrics.py
"""
Standardized Evaluation Metrics for TAMEV Benchmarks.
Calculates Accuracy, Calibration (ECE), Brier Score, NLL, Permutation Drift, and Latency.
"""

import math
from typing import Any

import numpy as np


def reliability_bins(
    conf_arr: np.ndarray, acc_arr: np.ndarray, num_bins: int = 10
) -> list[tuple[float, float, int, float, float]]:
    """Equal-width reliability-diagram bins as ``(lower, upper, count, mean_conf, mean_acc)``.

    The single binning used by both ECE and the published calibration curve, so the reported
    curve is exactly the one ECE was computed from. Bin ``i`` is right-closed ``(lo, hi]``
    (matching the long-standing ECE definition: a confidence of exactly 0.0 lands nowhere);
    empty bins carry count 0 with means reported as 0.0 rather than NaN.
    """
    boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bins = []
    for i in range(num_bins):
        bin_lower, bin_upper = boundaries[i], boundaries[i + 1]
        in_bin = (conf_arr > bin_lower) & (conf_arr <= bin_upper)
        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(acc_arr[in_bin]))
            bin_conf = float(np.mean(conf_arr[in_bin]))
        else:
            bin_acc, bin_conf = 0.0, 0.0
        bins.append((float(bin_lower), float(bin_upper), bin_count, bin_conf, bin_acc))
    return bins


def compute_calibration_curve(
    probs_list: list[np.ndarray], targets_list: list[int], num_bins: int = 10
) -> dict[str, Any]:
    """Raw reliability diagram + predicted-confidence distribution.

    Dumps the per-bin ``(bin_confidence, bin_accuracy, count)`` on the same bins ECE uses so
    the curve can be replotted offline, plus summary stats of the predicted-choice confidence
    (the histogram of *which* confidence bucket the model lands in, which ECE alone hides).
    """
    if not probs_list:
        return {
            "n": 0,
            "bin_edges": [],
            "bin_confidence": [],
            "bin_accuracy": [],
            "count": [],
            "confidence_mean": 0.0,
            "confidence_std": 0.0,
            "confidence_p10": 0.0,
            "confidence_p50": 0.0,
            "confidence_p90": 0.0,
        }

    conf_arr = np.array([float(p[int(np.argmax(p))]) for p in probs_list])
    acc_arr = np.array([int(int(np.argmax(p)) == y) for p, y in zip(probs_list, targets_list)])
    bins = reliability_bins(conf_arr, acc_arr, num_bins)
    q10, q50, q90 = (float(x) for x in np.percentile(conf_arr, [10, 50, 90]))
    return {
        "n": len(probs_list),
        "bin_edges": [b[0] for b in bins] + [bins[-1][1]],
        "bin_confidence": [round(b[3], 6) for b in bins],
        "bin_accuracy": [round(b[4], 6) for b in bins],
        "count": [b[2] for b in bins],
        "confidence_mean": round(float(conf_arr.mean()), 6),
        "confidence_std": round(float(conf_arr.std()), 6),
        "confidence_p10": round(q10, 6),
        "confidence_p50": round(q50, 6),
        "confidence_p90": round(q90, 6),
    }


def compute_classification_and_calibration(
    probs_list: list[np.ndarray], targets_list: list[int], num_bins: int = 10
) -> dict[str, Any]:
    """
    Computes classification accuracy and probability calibration metrics.
    """
    n = len(probs_list)
    if n == 0:
        return {"n": 0, "top1_accuracy": 0.0, "top3_accuracy": 0.0, "brier_score": 0.0, "ece": 0.0}

    top1_correct = 0
    top3_correct = 0
    brier_sum = 0.0
    nll_sum = 0.0
    confidences = []
    accuracies = []

    for p, y in zip(probs_list, targets_list):
        k = len(p)
        pred = int(np.argmax(p))
        is_top1 = 1 if pred == y else 0
        top1_correct += is_top1

        # Top-3
        top3_indices = np.argsort(p)[::-1][: min(3, k)]
        top3_correct += 1 if y in top3_indices else 0

        # Confidence of predicted choice
        conf = float(p[pred])
        confidences.append(conf)
        accuracies.append(is_top1)

        # Brier score: sum_{j=0}^{k-1} (p_j - y_j)^2
        one_hot = np.zeros(k, dtype=np.float32)
        if 0 <= y < k:
            one_hot[y] = 1.0
        brier_sum += float(np.sum((p - one_hot) ** 2))

        # NLL
        prob_true = max(float(p[y]) if 0 <= y < k else 1e-9, 1e-9)
        nll_sum += -math.log(prob_true)

    # Expected Calibration Error (ECE), accumulated over the shared reliability bins so this
    # and compute_calibration_curve can never drift apart.
    conf_arr = np.array(confidences)
    acc_arr = np.array(accuracies)
    ece = 0.0

    for _lo, _hi, bin_count, bin_conf, bin_acc in reliability_bins(conf_arr, acc_arr, num_bins):
        if bin_count > 0:
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    # High-confidence error rate (for p >= 0.9)
    high_conf = conf_arr >= 0.9
    high_conf_n = int(np.sum(high_conf))
    error_at_09 = float(1.0 - np.mean(acc_arr[high_conf])) if high_conf_n > 0 else 0.0

    return {
        "n": n,
        "top1_accuracy": round(top1_correct / n, 4),
        "top3_accuracy": round(top3_correct / n, 4),
        "brier_score": round(brier_sum / n, 4),
        "nll": round(nll_sum / n, 4),
        "ece": round(ece, 4),
        "mean_confidence": round(float(np.mean(conf_arr)), 4),
        "error_at_09": round(error_at_09, 4),
        "high_conf_count": high_conf_n,
    }


def bootstrap_ci(  # noqa: PLR0917
    probs_list: list[np.ndarray],
    targets_list: list[int],
    metric: str = "ece",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Percentile-bootstrap confidence interval for one metric of
    ``compute_classification_and_calibration``, resampling items with replacement.

    The published evaluation suites are small (~900 items), so a bare point estimate of ECE
    is not resolvable at the ~0.05 scale the project gates on. Report the interval alongside
    the point estimate instead of implying the gate was passed or failed on noise.
    """
    n = len(probs_list)
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        res = compute_classification_and_calibration(
            [probs_list[i] for i in idx], [targets_list[i] for i in idx]
        )
        values.append(res[metric])
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (round(float(lo), 4), round(float(hi), 4))


def compute_permutation_drift(
    orig_probs: list[np.ndarray], perm_aligned_probs: list[np.ndarray]
) -> tuple[float, float]:
    """
    Computes maximum absolute difference and flip rate under option order permutation.
    """
    if not orig_probs:
        return 0.0, 0.0

    drifts = []
    flips = []
    for p_orig, p_perm in zip(orig_probs, perm_aligned_probs):
        diff = np.max(np.abs(p_orig - p_perm))
        drifts.append(float(diff))
        flips.append(int(np.argmax(p_orig) != np.argmax(p_perm)))

    mean_max_drift = float(np.mean(drifts))
    flip_rate = float(np.mean(flips))
    return mean_max_drift, flip_rate


def compute_latency_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    """Calculates p50, p90, p95, p99, mean, min, max, and QPS throughput."""
    if not latencies_ms:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "qps": 0.0}

    arr = np.array(latencies_ms)
    mean_lat = float(np.mean(arr))
    return {
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "mean": round(mean_lat, 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
        "qps": round(1000.0 / max(mean_lat, 0.01), 1),
    }
