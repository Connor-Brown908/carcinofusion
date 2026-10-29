"""Metrics, confidence intervals and calibration.

One definition of every metric, used by every model, so comparisons are valid.

Two choices worth stating because they change the numbers:

* PR-AUC is scikit-learn's `average_precision_score`. That is average
  precision, which is related to but not numerically identical to the
  trapezoidal area under a precision-recall curve. Mixing the two across
  experiments would make models look different when only the definition
  changed.

* Bootstrap resampling is over SCAFFOLD GROUPS by default, not over
  compounds. Compounds sharing a scaffold are not independent, so resampling
  individual compounds understates the interval. The unit is recorded with
  every interval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)

LOG = logging.getLogger("metrics")

# Metrics computed from probabilities alone, independent of any threshold.
THRESHOLD_FREE = {"pr_auc", "roc_auc", "brier"}
# Metrics that require an operating threshold.
THRESHOLD_DEPENDENT = {
    "mcc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "accuracy",
}


@dataclass
class MetricResult:
    value: float | None
    ci_low: float | None = None
    ci_high: float | None = None
    ci_method: str | None = None
    n_bootstrap: int | None = None


@dataclass
class EvaluationResult:
    metrics: dict[str, MetricResult]
    threshold: float
    prevalence: float
    n: int
    n_positive: int
    n_negative: int
    confusion: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_rows(self) -> list[dict]:
        return [
            {
                "metric_name": name,
                "value": m.value,
                "ci_low": m.ci_low,
                "ci_high": m.ci_high,
                "ci_method": m.ci_method,
                "n_bootstrap": m.n_bootstrap,
            }
            for name, m in self.metrics.items()
        ]


def _safe(fn, *args, **kwargs) -> float | None:
    """Return None rather than raising when a metric is undefined.

    A metric is undefined when, for instance, the held-out set contains a
    single class. Returning None makes that visible; returning 0.0 would look
    like a terrible model rather than an unanswerable question.

    sklearn's warnings for these cases are suppressed because the condition is
    handled explicitly here; leaving them on would bury the real warnings the
    runner emits.
    """
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = float(fn(*args, **kwargs))
        return None if np.isnan(value) else value
    except (ValueError, ZeroDivisionError):
        return None


ALL_METRICS = (
    "pr_auc",
    "roc_auc",
    "brier",
    "mcc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "accuracy",
)
_CONFUSION_METRICS = frozenset(
    {"sensitivity", "specificity", "precision", "accuracy"}
)


def point_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    which: set[str] | list[str] | None = None,
) -> dict[str, float | None]:
    """Metrics at one threshold, without intervals.

    `which` restricts the computation. Each sklearn call costs around a
    millisecond, which is irrelevant once but not across tens of thousands of
    bootstrap resamples: computing nine metrics to use one made the paired
    comparison roughly nine times slower than it needed to be.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    wanted = set(which) if which else set(ALL_METRICS)

    # Ranking metrics are undefined when the held-out set holds one class.
    # average_precision_score returns 1.0 for an all-positive set, which reads
    # as a perfect model rather than an unanswerable question, so it is nulled
    # here alongside ROC-AUC, which sklearn already declines to compute.
    single_class = len(np.unique(y_true)) < 2

    out: dict[str, float | None] = {}
    if "pr_auc" in wanted:
        out["pr_auc"] = (
            None if single_class else _safe(average_precision_score, y_true, y_prob)
        )
    if "roc_auc" in wanted:
        out["roc_auc"] = (
            None if single_class else _safe(roc_auc_score, y_true, y_prob)
        )
    if "brier" in wanted:
        out["brier"] = _safe(brier_score_loss, y_true, y_prob)
    if "mcc" in wanted:
        out["mcc"] = _safe(matthews_corrcoef, y_true, y_pred)
    if "balanced_accuracy" in wanted:
        out["balanced_accuracy"] = _safe(balanced_accuracy_score, y_true, y_pred)

    if wanted & _CONFUSION_METRICS:
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            if "sensitivity" in wanted:
                out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else None
            if "specificity" in wanted:
                out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else None
            if "precision" in wanted:
                out["precision"] = float(tp / (tp + fp)) if (tp + fp) else None
            if "accuracy" in wanted:
                out["accuracy"] = (
                    float((tp + tn) / len(y_true)) if len(y_true) else None
                )
        except ValueError:
            for k in _CONFUSION_METRICS & wanted:
                out[k] = None
    return out


def choose_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, policy: str = "max_mcc_on_validation"
) -> float:
    """Pick an operating threshold. Must be called on VALIDATION only.

    Choosing a threshold on the test set is a quiet way of fitting to it: the
    threshold is a model parameter, and tuning it on held-out data makes the
    held-out score optimistic.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    if policy == "fixed_0.5" or len(np.unique(y_true)) < 2:
        return 0.5

    candidates = np.unique(np.round(y_prob, 4))
    if candidates.size == 0:
        return 0.5
    # Midpoints between observed probabilities, plus the extremes.
    grid = np.clip(np.concatenate([[0.01], candidates, [0.99]]), 0.01, 0.99)

    if policy == "max_mcc_on_validation":
        scorer = lambda t: _safe(  # noqa: E731
            matthews_corrcoef, y_true, (y_prob >= t).astype(int)
        )
    elif policy == "max_balanced_accuracy_on_validation":
        scorer = lambda t: _safe(  # noqa: E731
            balanced_accuracy_score, y_true, (y_prob >= t).astype(int)
        )
    else:
        raise ValueError(f"unknown threshold policy: {policy}")

    best_t, best_s = 0.5, -np.inf
    for t in grid:
        s = scorer(t)
        if s is not None and s > best_s:
            best_t, best_s = float(t), s
    return best_t


def bootstrap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
    groups: np.ndarray | None = None,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
    metrics: list[str] | None = None,
) -> dict[str, MetricResult]:
    """Percentile bootstrap intervals.

    `groups` is the resampling unit. When supplied, whole groups are resampled
    rather than individual rows, because compounds sharing a scaffold are not
    independent and resampling them separately gives intervals that are too
    narrow.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    point = point_metrics(y_true, y_prob, threshold)
    wanted = metrics or list(point.keys())
    rng = np.random.default_rng(seed)

    if n == 0:
        return {m: MetricResult(value=None) for m in wanted}

    if groups is None:
        unit = "row"
        unique_groups = np.arange(n)
        index_by_group = {g: np.array([g]) for g in unique_groups}
    else:
        unit = "group"
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        index_by_group = {g: np.where(groups == g)[0] for g in unique_groups}

    samples: dict[str, list[float]] = {m: [] for m in wanted}
    n_groups = len(unique_groups)

    for _ in range(n_bootstrap):
        picked = rng.choice(unique_groups, size=n_groups, replace=True)
        idx = np.concatenate([index_by_group[g] for g in picked])
        if len(np.unique(y_true[idx])) < 2:
            # A resample with one class leaves most metrics undefined. Skipping
            # it biases the interval slightly; including it as NaN would be
            # worse. The count of usable resamples is reported.
            continue
        boot = point_metrics(y_true[idx], y_prob[idx], threshold, which=wanted)
        for m in wanted:
            v = boot.get(m)
            if v is not None:
                samples[m].append(v)

    alpha = (1.0 - confidence_level) / 2.0
    out: dict[str, MetricResult] = {}
    for m in wanted:
        vals = np.array(samples[m], dtype=float)
        if vals.size < 20:
            out[m] = MetricResult(
                value=point.get(m),
                ci_method=f"bootstrap_percentile_by_{unit}_insufficient_resamples",
                n_bootstrap=int(vals.size),
            )
            continue
        out[m] = MetricResult(
            value=point.get(m),
            ci_low=float(np.quantile(vals, alpha)),
            ci_high=float(np.quantile(vals, 1 - alpha)),
            ci_method=f"bootstrap_percentile_by_{unit}",
            n_bootstrap=int(vals.size),
        )
    return out


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
    groups: np.ndarray | None = None,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
    metrics: list[str] | None = None,
) -> EvaluationResult:
    """Full evaluation at a fixed threshold, with intervals and prevalence."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    results = bootstrap_metrics(
        y_true,
        y_prob,
        threshold=threshold,
        groups=groups,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
        metrics=metrics,
    )

    n = len(y_true)
    n_pos = int(y_true.sum())
    notes: list[str] = []
    if n and len(np.unique(y_true)) < 2:
        notes.append(
            "held-out set contains a single class; ranking metrics are undefined"
        )

    confusion: dict[str, int] = {}
    try:
        tn, fp, fn, tp = confusion_matrix(
            y_true, (y_prob >= threshold).astype(int), labels=[0, 1]
        ).ravel()
        confusion = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    except ValueError:
        pass

    return EvaluationResult(
        metrics=results,
        threshold=float(threshold),
        prevalence=float(n_pos / n) if n else 0.0,
        n=n,
        n_positive=n_pos,
        n_negative=int(n - n_pos),
        confusion=confusion,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------


def paired_bootstrap_difference(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    *,
    metric: str = "pr_auc",
    threshold: float = 0.5,
    groups: np.ndarray | None = None,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Difference between two models on the SAME held-out compounds.

    Pairing matters: comparing two independent intervals is a weaker test than
    comparing the paired difference, and overlapping intervals do not imply the
    difference is not significant.
    """
    y_true = np.asarray(y_true).astype(int)
    a = np.asarray(y_prob_a, dtype=float)
    b = np.asarray(y_prob_b, dtype=float)
    if not (len(y_true) == len(a) == len(b)):
        raise ValueError("paired comparison needs identical held-out compounds")

    n = len(y_true)
    rng = np.random.default_rng(seed)
    if groups is None:
        unique_groups, index_by_group = np.arange(n), {i: np.array([i]) for i in range(n)}
    else:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        index_by_group = {g: np.where(groups == g)[0] for g in unique_groups}

    only = {metric}
    point_a = point_metrics(y_true, a, threshold, which=only).get(metric)
    point_b = point_metrics(y_true, b, threshold, which=only).get(metric)
    observed = (
        None if point_a is None or point_b is None else float(point_a - point_b)
    )

    diffs: list[float] = []
    for _ in range(n_bootstrap):
        picked = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([index_by_group[g] for g in picked])
        if len(np.unique(y_true[idx])) < 2:
            continue
        va = point_metrics(y_true[idx], a[idx], threshold, which=only).get(metric)
        vb = point_metrics(y_true[idx], b[idx], threshold, which=only).get(metric)
        if va is not None and vb is not None:
            diffs.append(va - vb)

    arr = np.array(diffs, dtype=float)
    if arr.size < 20:
        return {
            "metric": metric,
            "difference": observed,
            "ci_low": None,
            "ci_high": None,
            "n_bootstrap": int(arr.size),
            "note": "insufficient usable resamples for an interval",
        }

    alpha = (1.0 - confidence_level) / 2.0
    lo, hi = float(np.quantile(arr, alpha)), float(np.quantile(arr, 1 - alpha))
    return {
        "metric": metric,
        "difference": observed,
        "ci_low": lo,
        "ci_high": hi,
        "n_bootstrap": int(arr.size),
        # The interval excluding zero is the test. Non-overlap of two separate
        # intervals is not.
        "excludes_zero": bool(lo > 0 or hi < 0),
        "p_value_two_sided": float(2 * min((arr <= 0).mean(), (arr >= 0).mean())),
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibration_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> dict:
    """Reliability curve plus expected and maximum calibration error."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n == 0:
        return {"bins": [], "ece": None, "mce": None, "n_bins": n_bins}

    if strategy == "quantile":
        edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    if edges.size < 2:
        edges = np.array([0.0, 1.0])

    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, len(edges) - 2)
    bins, ece, mce = [], 0.0, 0.0
    for b in range(len(edges) - 1):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(y_prob[mask].mean())
        frac_pos = float(y_true[mask].mean())
        gap = abs(mean_pred - frac_pos)
        ece += (count / n) * gap
        mce = max(mce, gap)
        bins.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "count": count,
                "mean_predicted": mean_pred,
                "fraction_positive": frac_pos,
                "gap": gap,
            }
        )
    return {"bins": bins, "ece": float(ece), "mce": float(mce), "n_bins": len(bins)}


def brier_decomposition(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10) -> dict:
    """Murphy decomposition: Brier = reliability - resolution + uncertainty.

    Reliability is calibration error (lower is better); resolution is how far
    predictions move away from the base rate (higher is better); uncertainty is
    a property of the data alone and no model can change it.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n == 0:
        return {"brier": None, "reliability": None, "resolution": None, "uncertainty": None}

    base = float(y_true.mean())
    uncertainty = base * (1 - base)

    edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        edges = np.array([0.0, 1.0])
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, len(edges) - 2)

    reliability = resolution = 0.0
    for b in range(len(edges) - 1):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(y_prob[mask].mean())
        frac_pos = float(y_true[mask].mean())
        reliability += (count / n) * (mean_pred - frac_pos) ** 2
        resolution += (count / n) * (frac_pos - base) ** 2

    return {
        "brier": float(np.mean((y_prob - y_true) ** 2)),
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "base_rate": base,
    }


def pr_curve_points(y_true: np.ndarray, y_prob: np.ndarray, *, max_points: int = 200) -> dict:
    """Precision-recall curve, thinned for storage."""
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return {"precision": [], "recall": [], "prevalence": float(np.mean(y_true))}
    precision, recall, _ = precision_recall_curve(y_true, np.asarray(y_prob, dtype=float))
    if len(precision) > max_points:
        keep = np.linspace(0, len(precision) - 1, max_points).astype(int)
        precision, recall = precision[keep], recall[keep]
    return {
        "precision": [float(p) for p in precision],
        "recall": [float(r) for r in recall],
        "prevalence": float(np.mean(y_true)),
    }


__all__ = [
    "EvaluationResult",
    "MetricResult",
    "THRESHOLD_DEPENDENT",
    "THRESHOLD_FREE",
    "bootstrap_metrics",
    "brier_decomposition",
    "calibration_curve_data",
    "choose_threshold",
    "evaluate",
    "paired_bootstrap_difference",
    "point_metrics",
    "pr_curve_points",
]
