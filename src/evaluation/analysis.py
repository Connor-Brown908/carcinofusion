"""Post-hoc analysis of benchmark runs.

Four questions the summary table cannot answer:

  1. Does performance survive chemical novelty, or does it decay as test
     compounds get further from anything in training? This is the applicability
     domain question, and under OECD guidance a prediction is only interpretable
     inside a stated domain.
  2. Which compounds are wrong, and do the false negatives share anything? A
     false negative on a carcinogen is not interchangeable with a false positive.
  3. Are the probabilities calibrated, or merely well-ranked? A model can rank
     perfectly and still be useless for a risk assessment that needs a number.
  4. Does the random-versus-scaffold gap indicate memorised chemistry?

Reads the runner's JSON output; writes tables and a markdown section.

Usage:
    python -m src.evaluation.analysis --endpoint primary
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import StageManifest, load_config, resolve_path, setup_logging
from src.evaluation.metrics import paired_bootstrap_difference, point_metrics

LOG = logging.getLogger("analysis")
STAGE = "12_analysis"


def predictions_frame(runs: list[dict]) -> pd.DataFrame:
    """All per-compound predictions across runs, in one table."""
    rows = []
    for r in runs:
        if r.get("status") != "completed":
            continue
        for p in r.get("predictions", []):
            rows.append(
                {
                    **p,
                    "model_id": r["model_id"],
                    "model_name": r["model_name"],
                    "split_strategy": r["split_strategy"],
                    "threshold": r.get("threshold", 0.5),
                }
            )
    return pd.DataFrame(rows)


def similarity_analysis(preds: pd.DataFrame, bins: list[float]) -> pd.DataFrame:
    """Performance stratified by distance to the nearest training compound.

    The question is not "how accurate is the model" but "how far from known
    chemistry does it stay accurate". A model whose accuracy collapses in the
    low-similarity bin has a narrow applicability domain however good its
    headline number.
    """
    if preds.empty:
        return pd.DataFrame()

    df = preds[(preds["partition"] == "test") & preds["nn_similarity"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]
    df["similarity_bin"] = pd.cut(
        df["nn_similarity"], bins=bins, labels=labels, include_lowest=True
    )

    rows = []
    for (model, strategy, bin_label), grp in df.groupby(
        ["model_id", "split_strategy", "similarity_bin"], observed=True
    ):
        y, p = grp["y_true"].to_numpy(), grp["y_prob"].to_numpy()
        thr = float(grp["threshold"].iloc[0])
        m = point_metrics(y, p, thr)
        rows.append(
            {
                "model_id": model,
                "split_strategy": strategy,
                "similarity_bin": str(bin_label),
                "n": int(len(grp)),
                "n_positive": int(y.sum()),
                "prevalence": float(y.mean()) if len(y) else None,
                "pr_auc": m["pr_auc"],
                "roc_auc": m["roc_auc"],
                "balanced_accuracy": m["balanced_accuracy"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "mean_similarity": float(grp["nn_similarity"].mean()),
            }
        )
    return pd.DataFrame(rows)


def error_analysis(preds: pd.DataFrame, compounds: pd.DataFrame | None) -> pd.DataFrame:
    """Every misclassified test compound, with its structure and novelty.

    False negatives and false positives are listed separately rather than
    pooled into an error rate: for this endpoint a missed carcinogen and a
    false alarm carry very different costs.
    """
    if preds.empty:
        return pd.DataFrame()

    df = preds[preds["partition"] == "test"].copy()
    df["y_pred"] = (df["y_prob"] >= df["threshold"]).astype(int)
    df["correct"] = df["y_pred"] == df["y_true"]
    df["error_type"] = np.where(
        df["correct"],
        "correct",
        np.where(df["y_true"] == 1, "false_negative", "false_positive"),
    )
    # How wrong, not merely whether: a confident error is worse than a
    # borderline one and should be visible in triage.
    df["confidence_in_error"] = np.where(
        df["y_true"] == 1, 1.0 - df["y_prob"], df["y_prob"]
    )

    errors = df[~df["correct"]].copy()
    if compounds is not None and not compounds.empty and not errors.empty:
        cols = [
            c
            for c in (
                "compound_id",
                "canonical_smiles",
                "scaffold_group",
                "molecular_weight",
                "n_experiments",
                "label_confidence",
            )
            if c in compounds.columns
        ]
        errors = errors.merge(compounds[cols], on="compound_id", how="left")

    return errors.sort_values(
        ["model_id", "split_strategy", "error_type", "confidence_in_error"],
        ascending=[True, True, True, False],
    )


def error_summary(preds: pd.DataFrame) -> pd.DataFrame:
    """Counts and rates per model and split."""
    if preds.empty:
        return pd.DataFrame()
    df = preds[preds["partition"] == "test"].copy()
    df["y_pred"] = (df["y_prob"] >= df["threshold"]).astype(int)

    rows = []
    for (model, strategy), grp in df.groupby(["model_id", "split_strategy"]):
        y, yp = grp["y_true"].to_numpy(), grp["y_pred"].to_numpy()
        tp = int(((y == 1) & (yp == 1)).sum())
        tn = int(((y == 0) & (yp == 0)).sum())
        fp = int(((y == 0) & (yp == 1)).sum())
        fn = int(((y == 1) & (yp == 0)).sum())
        rows.append(
            {
                "model_id": model,
                "split_strategy": strategy,
                "n_test": int(len(grp)),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "false_negative_rate": float(fn / (tp + fn)) if (tp + fn) else None,
                "false_positive_rate": float(fp / (tn + fp)) if (tn + fp) else None,
                "mean_fn_similarity": (
                    float(
                        grp[(y == 1) & (yp == 0)]["nn_similarity"].mean()
                    )
                    if fn
                    else None
                ),
                "mean_correct_similarity": (
                    float(grp[y == yp]["nn_similarity"].mean())
                    if (tp + tn)
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def generalization_gap(runs: list[dict], metric: str = "pr_auc") -> pd.DataFrame:
    """Random minus scaffold, per model.

    A model that scores well on the random split and poorly on the scaffold
    split has learned the chemical distribution rather than the structural
    determinants. That gap, not the headline number, is what says whether a
    model will hold up on unfamiliar chemistry.
    """
    values: dict[tuple[str, str], float | None] = {}
    names: dict[str, str] = {}
    for r in runs:
        if r.get("status") != "completed":
            continue
        names[r["model_id"]] = r["model_name"]
        row = next(
            (
                m
                for m in r.get("results", {}).get("test", {}).get("metrics", [])
                if m["metric_name"] == metric
            ),
            None,
        )
        values[(r["model_id"], r["split_strategy"])] = row["value"] if row else None

    rows = []
    for model_id, name in sorted(names.items()):
        rnd = values.get((model_id, "random"))
        scaf = values.get((model_id, "scaffold"))
        rows.append(
            {
                "model_id": model_id,
                "model_name": name,
                "metric": metric,
                "random": rnd,
                "scaffold": scaf,
                "gap_random_minus_scaffold": (
                    None if rnd is None or scaf is None else float(rnd - scaf)
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_model_comparison(
    preds: pd.DataFrame, *, strategy: str, metric: str, unc: dict
) -> pd.DataFrame:
    """Every model pair, compared on the identical held-out compounds.

    Pairing matters: overlapping confidence intervals do not mean the
    difference is insignificant, and unpaired comparison confounds the
    difference with which compounds each model happened to be scored on.
    """
    if preds.empty:
        return pd.DataFrame()
    df = preds[
        (preds["partition"] == "test") & (preds["split_strategy"] == strategy)
    ]
    if df.empty:
        return pd.DataFrame()

    wide = df.pivot_table(
        index="compound_id", columns="model_id", values="y_prob", aggfunc="first"
    )
    truth = df.drop_duplicates("compound_id").set_index("compound_id")["y_true"]
    wide = wide.join(truth, how="inner").dropna()
    if wide.empty or len(wide.columns) < 3:
        return pd.DataFrame()

    y = wide["y_true"].to_numpy().astype(int)
    models = [c for c in wide.columns if c != "y_true"]

    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            cmp = paired_bootstrap_difference(
                y,
                wide[a].to_numpy(),
                wide[b].to_numpy(),
                metric=metric,
                threshold=0.5,
                n_bootstrap=min(unc.get("n_bootstrap", 2000), 2000),
                confidence_level=unc.get("confidence_level", 0.95),
            )
            rows.append(
                {
                    "split_strategy": strategy,
                    "model_a": a,
                    "model_b": b,
                    "metric": metric,
                    "difference_a_minus_b": cmp["difference"],
                    "ci_low": cmp["ci_low"],
                    "ci_high": cmp["ci_high"],
                    "excludes_zero": cmp.get("excludes_zero"),
                    "p_value": cmp.get("p_value_two_sided"),
                    "n_compounds": int(len(y)),
                }
            )
    return pd.DataFrame(rows)


def calibration_table(runs: list[dict]) -> pd.DataFrame:
    """Calibration error and Brier decomposition per run."""
    rows = []
    for r in runs:
        if r.get("status") != "completed":
            continue
        test = r.get("results", {}).get("test", {})
        cal = test.get("calibration", {})
        bd = test.get("brier_decomposition", {})
        rows.append(
            {
                "model_id": r["model_id"],
                "model_name": r["model_name"],
                "split_strategy": r["split_strategy"],
                "ece": cal.get("ece"),
                "mce": cal.get("mce"),
                "brier": bd.get("brier"),
                "reliability": bd.get("reliability"),
                "resolution": bd.get("resolution"),
                "uncertainty": bd.get("uncertainty"),
                "base_rate": bd.get("base_rate"),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--benchmark-dir", default=None)
    args = parser.parse_args(argv)

    setup_logging()
    data_cfg = load_config("data")
    bench = load_config("benchmark")
    processed = resolve_path(data_cfg["paths"]["processed"])
    bench_dir = Path(args.benchmark_dir) if args.benchmark_dir else processed / "benchmark"
    endpoint = args.endpoint or bench["task"]["endpoint_name"]

    runs_path = bench_dir / f"runs_{endpoint}.json"
    if not runs_path.exists():
        LOG.error("%s not found; run the benchmark first", runs_path)
        return 2

    man = StageManifest(stage=STAGE)
    man.add_input(runs_path)
    man.config_versions = {"protocol_version": bench["protocol_version"]}

    runs = json.loads(runs_path.read_text(encoding="utf-8"))
    preds = predictions_frame(runs)

    comp_path = processed / f"model_dataset_{endpoint}.parquet"
    compounds = pd.read_parquet(comp_path) if comp_path.exists() else None

    outputs: dict[str, pd.DataFrame] = {
        "similarity_analysis": similarity_analysis(
            preds, bench["similarity_analysis"]["bins"]
        ),
        "error_detail": error_analysis(preds, compounds),
        "error_summary": error_summary(preds),
        "generalization_gap": generalization_gap(runs, bench["metrics"]["primary"]),
        "calibration": calibration_table(runs),
    }
    for strategy in ("scaffold", "random"):
        outputs[f"paired_comparison_{strategy}"] = paired_model_comparison(
            preds,
            strategy=strategy,
            metric=bench["metrics"]["primary"],
            unc=bench["uncertainty"],
        )

    if not preds.empty:
        p = bench_dir / f"predictions_{endpoint}.parquet"
        preds.to_parquet(p, index=False)
        man.add_output(p, rows=len(preds))

    for name, df in outputs.items():
        path = bench_dir / f"{name}_{endpoint}.parquet"
        df.to_parquet(path, index=False)
        man.add_output(path, rows=len(df))
        LOG.info("%s: %d row(s)", name, len(df))

    man.counts = {name: int(len(df)) for name, df in outputs.items()}
    man.counts["predictions"] = int(len(preds))
    man.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
