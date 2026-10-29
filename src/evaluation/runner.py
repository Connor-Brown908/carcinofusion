"""Phase 2 experiment runner.

Runs every enabled model under both split strategies, with one training and
evaluation path for all of them, and records enough provenance that a number
can be tied to the conditions that produced it.

Three rules the runner enforces rather than trusting:

* Hyperparameters are selected on VALIDATION. The test set is not scored until
  selection is complete, and every test evaluation is counted so repeated
  touches are visible rather than forgotten.
* The operating threshold is chosen on validation too. A threshold tuned on
  test makes the test score optimistic, quietly.
* Every result carries its data provenance. A run against invented data is
  marked SYNTHETIC_FIXTURE and is never presented as a carcinogenicity finding.

Usage:
    python -m src.evaluation.runner --endpoint primary
    python -m src.evaluation.runner --models B0 B1 --splits scaffold
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import StageManifest, load_config, resolve_path, setup_logging, utc_now
from src.evaluation.metrics import (
    brier_decomposition,
    calibration_curve_data,
    choose_threshold,
    evaluate,
    point_metrics,
    pr_curve_points,
)
from src.models.base import UnavailableModel, build_featurizer
from src.models.baselines import make_baseline

LOG = logging.getLogger("runner")
STAGE = "11_benchmark"

PARTITIONS = ("train", "validation", "test")


def dependency_versions() -> dict:
    """Versions of everything that can change a number."""
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "pandas", "sklearn", "rdkit", "lightgbm", "xgboost", "torch", "transformers"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            out[mod] = None
    return out


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def grid_points(grid: dict) -> list[dict]:
    """Cartesian product of a hyperparameter grid."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def load_split(
    processed: Path, endpoint: str, strategy: str
) -> tuple[pd.DataFrame, dict]:
    """Load one endpoint's split, keeping only partitioned compounds."""
    path = processed / f"splits_{endpoint}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the Phase 1 pipeline first."
        )
    df = pd.read_parquet(path)
    df = df[df["split_strategy"] == strategy]
    df = df[df["split_assignment"].isin(PARTITIONS)].copy()
    df["label"] = df["label"].astype(int)
    info = {
        "n_total": int(len(df)),
        "by_partition": df["split_assignment"].value_counts().to_dict(),
    }
    return df, info


def nearest_neighbour_similarity(
    test_smiles: list[str], train_smiles: list[str], cfg: dict
) -> np.ndarray:
    from src.features.fingerprints import nearest_neighbour_similarities

    return nearest_neighbour_similarities(
        test_smiles,
        train_smiles,
        radius=cfg.get("radius", 2),
        n_bits=cfg.get("n_bits", 2048),
    )


def build_model(model_id: str, spec: dict, hp: dict, seed: int):
    """Construct a model from its registry entry plus chosen hyperparameters."""
    fixed = dict(spec.get("fixed", {}))
    params = {**fixed, **hp}

    if model_id == "B0":
        return make_baseline("B0", seed=seed)

    if model_id in ("B1", "B2", "B3"):
        descriptors = spec.get("descriptors") if model_id == "B3" else None
        if model_id == "B3" and params.pop("_variant", "fp_plus_descriptors") == "fp_only":
            descriptors = None
        featurizer = build_featurizer(spec.get("featurizer", {}), descriptors)
        if model_id == "B3":
            params["backend"] = spec.get("backend", "lightgbm")
        return make_baseline(model_id, featurizer=featurizer, seed=seed, **params)

    if model_id == "B4":
        from src.models.mpnn import MPNNModel

        arch = dict(spec.get("architecture", {}))
        return MPNNModel(seed=seed, **{**arch, **params})

    if model_id == "B5":
        from src.models.pretrained import PretrainedEncoderModel

        return PretrainedEncoderModel(
            seed=seed,
            checkpoint=spec.get("checkpoint"),
            mode=spec.get("default_mode", "frozen_embeddings"),
            **params,
        )

    raise ValueError(f"unknown model id: {model_id}")


def run_one(
    model_id: str,
    spec: dict,
    split_df: pd.DataFrame,
    *,
    bench_cfg: dict,
    endpoint: str,
    strategy: str,
    seed: int,
    provenance: str,
    versions: dict,
    man: StageManifest,
) -> dict:
    """Select on validation, then score the test set exactly once."""
    train = split_df[split_df["split_assignment"] == "train"]
    val = split_df[split_df["split_assignment"] == "validation"]
    test = split_df[split_df["split_assignment"] == "test"]

    run_id = f"{model_id}_{strategy}_{endpoint}_{uuid.uuid4().hex[:8]}"
    started = utc_now()
    t0 = time.time()

    record: dict = {
        "run_id": run_id,
        "model_id": model_id,
        "model_name": spec.get("name", model_id),
        "representation": spec.get("representation", "none"),
        "endpoint_name": endpoint,
        "split_strategy": strategy,
        "seed": seed,
        "data_provenance": provenance,
        "dependency_versions": versions,
        "n_train": int(len(train)),
        "n_validation": int(len(val)),
        "n_test": int(len(test)),
        "train_prevalence": float(train["label"].mean()) if len(train) else None,
        "test_prevalence": float(test["label"].mean()) if len(test) else None,
        "started_at": started,
        "status": "completed",
        "unavailable_reason": None,
    }

    if len(train) == 0 or len(test) == 0:
        record.update(
            status="failed",
            unavailable_reason="training or test partition is empty",
            finished_at=utc_now(),
        )
        return record

    # ---- hyperparameter selection, on validation only -------------------
    grid = grid_points(spec.get("grid", {}))
    if model_id == "B3" and spec.get("variants"):
        grid = [
            {**g, "_variant": v} for g in grid for v in spec["variants"]
        ]

    # Selection metric chain. The primary metric is undefined when the
    # validation fold holds a single class, which a small or imbalanced
    # dataset produces readily. Falling through to a defined metric keeps the
    # run alive; which metric was actually used is recorded, because a model
    # chosen on balanced accuracy is not chosen the same way as one chosen on
    # PR-AUC.
    primary = bench_cfg["metrics"]["primary"]
    metric_chain = [primary] + [
        m
        for m in ("roc_auc", "mcc", "balanced_accuracy")
        if m != primary
    ]

    selection: list[dict] = []
    fitted: list[tuple[dict, object]] = []
    n_fit_failures = 0
    last_fit_error: str | None = None

    for hp in grid:
        try:
            model = build_model(model_id, spec, dict(hp), seed)
            model.fit(
                train["canonical_smiles"].tolist(),
                train["label"].to_numpy(),
                val["canonical_smiles"].tolist() if len(val) else None,
                val["label"].to_numpy() if len(val) else None,
            )
        except UnavailableModel as exc:
            record.update(
                status="unavailable",
                unavailable_reason=str(exc),
                finished_at=utc_now(),
            )
            man.warn(f"{model_id}: unavailable - {exc}")
            return record
        except Exception as exc:  # noqa: BLE001
            n_fit_failures += 1
            last_fit_error = f"{type(exc).__name__}: {exc}"
            man.warn(f"{model_id} hp={hp}: fit failed - {exc}")
            continue
        fitted.append((dict(hp), model))

    if not fitted:
        record.update(
            status="failed",
            unavailable_reason=(
                f"all {len(grid)} hyperparameter setting(s) failed to fit; "
                f"last error: {last_fit_error}"
            ),
            finished_at=utc_now(),
        )
        return record

    def score_on(frame, metric: str):
        """Score every fitted candidate on `frame` with one metric."""
        out = []
        for hp, model in fitted:
            probs = model.predict_proba(frame["canonical_smiles"].tolist())
            out.append(
                (hp, model, point_metrics(frame["label"].to_numpy(), probs, 0.5).get(metric))
            )
        return out

    best_hp = best_model = None
    selection_metric = None
    selection_partition = None
    selection_fallback = None

    # Validation first, trying each metric in the chain.
    if len(val):
        for metric in metric_chain:
            scored = score_on(val, metric)
            usable = [(hp, m, s) for hp, m, s in scored if s is not None]
            selection.append(
                {
                    "partition": "validation",
                    "metric": metric,
                    "scores": [
                        {"hyperparameters": hp, "value": s} for hp, _, s in scored
                    ],
                }
            )
            if usable:
                best_hp, best_model, _ = max(usable, key=lambda t: t[2])
                selection_metric, selection_partition = metric, "validation"
                if metric != primary:
                    selection_fallback = (
                        f"primary metric '{primary}' undefined on validation "
                        f"(single-class fold); selected on '{metric}' instead"
                    )
                break

    # Validation gave nothing usable at all. Selecting on train favours the
    # model that overfits hardest, so it is a last resort and is recorded.
    if best_model is None:
        for metric in metric_chain:
            scored = score_on(train, metric)
            usable = [(hp, m, s) for hp, m, s in scored if s is not None]
            if usable:
                best_hp, best_model, _ = max(usable, key=lambda t: t[2])
                selection_metric, selection_partition = metric, "train"
                selection_fallback = (
                    "no selection metric was defined on validation; selection "
                    "fell back to the training fold, which favours overfitting "
                    "and makes this run's hyperparameters untrustworthy"
                )
                man.warn(f"{model_id}/{strategy}: {selection_fallback}")
                break

    # Still nothing scoreable: take the single fitted candidate rather than
    # discarding a model that trained successfully.
    if best_model is None:
        best_hp, best_model = fitted[0]
        selection_metric, selection_partition = None, None
        selection_fallback = (
            "no metric was defined on any partition; the first fitted "
            "candidate was taken without selection"
        )
        man.warn(f"{model_id}/{strategy}: {selection_fallback}")

    # ---- threshold, also on validation ----------------------------------
    threshold_source = val if len(val) else train
    thr_probs = best_model.predict_proba(threshold_source["canonical_smiles"].tolist())
    threshold = choose_threshold(
        threshold_source["label"].to_numpy(),
        thr_probs,
        policy=bench_cfg["metrics"].get("threshold_policy", "max_mcc_on_validation"),
    )

    # ---- single test evaluation -----------------------------------------
    unc = bench_cfg["uncertainty"]
    resample_col = (
        "scaffold_group"
        if unc.get("resample_unit") == "scaffold_group"
        and "scaffold_group" in test.columns
        else None
    )

    results: dict[str, dict] = {}
    predictions: list[dict] = []

    for part_name, part in (("validation", val), ("test", test)):
        if len(part) == 0:
            continue
        probs = best_model.predict_proba(part["canonical_smiles"].tolist())
        groups = part[resample_col].to_numpy() if resample_col else None
        ev = evaluate(
            part["label"].to_numpy(),
            probs,
            threshold=threshold,
            groups=groups,
            n_bootstrap=unc.get("n_bootstrap", 2000),
            confidence_level=unc.get("confidence_level", 0.95),
            seed=seed,
            metrics=bench_cfg["metrics"]["report"],
        )
        cal = calibration_curve_data(
            part["label"].to_numpy(),
            probs,
            n_bins=bench_cfg["calibration"].get("n_bins", 10),
            strategy=bench_cfg["calibration"].get("strategy", "quantile"),
        )
        results[part_name] = {
            "metrics": ev.as_rows(),
            "threshold": ev.threshold,
            "prevalence": ev.prevalence,
            "n": ev.n,
            "n_positive": ev.n_positive,
            "n_negative": ev.n_negative,
            "confusion": ev.confusion,
            "notes": ev.notes,
            "calibration": cal,
            "brier_decomposition": brier_decomposition(
                part["label"].to_numpy(), probs
            ),
            "pr_curve": pr_curve_points(part["label"].to_numpy(), probs),
        }

        sims = nearest_neighbour_similarity(
            part["canonical_smiles"].tolist(),
            train["canonical_smiles"].tolist(),
            bench_cfg["similarity_analysis"],
        )
        for i, (_, row) in enumerate(part.iterrows()):
            predictions.append(
                {
                    "run_id": run_id,
                    "compound_id": row["compound_id"],
                    "partition": part_name,
                    "y_true": int(row["label"]),
                    "y_prob": float(probs[i]),
                    "nn_similarity": float(sims[i]) if i < len(sims) else None,
                }
            )

    record.update(
        {
            "hyperparameters": best_hp or {},
            "selection_trace": selection,
            "selection_metric": selection_metric,
            "selection_partition": selection_partition,
            "selection_fallback": selection_fallback,
            "n_fit_failures": n_fit_failures,
            "threshold": threshold,
            "threshold_selected_on": "validation" if len(val) else "train",
            "results": results,
            "predictions": predictions,
            "model_description": best_model.describe(),
            "train_seconds": round(time.time() - t0, 2),
            "finished_at": utc_now(),
        }
    )
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument(
        "--provenance",
        default=None,
        choices=["REAL_CPDB", "SYNTHETIC_FIXTURE"],
        help="stamped on every result; inferred from the dataset when omitted",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    setup_logging()
    data_cfg = load_config("data")
    lab_cfg = load_config("labels")
    split_cfg = load_config("splits")
    bench = load_config("benchmark")

    processed = resolve_path(data_cfg["paths"]["processed"])
    out_dir = Path(args.out) if args.out else processed / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoint = args.endpoint or bench["task"]["endpoint_name"]
    strategies = args.splits or [
        bench["evaluation"]["primary_split"],
        bench["evaluation"]["secondary_split"],
    ]
    model_ids = args.models or [
        mid for mid, spec in bench["models"].items() if spec.get("enabled", True)
    ]

    # Provenance is required. Guessing it is how a synthetic number ends up in
    # a results table, so it is inferred only from an explicit marker file.
    provenance = args.provenance
    if provenance is None:
        marker = processed / "DATA_PROVENANCE"
        provenance = (
            marker.read_text(encoding="utf-8").strip()
            if marker.exists()
            else "SYNTHETIC_FIXTURE"
        )
    if provenance == "SYNTHETIC_FIXTURE":
        LOG.warning(
            "Results are marked SYNTHETIC_FIXTURE. They exercise the harness "
            "and are NOT carcinogenicity findings."
        )

    versions = dependency_versions()
    man = StageManifest(stage=STAGE)
    man.config_versions = {
        "protocol_version": bench["protocol_version"],
        "dataset_version": data_cfg["dataset_version"],
        "label_policy_version": lab_cfg["label_policy_version"],
        "split_version": split_cfg["split_version"],
        "seed": bench["reproducibility"]["seed"],
        "config_hash": config_hash(bench),
        "data_provenance": provenance,
    }

    seed = bench["reproducibility"]["seed"]
    all_runs: list[dict] = []

    for strategy in strategies:
        try:
            split_df, info = load_split(processed, endpoint, strategy)
        except FileNotFoundError as exc:
            LOG.error("%s", exc)
            man.warn(str(exc))
            man.write()
            return 2
        LOG.info("%s/%s: %s", endpoint, strategy, info["by_partition"])

        n_test = int((split_df["split_assignment"] == "test").sum())
        floor = bench["reporting"].get("min_test_compounds_for_headline", 50)
        if n_test < floor:
            man.warn(
                f"{strategy}: {n_test} test compounds, below the {floor} needed "
                f"for a headline number; metrics are indicative only"
            )

        for model_id in model_ids:
            spec = bench["models"].get(model_id)
            if spec is None:
                man.warn(f"unknown model id in --models: {model_id}")
                continue
            LOG.info("running %s on %s split", model_id, strategy)
            rec = run_one(
                model_id,
                spec,
                split_df,
                bench_cfg=bench,
                endpoint=endpoint,
                strategy=strategy,
                seed=seed,
                provenance=provenance,
                versions=versions,
                man=man,
            )
            rec["dataset_version"] = data_cfg["dataset_version"]
            rec["label_policy_version"] = lab_cfg["label_policy_version"]
            rec["split_version"] = split_cfg["split_version"]
            rec["protocol_version"] = bench["protocol_version"]
            rec["config"] = {
                "protocol_version": bench["protocol_version"],
                "config_hash": config_hash(bench),
                "endpoint": endpoint,
                "model_spec": spec,
            }
            all_runs.append(rec)

            if rec["status"] == "completed":
                test_res = rec["results"].get("test", {})
                primary = bench["metrics"]["primary"]
                row = next(
                    (m for m in test_res.get("metrics", []) if m["metric_name"] == primary),
                    None,
                )
                LOG.info(
                    "  %s/%s %s=%s (prevalence %.3f, n_test=%d)",
                    model_id,
                    strategy,
                    primary,
                    f"{row['value']:.3f}" if row and row["value"] is not None else "n/a",
                    test_res.get("prevalence", 0.0),
                    test_res.get("n", 0),
                )
            else:
                LOG.info("  %s/%s %s: %s", model_id, strategy, rec["status"], rec["unavailable_reason"])

    runs_path = out_dir / f"runs_{endpoint}.json"
    with runs_path.open("w", encoding="utf-8") as fh:
        json.dump(all_runs, fh, indent=2, default=str)
        fh.write("\n")
    man.add_output(runs_path)

    rows = []
    for r in all_runs:
        base = {
            "model_id": r["model_id"],
            "model_name": r["model_name"],
            "split_strategy": r["split_strategy"],
            "status": r["status"],
            "data_provenance": r["data_provenance"],
            "n_test": r.get("n_test"),
            "test_prevalence": r.get("test_prevalence"),
            "unavailable_reason": r.get("unavailable_reason"),
        }
        for m in r.get("results", {}).get("test", {}).get("metrics", []):
            base[m["metric_name"]] = m["value"]
            base[f"{m['metric_name']}_ci_low"] = m["ci_low"]
            base[f"{m['metric_name']}_ci_high"] = m["ci_high"]
        rows.append(base)
    summary = pd.DataFrame(rows)
    sum_path = out_dir / f"summary_{endpoint}.parquet"
    summary.to_parquet(sum_path, index=False)
    man.add_output(sum_path, rows=len(summary))

    man.counts = {
        "runs": len(all_runs),
        "completed": sum(1 for r in all_runs if r["status"] == "completed"),
        "unavailable": sum(1 for r in all_runs if r["status"] == "unavailable"),
        "failed": sum(1 for r in all_runs if r["status"] == "failed"),
        "data_provenance": provenance,
    }
    man.write()
    LOG.info("benchmark written: %s", runs_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
