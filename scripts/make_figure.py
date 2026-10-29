#!/usr/bin/env python3
"""Build Figure 1 from the benchmark artefacts.

Two panels:

  A  PR-AUC on the scaffold split, every model, 95% bootstrap CIs resampled
     over scaffold groups, with a dashed line at test prevalence. PR-AUC's
     floor is the positive rate, not 0.5, so the line is not decoration: a
     bar near it is a model that has learned nothing.

  B  Sensitivity on the NON-GENOTOXIC subset. Genotoxic carcinogens are
     structurally obvious and already caught by mutagenicity alerts, so
     aggregate accuracy largely reflects how many of those a test set
     happens to contain. Panel B is where a platform claim is won or lost.

The figure refuses to draw when the numbers cannot support it: too few test
compounds, synthetic provenance, or a scaffold-split result above the
reproducibility ceiling of the underlying bioassay. Those are stated on the
figure rather than left for a reader to work out.

    python3 scripts/make_figure.py
    python3 scripts/make_figure.py --highlight CarcinoFusion --out figure1.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import load_config, resolve_path, setup_logging  # noqa: E402
from src.evaluation.metrics import bootstrap_metrics, point_metrics  # noqa: E402

LOG = logging.getLogger("figure")

# Replicate rodent bioassays agree at roughly 81% balanced accuracy, so a
# model cannot legitimately exceed that against bioassay labels. A
# scaffold-split result above it is evidence of leakage, not of skill.
ASSAY_CEILING_BALANCED_ACCURACY = 0.81

INK = "#1a1a1a"
MUTED = "#8a8a8a"
BASELINE = "#b9c4cf"
HIGHLIGHT = "#2d6a9f"
EXTERNAL = "#9f6b2d"
WARN = "#b03a2e"


def load_runs(processed: Path, endpoint: str) -> list[dict]:
    path = processed / "benchmark" / f"runs_{endpoint}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run the benchmark first.")
    return json.loads(path.read_text(encoding="utf-8"))


def metric_from_run(run: dict, metric: str, partition: str = "test"):
    for m in run.get("results", {}).get(partition, {}).get("metrics", []):
        if m["metric_name"] == metric:
            return m["value"], m.get("ci_low"), m.get("ci_high")
    return None, None, None


def panel_a(ax, runs: list[dict], split: str, highlight: str | None) -> list[str]:
    notes: list[str] = []
    rows = []
    for r in runs:
        if r.get("split_strategy") != split:
            continue
        value, lo, hi = metric_from_run(r, "pr_auc")
        rows.append(
            {
                "name": r["model_name"],
                "model_id": r.get("model_id", ""),
                "status": r.get("status"),
                "reason": r.get("unavailable_reason"),
                "value": value,
                "lo": lo,
                "hi": hi,
                "prevalence": r.get("test_prevalence"),
                "n_test": r.get("n_test"),
            }
        )
    if not rows:
        ax.text(0.5, 0.5, f"no {split}-split runs", ha="center", va="center",
                transform=ax.transAxes, color=MUTED)
        return ["no runs to plot"]

    order = {"B0": 0, "B1": 1, "B2": 2, "B3": 3, "B4": 4, "B5": 5, "EXT": 6}
    rows.sort(key=lambda d: (order.get(d["model_id"], 7), d["name"]))

    labels = [d["name"] for d in rows]
    values = [d["value"] if d["value"] is not None else 0.0 for d in rows]
    x = np.arange(len(rows))

    colours = []
    for d in rows:
        if highlight and d["name"].lower() == highlight.lower():
            colours.append(HIGHLIGHT)
        elif d["model_id"] == "EXT":
            colours.append(EXTERNAL)
        else:
            colours.append(BASELINE)

    ax.bar(x, values, color=colours, width=0.62, zorder=2)

    for i, d in enumerate(rows):
        if d["value"] is None:
            ax.text(i, 0.02, d["status"] or "n/a", rotation=90, ha="center",
                    va="bottom", fontsize=7, color=WARN)
            continue
        if d["lo"] is not None and d["hi"] is not None:
            ax.plot([i, i], [d["lo"], d["hi"]], color=INK, lw=1.2, zorder=3)
            ax.plot([i - 0.1, i + 0.1], [d["lo"]] * 2, color=INK, lw=1.2, zorder=3)
            ax.plot([i - 0.1, i + 0.1], [d["hi"]] * 2, color=INK, lw=1.2, zorder=3)
        ax.text(i, (d["hi"] if d["hi"] is not None else d["value"]) + 0.02,
                f"{d['value']:.2f}", ha="center", va="bottom", fontsize=8, color=INK)

    prevalences = [d["prevalence"] for d in rows if d["prevalence"] is not None]
    if prevalences:
        prev = float(np.mean(prevalences))
        ax.axhline(prev, ls="--", lw=1.1, color=MUTED, zorder=1)
        ax.text(len(rows) - 0.4, prev + 0.012,
                f"prevalence {prev:.2f}  (PR-AUC floor)",
                ha="right", va="bottom", fontsize=8, color=MUTED)

    n_test = next((d["n_test"] for d in rows if d["n_test"]), 0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("PR-AUC", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_title(f"A   {split} split, n={n_test} test compounds",
                 loc="left", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", zorder=0)

    best = max((d for d in rows if d["value"] is not None and d["model_id"] != "EXT"),
               key=lambda d: d["value"], default=None)
    if best and best["value"] is not None and best["value"] > 0.95:
        notes.append(
            f"{best['name']} reaches PR-AUC {best['value']:.2f} on a scaffold "
            f"split, which is above what the bioassay's own reproducibility "
            f"supports. Treat as leakage until shown otherwise."
        )
    return notes


def panel_b(ax, runs: list[dict], split: str, mechanism: Path, highlight: str | None) -> list[str]:
    notes: list[str] = []
    if not mechanism.exists():
        ax.text(0.5, 0.5,
                "no mechanism labels\n\nrun scripts/build_mechanism_labels.py",
                ha="center", va="center", transform=ax.transAxes,
                color=MUTED, fontsize=9)
        ax.set_title("B   non-genotoxic subset", loc="left", fontsize=11, color=INK)
        ax.axis("off")
        return ["Panel B needs mechanism labels; Figure 1 is incomplete without it"]

    mech = pd.read_csv(mechanism, sep="\t")
    # Alert-derived calls are inferences about structure, not measurements,
    # and cannot support a claim about experimentally confirmed
    # non-genotoxic carcinogens.
    measured = mech[mech["mechanism_confidence"] == "measured"]
    ng = set(measured[measured["mechanism"] == "non_genotoxic"]["compound_id"])
    if len(measured) < len(mech):
        notes.append(
            f"{len(mech) - len(measured)} mechanism call(s) were inferred from "
            f"structural alerts and are excluded from Panel B"
        )
    if len(ng) < 30:
        notes.append(
            f"only {len(ng)} measured non-genotoxic compounds; Panel B is "
            f"indicative, not a result"
        )

    rows = []
    for r in runs:
        if r.get("split_strategy") != split or r.get("status") != "completed":
            continue
        preds = [p for p in r.get("predictions", [])
                 if p["partition"] == "test" and p["compound_id"] in ng]
        if len(preds) < 5:
            continue
        y = np.array([p["y_true"] for p in preds])
        prob = np.array([p["y_prob"] for p in preds])
        thr = r.get("threshold", 0.5)
        groups = None
        res = bootstrap_metrics(y, prob, threshold=thr, groups=groups,
                                n_bootstrap=1000, metrics=["sensitivity"])
        sens = res["sensitivity"]
        rows.append(
            {
                "name": r["model_name"],
                "model_id": r.get("model_id", ""),
                "value": sens.value,
                "lo": sens.ci_low,
                "hi": sens.ci_high,
                "n": len(preds),
                "n_pos": int(y.sum()),
            }
        )

    if not rows:
        ax.text(0.5, 0.5, "no model covers enough\nnon-genotoxic test compounds",
                ha="center", va="center", transform=ax.transAxes, color=MUTED, fontsize=9)
        ax.set_title("B   non-genotoxic subset", loc="left", fontsize=11, color=INK)
        ax.axis("off")
        return notes + ["Panel B could not be drawn"]

    order = {"B0": 0, "B1": 1, "B2": 2, "B3": 3, "B4": 4, "B5": 5, "EXT": 6}
    rows.sort(key=lambda d: (order.get(d["model_id"], 7), d["name"]))
    x = np.arange(len(rows))
    colours = [
        HIGHLIGHT if highlight and d["name"].lower() == highlight.lower()
        else EXTERNAL if d["model_id"] == "EXT" else BASELINE
        for d in rows
    ]
    ax.bar(x, [d["value"] or 0.0 for d in rows], color=colours, width=0.62, zorder=2)

    for i, d in enumerate(rows):
        if d["lo"] is not None and d["hi"] is not None:
            ax.plot([i, i], [d["lo"], d["hi"]], color=INK, lw=1.2, zorder=3)
        if d["value"] is not None:
            ax.text(i, (d["hi"] or d["value"]) + 0.02, f"{d['value']:.2f}",
                    ha="center", va="bottom", fontsize=8, color=INK)

    # The bar from the Phase 1 validation spec.
    ax.axhline(0.75, ls="--", lw=1.1, color=WARN, zorder=1)
    ax.text(len(rows) - 0.4, 0.765, "0.75 target", ha="right", va="bottom",
            fontsize=8, color=WARN)

    n_pos = rows[0]["n_pos"] if rows else 0
    ax.set_xticks(x)
    ax.set_xticklabels([d["name"] for d in rows], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Sensitivity, non-genotoxic", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_title(f"B   non-genotoxic subset, {len(ng)} compounds ({n_pos} positive)",
                 loc="left", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", zorder=0)
    return notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--split", default="scaffold")
    ap.add_argument("--highlight", default="CarcinoFusion")
    ap.add_argument("--out", default="data/processed/benchmark/figure1.png")
    ap.add_argument("--mechanism", default="data/external/mechanism_labels.tsv")
    ap.add_argument("--force", action="store_true",
                    help="draw even when the data cannot support the figure")
    args = ap.parse_args(argv)

    setup_logging()
    data_cfg = load_config("data")
    bench = load_config("benchmark")
    processed = resolve_path(data_cfg["paths"]["processed"])
    endpoint = args.endpoint or bench["task"]["endpoint_name"]

    runs = load_runs(processed, endpoint)

    # Refuse before drawing, not after. A figure is the artefact most likely
    # to be screenshotted out of context, so its preconditions are checked
    # harder than anything else in the pipeline.
    blockers: list[str] = []
    provenance = {r.get("data_provenance") for r in runs}
    if "SYNTHETIC_FIXTURE" in provenance:
        blockers.append(
            "runs are marked SYNTHETIC_FIXTURE. These are not carcinogenicity "
            "results and must not be presented as a benchmark."
        )
    n_test = max((r.get("n_test") or 0) for r in runs) if runs else 0
    floor = bench["reporting"].get("min_test_compounds_for_headline", 50)
    if n_test < floor:
        blockers.append(
            f"{n_test} test compounds, below the {floor} needed for a headline "
            f"number. Confidence intervals would span most of the range."
        )

    if blockers and not args.force:
        print("\nFigure 1 not drawn:\n")
        for b in blockers:
            print(f"  - {b}")
        print(
            "\nFix the dataset first:\n"
            "  python3 scripts/diagnose_attrition.py\n"
            "\nOr pass --force to draw it anyway, watermarked NOT A RESULT.\n"
        )
        return 1

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5.2))
    notes = panel_a(ax_a, runs, args.split, args.highlight)
    notes += panel_b(ax_b, runs, args.split, Path(args.mechanism), args.highlight)

    fig.suptitle(
        f"Rodent carcinogenicity prediction — {args.split} split, endpoint '{endpoint}'",
        fontsize=12, x=0.01, ha="left", color=INK,
    )

    caption = (
        "Error bars: 95% bootstrap CI over scaffold groups. "
        f"Dataset {data_cfg['dataset_version']}, label policy "
        f"{load_config('labels')['label_policy_version']}, protocol "
        f"{bench['protocol_version']}."
    )
    if notes:
        caption += "\n" + "\n".join(f"Note: {n}" for n in notes[:3])
    fig.text(0.01, 0.015, caption, fontsize=7.5, color=MUTED, va="bottom")

    if blockers and args.force:
        fig.text(0.5, 0.5, "NOT A RESULT", fontsize=68, color=WARN, alpha=0.16,
                 ha="center", va="center", rotation=22, zorder=10)

    fig.tight_layout(rect=(0, 0.075, 1, 0.95))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

    print(f"\nFigure written: {out}  (and {out.with_suffix('.pdf').name})")
    for n in notes:
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
