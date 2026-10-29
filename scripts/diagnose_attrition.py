#!/usr/bin/env python3
"""Trace where compounds are lost between the raw files and the test set.

CPDB holds roughly 1,500 chemicals. If the benchmark ends up with a handful
of test compounds, something discarded most of them, and the number alone
does not say what. This walks the funnel stage by stage and reports the drop
at each one, largest first, so the cause is visible rather than inferred.

Reads only artefacts the pipeline has already written. Changes nothing.

    python3 scripts/diagnose_attrition.py
    python3 scripts/diagnose_attrition.py --endpoint sensitivity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.common import load_config, resolve_path  # noqa: E402


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:5.1f}%" if d else "    -"


def step(label: str, n: int, prev: int | None, first: int) -> None:
    drop = "" if prev is None else f"  (-{prev - n:,})" if prev > n else ""
    print(f"  {label:<44} {n:>7,}  {pct(n, first)}{drop}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default=None)
    args = ap.parse_args(argv)

    data_cfg = load_config("data")
    lab_cfg = load_config("labels")
    bench = load_config("benchmark")
    endpoint = args.endpoint or bench["task"]["endpoint_name"]

    I = resolve_path(data_cfg["paths"]["interim"])
    P = resolve_path(data_cfg["paths"]["processed"])
    E = resolve_path(data_cfg["paths"]["external"])

    def load(p: Path) -> pd.DataFrame | None:
        return pd.read_parquet(p) if p.exists() else None

    exp = load(I / "carcinogenicity_experiments.parquet")
    comp = load(I / "compound_identity.parquet")
    review = load(I / "structure_review.parquet")
    ds = load(P / f"model_dataset_{endpoint}.parquet")
    splits = load(P / f"splits_{endpoint}.parquet")
    ledger = load(P / "curation_ledger.parquet")

    if exp is None:
        print("No parsed experiments found. Run the pipeline first.", file=sys.stderr)
        return 2

    print("=" * 72)
    print(f"ATTRITION FUNNEL   endpoint={endpoint}   policy={lab_cfg['label_policy_version']}")
    print("=" * 72)

    n_records = len(exp)
    n_chem = exp["chemcode"].nunique(dropna=True)
    print(f"\n  {'experimental records parsed':<44} {n_records:>7,}")
    print(f"  {'distinct source chemicals (chemcode)':<44} {n_chem:>7,}")
    print()

    first = n_chem
    prev = n_chem
    step("source chemicals", n_chem, None, first)

    # Structure resolution
    res_path = E / "structure_resolution.tsv"
    if res_path.exists():
        res = pd.read_csv(res_path, sep="\t", dtype=str)
        with_smiles = int(res["smiles"].notna().sum()) if "smiles" in res else 0
        step("  with a SMILES in the resolution table", with_smiles, prev, first)
        prev = with_smiles

    if comp is not None:
        step("  standardised into a compound", len(comp), prev, first)
        prev = len(comp)
        if review is not None and len(review):
            print(f"\n    structures sent to review: {len(review):,}")
            for reason, n in review["structure_status"].value_counts().items():
                print(f"      {reason:<40} {n:>6,}")

    if ds is None:
        print("\nNo model dataset. Stop here and check the label stage.")
        return 1

    labelled = int(ds["label"].notna().sum())
    eligible = int((ds["training_eligibility"] == "eligible").sum())
    print()
    step("  received a usable label", labelled, prev, first)
    step("  eligible for training", eligible, labelled, first)

    conflicted = int((ds["training_eligibility"] == "excluded_conflicting_evidence").sum())
    unresolved = int((ds["training_eligibility"] == "excluded_unresolved_label").sum())
    if conflicted or unresolved:
        print(f"\n    excluded, conflicting evidence: {conflicted:,}")
        print(f"    excluded, unresolved label:     {unresolved:,}")

    if eligible:
        pos = int((ds[ds.training_eligibility == "eligible"]["label"] == 1).sum())
        neg = eligible - pos
        print(f"\n    positive {pos:,}   negative {neg:,}   prevalence {pos / eligible:.3f}")

    if splits is not None:
        print("\n  partitions")
        for strat, grp in splits.groupby("split_strategy"):
            counts = (
                grp[grp.split_assignment.isin(["train", "validation", "test"])]
                .split_assignment.value_counts()
                .to_dict()
            )
            line = "  ".join(f"{k} {v:,}" for k, v in sorted(counts.items()))
            print(f"    {strat:<12} {line}")

    # ---- where the losses actually came from -------------------------------
    print("\n" + "=" * 72)
    print("LARGEST LOSSES")
    print("=" * 72)

    losses: list[tuple[int, str]] = []
    if comp is not None:
        lost_structure = n_chem - len(comp)
        if lost_structure > 0:
            losses.append((lost_structure, "no usable structure (see structure_review)"))
    if ds is not None:
        if unresolved:
            losses.append((unresolved, "label unresolved under the policy"))
        if conflicted:
            losses.append((conflicted, "conflicting experimental evidence"))

    if ledger is not None and len(ledger):
        excl = ledger[ledger.event_type == "exclusion"]
        by_reason = excl["reason_code"].value_counts()
        print("\n  curation ledger, exclusions by reason:")
        for reason, n in by_reason.items():
            print(f"    {reason:<44} {n:>7,}")

    if losses:
        print()
        for n, why in sorted(losses, reverse=True):
            print(f"  {n:>7,}  {why}")

    # ---- the specific checks worth making ----------------------------------
    print("\n" + "=" * 72)
    print("CHECKS")
    print("=" * 72)

    problems = []

    # Species eligibility is the quietest filter: it removes records before
    # any label is formed, so a mismatch shows up as "no eligible record"
    # rather than as a filtering step.
    eligible_species = {s.lower() for s in lab_cfg["aggregation"]["eligible_species"]}
    seen = exp["species"].dropna().str.lower().value_counts()
    print("\n  species values seen in the data:")
    for sp, n in seen.head(12).items():
        mark = "eligible" if sp in eligible_species else "EXCLUDED"
        print(f"    {sp:<28} {n:>7,}  {mark}")
    unmatched = sum(n for sp, n in seen.items() if sp not in eligible_species)
    if unmatched > 0.5 * len(exp):
        problems.append(
            f"{unmatched:,} of {len(exp):,} records are in species the policy "
            f"does not accept ({sorted(eligible_species)}). If those values are "
            f"CPDB codes rather than names, the vocabulary in "
            f"configs/cpdb_columns.yaml is not decoding them."
        )

    # Assessments that matched no policy rule become null, which looks like
    # missing data rather than a mapping gap.
    assess = exp["author_assessment"].dropna().value_counts()
    print("\n  assessment values seen (top 12):")
    for a, n in assess.head(12).items():
        print(f"    {str(a)[:44]:<44} {n:>7,}")

    mapped = set()
    for ep in lab_cfg["endpoints"].values():
        for bucket in ("positive", "negative", "unresolved"):
            mapped |= {str(t).strip().lower() for t in ep.get(bucket, [])}
    unmapped_n = sum(
        n for a, n in assess.items() if str(a).strip().lower() not in mapped
    )
    if unmapped_n > 0.2 * len(exp):
        problems.append(
            f"{unmapped_n:,} of {len(exp):,} records carry an assessment that "
            f"matches no rule in configs/labels.yaml, so they became null. "
            f"Add the real values to the policy's positive/negative/unresolved "
            f"lists."
        )

    if ds is not None and eligible < 100:
        problems.append(
            f"only {eligible:,} compounds are eligible for training, which is "
            f"too few for a meaningful benchmark. Fix the causes above before "
            f"reading any metric."
        )

    print()
    if problems:
        for i, p in enumerate(problems, 1):
            print(f"  [{i}] {p}\n")
        return 1

    print("  Nothing obviously wrong in the funnel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
