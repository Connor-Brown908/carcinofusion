#!/usr/bin/env python3
"""Build a head-to-head test set on which BOTH models are out of sample.

The problem
-----------
Nearly every free carcinogenicity predictor was fitted to CPDB or to a
database derived from it. VEGA's CAESAR carcinogenicity model states its
training set plainly in its QMRF: 645 chemicals from the DSSTox `sdf_cpdbas`
file, "the L. Gold database", which is CPDB. Our test set is also drawn from
CPDB. So a comparator has very likely seen a large share of our held-out
compounds during its own training.

Comparing our out-of-sample performance against a comparator's partly
in-sample performance is not a comparison. It happens to bias AGAINST us —
the comparator is flattered — but it is still invalid, and it is the first
thing a reviewer will ask about.

The fix
-------
The comparator's training set is published, and every compound here already
carries a standardised InChIKey. So the overlap can be measured exactly
rather than assumed, and a subset defined on which neither model had an
advantage.

Three sets come out of this, and the distinction matters:

  full        every test compound. Our models' honest number; the
              comparator's is inflated by whatever it memorised.
  head_to_head  test compounds absent from the comparator's training set.
              The only subset on which a head-to-head claim is defensible.
  overlap     test compounds the comparator trained on. Reported, not
              hidden: the gap between a comparator's score here and on
              head_to_head measures how much of its apparent skill was
              recall.

Matching is on the first 14 characters of the InChIKey (the skeleton block)
as well as the full key. The skeleton ignores stereochemistry and
protonation, which a comparator's training set may record differently from
CPDB while still being the same substance for a model that cannot see
stereochemistry anyway. Both counts are reported; the skeleton match is used
for the exclusion because the conservative choice is to remove a compound
that might have been seen.

Usage
-----
    # 1. Export our test compounds for the comparator to predict
    python3 scripts/build_headtohead_set.py --export-smiles > test.smi

    # 2. Having obtained the comparator's training set, measure the overlap
    python3 scripts/build_headtohead_set.py \\
        --comparator vega_caesar \\
        --training-set external/vega_caesar_training.smi \\
        --citation "QMRF_VEGA_CAR_CAESAR.pdf, 645 chemicals from DSSTox sdf_cpdbas"

Obtaining a comparator's training set is a manual step and deliberately so:
it is the evidence the whole comparison rests on, and it should be fetched
from the model's own documentation rather than guessed at by a script.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.common import load_config, resolve_path, setup_logging

LOG = logging.getLogger("headtohead")

SKELETON_LEN = 14

# Below this, a difference cannot be resolved and the subset should not carry
# a claim. See the power note printed at the end.
MIN_USABLE = 40


def _rdkit():
    try:
        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        return Chem
    except ImportError:  # pragma: no cover - environment dependent
        raise SystemExit(
            "RDKit is required to standardise the comparator's training set.\n"
            "  pip install rdkit --break-system-packages"
        )


def inchikey_of(smiles: str) -> str | None:
    Chem = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol) or None
    except Exception:
        return None


def read_training_set(path: Path) -> tuple[set[str], set[str], dict]:
    """Read a comparator's training set as InChIKeys.

    Accepts a .smi / .txt / .csv / .tsv file. A column named inchikey is used
    directly; otherwise SMILES are standardised through RDKit. Unparseable
    rows are counted and reported, never silently dropped: a training set we
    could only half read would understate the overlap and overstate the
    head-to-head set.
    """
    if not path.exists():
        raise SystemExit(f"{path} not found.")

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise SystemExit(f"{path} is empty.")

    delim = "\t" if "\t" in lines[0] else ("," if "," in lines[0] else None)
    header = [h.strip().lower() for h in (lines[0].split(delim) if delim else [lines[0]])]
    has_header = any(h in {"smiles", "inchikey", "name", "cas", "id"} for h in header)

    keys: set[str] = set()
    stats = {"rows": 0, "unparsed": 0, "source": str(path)}

    if delim and has_header and "inchikey" in header:
        col = header.index("inchikey")
        for ln in lines[1:]:
            parts = ln.split(delim)
            stats["rows"] += 1
            if col < len(parts) and parts[col].strip():
                keys.add(parts[col].strip().upper())
            else:
                stats["unparsed"] += 1
    else:
        smi_col = header.index("smiles") if (delim and has_header and "smiles" in header) else 0
        body = lines[1:] if has_header else lines
        for ln in body:
            parts = ln.split(delim) if delim else ln.split()
            stats["rows"] += 1
            if not parts or smi_col >= len(parts):
                stats["unparsed"] += 1
                continue
            key = inchikey_of(parts[smi_col].strip())
            if key:
                keys.add(key)
            else:
                stats["unparsed"] += 1

    skeletons = {k[:SKELETON_LEN] for k in keys if k}
    stats["distinct_inchikeys"] = len(keys)
    stats["distinct_skeletons"] = len(skeletons)
    if stats["unparsed"]:
        LOG.warning(
            "%d of %d training-set row(s) could not be read as a structure. "
            "The overlap below is therefore a LOWER bound and the "
            "head-to-head set an upper bound.",
            stats["unparsed"],
            stats["rows"],
        )
    return keys, skeletons, stats


# Column aliases, resolved against the file rather than assumed. Writing a
# bare column name here is how this project has lost the most time: the names
# in the parquet are the pipeline's, not the ones a script's author remembers.
COLUMNS = {
    "inchikey": ["inchi_key", "inchikey", "InChIKey"],
    # The stereo- and protonation-insensitive key the pipeline already
    # computes. Better than slicing an InChIKey: it is derived once, by the
    # standardiser, under a recorded version.
    "connectivity": ["connectivity_key", "skeleton_key"],
    "smiles": ["canonical_smiles", "smiles", "isomeric_smiles"],
    "split": ["split_assignment", "split"],
    "strategy": ["split_strategy"],
    "label": ["label", "harmonized_label"],
}


def resolve(frame: pd.DataFrame, concept: str, *, required: bool = True) -> str | None:
    for candidate in COLUMNS[concept]:
        if candidate in frame.columns:
            return candidate
    if not required:
        return None
    raise SystemExit(
        f"no column for {concept!r} in the split table.\n"
        f"  tried: {COLUMNS[concept]}\n"
        f"  found: {sorted(frame.columns)[:25]}\n"
        f"Add the real name to COLUMNS in this script rather than renaming "
        f"the pipeline's output."
    )


def load_test_set(processed: Path, endpoint: str, strategy: str) -> tuple[pd.DataFrame, dict]:
    splits = processed / f"splits_{endpoint}.parquet"
    if not splits.exists():
        raise SystemExit(f"{splits} not found. Run the pipeline first.")

    sp = pd.read_parquet(splits)
    cols = {
        name: resolve(sp, name, required=name not in {"connectivity", "label"})
        for name in COLUMNS
    }

    mask = (sp[cols["strategy"]] == strategy) & (sp[cols["split"]] == "test")
    test = sp[mask]
    if test.empty:
        raise SystemExit(
            f"no {strategy}-split test rows in {splits.name}.\n"
            f"  strategies present: {sorted(sp[cols['strategy']].dropna().unique())}\n"
            f"  splits present:     {sorted(sp[cols['split']].dropna().unique())}"
        )
    return test.copy(), cols


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--endpoint", default="primary")
    ap.add_argument("--split", default="scaffold", choices=["scaffold", "random"])
    ap.add_argument("--comparator", default=None,
                    help="short name, e.g. vega_caesar")
    ap.add_argument("--training-set", type=Path, default=None,
                    help="the comparator's published training set (.smi/.csv/.tsv)")
    ap.add_argument("--citation", default="",
                    help="where the training set came from; recorded in the output")
    ap.add_argument("--export-smiles", action="store_true",
                    help="write the test compounds as SMILES for the comparator")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config("data")
    processed = resolve_path(cfg["paths"]["processed"])
    test, cols = load_test_set(processed, args.endpoint, args.split)

    if args.export_smiles:
        w = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
        for _, r in test.iterrows():
            smi = r.get(cols["smiles"])
            if isinstance(smi, str) and smi:
                w.writerow([smi, r["compound_id"]])
        LOG.info("exported %d test compound(s)", len(test))
        return 0

    if not args.comparator or not args.training_set:
        ap.error("--comparator and --training-set are required "
                 "(or use --export-smiles)")

    out_dir = args.out_dir or resolve_path(cfg["paths"]["processed"]) / "headtohead"
    out_dir.mkdir(parents=True, exist_ok=True)

    keys, skeletons, tstats = read_training_set(args.training_set)

    test["_key"] = test[cols["inchikey"]].astype(str).str.upper()
    test["exact_match"] = test["_key"].isin(keys)
    # Stereo-insensitive match. Prefer the pipeline's own connectivity key,
    # which the standardiser derives under a recorded version; fall back to
    # the InChIKey skeleton block only if that column is absent.
    if cols["connectivity"]:
        test["_skel"] = (
            test[cols["connectivity"]].astype(str).str.upper().str[:SKELETON_LEN]
        )
        skeleton_basis = f"{cols['connectivity']} (first {SKELETON_LEN} chars)"
    else:
        test["_skel"] = test["_key"].str[:SKELETON_LEN]
        skeleton_basis = f"InChIKey skeleton (first {SKELETON_LEN} chars)"
    test["skeleton_match"] = test["_skel"].isin(skeletons)
    # Conservative: a skeleton match may be the same substance to a model that
    # cannot see stereochemistry, so it counts as seen.
    test["seen_by_comparator"] = test["exact_match"] | test["skeleton_match"]

    n = len(test)
    n_exact = int(test["exact_match"].sum())
    n_skel = int(test["skeleton_match"].sum())
    h2h = test[~test["seen_by_comparator"]]
    overlap = test[test["seen_by_comparator"]]

    label_col = cols["label"]

    def prev(df):
        if not label_col or df.empty:
            return None
        v = pd.to_numeric(df[label_col], errors="coerce").dropna()
        return float(v.mean()) if len(v) else None

    report = {
        "endpoint": args.endpoint,
        "split_strategy": args.split,
        "comparator": args.comparator,
        "training_set": tstats,
        "citation": args.citation,
        "n_test_full": n,
        "n_overlap_exact_inchikey": n_exact,
        "n_overlap_skeleton": n_skel,
        "n_head_to_head": len(h2h),
        "overlap_fraction": (n_skel / n) if n else None,
        "prevalence_full": prev(test),
        "prevalence_head_to_head": prev(h2h),
        "prevalence_overlap": prev(overlap),
        "matching": {
            "basis": skeleton_basis,
            "rationale": "ignores stereochemistry and protonation, which a "
                         "comparator's training set may record differently "
                         "while being the same substance to the model",
        },
    }

    h2h_path = out_dir / f"headtohead_{args.endpoint}_{args.split}_{args.comparator}.csv"
    keep = [c for c in ("compound_id", cols["inchikey"], cols["smiles"], label_col,
                        "exact_match", "skeleton_match", "seen_by_comparator")
            if c and c in test.columns]
    test[keep].to_csv(h2h_path, index=False)
    report_path = out_dir / f"report_{args.endpoint}_{args.split}_{args.comparator}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print()
    print("=" * 70)
    print(f"HEAD-TO-HEAD SET  —  {args.comparator}  ({args.endpoint}/{args.split})")
    print("=" * 70)
    print(f"  comparator training set   {tstats['distinct_inchikeys']:>5,} structures"
          f"   ({args.training_set.name})")
    if tstats["unparsed"]:
        print(f"    unreadable rows         {tstats['unparsed']:>5,}"
              f"   <- overlap is a LOWER bound")
    print()
    print(f"  our test set              {n:>5,} compounds")
    print(f"    seen by comparator      {n_skel:>5,}"
          f"   ({n_skel/n:.1%})   exact InChIKey: {n_exact:,}")
    print(f"    HEAD-TO-HEAD            {len(h2h):>5,}"
          f"   <- the only defensible comparison")
    print()
    if label_col:
        def fmt(key: str) -> str:
            v = report[key]
            return "n/a" if v is None else f"{v:.3f}"

        print(f"  prevalence  full {fmt('prevalence_full')}"
              f"   head-to-head {fmt('prevalence_head_to_head')}"
              f"   overlap {fmt('prevalence_overlap')}")
        # A prevalence shift between the two subsets means the overlap is not
        # a random sample of the test set, so the head-to-head number is not
        # directly comparable to the full-set number either.
        pf, ph = report["prevalence_full"], report["prevalence_head_to_head"]
        if pf is not None and ph is not None and abs(pf - ph) > 0.10:
            print(f"    !! prevalence shifts by {abs(pf - ph):.3f} between the")
            print(f"       full and head-to-head sets, so the comparator's")
            print(f"       training set is not a random slice of our test set.")
            print(f"       Compare models WITHIN a subset, never across them.")
    print()
    print(f"  written: {h2h_path}")
    print(f"           {report_path}")
    print()

    if len(h2h) < MIN_USABLE:
        print(f"  !! {len(h2h)} compounds is too few to support a head-to-head")
        print(f"     claim. Below about {MIN_USABLE}, a paired bootstrap cannot")
        print(f"     resolve the differences at stake here. Report the")
        print(f"     comparator on the full set WITH its overlap stated, and")
        print(f"     treat the head-to-head as indicative only.")
    else:
        print(f"  Power: a paired bootstrap over ~{len(h2h)} compounds resolves")
        print(f"  a balanced-accuracy gap of roughly "
              f"{0.35/max(len(h2h),1)**0.5:.03f} at 95%. A smaller true")
        print(f"  difference will not be distinguishable from zero, which is a")
        print(f"  finding, not a failure.")
    print()
    print("  Next: predict these compounds with the comparator, then")
    print(f"    python3 scripts/import_external_predictions.py \\")
    print(f"        --tool {args.comparator} --predictions <output> \\")
    print(f"        --restrict-to {h2h_path}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
