#!/usr/bin/env python3
"""Import an external tool's predictions as a benchmark run.

This is what turns "we beat the published numbers" into a real comparison.
Published sensitivity and specificity came from other compounds under other
splits; quoting them beside your own is not a comparison and a reviewer will
say so. Running the incumbent on YOUR frozen test set is.

Two steps:

    # 1. export the test compounds in the tool's input format
    python3 scripts/import_external_predictions.py --export vega   > test.smi
    python3 scripts/import_external_predictions.py --export csv    > test.csv

    # 2. run the tool yourself, then bring its output back
    python3 scripts/import_external_predictions.py \\
        --tool vega_caesar --predictions vega_out.csv \\
        --smiles-col SMILES --prediction-col "Carcinogenicity"

The import writes a run into the same benchmark store the ladder uses, so the
analysis module pairs it against your models on identical compounds without
knowing it came from elsewhere.

Free tools worth running:
  VEGA (CAESAR, ISS, IRFMN)  https://www.vegahub.eu/        has a batch CLI
  EPA TEST                   https://www.epa.gov/chemical-research/toxicity-estimation-software-tool-test
  OECD QSAR Toolbox          https://qsartoolbox.org/

Matching is by InChIKey where the tool echoes structures, falling back to
input order only when you pass --trust-order, because a tool that silently
drops unparseable rows will otherwise shift every prediction by one.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import inchi as rd_inchi  # noqa: E402

from src.common import StageManifest, load_config, resolve_path, setup_logging, utc_now  # noqa: E402

LOG = logging.getLogger("external")
STAGE = "15_external_predictions"
RDLogger.DisableLog("rdApp.*")

# Verdicts are decoded in three passes, and the order matters.
#
# Short words must only ever match a WHOLE value. "no" as a substring matches
# inside "carcinogen" (car-ci-NO-gen), which silently inverted every positive
# VEGA call until it was caught. Anything this short is exact-match only.
EXACT_POSITIVE = {"1", "y", "yes", "true", "t", "+", "p", "pos"}
EXACT_NEGATIVE = {"0", "n", "no", "false", "f", "-", "neg"}

# Negative phrases are checked before positive ones, because almost every one
# of them contains a positive word: "non-carcinogen" contains "carcinogen",
# "inactive" contains "active".
SUBSTRING_NEGATIVE = [
    "non-carcinogen", "noncarcinogen", "non carcinogen",
    "non-mutagen", "nonmutagen", "non mutagen",
    "non-toxic", "nontoxic", "non toxic",
    "not carcinogenic", "not mutagenic",
    "inactive", "negative", "no alert", "no evidence",
]

# How each tool words a positive call, as substrings, since these carry
# version-dependent qualifiers: "Carcinogen (good reliability)".
POSITIVE_TOKENS = {
    "vega_caesar": ["carcinogen", "mutagen", "active", "positive", "toxic"],
    "vega_iss": ["carcinogen", "active", "positive"],
    "vega_irfmn": ["carcinogen", "active", "positive"],
    "epa_test": ["positive", "active", "carcinogen", "mutagenic"],
    "qsar_toolbox": ["positive", "active", "carcinogen"],
    "generic": ["positive", "active", "carcinogen", "mutagen", "toxic"],
}


def inchikey(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    try:
        s = rd_inchi.MolToInchi(mol)
        return rd_inchi.InchiToInchiKey(s) if s else None
    except Exception:  # noqa: BLE001
        return None


def to_probability(value, tool: str) -> float | None:
    """Map a tool's verdict to P(carcinogenic).

    Many of these emit a class, not a probability. A hard 0/1 is honest about
    that: inventing 0.9 would flatter the tool's calibration and corrupt the
    Brier comparison. Where a genuine numeric score is given, it is used.
    """
    if value is None or (isinstance(value, float) and value != value):
        return None

    # A real numeric score in [0, 1] is used as-is.
    try:
        num = float(str(value).strip())
        if 0.0 <= num <= 1.0:
            return num
        if num in (0.0, 1.0):
            return num
    except (TypeError, ValueError):
        pass

    text = str(value).strip().lower()
    if not text:
        return None

    # 1. whole-value matches, for tokens too short to be safe as substrings
    if text in EXACT_POSITIVE:
        return 1.0
    if text in EXACT_NEGATIVE:
        return 0.0

    # 2. negative phrases, before positive ones: "non-carcinogen" contains
    #    "carcinogen" and "inactive" contains "active"
    for token in SUBSTRING_NEGATIVE:
        if token in text:
            return 0.0

    # 3. positive phrases
    for token in POSITIVE_TOKENS.get(tool, POSITIVE_TOKENS["generic"]):
        if token in text:
            return 1.0

    # Unrecognised. Returning None keeps it out of the comparison and gets it
    # counted and reported, rather than guessed at.
    return None


def load_test_set(processed: Path, endpoint: str, strategy: str) -> pd.DataFrame:
    path = processed / f"splits_{endpoint}.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run the pipeline and the split first.")
    df = pd.read_parquet(path)
    df = df[(df.split_strategy == strategy) & (df.split_assignment == "test")]
    if df.empty:
        raise SystemExit(f"no test compounds for {endpoint}/{strategy}")
    return df.copy()


def export(test: pd.DataFrame, fmt: str) -> None:
    """Write the test compounds in a tool-ingestible format, to stdout."""
    if fmt in ("vega", "smi"):
        # SMILES then identifier: VEGA and most batch tools echo the second
        # column, which is what lets predictions be matched back by id.
        for _, r in test.iterrows():
            print(f"{r['canonical_smiles']}\t{r['compound_id']}")
    elif fmt == "csv":
        cols = ["compound_id", "canonical_smiles", "inchi_key", "label"]
        print(",".join(cols))
        for _, r in test.iterrows():
            print(",".join(str(r.get(c, "")) for c in cols))
    elif fmt == "sdf":
        for _, r in test.iterrows():
            mol = Chem.MolFromSmiles(r["canonical_smiles"])
            if mol is None:
                continue
            mol.SetProp("_Name", str(r["compound_id"]))
            mol.SetProp("compound_id", str(r["compound_id"]))
            print(Chem.MolToMolBlock(mol), end="")
            print("$$$$")
    else:
        raise SystemExit(f"unknown export format: {fmt}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", choices=["vega", "smi", "csv", "sdf"],
                    help="print the test compounds and exit")
    ap.add_argument("--tool", default=None, help="e.g. vega_caesar, epa_test")
    ap.add_argument("--tool-name", default=None, help="label for the figure")
    ap.add_argument("--predictions", default=None, help="the tool's output file")
    ap.add_argument("--smiles-col", default=None)
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--prediction-col", default=None)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--split", default="scaffold")
    ap.add_argument("--trust-order", action="store_true",
                    help="match by row order (only if the tool kept every row)")
    ap.add_argument("--version", default=None, help="the tool's version, recorded")
    args = ap.parse_args(argv)

    setup_logging()
    data_cfg = load_config("data")
    lab_cfg = load_config("labels")
    split_cfg = load_config("splits")
    bench = load_config("benchmark")
    processed = resolve_path(data_cfg["paths"]["processed"])
    endpoint = args.endpoint or bench["task"]["endpoint_name"]

    test = load_test_set(processed, endpoint, args.split)

    if args.export:
        export(test, args.export)
        return 0

    if not args.tool or not args.predictions:
        ap.error("--tool and --predictions are required unless --export is used")

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        LOG.error("predictions file not found: %s", pred_path)
        return 2

    sep = "\t" if pred_path.suffix.lower() in (".tsv", ".tab", ".txt") else ","
    preds = pd.read_csv(pred_path, sep=sep, dtype=str, engine="python")
    LOG.info("%s: %d row(s), columns %s", pred_path.name, len(preds), list(preds.columns))

    lower = {c.lower().strip(): c for c in preds.columns}

    def pick(explicit, *candidates):
        if explicit:
            if explicit not in preds.columns:
                raise SystemExit(f"column {explicit!r} not in {list(preds.columns)}")
            return explicit
        for c in candidates:
            if c in lower:
                return lower[c]
        return None

    pred_col = pick(args.prediction_col, "prediction", "carcinogenicity", "result",
                    "class", "activity", "pred", "value", "assessment")
    if pred_col is None:
        LOG.error(
            "could not find the prediction column; pass --prediction-col. "
            "Columns present: %s", list(preds.columns)
        )
        return 2
    id_col = pick(args.id_col, "compound_id", "id", "name", "molecule", "no.")
    smi_col = pick(args.smiles_col, "smiles", "canonical_smiles", "structure")

    # Match by compound id, then InChIKey, then row order on request. Order is
    # last because tools routinely drop rows they cannot parse, and a silent
    # off-by-one assigns every prediction to the wrong compound.
    matched: dict[str, float] = {}
    method = None

    if id_col is not None:
        by_id = {str(r[id_col]).strip(): r[pred_col] for _, r in preds.iterrows()}
        hits = {c: by_id[c] for c in test["compound_id"] if c in by_id}
        if len(hits) >= 0.5 * len(test):
            method = f"compound_id ({id_col})"
            matched = {k: v for k, v in hits.items()}

    if not matched and smi_col is not None:
        by_key: dict[str, object] = {}
        for _, r in preds.iterrows():
            k = inchikey(str(r[smi_col]))
            if k:
                by_key.setdefault(k, r[pred_col])
        hits = {
            r["compound_id"]: by_key[r["inchi_key"]]
            for _, r in test.iterrows()
            if r["inchi_key"] in by_key
        }
        if hits:
            method = f"InChIKey ({smi_col})"
            matched = hits

    if not matched and args.trust_order:
        if len(preds) != len(test):
            LOG.error(
                "--trust-order needs equal lengths: %d predictions vs %d test "
                "compounds. The tool dropped rows, so order cannot be trusted.",
                len(preds), len(test),
            )
            return 2
        method = "row order (requested)"
        matched = {
            cid: preds.iloc[i][pred_col]
            for i, cid in enumerate(test["compound_id"])
        }

    if not matched:
        LOG.error(
            "could not match any prediction to a test compound. Pass --id-col "
            "or --smiles-col, or --trust-order if the tool preserved every row."
        )
        return 2

    LOG.info("matched %d/%d test compounds by %s", len(matched), len(test), method)

    tool = args.tool
    rows, undecodable = [], 0
    for _, r in test.iterrows():
        raw = matched.get(r["compound_id"])
        if raw is None:
            continue
        prob = to_probability(raw, tool)
        if prob is None:
            undecodable += 1
            continue
        rows.append(
            {
                "compound_id": r["compound_id"],
                "partition": "test",
                "y_true": int(r["label"]),
                "y_prob": float(prob),
                "nn_similarity": None,
                "raw_prediction": str(raw),
            }
        )

    if undecodable:
        examples = sorted({str(matched[c]) for c in list(matched)[:40]})[:5]
        LOG.warning(
            "%d prediction(s) could not be read as positive or negative. "
            "Example values: %s. Extend POSITIVE_TOKENS/NEGATIVE_TOKENS or map "
            "them before import.", undecodable, examples
        )
    if not rows:
        LOG.error("no predictions were decodable; nothing imported")
        return 2

    run_id = f"EXT_{tool}_{args.split}_{endpoint}_{uuid.uuid4().hex[:8]}"
    covered = len(rows)
    record = {
        "run_id": run_id,
        "model_id": "EXT",
        "model_name": args.tool_name or tool,
        "representation": "external_tool",
        "endpoint_name": endpoint,
        "split_strategy": args.split,
        "dataset_version": data_cfg["dataset_version"],
        "label_policy_version": lab_cfg["label_policy_version"],
        "split_version": split_cfg["split_version"],
        "protocol_version": bench["protocol_version"],
        "seed": 0,
        "data_provenance": (
            (processed / "DATA_PROVENANCE").read_text().strip()
            if (processed / "DATA_PROVENANCE").exists()
            else "SYNTHETIC_FIXTURE"
        ),
        "dependency_versions": {"external_tool": tool, "version": args.version},
        "config": {
            "source_file": str(pred_path),
            "match_method": method,
            "prediction_column": pred_col,
            "tool_version": args.version,
        },
        "hyperparameters": {},
        "n_train": None,
        "n_validation": None,
        "n_test": int(len(test)),
        "test_prevalence": float(test["label"].mean()),
        "status": "completed",
        "unavailable_reason": None,
        "started_at": utc_now(),
        "finished_at": utc_now(),
        "predictions": rows,
        # An external tool predicts a class, so there is nothing to tune and
        # 0.5 is the only defensible threshold.
        "threshold": 0.5,
        "threshold_selected_on": "not_applicable_external_tool",
        "coverage": round(covered / len(test), 4),
        "results": {},
    }

    man = StageManifest(stage=STAGE)
    man.add_input(pred_path)
    man.config_versions = {"tool": tool, "endpoint": endpoint, "split": args.split}

    runs_path = processed / "benchmark" / f"runs_{endpoint}.json"
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(runs_path.read_text()) if runs_path.exists() else []
    existing = [r for r in existing if r.get("model_name") != record["model_name"]
                or r.get("split_strategy") != args.split]
    existing.append(record)
    runs_path.write_text(json.dumps(existing, indent=2, default=str) + "\n")
    man.add_output(runs_path)

    man.counts = {
        "matched": covered,
        "test_compounds": int(len(test)),
        "coverage": record["coverage"],
        "undecodable": undecodable,
        "match_method": method,
    }
    if covered < len(test):
        man.warn(
            f"{tool} covers {covered}/{len(test)} test compounds. Coverage is "
            f"part of the comparison: a tool that declines the hard cases is "
            f"not equivalent to one that predicts them all."
        )
    man.write()

    print(f"\nImported {covered}/{len(test)} predictions for {record['model_name']}")
    print(f"  matched by: {method}")
    print(f"  coverage:   {record['coverage']:.1%}")
    print(f"  run_id:     {run_id}")
    print("\nRe-run the analysis to include it in the paired comparison:")
    print(f"  python3 -m src.evaluation.analysis --endpoint {endpoint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
