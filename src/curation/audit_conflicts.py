"""Step 5 - audit the curated dataset.

Produces the Step 5 report from the protocol: counts, resolution outcomes,
label distributions and, most importantly, the detail of every compound whose
experimental evidence disagrees with itself.

Conflicting evidence is reported along the axis it disagrees on — species, sex,
exposure route or target organ — because a compound positive only in one sex of
one species is a different scientific situation from one positive across the
board, and a majority vote would erase that distinction.

Outputs:
    data/processed/audit_report.json          machine-readable summary
    data/processed/audit_report.md            the same, readable
    data/processed/conflicts_detail.parquet   per-compound conflict breakdown

Usage:
    python -m src.curation.audit_conflicts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.common import StageManifest, load_config, resolve_path, setup_logging, utc_now

LOG = logging.getLogger("audit_conflicts")
STAGE = "05_audit"

CONFLICT_AXES = ["species", "sex", "exposure_route", "target_organ"]


def conflict_detail(observations: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Per-compound breakdown of disagreeing evidence."""
    rows: list[dict] = []
    for cid, grp in observations.groupby("compound_id", dropna=True):
        pos = grp[grp[label_col] == 1]
        neg = grp[grp[label_col] == 0]
        if pos.empty or neg.empty:
            continue

        row: dict = {
            "compound_id": cid,
            "n_positive": int(len(pos)),
            "n_negative": int(len(neg)),
            "n_indeterminate": int(grp[label_col].isna().sum()),
        }
        for axis in CONFLICT_AXES:
            if axis not in grp.columns:
                continue
            p = set(pos[axis].dropna().astype(str))
            n = set(neg[axis].dropna().astype(str))
            row[f"positive_{axis}"] = "; ".join(sorted(p)) or None
            row[f"negative_{axis}"] = "; ".join(sorted(n)) or None
            # Disagreement WITHIN one value of the axis is the strongest kind:
            # the same species (or sex, route, organ) gave both answers.
            row[f"{axis}_overlap"] = "; ".join(sorted(p & n)) or None

        # Classify the conflict so it can be triaged rather than just counted.
        overlaps = [
            axis for axis in CONFLICT_AXES if row.get(f"{axis}_overlap")
        ]
        if "species" in overlaps and "sex" in overlaps:
            row["conflict_type"] = "within_species_and_sex"
        elif "species" in overlaps:
            row["conflict_type"] = "within_species_across_other_factors"
        else:
            row["conflict_type"] = "across_species"
        row["conflicting_axes"] = "; ".join(overlaps) or "none_shared"
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim", default=None)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args(argv)

    setup_logging()
    data_cfg = load_config("data")
    lab_cfg = load_config("labels")
    interim = resolve_path(args.interim or data_cfg["paths"]["interim"])
    processed = resolve_path(args.outdir or data_cfg["paths"]["processed"])
    processed.mkdir(parents=True, exist_ok=True)

    man = StageManifest(stage=STAGE)
    man.config_versions = {
        "dataset_version": data_cfg.get("dataset_version"),
        "label_policy_version": lab_cfg["label_policy_version"],
    }

    paths = {
        "experiments": interim / "carcinogenicity_experiments.parquet",
        "compounds": interim / "compound_identity.parquet",
        "review": interim / "structure_review.parquet",
        "observations": interim / "observations_labeled.parquet",
        "labels": interim / "compound_labels.parquet",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        LOG.error("missing inputs: %s — run the earlier stages first", missing)
        return 2
    for p in paths.values():
        man.add_input(p)

    experiments = pd.read_parquet(paths["experiments"])
    compounds = pd.read_parquet(paths["compounds"])
    review = pd.read_parquet(paths["review"])
    observations = pd.read_parquet(paths["observations"])
    labels = pd.read_parquet(paths["labels"])

    endpoint_names = list(lab_cfg["endpoints"].keys())
    linked = observations[observations["compound_id"].notna()]

    report: dict = {
        "generated_at": utc_now(),
        "dataset_version": data_cfg.get("dataset_version"),
        "label_policy_version": lab_cfg["label_policy_version"],
        "endpoint_definition": lab_cfg["endpoint_definition"],
        "source_records": {
            "total_experiment_records": int(len(experiments)),
            "unique_source_chemicals": int(experiments["chemcode"].nunique(dropna=True)),
            "records_by_source": experiments["source_database"]
            .value_counts()
            .to_dict(),
            "records_by_species": experiments["species"]
            .fillna("<null>")
            .value_counts()
            .to_dict(),
            "records_missing_assessment": int(
                experiments["author_assessment"].isna().sum()
            ),
            "records_with_td50": int(experiments["td50"].notna().sum()),
        },
        "structure_resolution": {
            "compounds_standardized": int(len(compounds)),
            "structures_failed_or_unresolved": int(len(review)),
            "failure_reasons": (
                review["structure_status"].value_counts().to_dict()
                if not review.empty
                else {}
            ),
            "compounds_merged_from_multiple_source_chemicals": (
                int((compounds["n_source_chemicals"] > 1).sum())
                if not compounds.empty
                else 0
            ),
            "compounds_with_unassigned_stereocenters": (
                int((compounds["n_unassigned_stereocenters"] > 0).sum())
                if not compounds.empty
                else 0
            ),
            "unique_scaffold_groups": (
                int(compounds["scaffold_group"].nunique()) if not compounds.empty else 0
            ),
            "duplicate_inchikeys": (
                int(compounds["inchi_key"].duplicated().sum())
                if not compounds.empty
                else 0
            ),
        },
        "linkage": {
            "records_linked_to_compound": int(len(linked)),
            "records_unlinked": int(len(observations) - len(linked)),
        },
        "endpoints": {},
    }

    conflict_frames: list[pd.DataFrame] = []
    for name in endpoint_names:
        rec_col = f"label_{name}"
        comp_col = f"label_{name}"
        dist = labels[comp_col].value_counts(dropna=False)
        detail = conflict_detail(linked, rec_col)
        if not detail.empty:
            detail = detail.assign(endpoint=name)
            conflict_frames.append(detail)

        by_source = (
            linked.groupby("source_database")[rec_col]
            .value_counts(dropna=False)
            .unstack(fill_value=0)
        )
        by_species = (
            linked.groupby(linked["species"].fillna("<null>"))[rec_col]
            .value_counts(dropna=False)
            .unstack(fill_value=0)
        )

        report["endpoints"][name] = {
            "description": lab_cfg["endpoints"][name]["description"].strip(),
            "record_labels": {
                str(k): int(v)
                for k, v in linked[rec_col].value_counts(dropna=False).items()
            },
            "compound_labels": {
                "positive": int(dist.get(1, 0)),
                "negative": int(dist.get(0, 0)),
                "unresolved": int(labels[comp_col].isna().sum()),
            },
            "conflicted_compounds": int(labels[f"is_conflicted_{name}"].sum()),
            "conflict_types": (
                detail["conflict_type"].value_counts().to_dict()
                if not detail.empty
                else {}
            ),
            "label_confidence": {
                str(k): int(v)
                for k, v in labels[f"label_confidence_{name}"]
                .value_counts(dropna=False)
                .items()
            },
            "record_labels_by_source": {
                str(idx): {str(c): int(v) for c, v in row.items()}
                for idx, row in by_source.iterrows()
            },
            "record_labels_by_species": {
                str(idx): {str(c): int(v) for c, v in row.items()}
                for idx, row in by_species.iterrows()
            },
        }

    conflicts = (
        pd.concat(conflict_frames, ignore_index=True)
        if conflict_frames
        else pd.DataFrame(columns=["compound_id", "endpoint", "conflict_type"])
    )
    conflicts_path = processed / "conflicts_detail.parquet"
    conflicts.to_parquet(conflicts_path, index=False)

    json_path = processed / "audit_report.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    md_path = processed / "audit_report.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")

    for p, n in ((conflicts_path, len(conflicts)), (json_path, None), (md_path, None)):
        man.add_output(p, rows=n)
    man.counts = {
        "compounds": int(len(compounds)),
        "conflicted_rows": int(len(conflicts)),
        "structures_in_review": int(len(review)),
    }
    man.write()

    LOG.info("audit written: %s and %s", json_path, md_path)
    return 0


def render_markdown(report: dict) -> str:
    """Render the audit as a readable report."""
    L: list[str] = []
    L.append("# Dataset audit report")
    L.append("")
    L.append(f"Generated: {report['generated_at']}  ")
    L.append(f"Dataset version: `{report['dataset_version']}`  ")
    L.append(f"Label policy version: `{report['label_policy_version']}`")
    L.append("")
    L.append(f"**Endpoint.** {report['endpoint_definition'].strip()}")
    L.append("")

    src = report["source_records"]
    L.append("## Source records")
    L.append("")
    L.append("| Quantity | Count |")
    L.append("| --- | --- |")
    L.append(f"| Total experiment records | {src['total_experiment_records']} |")
    L.append(f"| Unique source chemicals | {src['unique_source_chemicals']} |")
    L.append(f"| Records missing an assessment | {src['records_missing_assessment']} |")
    L.append(f"| Records carrying a TD50 | {src['records_with_td50']} |")
    L.append("")

    res = report["structure_resolution"]
    L.append("## Structure resolution")
    L.append("")
    L.append("| Quantity | Count |")
    L.append("| --- | --- |")
    L.append(f"| Compounds standardised | {res['compounds_standardized']} |")
    L.append(f"| Unresolved or failed structures | {res['structures_failed_or_unresolved']} |")
    L.append(f"| Compounds merged from several source chemicals | {res['compounds_merged_from_multiple_source_chemicals']} |")
    L.append(f"| Compounds with unassigned stereocentres | {res['compounds_with_unassigned_stereocenters']} |")
    L.append(f"| Duplicate InChIKeys (should be 0) | {res['duplicate_inchikeys']} |")
    L.append(f"| Unique scaffold groups | {res['unique_scaffold_groups']} |")
    L.append("")
    if res["failure_reasons"]:
        L.append("Failure reasons:")
        L.append("")
        for reason, n in sorted(res["failure_reasons"].items(), key=lambda kv: -kv[1]):
            L.append(f"- `{reason}`: {n}")
        L.append("")

    for name, ep in report["endpoints"].items():
        L.append(f"## Endpoint: {name}")
        L.append("")
        L.append(ep["description"])
        L.append("")
        cl = ep["compound_labels"]
        L.append("| Compound label | Count |")
        L.append("| --- | --- |")
        L.append(f"| Positive | {cl['positive']} |")
        L.append(f"| Negative | {cl['negative']} |")
        L.append(f"| Unresolved (null) | {cl['unresolved']} |")
        L.append(f"| Conflicted | {ep['conflicted_compounds']} |")
        L.append("")
        if ep["conflict_types"]:
            L.append("Conflict types:")
            L.append("")
            for t, n in sorted(ep["conflict_types"].items(), key=lambda kv: -kv[1]):
                L.append(f"- `{t}`: {n}")
            L.append("")
    L.append("---")
    L.append("")
    L.append(
        "Unresolved is not negative. Compounds with conflicting evidence are "
        "excluded from the primary binary training set and retained in full."
    )
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
