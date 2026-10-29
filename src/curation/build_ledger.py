"""Stage 8 - build the curation ledger.

A single machine-readable record of every exclusion and every transformation
the pipeline applied, one row per event, with the entity it applied to and a
structured reason code.

The information exists across several artefacts — the structure review table,
the training-eligibility column, the standardisation notes, the unmapped
assessment list. Scattered, it cannot be queried: answering "why is this
compound absent from the training set" means opening four files and knowing
which to look in. The ledger consolidates it so that question is one filter.

Runs after the split, because exclusions from the training set are only
decided at build and split time.

Outputs:
    data/processed/curation_ledger.parquet
    data/processed/curation_ledger_summary.json

Usage:
    python -m src.curation.build_ledger
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from src.common import StageManifest, load_config, resolve_path, setup_logging, utc_now

LOG = logging.getLogger("build_ledger")
STAGE = "08_ledger"

LEDGER_COLUMNS = [
    "event_id",
    "event_type",
    "stage",
    "entity_type",
    "entity_id",
    "reason_code",
    "detail",
    "endpoint_name",
    "dataset_version",
    "label_policy_version",
    "standardization_version",
]

# Reason codes are a closed vocabulary so the ledger can be aggregated.
EXCLUSION_REASONS = {
    "structure_parse_failed",
    "structure_no_organic_fragment",
    "structure_standardize_failed",
    "structure_unresolved_no_structure",
    "structure_unresolved_empty_smiles",
    "record_unlinked_no_compound",
    "record_label_unresolved",
    "compound_conflicting_evidence",
    "compound_unresolved_label",
    "compound_in_external_set",
}
TRANSFORMATION_REASONS = {
    "fragment_discarded",
    "stereocentre_unassigned",
    "structure_standardized",
    "compounds_merged",
    "value_normalized_species",
    "value_normalized_sex",
    "cas_filled_from_identity_table",
    "td50_non_numeric_nulled",
    "assessment_unmapped_to_null",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim", default=None)
    parser.add_argument("--processed", default=None)
    args = parser.parse_args(argv)

    setup_logging()
    data_cfg = load_config("data")
    lab_cfg = load_config("labels")
    interim = resolve_path(args.interim or data_cfg["paths"]["interim"])
    processed = resolve_path(args.processed or data_cfg["paths"]["processed"])

    man = StageManifest(stage=STAGE)
    man.config_versions = {
        "dataset_version": data_cfg["dataset_version"],
        "label_policy_version": lab_cfg["label_policy_version"],
    }

    paths = {
        "review": interim / "structure_review.parquet",
        "compounds": interim / "compound_identity.parquet",
        "observations": interim / "observations_labeled.parquet",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        LOG.error("missing inputs: %s — run the earlier stages first", missing)
        return 2
    for p in paths.values():
        man.add_input(p)

    review = pd.read_parquet(paths["review"])
    compounds = pd.read_parquet(paths["compounds"])
    observations = pd.read_parquet(paths["observations"])

    stamp = utc_now()
    # The ledger carries no wall-clock timestamp per row. When the ledger was
    # built is recorded once, in the stage manifest and the summary; stamping
    # it on every row would add nothing and would make the parquet differ
    # between two runs of identical inputs, breaking reproducibility.
    base = {
        "dataset_version": data_cfg["dataset_version"],
        "label_policy_version": lab_cfg["label_policy_version"],
        "standardization_version": (
            compounds["standardization_version"].iloc[0] if not compounds.empty else None
        ),
    }
    rows: list[dict] = []

    def add(event_type, stage, entity_type, entity_id, reason, detail, endpoint=None):
        rows.append(
            {
                "event_type": event_type,
                "stage": stage,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "reason_code": reason,
                "detail": detail,
                "endpoint_name": endpoint,
                **base,
            }
        )

    # ---------------- exclusions: structures ----------------
    for rec in review.to_dict("records"):
        status = rec.get("structure_status") or "unknown"
        add(
            "exclusion",
            "03_resolve_identity",
            "source_chemical",
            rec.get("chemcode"),
            f"structure_{status}",
            rec.get("reason"),
        )

    # ---------------- exclusions: unlinked records ----------------
    unlinked = observations[observations["compound_id"].isna()]
    for rec in unlinked.to_dict("records"):
        add(
            "exclusion",
            "03_resolve_identity",
            "experiment_record",
            rec.get("experiment_id"),
            "record_unlinked_no_compound",
            f"chemcode {rec.get('chemcode')} has no resolved structure",
        )

    # ---------------- transformations: standardisation ----------------
    for rec in compounds.to_dict("records"):
        raw = rec.get("transformations")
        if raw:
            try:
                for t in json.loads(raw):
                    add(
                        "transformation",
                        "03_resolve_identity",
                        "compound",
                        rec["compound_id"],
                        t.get("type", "structure_standardized"),
                        t.get("detail"),
                    )
            except (TypeError, ValueError):  # pragma: no cover - defensive
                man.warn(f"unparseable transformations for {rec['compound_id']}")
        if (rec.get("n_source_chemicals") or 1) > 1:
            add(
                "transformation",
                "03_resolve_identity",
                "compound",
                rec["compound_id"],
                "compounds_merged",
                f"{rec['n_source_chemicals']} source chemicals standardised to one "
                f"structure: {rec.get('source_chemcodes')}",
            )

    # ---------------- transformations: value normalisation ----------------
    for raw_col, std_col, reason in (
        ("species_raw", "species", "value_normalized_species"),
        ("sex_raw", "sex", "value_normalized_sex"),
    ):
        if raw_col not in observations.columns:
            continue
        changed = observations[
            observations[raw_col].notna()
            & (observations[raw_col] != observations[std_col])
        ]
        # One event per distinct mapping, not per record: the mapping is the
        # transformation, and 70 identical rows would bury the real events.
        for (src, dst), grp in changed.groupby([raw_col, std_col]):
            add(
                "transformation",
                "02_parse_cpdb",
                "field",
                std_col,
                reason,
                f"'{src}' -> '{dst}' applied to {len(grp)} record(s)",
            )

    # ---------------- exclusions: labels ----------------
    for endpoint in lab_cfg["endpoints"]:
        col = f"label_{endpoint}"
        if col not in observations.columns:
            continue
        unresolved = observations[observations[col].isna()]
        for rec in unresolved.to_dict("records"):
            assessment = rec.get("author_assessment_raw")
            detail = (
                f"assessment '{assessment}' is unresolved under this endpoint"
                if assessment and not pd.isna(assessment)
                else "no assessment recorded in the source"
            )
            add(
                "exclusion",
                "04_harmonize_labels",
                "experiment_record",
                rec.get("experiment_id"),
                "record_label_unresolved",
                detail,
                endpoint,
            )

        ds_path = processed / f"model_dataset_{endpoint}.parquet"
        if not ds_path.exists():
            man.warn(f"model dataset absent for endpoint {endpoint}; compound-level "
                     f"exclusions not recorded")
            continue
        man.add_input(ds_path)
        ds = pd.read_parquet(ds_path)
        for rec in ds[ds["training_eligibility"] != "eligible"].to_dict("records"):
            reason = (
                "compound_conflicting_evidence"
                if rec["training_eligibility"] == "excluded_conflicting_evidence"
                else "compound_unresolved_label"
            )
            add(
                "exclusion",
                "06_build_dataset",
                "compound",
                rec["compound_id"],
                reason,
                rec.get("aggregation_reason"),
                endpoint,
            )

        sp_path = processed / f"splits_{endpoint}.parquet"
        if sp_path.exists():
            man.add_input(sp_path)
            sp = pd.read_parquet(sp_path)
            ext = sp[sp["split_assignment"] == "external"]["compound_id"].unique()
            for cid in ext:
                add(
                    "exclusion",
                    "07_split_dataset",
                    "compound",
                    cid,
                    "compound_in_external_set",
                    "held out of the internal partitions for final validation",
                    endpoint,
                )

    ledger = pd.DataFrame(rows)
    if not ledger.empty:
        ledger = ledger.sort_values(
            ["event_type", "stage", "entity_type", "reason_code", "entity_id"]
        ).reset_index(drop=True)
        ledger.insert(0, "event_id", [f"EVT{i:06d}" for i in range(1, len(ledger) + 1)])
    for col in LEDGER_COLUMNS:
        if col not in ledger.columns:
            ledger[col] = None
    ledger = ledger[LEDGER_COLUMNS]

    unknown = set(ledger["reason_code"].dropna()) - EXCLUSION_REASONS - TRANSFORMATION_REASONS
    if unknown:
        man.warn(f"reason codes outside the declared vocabulary: {sorted(unknown)}")

    out = processed / "curation_ledger.parquet"
    ledger.to_parquet(out, index=False)
    man.add_output(out, rows=len(ledger))

    summary = {
        "generated_at": stamp,
        "dataset_version": data_cfg["dataset_version"],
        "label_policy_version": lab_cfg["label_policy_version"],
        "total_events": int(len(ledger)),
        "exclusions": {
            "total": int((ledger["event_type"] == "exclusion").sum()),
            "by_reason": ledger[ledger["event_type"] == "exclusion"]["reason_code"]
            .value_counts()
            .to_dict(),
            "by_entity_type": ledger[ledger["event_type"] == "exclusion"]["entity_type"]
            .value_counts()
            .to_dict(),
        },
        "transformations": {
            "total": int((ledger["event_type"] == "transformation").sum()),
            "by_reason": ledger[ledger["event_type"] == "transformation"]["reason_code"]
            .value_counts()
            .to_dict(),
        },
        "reason_code_vocabulary": {
            "exclusion": sorted(EXCLUSION_REASONS),
            "transformation": sorted(TRANSFORMATION_REASONS),
        },
    }
    sum_path = processed / "curation_ledger_summary.json"
    with sum_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    man.add_output(sum_path)

    man.counts = {
        "events": int(len(ledger)),
        "exclusions": summary["exclusions"]["total"],
        "transformations": summary["transformations"]["total"],
    }
    man.write()

    LOG.info(
        "curation ledger written: %s (%d events: %d exclusions, %d transformations)",
        out,
        len(ledger),
        summary["exclusions"]["total"],
        summary["transformations"]["total"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
