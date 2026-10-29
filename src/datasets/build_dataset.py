"""Step 6a - assemble the model-ready dataset.

Joins standardised structures to compound-level labels and emits one table per
endpoint. Compounds that are unresolved or conflicted are carried through with
their reason recorded rather than dropped, so that the eventual training
selection is explicit and auditable.

Output:
    data/processed/model_dataset_{endpoint}.parquet

Usage:
    python -m src.datasets.build_dataset
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.common import StageManifest, load_config, resolve_path, setup_logging

LOG = logging.getLogger("build_dataset")
STAGE = "06_build_dataset"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument(
        "--with-descriptors",
        action="store_true",
        help="also compute physicochemical descriptors (slower)",
    )
    args = parser.parse_args(argv)

    setup_logging()
    data_cfg = load_config("data")
    lab_cfg = load_config("labels")
    interim = resolve_path(args.interim or data_cfg["paths"]["interim"])
    processed = resolve_path(args.outdir or data_cfg["paths"]["processed"])
    processed.mkdir(parents=True, exist_ok=True)

    comp_path = interim / "compound_identity.parquet"
    lab_path = interim / "compound_labels.parquet"
    for p in (comp_path, lab_path):
        if not p.exists():
            LOG.error("missing input: %s — run the earlier stages first", p)
            return 2

    man = StageManifest(stage=STAGE)
    man.config_versions = {
        "dataset_version": data_cfg.get("dataset_version"),
        "label_policy_version": lab_cfg["label_policy_version"],
    }
    man.add_input(comp_path)
    man.add_input(lab_path)

    compounds = pd.read_parquet(comp_path)
    labels = pd.read_parquet(lab_path)

    merged = compounds.merge(labels, on="compound_id", how="left", validate="one_to_one")
    LOG.info("compounds: %d, with labels: %d", len(compounds), len(labels))

    unlabelled = int(merged["label_policy_version"].isna().sum())
    if unlabelled:
        man.warn(
            f"{unlabelled} standardised compound(s) have no experimental record "
            f"and therefore no label; they are retained with label null"
        )

    structure_cols = [
        "compound_id",
        "canonical_smiles",
        "isomeric_smiles",
        "inchi_key",
        "connectivity_key",
        "scaffold_smiles",
        "scaffold_group",
        "molecular_formula",
        "molecular_weight",
        "n_stereocenters",
        "n_unassigned_stereocenters",
        "structure_status",
        "standardization_version",
        "resolution_source",
    ]

    for name in lab_cfg["endpoints"]:
        cols = structure_cols + [
            c
            for c in (
                f"label_{name}",
                f"label_confidence_{name}",
                f"is_conflicted_{name}",
                f"n_positive_{name}",
                f"n_negative_{name}",
                f"n_indeterminate_{name}",
                f"aggregation_reason_{name}",
                "n_experiments",
                "n_eligible_experiments",
                "label_policy_version",
                "endpoint_definition",
            )
            if c in merged.columns
        ]
        ds = merged[cols].copy()
        ds = ds.rename(
            columns={
                f"label_{name}": "label",
                f"label_confidence_{name}": "label_confidence",
                f"is_conflicted_{name}": "is_conflicted",
                f"n_positive_{name}": "n_positive",
                f"n_negative_{name}": "n_negative",
                f"n_indeterminate_{name}": "n_indeterminate",
                f"aggregation_reason_{name}": "aggregation_reason",
            }
        )
        ds["endpoint_name"] = name
        ds["is_conflicted"] = ds["is_conflicted"].fillna(False).astype(bool)

        # Why a compound is or is not eligible for the binary training set.
        def eligibility(row) -> str:
            if row["label"] is None or pd.isna(row["label"]):
                if row["is_conflicted"]:
                    return "excluded_conflicting_evidence"
                return "excluded_unresolved_label"
            return "eligible"

        ds["training_eligibility"] = ds.apply(eligibility, axis=1)

        if args.with_descriptors:
            from src.features.descriptors import compute_frame  # noqa: PLC0415

            desc = compute_frame(ds["canonical_smiles"].fillna("").tolist())
            ds = pd.concat([ds.reset_index(drop=True), desc.reset_index(drop=True)], axis=1)

        out = processed / f"model_dataset_{name}.parquet"
        ds.to_parquet(out, index=False)
        man.add_output(out, rows=len(ds))

        eligible = int((ds["training_eligibility"] == "eligible").sum())
        man.counts[f"endpoint_{name}"] = {
            "compounds": int(len(ds)),
            "eligible_for_training": eligible,
            "positive": int((ds["label"] == 1).sum()),
            "negative": int((ds["label"] == 0).sum()),
            "excluded_conflicting": int(
                (ds["training_eligibility"] == "excluded_conflicting_evidence").sum()
            ),
            "excluded_unresolved": int(
                (ds["training_eligibility"] == "excluded_unresolved_label").sum()
            ),
        }
        LOG.info(
            "endpoint %s: %d compounds, %d eligible (%d pos / %d neg)",
            name,
            len(ds),
            eligible,
            int((ds["label"] == 1).sum()),
            int((ds["label"] == 0).sum()),
        )

    man.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
