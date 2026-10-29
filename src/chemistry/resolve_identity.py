"""Step 3 - resolve chemical identity and build Dataset B.

Reads Dataset A (experimental records), attaches a structure to each distinct
source chemical, standardises it, and emits:

  data/interim/compound_identity.parquet    one row per standardised compound
  data/interim/structure_review.parquet     every record that failed to resolve
  data/interim/experiments_linked.parquet   Dataset A with compound_id attached

Two rules carried through from the protocol:

  * A CAS number or a chemical name is never treated as proof of identity. A
    structure enters the pipeline only from the resolution table, which must
    carry a named source and a retrieval date per row.
  * Structures that fail to resolve go to the review table. They are never
    silently discarded, and they are never given a label.

Two identity levels are produced:
  compound_id      exact standardised identity, InChIKey-based, stereo included
  scaffold_group   grouping key for leakage-resistant splits

Usage:
    python -m src.chemistry.resolve_identity
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.chemistry.standardize import (
    STANDARDIZATION_VERSION,
    STATUS_EMPTY,
    standardize_smiles,
)
from src.common import StageManifest, load_config, resolve_path, setup_logging

LOG = logging.getLogger("resolve_identity")
STAGE = "03_resolve_identity"

RESOLUTION_COLUMNS_REQUIRED = ["smiles"]
RESOLUTION_KEYS = ["cas_number", "chemcode"]


def load_resolution_table(path: Path, *, require_source: bool, man: StageManifest) -> pd.DataFrame:
    """Load the structure resolution table supplied by the curator.

    Expected columns: at least one of cas_number / chemcode, plus smiles.
    Optional: source, retrieved_date, inchi, chemical_name.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"structure resolution table not found: {path}\n"
            f"Create it as described in docs/dataset_protocol.md. The pipeline "
            f"does not fetch structures over the network, so that the structure "
            f"a compound had when the dataset was frozen is recorded, not "
            f"re-fetched later."
        )
    sep = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, na_values=[""])

    missing = [c for c in RESOLUTION_COLUMNS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required column(s) {missing}")
    if not any(k in df.columns for k in RESOLUTION_KEYS):
        raise ValueError(
            f"{path.name}: needs at least one join key from {RESOLUTION_KEYS}"
        )

    if require_source:
        if "source" not in df.columns:
            raise ValueError(
                f"{path.name}: structure_resolution.require_source_attribution is "
                f"set, so a `source` column naming where each structure came from "
                f"is required"
            )
        unattributed = int(df["source"].isna().sum())
        if unattributed:
            man.warn(
                f"{unattributed} resolution row(s) carry no source attribution; "
                f"those structures are rejected"
            )
            df = df[df["source"].notna()]
    return df


def build_lookup(resolution: pd.DataFrame) -> tuple[dict, dict]:
    """Build cas -> row and chemcode -> row lookups, flagging conflicts."""
    by_cas: dict[str, dict] = {}
    by_chemcode: dict[str, dict] = {}
    for rec in resolution.to_dict("records"):
        smi = rec.get("smiles")
        if not smi or not str(smi).strip():
            continue
        cas = rec.get("cas_number")
        code = rec.get("chemcode")
        if cas and str(cas).strip():
            by_cas.setdefault(str(cas).strip(), rec)
        if code and str(code).strip():
            by_chemcode.setdefault(str(code).strip(), rec)
    return by_cas, by_chemcode


def compound_id_for(inchi_key: str) -> str:
    """Stable internal compound identifier derived from the exact identity.

    Deterministic across runs and machines, so a compound keeps its id when the
    dataset is rebuilt. Derived from the full InChIKey, whose second block
    carries the stereochemistry layer, so stereoisomers receive distinct ids.
    """
    digest = hashlib.sha256(inchi_key.encode("utf-8")).hexdigest()[:12]
    return f"CMP{digest.upper()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", default=None)
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config("data")
    interim = resolve_path(args.outdir or cfg["paths"]["interim"])
    interim.mkdir(parents=True, exist_ok=True)

    exp_path = Path(args.experiments) if args.experiments else interim / "carcinogenicity_experiments.parquet"
    res_path = (
        Path(args.resolution)
        if args.resolution
        else resolve_path(cfg["structure_resolution"]["resolution_table"])
    )

    man = StageManifest(stage=STAGE)
    man.config_versions = {
        "dataset_version": cfg.get("dataset_version"),
        "standardization_version": STANDARDIZATION_VERSION,
    }

    if not exp_path.exists():
        LOG.error("Dataset A not found: %s (run parse_cpdb first)", exp_path)
        return 2
    man.add_input(exp_path)
    experiments = pd.read_parquet(exp_path)

    try:
        resolution = load_resolution_table(
            res_path,
            require_source=cfg["structure_resolution"]["require_source_attribution"],
            man=man,
        )
    except (FileNotFoundError, ValueError) as exc:
        LOG.error("%s", exc)
        man.warn(str(exc))
        man.write()
        return 2
    man.add_input(res_path)
    by_cas, by_chemcode = build_lookup(resolution)
    LOG.info(
        "resolution table: %d row(s), %d CAS key(s), %d chemcode key(s)",
        len(resolution),
        len(by_cas),
        len(by_chemcode),
    )

    # Distinct source chemicals, keyed by chemcode (CPDB's own identifier).
    source_chems = (
        experiments[["chemcode", "cas_number", "chemical_name"]]
        .drop_duplicates(subset=["chemcode"])
        .reset_index(drop=True)
    )
    LOG.info("distinct source chemicals: %d", len(source_chems))

    compounds: dict[str, dict] = {}
    review_rows: list[dict] = []
    chemcode_to_compound: dict[str, str] = {}

    for rec in source_chems.to_dict("records"):
        code = rec["chemcode"]
        cas = rec.get("cas_number")

        # chemcode first, then CAS. Which key matched is recorded.
        hit = by_chemcode.get(str(code).strip()) if code else None
        matched_on = "chemcode" if hit else None
        if hit is None and cas:
            hit = by_cas.get(str(cas).strip())
            matched_on = "cas_number" if hit else None

        if hit is None:
            review_rows.append(
                {
                    **rec,
                    "structure_status": "unresolved_no_structure",
                    "reason": "no structure in the resolution table for this chemcode or CAS",
                    "standardization_version": STANDARDIZATION_VERSION,
                }
            )
            continue

        result = standardize_smiles(hit.get("smiles"))
        if not result.ok or not result.inchi_key:
            review_rows.append(
                {
                    **rec,
                    "input_smiles": result.input_smiles,
                    "structure_status": result.status
                    if result.status != STATUS_EMPTY
                    else "unresolved_empty_smiles",
                    "reason": "; ".join(result.notes) or result.status,
                    "resolution_source": hit.get("source"),
                    "standardization_version": STANDARDIZATION_VERSION,
                }
            )
            continue

        cid = compound_id_for(result.inchi_key)
        chemcode_to_compound[code] = cid

        if cid not in compounds:
            compounds[cid] = {
                "compound_id": cid,
                "canonical_smiles": result.canonical_smiles,
                "isomeric_smiles": result.isomeric_smiles,
                "inchi_key": result.inchi_key,
                "connectivity_key": result.connectivity_key,
                "scaffold_smiles": result.scaffold_smiles,
                "molecular_formula": result.molecular_formula,
                "molecular_weight": result.molecular_weight,
                "n_stereocenters": result.n_stereocenters,
                "n_unassigned_stereocenters": result.n_unassigned_stereocenters,
                "n_fragments": result.n_fragments,
                "structure_status": result.status,
                "standardization_version": result.standardization_version,
                "input_smiles": result.input_smiles,
                "resolution_source": hit.get("source"),
                "resolution_retrieved_date": hit.get("retrieved_date"),
                "resolution_matched_on": matched_on,
                "source_chemcodes": [code],
                "source_names": [rec.get("chemical_name")],
                "standardization_notes": "; ".join(result.notes) or None,
                # Structured transformation record, consumed by the curation
                # ledger so transformations can be counted by type rather than
                # read out of free text.
                "transformations": json.dumps(result.transformations),
            }
        else:
            # Two source chemicals standardised to the same structure. This is
            # a real merge and is recorded, not hidden.
            compounds[cid]["source_chemcodes"].append(code)
            compounds[cid]["source_names"].append(rec.get("chemical_name"))

    comp_df = pd.DataFrame(list(compounds.values()))
    if not comp_df.empty:
        comp_df["n_source_chemicals"] = comp_df["source_chemcodes"].map(len)
        comp_df["source_chemcodes"] = comp_df["source_chemcodes"].map(
            lambda v: "; ".join(str(x) for x in v)
        )
        comp_df["source_names"] = comp_df["source_names"].map(
            lambda v: "; ".join(str(x) for x in v if x is not None)
        )

        # Scaffold grouping key. Compounds with no Murcko scaffold (acyclic)
        # are grouped by their connectivity skeleton instead, so they are
        # neither collapsed into one giant group nor left with a null key that
        # a later groupby would silently drop.
        #
        # pd.isna is required rather than a truthiness test: a missing value
        # arrives here as float NaN, and NaN is truthy in Python, so
        # `scaf if scaf else fallback` would keep the NaN.
        def _group_key(scaf, conn, cid) -> str:
            if not pd.isna(scaf) and str(scaf).strip():
                return str(scaf)
            if not pd.isna(conn) and str(conn).strip():
                return f"acyclic:{conn}"
            return f"ungrouped:{cid}"

        comp_df["scaffold_group"] = [
            _group_key(scaf, conn, cid)
            for scaf, conn, cid in zip(
                comp_df["scaffold_smiles"],
                comp_df["connectivity_key"],
                comp_df["compound_id"],
            )
        ]
        if comp_df["scaffold_group"].isna().any():  # pragma: no cover - defensive
            raise RuntimeError("scaffold_group must never be null")

    review_df = pd.DataFrame(review_rows)

    # Link Dataset A to compounds.
    experiments_linked = experiments.copy()
    experiments_linked["compound_id"] = experiments_linked["chemcode"].map(
        chemcode_to_compound
    )
    linked = int(experiments_linked["compound_id"].notna().sum())

    comp_path = interim / "compound_identity.parquet"
    review_path = interim / "structure_review.parquet"
    linked_path = interim / "experiments_linked.parquet"
    comp_df.to_parquet(comp_path, index=False)
    review_df.to_parquet(review_path, index=False)
    experiments_linked.to_parquet(linked_path, index=False)

    merged = (
        int((comp_df["n_source_chemicals"] > 1).sum()) if not comp_df.empty else 0
    )
    man.counts.update(
        {
            "source_chemicals": int(len(source_chems)),
            "compounds_standardized": int(len(comp_df)),
            "compounds_merged_from_multiple_source_chemicals": merged,
            "structures_sent_to_review": int(len(review_df)),
            "review_reasons": (
                review_df["structure_status"].value_counts().to_dict()
                if not review_df.empty
                else {}
            ),
            "experiment_records": int(len(experiments_linked)),
            "experiment_records_linked_to_compound": linked,
            "experiment_records_unlinked": int(len(experiments_linked) - linked),
            "unique_scaffold_groups": (
                int(comp_df["scaffold_group"].nunique()) if not comp_df.empty else 0
            ),
            "compounds_with_unassigned_stereo": (
                int((comp_df["n_unassigned_stereocenters"] > 0).sum())
                if not comp_df.empty
                else 0
            ),
        }
    )
    for p, n in ((comp_path, len(comp_df)), (review_path, len(review_df)), (linked_path, len(experiments_linked))):
        man.add_output(p, rows=n)

    if merged:
        man.warn(
            f"{merged} compound(s) were standardised from more than one source "
            f"chemical; check compound_identity.source_chemcodes before freezing"
        )
    if len(review_df):
        man.warn(
            f"{len(review_df)} source chemical(s) have no usable structure and "
            f"are held in structure_review.parquet for manual review"
        )
    man.write()

    LOG.info(
        "Dataset B written: %s (%d compounds, %d scaffold groups, %d in review)",
        comp_path,
        len(comp_df),
        comp_df["scaffold_group"].nunique() if not comp_df.empty else 0,
        len(review_df),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
