"""Step 2 - parse CPDB into Dataset A: one row per experimental record.

Produces data/interim/carcinogenicity_experiments.parquet with the original
source values preserved alongside their standardised forms. Every standardised
column has a `*_raw` companion holding exactly what the source said, because
auditing a conflicting result is impossible once the original value is gone.

Joins use the documented CPDB chemcode. Row order is never used to associate
records across files. Dose/incidence detail is attached on the documented key
set and, where a record cannot be matched unambiguously, it is left unattached
and counted rather than guessed at.

Usage:
    python -m src.ingestion.parse_cpdb
    python -m src.ingestion.parse_cpdb --raw-dir data/raw/cpdb_test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.common import (
    ColumnMappingError,
    StageManifest,
    load_config,
    map_columns,
    normalise_null,
    normalise_vocab,
    resolve_path,
    setup_logging,
)

LOG = logging.getLogger("parse_cpdb")
STAGE = "02_parse_cpdb"

# Canonical experiment columns, standardised form plus preserved original.
EXPERIMENT_COLUMNS = [
    "experiment_id",
    "source_database",
    "source_record_id",
    "chemcode",
    "chemical_name",
    "cas_number",
    "species",
    "species_raw",
    "sex",
    "sex_raw",
    "strain",
    "exposure_route",
    "duration",
    "dose_duration",
    "target_organ",
    "tumor_endpoint",
    "author_assessment",
    "author_assessment_raw",
    "td50",
    "tumor_incidence",
    "dose_response",
    "reference",
    "notes",
    "curation_status",
]


def read_tab(path: Path) -> pd.DataFrame:
    """Read a CPDB tab-separated file as raw strings.

    Everything is read as str. Type coercion happens after null tokens have
    been handled, so that a '.' placeholder never becomes NaN-as-zero.
    """
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        encoding="latin-1",
        on_bad_lines="warn",
    )


def _series(df: pd.DataFrame, resolved: dict[str, str], canonical: str) -> pd.Series:
    """Column by canonical name, or an all-None series when absent."""
    col = resolved.get(canonical)
    if col is None:
        return pd.Series([None] * len(df), index=df.index, dtype="object")
    return df[col]


def parse_experiment_table(
    path: Path,
    *,
    source_database: str,
    colcfg: dict,
    man: StageManifest,
) -> pd.DataFrame:
    """Parse one experiment table (ncintp or lit) into canonical form."""
    raw = read_tab(path)
    resolved, aliases, missing = map_columns(
        list(raw.columns),
        colcfg["experiments"],
        required=colcfg["experiments"]["required"],
        table_name=path.name,
    )
    if missing:
        man.warn(
            f"{path.name}: optional column(s) not present and left null: {missing}"
        )
    LOG.info("%s: %d rows, mapped columns %s", path.name, len(raw), aliases)

    nulls = colcfg["null_tokens"]
    vocab = colcfg["vocabularies"]
    out = pd.DataFrame(index=raw.index)

    out["source_database"] = source_database
    # A stable per-file record identifier: the source's own row position.
    # Recorded so that any derived row can be traced back to the exact line.
    out["source_record_id"] = [f"{path.name}:{i}" for i in range(len(raw))]

    def clean(canonical: str) -> pd.Series:
        return _series(raw, resolved, canonical).map(lambda v: normalise_null(v, nulls))

    out["chemcode"] = clean("chemcode")
    out["cas_number"] = clean("cas_number")

    # Preserve the original, then standardise beside it.
    out["species_raw"] = clean("species")
    out["species"] = out["species_raw"].map(lambda v: normalise_vocab(v, vocab["species"]))
    out["sex_raw"] = clean("sex")
    out["sex"] = out["sex_raw"].map(lambda v: normalise_vocab(v, vocab["sex"]))
    out["author_assessment_raw"] = clean("author_assessment")
    out["author_assessment"] = out["author_assessment_raw"]

    out["strain"] = clean("strain")
    out["exposure_route"] = clean("exposure_route")
    out["duration"] = clean("study_duration")
    out["dose_duration"] = clean("dose_duration")
    out["target_organ"] = clean("target_organ")
    out["tumor_endpoint"] = clean("tumor_endpoint")
    out["reference"] = clean("reference")
    out["notes"] = clean("notes")

    td50 = clean("td50")
    out["td50"] = pd.to_numeric(td50, errors="coerce")
    bad_td50 = int((td50.notna() & out["td50"].isna()).sum())
    if bad_td50:
        man.warn(
            f"{path.name}: {bad_td50} td50 value(s) present but not numeric; "
            f"set to null and retained in the raw record"
        )

    out["chemical_name"] = None
    out["tumor_incidence"] = None
    out["dose_response"] = None
    out["curation_status"] = "parsed"
    return out


def parse_chemname(path: Path, colcfg: dict, man: StageManifest) -> pd.DataFrame:
    """Parse the chemical-name table into a chemcode lookup."""
    raw = read_tab(path)
    resolved, aliases, missing = map_columns(
        list(raw.columns),
        colcfg["chemname"],
        required=colcfg["chemname"]["required"],
        table_name=path.name,
    )
    if missing:
        man.warn(f"{path.name}: optional column(s) left null: {missing}")
    LOG.info("%s: %d rows, mapped columns %s", path.name, len(raw), aliases)

    nulls = colcfg["null_tokens"]
    out = pd.DataFrame(index=raw.index)
    for canonical in ("chemcode", "cas_number", "chemical_name"):
        out[canonical] = _series(raw, resolved, canonical).map(
            lambda v: normalise_null(v, nulls)
        )
    out = out[out["chemcode"].notna()]

    dupes = int(out["chemcode"].duplicated().sum())
    if dupes:
        man.warn(
            f"{path.name}: {dupes} duplicate chemcode row(s); "
            f"first occurrence used for name lookup, all retained in source"
        )
        out = out.drop_duplicates(subset=["chemcode"], keep="first")
    return out


def parse_dose(path: Path, colcfg: dict, man: StageManifest) -> pd.DataFrame:
    """Parse a dose/incidence table into canonical form."""
    raw = read_tab(path)
    resolved, aliases, missing = map_columns(
        list(raw.columns),
        colcfg["dose"],
        required=colcfg["dose"]["required"],
        table_name=path.name,
    )
    if missing:
        man.warn(f"{path.name}: optional column(s) left null: {missing}")
    LOG.info("%s: %d rows, mapped columns %s", path.name, len(raw), aliases)

    nulls = colcfg["null_tokens"]
    vocab = colcfg["vocabularies"]
    out = pd.DataFrame(index=raw.index)
    for canonical in colcfg["dose"]:
        if canonical == "required":
            continue
        out[canonical] = _series(raw, resolved, canonical).map(
            lambda v: normalise_null(v, nulls)
        )
    if "species" in out:
        out["species"] = out["species"].map(lambda v: normalise_vocab(v, vocab["species"]))
    if "sex" in out:
        out["sex"] = out["sex"].map(lambda v: normalise_vocab(v, vocab["sex"]))
    return out[out["chemcode"].notna()]


def attach_dose(
    experiments: pd.DataFrame,
    dose: pd.DataFrame,
    *,
    man: StageManifest,
) -> pd.DataFrame:
    """Attach dose/incidence detail on the documented key set.

    The key is chemcode plus whichever of species, sex, strain, exposure_route
    and target_organ are present in BOTH tables. A dose group that matches more
    than one experiment, or none, is left unattached and counted. Nothing is
    matched by row order.
    """
    if dose.empty:
        return experiments

    candidate_keys = ["chemcode", "species", "sex", "strain", "exposure_route", "target_organ"]
    keys = [
        k
        for k in candidate_keys
        if k in experiments.columns
        and k in dose.columns
        and dose[k].notna().any()
        and experiments[k].notna().any()
    ]
    if "chemcode" not in keys:
        man.warn("dose table has no usable chemcode; dose detail not attached")
        return experiments

    LOG.info("attaching dose detail on key set: %s", keys)

    # Group dose rows by the key set, keeping a compact summary per group.
    grouped = dose.groupby(keys, dropna=False)
    summary: dict[tuple, dict] = {}
    for key_vals, grp in grouped:
        key_tuple = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        treated = pd.to_numeric(grp.get("n_treated"), errors="coerce")
        tumour = pd.to_numeric(grp.get("n_with_tumor"), errors="coerce")
        levels = pd.to_numeric(grp.get("dose_level"), errors="coerce")
        incidence = None
        if treated is not None and tumour is not None:
            pairs = [
                f"{int(t)}/{int(n)}"
                for t, n in zip(tumour, treated)
                if pd.notna(t) and pd.notna(n)
            ]
            incidence = "; ".join(pairs) if pairs else None
        dose_resp = None
        if levels is not None:
            vals = [f"{v:g}" for v in levels.dropna()]
            dose_resp = "; ".join(vals) if vals else None
        summary[key_tuple] = {
            "tumor_incidence": incidence,
            "dose_response": dose_resp,
            "n_dose_rows": len(grp),
        }

    exp_keys = list(experiments[keys].itertuples(index=False, name=None))
    # Count how many experiments claim each key, so an ambiguous attachment is
    # detected rather than silently duplicated onto several records.
    key_counts: dict[tuple, int] = {}
    for k in exp_keys:
        key_counts[k] = key_counts.get(k, 0) + 1

    attached = 0
    ambiguous = 0
    unmatched_groups = set(summary) - set(key_counts)

    incidence_col: list[str | None] = []
    dose_col: list[str | None] = []
    for k in exp_keys:
        info = summary.get(k)
        if info is None:
            incidence_col.append(None)
            dose_col.append(None)
        elif key_counts[k] > 1:
            # The key set does not uniquely identify this experiment.
            ambiguous += 1
            incidence_col.append(None)
            dose_col.append(None)
        else:
            attached += 1
            incidence_col.append(info["tumor_incidence"])
            dose_col.append(info["dose_response"])

    experiments = experiments.copy()
    experiments["tumor_incidence"] = incidence_col
    experiments["dose_response"] = dose_col

    man.counts.setdefault("dose_attachment", {})
    man.counts["dose_attachment"] = {
        "key_set": keys,
        "experiments_with_dose_detail": attached,
        "experiments_ambiguous_key_left_unattached": ambiguous,
        "dose_groups_with_no_matching_experiment": len(unmatched_groups),
    }
    if ambiguous:
        man.warn(
            f"{ambiguous} experiment(s) share a dose key with another record; "
            f"dose detail left unattached rather than duplicated"
        )
    if unmatched_groups:
        man.warn(
            f"{len(unmatched_groups)} dose group(s) matched no experiment record"
        )
    return experiments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=None, help="override data.yaml raw_dir")
    parser.add_argument("--out", default=None, help="override output parquet path")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config("data")
    colcfg = load_config("cpdb_columns")
    src = cfg["sources"]["cpdb"]
    raw_dir = resolve_path(args.raw_dir or src["raw_dir"])
    interim = resolve_path(cfg["paths"]["interim"])
    interim.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else interim / "carcinogenicity_experiments.parquet"

    man = StageManifest(stage=STAGE)
    man.config_versions = {
        "dataset_version": cfg.get("dataset_version"),
        "join_key": colcfg["join_key"],
    }

    frames: list[pd.DataFrame] = []
    try:
        for key, label in (("ncintp", "CPDB:NCI/NTP"), ("lit", "CPDB:literature")):
            spec = src["files"][key]
            path = raw_dir / spec["filename"]
            if not path.exists():
                if spec.get("required", False):
                    LOG.error("required file missing: %s", path)
                    man.warn(f"required file missing: {path}")
                    man.write()
                    return 2
                man.warn(f"optional experiment file missing: {path}")
                continue
            man.add_input(path, file_key=key)
            frames.append(
                parse_experiment_table(
                    path, source_database=label, colcfg=colcfg, man=man
                )
            )
    except ColumnMappingError as exc:
        LOG.error("%s", exc)
        man.warn(str(exc))
        man.write()
        return 3

    if not frames:
        LOG.error("no experiment records parsed")
        man.write()
        return 2

    experiments = pd.concat(frames, ignore_index=True)

    # Chemical names, joined on chemcode.
    chem_path = raw_dir / src["files"]["chemname"]["filename"]
    if chem_path.exists():
        man.add_input(chem_path, file_key="chemname")
        chem = parse_chemname(chem_path, colcfg, man)
        name_map = dict(zip(chem["chemcode"], chem["chemical_name"]))
        cas_map = {
            k: v for k, v in zip(chem["chemcode"], chem["cas_number"]) if v is not None
        }
        experiments["chemical_name"] = experiments["chemcode"].map(name_map)
        # Fill a missing CAS from the identity table; never overwrite one the
        # experiment record already carried.
        filled = experiments["cas_number"].isna() & experiments["chemcode"].map(
            lambda c: c in cas_map
        )
        experiments.loc[filled, "cas_number"] = experiments.loc[
            filled, "chemcode"
        ].map(cas_map)
        man.counts["cas_filled_from_chemname"] = int(filled.sum())
        unnamed = int(experiments["chemical_name"].isna().sum())
        if unnamed:
            man.warn(f"{unnamed} experiment record(s) have no chemical name on chemcode")
    else:
        man.warn(f"chemname table missing: {chem_path}; names left null")

    # Dose / incidence detail.
    dose_frames = []
    for key in ("ncntdose", "litdose"):
        spec = src["files"].get(key)
        if spec is None:
            continue
        path = raw_dir / spec["filename"]
        if path.exists():
            man.add_input(path, file_key=key)
            dose_frames.append(parse_dose(path, colcfg, man))
    if dose_frames:
        experiments = attach_dose(experiments, pd.concat(dose_frames, ignore_index=True), man=man)
    else:
        man.warn("no dose/incidence tables present; dose detail unavailable")

    # Stable internal experiment identifier.
    experiments = experiments.reset_index(drop=True)
    experiments.insert(0, "experiment_id", [f"EXP{i:07d}" for i in range(1, len(experiments) + 1)])
    for col in EXPERIMENT_COLUMNS:
        if col not in experiments.columns:
            experiments[col] = None
    experiments = experiments[EXPERIMENT_COLUMNS]

    experiments.to_parquet(out_path, index=False)

    assessments = (
        experiments["author_assessment"].fillna("<null>").value_counts().to_dict()
    )
    man.counts.update(
        {
            "experiment_records": int(len(experiments)),
            "unique_chemcodes": int(experiments["chemcode"].nunique(dropna=True)),
            "unique_cas_numbers": int(experiments["cas_number"].nunique(dropna=True)),
            "records_missing_cas": int(experiments["cas_number"].isna().sum()),
            "records_missing_assessment": int(experiments["author_assessment"].isna().sum()),
            "records_with_td50": int(experiments["td50"].notna().sum()),
            "by_source": experiments["source_database"].value_counts().to_dict(),
            "by_species": experiments["species"].fillna("<null>").value_counts().to_dict(),
            "assessment_values": assessments,
        }
    )
    man.add_output(out_path, rows=len(experiments))
    man.write()

    LOG.info(
        "Dataset A written: %s (%d records, %d unique chemcodes)",
        out_path,
        len(experiments),
        experiments["chemcode"].nunique(dropna=True),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
