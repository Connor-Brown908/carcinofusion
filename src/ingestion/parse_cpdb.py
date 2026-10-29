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
    "idnum",
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
    "td50_lower",
    "td50_upper",
    "p_value",
    "n_control",
    "n_control_tumors",
    "n_doses",
    "inadequate_flag",
    "tumor_incidence",
    "dose_response",
    "n_dose_groups",
    "reference",
    "notes",
    "curation_status",
]


def _null_tokens_resolver(colcfg: dict):
    """Return a function giving the no-value tokens for one canonical column.

    Falls back to the global list. A column listed in
    `null_tokens_by_column` overrides it entirely, which is how a column
    whose vocabulary uses a glyph the global list calls blank keeps its
    values.
    """
    global_tokens = list(colcfg["null_tokens"])
    overrides = colcfg.get("null_tokens_by_column") or {}

    def tokens_for(canonical: str) -> list[str]:
        if canonical in overrides:
            return list(overrides[canonical] or [])
        return global_tokens

    return tokens_for


def check_null_tokens_against_labels(colcfg: dict, labcfg: dict) -> None:
    """Refuse to parse if a no-value token is also a label code.

    This is the general form of a bug that cost this project a week. The
    parser nulls tokens before the label policy runs, so a glyph that is
    blank globally and meaningful in the assessment column is destroyed
    upstream of every check the policy makes: the audit sees no unmapped
    values, the vocabulary guard sees nothing to complain about, and the
    dataset simply comes out one class short.

    Checking the two configs against each other catches the whole class at
    the only point where both are in hand.
    """
    tokens_for = _null_tokens_resolver(colcfg)
    assessment_nulls = {str(t).strip().lower() for t in tokens_for("author_assessment")}

    codes: dict[str, list[str]] = {}
    for endpoint, cfg in (labcfg.get("endpoints") or {}).items():
        for bucket in ("positive", "negative", "unresolved"):
            for term in cfg.get(bucket, []):
                token = str(term).strip().lower()
                if token and token in assessment_nulls:
                    codes.setdefault(token, []).append(f"{endpoint}/{bucket}")

    if not codes:
        return

    lines = "\n".join(
        f"    {token!r:<12} is a label code in {', '.join(where)}"
        for token, where in sorted(codes.items())
    )
    raise ValueError(
        "configs/cpdb_columns.yaml and configs/labels.yaml contradict each "
        "other on the author_assessment column:\n\n"
        f"{lines}\n\n"
        "  The parser nulls these tokens before the label policy sees them, "
        "so every\n"
        "  record carrying one loses its label and its compound drops out, "
        "with no\n"
        "  unmapped value for the vocabulary check to report. CPDB writes a "
        "negative\n"
        "  author opinion as '-', which is exactly this collision.\n\n"
        "  Fix it by overriding the column in cpdb_columns.yaml:\n\n"
        "    null_tokens_by_column:\n"
        "      author_assessment: ['']\n\n"
        "  The label policy's own missing_assessment_values decides what "
        "counts as\n"
        "  absent evidence. The parser's job is to preserve the raw value."
    )


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
    # Per-column overrides. A token can be a blank in one column and a
    # meaningful code in another: CPDB writes a negative author opinion as
    # '-', which the global list treats as no-value. One list applied to
    # every column silently destroyed 6,036 negative records — a quarter of
    # the database — leaving a positives-only dataset as the only symptom.
    nulls_for = _null_tokens_resolver(colcfg)
    vocab = colcfg["vocabularies"]
    out = pd.DataFrame(index=raw.index)

    out["source_database"] = source_database
    # Row position, recorded so a derived row traces back to the exact line.
    # CPDB's own idnum is the better identifier and is kept separately; this
    # stays because it is the only thing that still works when idnum is null.
    out["source_record_id"] = [f"{path.name}:{i}" for i in range(len(raw))]

    def clean(canonical: str) -> pd.Series:
        tokens = nulls_for(canonical)
        return _series(raw, resolved, canonical).map(
            lambda v: normalise_null(v, tokens)
        )

    out["idnum"] = clean("idnum")
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
    out["inadequate_flag"] = clean("inadequate_flag")

    # Numeric fields carried through for potency work and for reading a
    # treated-group tumour count against its control.
    for canonical in ("td50_lower", "td50_upper", "p_value", "n_control",
                      "n_control_tumors", "n_doses"):
        out[canonical] = pd.to_numeric(clean(canonical), errors="coerce")

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
    out["n_dose_groups"] = None
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
    # Filter on the dose table's own key. These files carry no chemcode, so
    # filtering on one silently dropped every row before the join.
    key = colcfg.get("dose_join_key", "idnum")
    if key in out.columns:
        return out[out[key].notna()]
    return out


def attach_dose(
    experiments: pd.DataFrame,
    dose: pd.DataFrame,
    *,
    join_key: str = "idnum",
    man: StageManifest,
) -> pd.DataFrame:
    """Attach dose and tumour-incidence detail on the experiment identifier.

    CPDB's dose tables carry no chemcode. Each row is one dose group belonging
    to the experiment named by `idnum`, with `order` giving its position in the
    dose series. So the join is idnum -> experiment.idnum: exact, one-to-many,
    and needing no heuristics.

    Dose groups are emitted in `order`, not in file order, so the dose series
    and the matching incidence series read low-to-high and line up with each
    other.
    """
    if dose.empty:
        return experiments
    if join_key not in experiments.columns or join_key not in dose.columns:
        man.warn(
            f"dose detail not attached: '{join_key}' missing from "
            f"{'experiments' if join_key not in experiments.columns else 'dose table'}"
        )
        return experiments

    usable = dose[dose[join_key].notna()].copy()
    if usable.empty:
        man.warn(f"every dose row has a null {join_key}; dose detail not attached")
        return experiments

    if "dose_order" in usable.columns:
        usable["_order"] = pd.to_numeric(usable["dose_order"], errors="coerce")
        usable = usable.sort_values([join_key, "_order"], na_position="last")

    summary: dict[str, dict] = {}
    for key, grp in usable.groupby(join_key, sort=False):
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
            vals = [f"{v:g}" for v in levels if pd.notna(v)]
            dose_resp = "; ".join(vals) if vals else None

        summary[str(key)] = {
            "tumor_incidence": incidence,
            "dose_response": dose_resp,
            "n_dose_groups": int(len(grp)),
        }

    experiments = experiments.copy()
    keys = experiments[join_key].map(lambda v: None if pd.isna(v) else str(v))
    experiments["tumor_incidence"] = keys.map(
        lambda k: summary.get(k, {}).get("tumor_incidence") if k else None
    )
    experiments["dose_response"] = keys.map(
        lambda k: summary.get(k, {}).get("dose_response") if k else None
    )
    experiments["n_dose_groups"] = keys.map(
        lambda k: summary.get(k, {}).get("n_dose_groups") if k else None
    )

    attached = int(experiments["dose_response"].notna().sum())
    orphan_groups = set(summary) - set(k for k in keys if k)
    duplicate_ids = int(experiments[join_key].duplicated().sum())

    man.counts["dose_attachment"] = {
        "join_key": join_key,
        "experiments_with_dose_detail": attached,
        "experiments_without_dose_detail": int(len(experiments) - attached),
        "dose_groups_with_no_matching_experiment": len(orphan_groups),
        "distinct_experiments_in_dose_table": len(summary),
    }
    if duplicate_ids:
        # idnum is supposed to identify an experiment uniquely. If it does not,
        # the same dose series is being attached to several records and the
        # assumption behind this join is wrong.
        man.warn(
            f"{duplicate_ids} experiment record(s) share an {join_key}, which "
            f"should be unique; dose detail is attached to each of them"
        )
    if orphan_groups:
        man.warn(
            f"{len(orphan_groups)} dose group(s) reference an {join_key} with "
            f"no matching experiment record"
        )
    if attached == 0:
        man.warn(
            f"no experiment matched any dose group on {join_key}; check that "
            f"the two files come from the same CPDB release"
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
    # Cross-check the two configs before reading a byte. A no-value token that
    # is also a label code destroys records upstream of every later guard.
    check_null_tokens_against_labels(colcfg, load_config("labels"))
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
        experiments = attach_dose(
            experiments,
            pd.concat(dose_frames, ignore_index=True),
            join_key=colcfg.get("dose_join_key", "idnum"),
            man=man,
        )
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
