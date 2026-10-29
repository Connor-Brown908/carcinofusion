"""Export the curated tables as PostgreSQL-loadable CSVs.

Emits one CSV per schema table plus a load.sql that uses \\copy, so loading
needs psql alone and no Python database driver.

    python -m src.datasets.export_for_postgres
    psql -d carcinofusion -f sql/schema.sql
    psql -d carcinofusion -f data/processed/pg/load.sql

Empty strings are written as \\N so that a missing value becomes SQL NULL
rather than an empty string. For `harmonized_label` this is the difference
between "unresolved" and a value, so it is not a formatting detail.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from src.common import StageManifest, load_config, resolve_path, setup_logging, utc_now

LOG = logging.getLogger("export_for_postgres")
STAGE = "10_export_postgres"

NULL_TOKEN = r"\N"


# Columns that must reach PostgreSQL as integers. A label held in a pandas
# object or float column round-trips as "1.0", which smallint rejects, so each
# is cast to the nullable Int64 dtype: it writes 1 and keeps missing as NULL.
INTEGER_COLUMNS = {
    "harmonized_label",
    "label",
    "n_stereocenters",
    "n_unassigned_stereocenters",
    "n_fragments",
    "n_experiments",
    "n_positive",
    "n_negative",
    "n_indeterminate",
}


def write_csv(df: pd.DataFrame, path, columns: list[str]) -> int:
    """Write a CSV with explicit NULL markers and a fixed column order."""
    out = df.reindex(columns=columns).copy()
    for col in out.columns:
        if col in INTEGER_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
        elif col == "is_conflicted":
            out[col] = out[col].fillna(False).astype(bool)
    out.to_csv(path, index=False, na_rep=NULL_TOKEN)
    return len(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="primary")
    parser.add_argument("--strategy", default="scaffold")
    args = parser.parse_args(argv)

    setup_logging()
    data_cfg = load_config("data")
    lab_cfg = load_config("labels")
    split_cfg = load_config("splits")
    interim = resolve_path(data_cfg["paths"]["interim"])
    processed = resolve_path(data_cfg["paths"]["processed"])
    out_dir = processed / "pg"
    out_dir.mkdir(parents=True, exist_ok=True)

    man = StageManifest(stage=STAGE)
    man.config_versions = {
        "dataset_version": data_cfg["dataset_version"],
        "label_policy_version": lab_cfg["label_policy_version"],
        "split_version": split_cfg["split_version"],
    }

    needed = {
        "compounds": interim / "compound_identity.parquet",
        "observations": interim / "observations_labeled.parquet",
        "splits": processed / f"splits_{args.endpoint}.parquet",
    }
    for name, p in needed.items():
        if not p.exists():
            LOG.error("missing %s: %s — run the pipeline first", name, p)
            return 2
        man.add_input(p)

    compounds = pd.read_parquet(needed["compounds"])
    observations = pd.read_parquet(needed["observations"])
    splits = pd.read_parquet(needed["splits"])

    # sources. The checksum of the raw file each source came from is read back
    # from the parse manifest, so a row in the database ties to the exact bytes
    # that produced it rather than to a file name that may have been replaced.
    raw_checksums: dict[str, str] = {}
    parse_manifest = resolve_path(data_cfg["paths"]["manifests"]) / "02_parse_cpdb_latest.json"
    if parse_manifest.exists():
        import json  # noqa: PLC0415

        rec = json.loads(parse_manifest.read_text(encoding="utf-8"))
        for entry in rec.get("inputs", []):
            key = entry.get("file_key")
            if key and entry.get("sha256"):
                raw_checksums[key] = entry["sha256"]
        man.add_input(parse_manifest, checksum=False)
    else:
        man.warn(
            "parse manifest absent; source rows will carry no raw-file checksum"
        )

    source_to_key = {"CPDB:NCI/NTP": "ncintp", "CPDB:literature": "lit"}
    sources = (
        observations[["source_database"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .assign(
            source_id=lambda d: ["SRC%03d" % (i + 1) for i in range(len(d))],
            retrieval_date=utc_now(),
            source_version=data_cfg["sources"]["cpdb"].get("source_version"),
            license=data_cfg["sources"]["cpdb"].get("license", "").strip(),
            source_url_or_reference=data_cfg["sources"]["cpdb"].get("base_url"),
            source_record_id=None,
            notes=None,
        )
    )
    sources["file_sha256"] = sources["source_database"].map(
        lambda s: raw_checksums.get(source_to_key.get(s, ""))
    )
    src_map = dict(zip(sources["source_database"], sources["source_id"]))

    n = {}
    n["sources"] = write_csv(
        sources,
        out_dir / "sources.csv",
        [
            "source_id", "source_database", "source_record_id",
            "source_url_or_reference", "retrieval_date", "source_version",
            "license", "file_sha256", "notes",
        ],
    )

    chem = compounds.assign(cas_number=None, chemical_name=compounds.get("source_names"))
    n["chemicals"] = write_csv(
        chem,
        out_dir / "chemicals.csv",
        [
            "compound_id", "canonical_smiles", "isomeric_smiles", "inchi_key",
            "connectivity_key", "cas_number", "chemical_name",
            "molecular_formula", "molecular_weight", "scaffold_smiles",
            "scaffold_group", "n_stereocenters", "n_unassigned_stereocenters",
            "n_fragments", "structure_status", "standardization_version",
            "input_smiles", "resolution_source", "resolution_retrieved_date",
            "standardization_notes",
        ],
    )

    # Only experiments whose structure resolved can satisfy the FK.
    exp = observations[observations["compound_id"].notna()].copy()
    exp["source_id"] = exp["source_database"].map(src_map)
    exp["study_duration"] = exp["duration"]
    exp["study_reference"] = exp["reference"]
    n["experiments"] = write_csv(
        exp.drop_duplicates(subset=["experiment_id"]),
        out_dir / "experiments.csv",
        [
            "experiment_id", "compound_id", "source_id", "source_record_id",
            "chemcode", "species", "species_raw", "sex", "sex_raw", "strain",
            "exposure_route", "study_duration", "dose_duration",
            "study_reference", "notes",
        ],
    )

    obs_frames = []
    for endpoint in lab_cfg["endpoints"]:
        col = f"label_{endpoint}"
        if col not in exp.columns:
            continue
        o = exp.copy()
        o["endpoint_name"] = endpoint
        o["harmonized_label"] = o[col]
        o["tumor_type"] = o["tumor_endpoint"]
        obs_frames.append(o)
    obs = pd.concat(obs_frames, ignore_index=True) if obs_frames else pd.DataFrame()
    n["observations"] = write_csv(
        obs,
        out_dir / "observations.csv",
        [
            "experiment_id", "target_organ", "tumor_type", "author_assessment",
            "author_assessment_raw", "endpoint_name", "harmonized_label",
            "dose_response", "tumor_incidence", "td50", "label_policy_version",
            "curation_status",
        ],
    )

    md = splits[splits["split_strategy"] == args.strategy].copy()
    md["dataset_version"] = data_cfg["dataset_version"]
    md["endpoint_name"] = args.endpoint
    md["frozen_at"] = utc_now()
    n["model_datasets"] = write_csv(
        md,
        out_dir / "model_datasets.csv",
        [
            "dataset_version", "compound_id", "endpoint_definition",
            "endpoint_name", "label", "label_confidence", "is_conflicted",
            "n_experiments", "n_positive", "n_negative", "n_indeterminate",
            "split_group", "split_strategy", "split_assignment",
            "label_policy_version", "split_version", "frozen_at",
        ],
    )

    load_sql = out_dir / "load.sql"
    load_sql.write_text(
        "-- Load the exported CSVs. Apply sql/schema.sql first.\n"
        "-- Run from the repository root:  psql -d carcinofusion -f data/processed/pg/load.sql\n"
        "\\set ON_ERROR_STOP on\n"
        "BEGIN;\n"
        "TRUNCATE model_datasets, observations, experiments, chemicals, sources RESTART IDENTITY CASCADE;\n"
        "\\copy sources FROM 'data/processed/pg/sources.csv' WITH (FORMAT csv, HEADER true, NULL '\\N')\n"
        "\\copy chemicals FROM 'data/processed/pg/chemicals.csv' WITH (FORMAT csv, HEADER true, NULL '\\N')\n"
        "\\copy experiments FROM 'data/processed/pg/experiments.csv' WITH (FORMAT csv, HEADER true, NULL '\\N')\n"
        "\\copy observations(experiment_id,target_organ,tumor_type,author_assessment,author_assessment_raw,"
        "endpoint_name,harmonized_label,dose_response,tumor_incidence,td50,label_policy_version,curation_status) "
        "FROM 'data/processed/pg/observations.csv' WITH (FORMAT csv, HEADER true, NULL '\\N')\n"
        "\\copy model_datasets FROM 'data/processed/pg/model_datasets.csv' WITH (FORMAT csv, HEADER true, NULL '\\N')\n"
        "COMMIT;\n"
        "SELECT 'chemicals' t, count(*) FROM chemicals\n"
        "UNION ALL SELECT 'experiments', count(*) FROM experiments\n"
        "UNION ALL SELECT 'observations', count(*) FROM observations\n"
        "UNION ALL SELECT 'unresolved labels', count(*) FROM observations WHERE harmonized_label IS NULL\n"
        "UNION ALL SELECT 'training set', count(*) FROM v_training_set;\n",
        encoding="utf-8",
    )

    for name, count in n.items():
        man.add_output(out_dir / f"{name}.csv", rows=count)
    man.add_output(load_sql)
    man.counts = n
    man.write()

    LOG.info("exported to %s: %s", out_dir, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
