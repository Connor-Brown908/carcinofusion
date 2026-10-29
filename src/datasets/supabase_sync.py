"""Sync curated data and benchmark results to Supabase.

The workspace that produces these artefacts is ephemeral. Supabase is where
the dataset and the benchmark results live afterwards, so a result can be
queried next week without re-running anything.

This module emits batched, idempotent SQL rather than opening its own
connection. Two reasons: it works whether the SQL is applied through the
Supabase MCP tools, `psql`, or the SQL editor; and the statements are
inspectable before anything is written to a shared database.

Every statement is `INSERT ... ON CONFLICT DO UPDATE`, so re-running a sync is
safe and updates in place instead of duplicating.

Target schema is `carcinofusion`, deliberately not `public`: Supabase exposes
the public schema through PostgREST, and a research dataset sitting there with
no row-level policies would be readable by anyone holding the anon key.

Usage:
    python -m src.datasets.supabase_sync                 # all tables
    python -m src.datasets.supabase_sync --only benchmark
    python -m src.datasets.supabase_sync --batch-size 200
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import pandas as pd

from src.common import StageManifest, load_config, resolve_path, setup_logging

LOG = logging.getLogger("supabase_sync")
STAGE = "13_supabase_sync"
SCHEMA = "carcinofusion"


def sql_literal(value) -> str:
    """Render a Python value as a SQL literal.

    NULL is preserved as NULL. For `harmonized_label` and `label` that is the
    difference between "unresolved" and a value, so it is not a formatting
    detail: a null that became 0 here would silently turn an untested chemical
    into a negative one.
    """
    if value is None:
        return "NULL"
    if isinstance(value, float) and math.isnan(value):
        return "NULL"
    if value is pd.NaT:
        return "NULL"
    try:
        if pd.isna(value):
            return "NULL"
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (dict, list)):
        return "'" + json.dumps(value, default=str).replace("'", "''") + "'::jsonb"
    return "'" + str(value).replace("'", "''") + "'"


def insert_statements(
    table: str,
    df: pd.DataFrame,
    columns: list[str],
    conflict_keys: list[str],
    *,
    batch_size: int = 200,
) -> list[str]:
    """Idempotent batched INSERTs for one table."""
    if df.empty:
        return []
    present = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        LOG.debug("%s: columns absent from the frame, left to default: %s", table, missing)

    updatable = [c for c in present if c not in conflict_keys]
    conflict = ", ".join(conflict_keys)
    if updatable:
        action = "DO UPDATE SET " + ", ".join(
            f"{c} = EXCLUDED.{c}" for c in updatable
        )
    else:
        action = "DO NOTHING"

    statements: list[str] = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start : start + batch_size]
        values = ",\n  ".join(
            "(" + ", ".join(sql_literal(row[c]) for c in present) + ")"
            for _, row in chunk.iterrows()
        )
        statements.append(
            f"INSERT INTO {SCHEMA}.{table} ({', '.join(present)})\nVALUES\n  {values}\n"
            f"ON CONFLICT ({conflict}) {action};"
        )
    return statements


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def build_core_statements(processed: Path, interim: Path, batch_size: int) -> dict:
    """Phase 1 tables, in foreign-key order."""
    out: dict[str, list[str]] = {}

    pg = processed / "pg"
    comp_path = interim / "compound_identity.parquet"
    obs_path = interim / "observations_labeled.parquet"

    if (pg / "sources.csv").exists():
        sources = pd.read_csv(pg / "sources.csv", na_values=[r"\N"])
        out["sources"] = insert_statements(
            "sources",
            sources,
            [
                "source_id", "source_database", "source_record_id",
                "source_url_or_reference", "retrieval_date", "source_version",
                "license", "file_sha256", "notes",
            ],
            ["source_id"],
            batch_size=batch_size,
        )

    if comp_path.exists():
        chem = pd.read_parquet(comp_path)
        chem = chem.assign(chemical_name=chem.get("source_names"))
        out["chemicals"] = insert_statements(
            "chemicals",
            chem,
            [
                "compound_id", "canonical_smiles", "isomeric_smiles", "inchi_key",
                "connectivity_key", "cas_number", "chemical_name",
                "molecular_formula", "molecular_weight", "scaffold_smiles",
                "scaffold_group", "n_stereocenters", "n_unassigned_stereocenters",
                "n_fragments", "structure_status", "standardization_version",
                "input_smiles", "resolution_source", "resolution_retrieved_date",
                "standardization_notes",
            ],
            ["compound_id"],
            batch_size=batch_size,
        )

    if obs_path.exists() and (pg / "sources.csv").exists():
        obs = pd.read_parquet(obs_path)
        obs = obs[obs["compound_id"].notna()].copy()
        sources = pd.read_csv(pg / "sources.csv", na_values=[r"\N"])
        src_map = dict(zip(sources["source_database"], sources["source_id"]))
        obs["source_id"] = obs["source_database"].map(src_map)
        obs["study_duration"] = obs.get("duration")
        obs["study_reference"] = obs.get("reference")

        out["experiments"] = insert_statements(
            "experiments",
            obs.drop_duplicates(subset=["experiment_id"]),
            [
                "experiment_id", "compound_id", "source_id", "source_record_id",
                "chemcode", "species", "species_raw", "sex", "sex_raw", "strain",
                "exposure_route", "study_duration", "dose_duration",
                "study_reference", "notes",
            ],
            ["experiment_id"],
            batch_size=batch_size,
        )

        lab_cfg = load_config("labels")
        frames = []
        for endpoint in lab_cfg["endpoints"]:
            col = f"label_{endpoint}"
            if col not in obs.columns:
                continue
            o = obs.copy()
            o["endpoint_name"] = endpoint
            o["harmonized_label"] = pd.to_numeric(o[col], errors="coerce").astype("Int64")
            o["tumor_type"] = o.get("tumor_endpoint")
            frames.append(o)
        if frames:
            out["observations"] = insert_statements(
                "observations",
                pd.concat(frames, ignore_index=True),
                [
                    "experiment_id", "target_organ", "tumor_type",
                    "author_assessment", "author_assessment_raw", "endpoint_name",
                    "harmonized_label", "dose_response", "tumor_incidence", "td50",
                    "label_policy_version", "curation_status",
                ],
                ["experiment_id", "endpoint_name"],
                batch_size=batch_size,
            )

    data_cfg = load_config("data")
    lab_cfg = load_config("labels")
    md_frames = []
    for endpoint in lab_cfg["endpoints"]:
        sp = processed / f"splits_{endpoint}.parquet"
        if not sp.exists():
            continue
        df = pd.read_parquet(sp)
        df = df.assign(
            dataset_version=data_cfg["dataset_version"], endpoint_name=endpoint
        )
        df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
        md_frames.append(df)
    if md_frames:
        out["model_datasets"] = insert_statements(
            "model_datasets",
            pd.concat(md_frames, ignore_index=True),
            [
                "dataset_version", "compound_id", "endpoint_definition",
                "endpoint_name", "label", "label_confidence", "is_conflicted",
                "n_experiments", "n_positive", "n_negative", "n_indeterminate",
                "split_group", "split_strategy", "split_assignment",
                "label_policy_version", "split_version",
            ],
            ["dataset_version", "endpoint_name", "split_strategy", "compound_id"],
            batch_size=batch_size,
        )

    ledger_path = processed / "curation_ledger.parquet"
    if ledger_path.exists():
        out["curation_ledger"] = insert_statements(
            "curation_ledger",
            pd.read_parquet(ledger_path),
            [
                "event_id", "event_type", "stage", "entity_type", "entity_id",
                "reason_code", "detail", "endpoint_name", "dataset_version",
                "label_policy_version", "standardization_version",
            ],
            ["event_id"],
            batch_size=batch_size,
        )
    return out


def build_benchmark_statements(processed: Path, endpoint: str, batch_size: int) -> dict:
    """Phase 2 benchmark runs, metrics and predictions."""
    runs_path = processed / "benchmark" / f"runs_{endpoint}.json"
    if not runs_path.exists():
        return {}
    runs = json.loads(runs_path.read_text(encoding="utf-8"))

    run_rows, metric_rows, pred_rows = [], [], []
    for r in runs:
        run_rows.append(
            {
                "run_id": r["run_id"],
                "model_id": r["model_id"],
                "model_name": r["model_name"],
                "representation": r.get("representation", "none"),
                "endpoint_name": r["endpoint_name"],
                "split_strategy": r["split_strategy"],
                "dataset_version": r.get("dataset_version"),
                "label_policy_version": r.get("label_policy_version"),
                "split_version": r.get("split_version"),
                "seed": r.get("seed"),
                "config": r.get("config", {}),
                "hyperparameters": r.get("hyperparameters", {}),
                "dependency_versions": r.get("dependency_versions", {}),
                "data_provenance": r.get("data_provenance"),
                "n_train": r.get("n_train"),
                "n_validation": r.get("n_validation"),
                "n_test": r.get("n_test"),
                "train_prevalence": r.get("train_prevalence"),
                "test_prevalence": r.get("test_prevalence"),
                "status": r.get("status"),
                "unavailable_reason": r.get("unavailable_reason"),
                "train_seconds": r.get("train_seconds"),
                "started_at": r.get("started_at"),
                "finished_at": r.get("finished_at"),
            }
        )
        for partition, res in (r.get("results") or {}).items():
            for m in res.get("metrics", []):
                metric_rows.append({"run_id": r["run_id"], "partition": partition, **m})
        for p in r.get("predictions", []):
            pred_rows.append(p)

    out: dict[str, list[str]] = {}
    if run_rows:
        out["benchmark_runs"] = insert_statements(
            "benchmark_runs",
            pd.DataFrame(run_rows),
            list(run_rows[0].keys()),
            ["run_id"],
            batch_size=batch_size,
        )
    if metric_rows:
        out["benchmark_metrics"] = insert_statements(
            "benchmark_metrics",
            pd.DataFrame(metric_rows),
            ["run_id", "partition", "metric_name", "value", "ci_low", "ci_high",
             "ci_method", "n_bootstrap"],
            ["run_id", "partition", "metric_name"],
            batch_size=batch_size,
        )
    if pred_rows:
        out["benchmark_predictions"] = insert_statements(
            "benchmark_predictions",
            pd.DataFrame(pred_rows),
            ["run_id", "compound_id", "partition", "y_true", "y_prob", "nn_similarity"],
            ["run_id", "compound_id", "partition"],
            batch_size=batch_size,
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["core", "benchmark"], default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    setup_logging()
    data_cfg = load_config("data")
    bench = load_config("benchmark")
    processed = resolve_path(data_cfg["paths"]["processed"])
    interim = resolve_path(data_cfg["paths"]["interim"])
    out_dir = Path(args.out) if args.out else processed / "supabase"
    out_dir.mkdir(parents=True, exist_ok=True)
    endpoint = args.endpoint or bench["task"]["endpoint_name"]

    man = StageManifest(stage=STAGE)
    man.config_versions = {
        "dataset_version": data_cfg["dataset_version"],
        "schema": SCHEMA,
    }

    statements: dict[str, list[str]] = {}
    if args.only != "benchmark":
        statements.update(build_core_statements(processed, interim, args.batch_size))
    if args.only != "core":
        statements.update(
            build_benchmark_statements(processed, endpoint, args.batch_size)
        )

    if not statements:
        LOG.error("nothing to sync; run the pipeline and benchmark first")
        return 2

    # Foreign keys constrain the order: a child row cannot land before its
    # parent. Writing them to numbered files makes the order explicit rather
    # than relying on dict iteration order holding.
    order = [
        "sources",
        "chemicals",
        "experiments",
        "observations",
        "model_datasets",
        "curation_ledger",
        "benchmark_runs",
        "benchmark_metrics",
        "benchmark_predictions",
    ]
    written, total = [], 0
    for i, table in enumerate([t for t in order if t in statements], start=1):
        stmts = statements[table]
        if not stmts:
            continue
        path = out_dir / f"{i:02d}_{table}.sql"
        path.write_text(
            f"-- {SCHEMA}.{table}: {len(stmts)} batch(es), idempotent upsert\n"
            + "\n\n".join(stmts)
            + "\n",
            encoding="utf-8",
        )
        man.add_output(path)
        written.append(path.name)
        total += len(stmts)
        LOG.info("%s: %d statement(s) -> %s", table, len(stmts), path.name)

    combined = out_dir / "apply_all.sql"
    combined.write_text(
        "-- Apply in this order; foreign keys require it.\n"
        "-- psql \"$SUPABASE_DB_URL\" -f data/processed/supabase/apply_all.sql\n"
        "BEGIN;\n"
        + "\n".join(
            (out_dir / name).read_text(encoding="utf-8") for name in written
        )
        + "\nCOMMIT;\n",
        encoding="utf-8",
    )
    man.add_output(combined)

    man.counts = {
        "tables": len(written),
        "statements": total,
        "files": written,
    }
    man.write()
    LOG.info("wrote %d statement(s) across %d table(s) to %s", total, len(written), out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
