"""Step 7 - freeze Dataset v1.

Publishes an immutable version of the curated dataset. Once frozen, a
correction never edits these files: it produces a new version with an entry in
the change log.

The freeze writes a lock file recording the SHA-256 of every published file
plus the config versions in force, so a later run can prove whether anything
drifted.

Outputs:
    data/processed/frozen/{dataset_version}/...
    data/processed/frozen/{dataset_version}/DATASET_LOCK.json
    data/processed/frozen/CHANGELOG.md   (appended)

Usage:
    python -m src.datasets.freeze_dataset --note "initial Phase 1 freeze"
    python -m src.datasets.freeze_dataset --verify v1
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from src.common import (
    StageManifest,
    load_config,
    resolve_path,
    setup_logging,
    sha256_file,
    utc_now,
)

LOG = logging.getLogger("freeze_dataset")
STAGE = "09_freeze"

PUBLISH_GLOBS = [
    "model_dataset_*.parquet",
    "splits_*.parquet",
    "split_report_*.json",
    "audit_report.json",
    "audit_report.md",
    "conflicts_detail.parquet",
    "curation_ledger.parquet",
    "curation_ledger_summary.json",
]
INTERIM_PUBLISH = [
    "carcinogenicity_experiments.parquet",
    "compound_identity.parquet",
    "observations_labeled.parquet",
    "compound_labels.parquet",
    "structure_review.parquet",
]


def verify(frozen_dir: Path) -> int:
    """Re-check every published file against the lock."""
    lock_path = frozen_dir / "DATASET_LOCK.json"
    if not lock_path.exists():
        LOG.error("no lock file at %s", lock_path)
        return 2
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    drift = []
    for entry in lock["files"]:
        p = frozen_dir / entry["name"]
        if not p.exists():
            drift.append((entry["name"], "missing"))
            continue
        actual = sha256_file(p)
        if actual != entry["sha256"]:
            drift.append((entry["name"], f"sha256 {actual[:12]} != {entry['sha256'][:12]}"))

    if drift:
        LOG.error("frozen dataset has drifted:")
        for name, why in drift:
            LOG.error("  %s: %s", name, why)
        return 5

    LOG.info(
        "verified: %d file(s) match the lock for dataset_version=%s",
        len(lock["files"]),
        lock["dataset_version"],
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--note", default="", help="change-log entry for this freeze")
    parser.add_argument("--verify", default=None, metavar="VERSION")
    parser.add_argument("--force", action="store_true", help="overwrite an existing freeze")
    args = parser.parse_args(argv)

    setup_logging()
    data_cfg = load_config("data")
    lab_cfg = load_config("labels")
    split_cfg = load_config("splits")
    processed = resolve_path(data_cfg["paths"]["processed"])
    interim = resolve_path(data_cfg["paths"]["interim"])
    frozen_root = processed / "frozen"

    if args.verify:
        return verify(frozen_root / args.verify)

    version = data_cfg["dataset_version"]
    target = frozen_root / version

    if target.exists() and not args.force:
        LOG.error(
            "dataset_version %s is already frozen at %s.\n"
            "A frozen dataset is immutable. Bump dataset_version in "
            "configs/data.yaml and re-run, or pass --force only if this freeze "
            "was never distributed.",
            version,
            target,
        )
        return 3

    man = StageManifest(stage=STAGE)
    man.config_versions = {
        "dataset_version": version,
        "label_policy_version": lab_cfg["label_policy_version"],
        "split_version": split_cfg["split_version"],
        "random_seed": split_cfg["random_seed"],
    }

    target.mkdir(parents=True, exist_ok=True)
    published: list[dict] = []

    for pattern in PUBLISH_GLOBS:
        for src in sorted(processed.glob(pattern)):
            if "frozen" in src.parts:
                continue
            dst = target / src.name
            shutil.copy2(src, dst)
            published.append({"name": src.name, "sha256": sha256_file(dst), "bytes": dst.stat().st_size})
            man.add_input(src)

    for fname in INTERIM_PUBLISH:
        src = interim / fname
        if src.exists():
            dst = target / src.name
            shutil.copy2(src, dst)
            published.append({"name": src.name, "sha256": sha256_file(dst), "bytes": dst.stat().st_size})
            man.add_input(src)
        else:
            man.warn(f"expected interim file not present at freeze time: {fname}")

    if not published:
        LOG.error("nothing to freeze — run the pipeline first")
        return 2

    lock = {
        "dataset_version": version,
        "frozen_at": utc_now(),
        "note": args.note,
        "label_policy_version": lab_cfg["label_policy_version"],
        "endpoint_definition": lab_cfg["endpoint_definition"].strip(),
        "split_version": split_cfg["split_version"],
        "random_seed": split_cfg["random_seed"],
        "environment": man.environment,
        "files": published,
    }
    lock_path = target / "DATASET_LOCK.json"
    with lock_path.open("w", encoding="utf-8") as fh:
        json.dump(lock, fh, indent=2)
        fh.write("\n")

    changelog = frozen_root / "CHANGELOG.md"
    header = "" if changelog.exists() else "# Dataset change log\n\n"
    entry = (
        f"## {version} — {lock['frozen_at']}\n\n"
        f"- Label policy: `{lab_cfg['label_policy_version']}`\n"
        f"- Split version: `{split_cfg['split_version']}`, seed `{split_cfg['random_seed']}`\n"
        f"- Files: {len(published)}\n"
        f"- Note: {args.note or '(none)'}\n\n"
    )
    with changelog.open("a", encoding="utf-8") as fh:
        fh.write(header + entry)

    man.add_output(lock_path)
    man.counts = {"files_frozen": len(published), "dataset_version": version}
    man.write()

    LOG.info("froze %d file(s) as dataset_version=%s at %s", len(published), version, target)
    LOG.info("verify later with: python -m src.datasets.freeze_dataset --verify %s", version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
