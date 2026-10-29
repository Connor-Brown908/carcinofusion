#!/usr/bin/env bash
# End-to-end Phase 1 run against the synthetic fixture.
#
# This exercises every stage in the order the protocol specifies. It uses
# invented data: it proves the pipeline works, not anything about chemistry.
#
# Usage:  bash tests/run_pipeline.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== clean ==="
rm -rf data/interim/* data/processed/* manifests/*.json data/raw/cpdb_test
mkdir -p data/interim data/processed manifests

echo "=== 0. build fixture ==="
python3 tests/make_fixture.py --outdir data/raw/cpdb_test \
    --resolution data/external/structure_resolution_test.tsv

echo "=== 1. acquire (register raw files) ==="
python3 -m src.ingestion.download_cpdb || echo "  (expected: real CPDB files absent)"

echo "=== 2. parse ==="
python3 -m src.ingestion.parse_cpdb --raw-dir data/raw/cpdb_test

echo "=== 3. resolve identity ==="
python3 -m src.chemistry.resolve_identity \
    --resolution data/external/structure_resolution_test.tsv

echo "=== 4. harmonize labels ==="
python3 -m src.curation.harmonize_labels

echo "=== 5. audit ==="
python3 -m src.curation.audit_conflicts

echo "=== 6. build dataset ==="
python3 -m src.datasets.build_dataset --with-descriptors

echo "=== 7. split ==="
python3 -m src.datasets.split_dataset

echo "=== 8. curation ledger ==="
python3 -m src.curation.build_ledger

echo "=== 9. freeze ==="
python3 -m src.datasets.freeze_dataset --note "fixture end-to-end run" --force

echo "=== 10. verify freeze ==="
# Read the version from config rather than hardcoding it: a policy change
# bumps dataset_version, and a hardcoded value here would verify a version
# that no longer exists.
DATASET_VERSION="$(python3 -c 'import sys; sys.path.insert(0,"."); from src.common import load_config; print(load_config("data")["dataset_version"])')"
python3 -m src.datasets.freeze_dataset --verify "$DATASET_VERSION"

echo "=== 11. export for postgres ==="
python3 -m src.datasets.export_for_postgres

echo "=== 12. tests ==="
python3 -m pytest tests/ -q

echo
echo "Pipeline completed."
