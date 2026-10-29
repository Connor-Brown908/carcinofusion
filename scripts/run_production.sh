#!/usr/bin/env bash
#
# Run the curation pipeline on REAL CPDB data.
#
# Distinct from tests/run_pipeline.sh, which deletes data/interim and
# data/processed and rebuilds the synthetic fixture. Running that against real
# data would destroy it. This script deletes nothing and builds no fixture.
#
#   bash scripts/run_production.sh            # stop before the freeze for review
#   bash scripts/run_production.sh --freeze   # also freeze, after you have reviewed
#   bash scripts/run_production.sh --from 4   # resume at stage 4
#
# Stages:
#   1 register   record checksums of the raw files
#   2 parse      Dataset A, one row per experimental record
#   3 identity   Dataset B, standardised structures
#   4 labels     apply the label policy
#   5 audit      the report you must read before freezing
#   6 build      model-ready table per endpoint
#   7 split      locked random and scaffold partitions
#   8 ledger     every exclusion and transformation, enumerated
#   9 freeze     immutable dataset version  (only with --freeze)

set -uo pipefail
cd "$(dirname "$0")/.."

FREEZE=0
FROM=1
while [ $# -gt 0 ]; do
  case "$1" in
    --freeze) FREEZE=1; shift ;;
    --from) FROM="${2:-1}"; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 64 ;;
  esac
done

run_stage() {
  local n="$1" name="$2"; shift 2
  [ "$n" -lt "$FROM" ] && { printf '  [%d] %-10s skipped\n' "$n" "$name"; return 0; }
  printf '\n=== [%d] %s ===\n' "$n" "$name"
  if ! "$@"; then
    echo >&2
    echo "Stage $n ($name) failed. Nothing after it has run." >&2
    echo "Fix the cause, then resume with: bash scripts/run_production.sh --from $n" >&2
    exit 1
  fi
}

# Guard against running on fixture leftovers and calling the result real.
if [ -d data/raw/cpdb_test ]; then
  echo "WARNING: data/raw/cpdb_test exists. This script reads data/raw/cpdb;" >&2
  echo "         make sure that holds the real download, not fixture output." >&2
fi

echo "Pre-flight: checking the column mapping against your files"
if ! python3 scripts/check_columns.py >/tmp/cf_cols.$$ 2>&1; then
  sed -n '/required column/,$p' /tmp/cf_cols.$$
  rm -f /tmp/cf_cols.$$
  echo >&2
  echo "Column mapping unresolved. Fix configs/cpdb_columns.yaml first:" >&2
  echo "  python3 scripts/check_columns.py" >&2
  exit 1
fi
rm -f /tmp/cf_cols.$$
echo "  column mapping OK"

# Check the freeze BEFORE the expensive stages, not at stage 9 after them.
# The common trap: the README suggests running the fixture pipeline first to
# prove the install, and that freezes a version from synthetic data. The real
# run then collides with it an hour later.
if [ "$FREEZE" -eq 1 ]; then
  VER_PRE="$(python3 -c 'import sys;sys.path.insert(0,".");from src.common import load_config;print(load_config("data")["dataset_version"])' 2>/dev/null)"
  if [ -n "${VER_PRE:-}" ] && [ -d "data/processed/frozen/$VER_PRE" ]; then
    echo >&2
    echo "dataset_version '$VER_PRE' is already frozen at data/processed/frozen/$VER_PRE" >&2
    echo >&2
    if grep -q "fixture" "data/processed/frozen/$VER_PRE/DATASET_LOCK.json" 2>/dev/null; then
      echo "  Its note mentions a fixture, so it is almost certainly the synthetic" >&2
      echo "  run from tests/run_pipeline.sh. Delete it and continue:" >&2
      echo "    rm -rf data/processed/frozen/$VER_PRE" >&2
    else
      echo "  Bump dataset_version in configs/data.yaml. A frozen dataset is" >&2
      echo "  immutable; a correction becomes a new version with a change log." >&2
    fi
    exit 1
  fi
fi

RES=data/external/structure_resolution.tsv
if [ ! -s "$RES" ]; then
  echo >&2
  echo "Missing $RES — nothing can be standardised without structures." >&2
  echo "  python3 scripts/build_resolution_table.py" >&2
  exit 1
fi
echo "  structure resolution table: $(( $(wc -l <"$RES") - 1 )) row(s)"

run_stage 1 register python3 -m src.ingestion.download_cpdb
run_stage 2 parse    python3 -m src.ingestion.parse_cpdb
run_stage 3 identity python3 -m src.chemistry.resolve_identity
run_stage 4 labels   python3 -m src.curation.harmonize_labels
run_stage 5 audit    python3 -m src.curation.audit_conflicts
run_stage 6 build    python3 -m src.datasets.build_dataset
run_stage 7 split    python3 -m src.datasets.split_dataset
run_stage 8 ledger   python3 -m src.curation.build_ledger

echo
echo "========================================================================"
python3 - <<'PY'
import json, pandas as pd
from pathlib import Path
P, I = Path("data/processed"), Path("data/interim")

def n(p):
    return len(pd.read_parquet(p)) if p.exists() else 0

print("DATASET SUMMARY")
print()
if (I / "carcinogenicity_experiments.parquet").exists():
    e = pd.read_parquet(I / "carcinogenicity_experiments.parquet")
    print(f"  experimental records      {len(e):>7,}")
    print(f"  distinct source chemicals {e['chemcode'].nunique():>7,}")
if (I / "compound_identity.parquet").exists():
    c = pd.read_parquet(I / "compound_identity.parquet")
    print(f"  standardised compounds    {len(c):>7,}")
    print(f"  scaffold groups           {c['scaffold_group'].nunique():>7,}")
print(f"  structures in review      {n(I / 'structure_review.parquet'):>7,}")

for ep in ("primary", "sensitivity"):
    p = P / f"model_dataset_{ep}.parquet"
    if not p.exists():
        continue
    d = pd.read_parquet(p)
    el = d[d.training_eligibility == "eligible"]
    pos, neg = int((el.label == 1).sum()), int((el.label == 0).sum())
    tot = pos + neg
    print(f"\n  endpoint: {ep}")
    print(f"    eligible for training   {tot:>7,}   positive {pos:,}  negative {neg:,}")
    if tot:
        print(f"    prevalence              {pos / tot:>7.3f}   <- the PR-AUC baseline")
    print(f"    excluded, conflicted    {int((d.training_eligibility == 'excluded_conflicting_evidence').sum()):>7,}")
    print(f"    excluded, unresolved    {int((d.training_eligibility == 'excluded_unresolved_label').sum()):>7,}")

for ep in ("primary",):
    r = P / f"split_report_{ep}.json"
    if not r.exists():
        continue
    rep = json.loads(r.read_text())
    for strat, info in rep["strategies"].items():
        parts = info["partitions"]
        print(f"\n  {ep} / {strat} split")
        for name in ("train", "validation", "test"):
            q = parts.get(name, {})
            print(f"    {name:<11} {q.get('compounds', 0):>6,}  pos {q.get('positive', 0):>5,}  neg {q.get('negative', 0):>5,}")
        sim = info["nearest_neighbour_similarity_to_train"].get("test", {})
        if sim.get("n"):
            print(f"    test nearest-neighbour similarity to train: median {sim['median']:.3f}, "
                  f"{sim['fraction_identical_fp'] * 100:.1f}% identical fingerprint")
PY
echo "========================================================================"

cat <<'EOF'

READ BEFORE FREEZING
--------------------
  data/processed/audit_report.md          counts, resolution, conflicts
  data/processed/conflicts_detail.parquet which compounds disagree, and on what
  data/external/structure_unresolved.tsv  chemicals with no structure
  data/processed/curation_ledger.parquet  every exclusion, with its reason

What to look for:
  - Conflicts of type within_species_and_sex. The same species and sex gave
    both answers, which species differences cannot explain, so either a study
    is unreliable or the curation is wrong.
  - A large excluded-unresolved count. min_negative_records is 2, so
    compounds resting on a single negative bioassay land here by design.
  - Test-set nearest-neighbour similarity. A high median, or any identical
    fingerprints, means the held-out set is not measuring generalisation.
EOF

if [ "$FREEZE" -eq 1 ]; then
  run_stage 9 freeze python3 -m src.datasets.freeze_dataset --note "first real CPDB freeze"
  VER="$(python3 -c 'import sys;sys.path.insert(0,".");from src.common import load_config;print(load_config("data")["dataset_version"])')"
  python3 -m src.datasets.freeze_dataset --verify "$VER"
  cat <<EOF

Frozen as $VER. Benchmark next:

  echo "REAL_CPDB" > data/processed/DATA_PROVENANCE
  python3 -m src.evaluation.runner   --endpoint primary
  python3 -m src.evaluation.analysis --endpoint primary
EOF
else
  cat <<'EOF'

Not frozen. Review the files above, then:

  bash scripts/run_production.sh --freeze --from 9
EOF
fi
