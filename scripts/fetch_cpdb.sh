#!/usr/bin/env bash
#
# Download the Carcinogenic Potency Database (CPDB) tab-separated files.
#
# Standalone: needs only bash and curl (or wget). Run it on a machine with
# ordinary internet access, then point the pipeline at the result.
#
#   bash scripts/fetch_cpdb.sh                  # into data/raw/cpdb/
#   bash scripts/fetch_cpdb.sh /some/other/dir
#   MIRROR=toxplanet bash scripts/fetch_cpdb.sh # force one mirror
#
# CPDB is archival. NLM's catalogue entry states the dataset is no longer
# updated, so these files should be stable; the checksums printed at the end
# are how you prove that later.
#
# Sources
#   NLM (authoritative)  https://ftp.nlm.nih.gov/projects/SISFTP/CPDB/Data%20Files/
#   ToxPlanet (mirror)   https://files.toxplanet.com/cpdb/tab/   (adds .txt)
#   Catalogue            https://catalog.data.gov/dataset/carcinogenic-potency-database-cpdb
#
# Documentation — read before the first production run, because
# configs/cpdb_columns.yaml must be checked against the real field names:
#   https://ftp.nlm.nih.gov/projects/SISFTP/CPDB/Documentation/Tab_Separated_CPDB.pdf
#   https://files.toxplanet.com/cpdb/pdfs/Documentation_for_tab.pdf

set -uo pipefail

DEST="${1:-data/raw/cpdb}"
MIRROR="${MIRROR:-}"
MIN_BYTES=512

NLM_BASE="https://ftp.nlm.nih.gov/projects/SISFTP/CPDB/Data%20Files"
TOX_BASE="https://files.toxplanet.com/cpdb/tab"

# name|required
FILES=(
  "cpdb.ncintp.tab|required"     # NCI/NTP experimental records
  "cpdb.lit.tab|required"        # literature experimental records
  "cpdb.chemname.tab|required"   # names, chemcodes, CAS numbers
  "cpdb.ncntdose.tab|optional"   # NCI/NTP dose and tumour incidence
  "cpdb.litdose.tab|optional"    # literature dose and tumour incidence
  "cpdb.cit.tab|optional"        # literature citations
  "cpdb.species.tab|optional"    # appendix: species codes
  "cpdb.strain.tab|optional"     # appendix: strain codes
  "cpdb.route.tab|optional"      # appendix: exposure route codes
  "cpdb.tissue.tab|optional"     # appendix: target site codes
  "cpdb.tumor.tab|optional"      # appendix: histopathology codes
  "cpdb.journal.tab|optional"    # appendix: journal codes
)

# Per-attempt errors are suppressed: each file is tried against two mirrors,
# so a first-mirror failure is routine and this script reports the outcome
# itself. Set VERBOSE=1 to see the underlying transfer errors.
VERBOSE="${VERBOSE:-}"
_q() { if [ -n "$VERBOSE" ]; then "$@"; else "$@" 2>/dev/null; fi; }

if command -v curl >/dev/null 2>&1; then
  get() { _q curl -fsSL --retry 3 --retry-delay 2 --max-time 180 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
  get() { _q wget -q --tries=3 --timeout=180 -O "$2" "$1"; }
else
  echo "error: neither curl nor wget is available" >&2
  exit 1
fi

filesize() { wc -c <"$1" 2>/dev/null | tr -d ' '; }

# A small or HTML response is an error page, not data. Saving it would send a
# broken file into the pipeline, where it fails later and less clearly.
plausible() {
  local f="$1"
  [ -s "$f" ] || return 1
  [ "$(filesize "$f")" -ge "$MIN_BYTES" ] || return 1
  head -c 200 "$f" | tr '[:upper:]' '[:lower:]' | grep -q "<!doctype html\|<html" && return 1
  return 0
}

mkdir -p "$DEST"
echo "Downloading CPDB into: $DEST"
echo

ok_count=0; skip_count=0; fail_required=0; failed=()

for entry in "${FILES[@]}"; do
  name="${entry%%|*}"
  req="${entry##*|}"
  out="$DEST/$name"

  if [ -s "$out" ]; then
    printf '  %-22s already present (%s bytes)\n' "$name" "$(filesize "$out")"
    skip_count=$((skip_count + 1))
    continue
  fi

  got=0
  # NLM serves plain names; ToxPlanet appends .txt. Either way the file is
  # saved under its plain name so the parser sees one consistent set.
  for src in nlm toxplanet; do
    [ -n "$MIRROR" ] && [ "$MIRROR" != "$src" ] && continue
    case "$src" in
      nlm)       url="$NLM_BASE/$name" ;;
      toxplanet) url="$TOX_BASE/$name.txt" ;;
    esac
    tmp="$(mktemp)"
    if get "$url" "$tmp" && plausible "$tmp"; then
      mv "$tmp" "$out"
      printf '  %-22s %10s bytes  [%s]\n' "$name" "$(filesize "$out")" "$src"
      ok_count=$((ok_count + 1)); got=1
      break
    fi
    rm -f "$tmp"
  done

  if [ "$got" -eq 0 ]; then
    printf '  %-22s FAILED (%s)\n' "$name" "$req"
    failed+=("$name")
    [ "$req" = "required" ] && fail_required=$((fail_required + 1))
  fi
done

echo
echo "Downloaded $ok_count, already present $skip_count, failed ${#failed[@]}"

if [ "${#failed[@]}" -gt 0 ]; then
  echo
  echo "Could not fetch: ${failed[*]}"
  echo "Download those by hand from one of:"
  echo "  $NLM_BASE/"
  echo "  $TOX_BASE/            (strip the trailing .txt when saving)"
  echo "and place them in $DEST/"
fi

echo
echo "SHA-256 checksums (record these; the pipeline records them too):"
if command -v sha256sum >/dev/null 2>&1; then
  (cd "$DEST" && sha256sum ./*.tab 2>/dev/null) || true
elif command -v shasum >/dev/null 2>&1; then
  (cd "$DEST" && shasum -a 256 ./*.tab 2>/dev/null) || true
fi

if [ "$fail_required" -gt 0 ]; then
  echo
  echo "ERROR: $fail_required required file(s) missing. The pipeline cannot run." >&2
  exit 2
fi

cat <<'EOF'

Next steps
----------
1. Read the CPDB documentation and CHECK configs/cpdb_columns.yaml against the
   real field names. The defaults there are unverified; the parser fails loudly
   on a mismatch rather than guessing, but correcting the config is on you.

     python -m src.ingestion.download_cpdb --print-urls   # documentation links

2. Register the files and record their checksums:

     python -m src.ingestion.download_cpdb

3. Supply a structure resolution table at
   data/external/structure_resolution.tsv with columns
   chemcode or cas_number, smiles, source, retrieved_date.
   See docs/dataset_protocol.md — the pipeline does no network lookup, so the
   structure a compound had at freeze time is recorded rather than re-fetched.

4. Run the pipeline:

     python -m src.ingestion.parse_cpdb
     python -m src.chemistry.resolve_identity
     python -m src.curation.harmonize_labels
     python -m src.curation.audit_conflicts
     python -m src.datasets.build_dataset
     python -m src.datasets.split_dataset
     python -m src.curation.build_ledger
     python -m src.datasets.freeze_dataset --note "first real CPDB freeze"

5. Mark the provenance before benchmarking, or results are stamped synthetic:

     echo "REAL_CPDB" > data/processed/DATA_PROVENANCE
     python -m src.evaluation.runner --endpoint primary
EOF
