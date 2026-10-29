#!/usr/bin/env python3
"""Check configs/cpdb_columns.yaml against the real CPDB file headers.

The parser refuses to guess: if a declared column is absent it fails and names
it. That is the right behaviour, but it fails one file at a time and only once
you run the pipeline. This checks every file up front and says exactly what to
change.

Reports, per file:
  matched   canonical name -> the source column it found, and via which alias
  MISSING   declared but absent (fatal if the column is required)
  unmapped  present in the file but named nowhere in the config

An unmapped column is not necessarily a problem — CPDB carries fields this
pipeline does not use — but it is worth reading, because a column you wanted
usually turns up there under a name the config did not anticipate.

Usage:
    python3 scripts/check_columns.py
    python3 scripts/check_columns.py --raw-dir data/raw/cpdb
    python3 scripts/check_columns.py --show-sample 3
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import load_config  # noqa: E402

# Which config block governs which file.
FILE_BLOCKS = {
    "ncintp": "experiments",
    "lit": "experiments",
    "chemname": "chemname",
    "ncntdose": "dose",
    "litdose": "dose",
}


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="latin-1", newline="") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if row and any(c.strip() for c in row):
                return [c.strip() for c in row]
    return []


def read_sample(path: Path, n: int) -> list[list[str]]:
    rows = []
    with path.open("r", encoding="latin-1", newline="") as fh:
        for i, row in enumerate(csv.reader(fh, delimiter="\t")):
            if i == 0:
                continue
            rows.append(row)
            if len(rows) >= n:
                break
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--show-sample", type=int, default=0, help="print N data rows per file")
    args = ap.parse_args(argv)

    data_cfg = load_config("data")
    colcfg = load_config("cpdb_columns")
    src = data_cfg["sources"]["cpdb"]
    raw_dir = Path(args.raw_dir or src["raw_dir"])

    if not raw_dir.exists():
        print(f"error: {raw_dir} does not exist. Run: bash scripts/fetch_cpdb.sh", file=sys.stderr)
        return 2

    problems = 0
    checked = 0

    for key, block_name in FILE_BLOCKS.items():
        spec = src["files"].get(key)
        if spec is None:
            continue
        path = raw_dir / spec["filename"]
        if not path.exists():
            mark = "REQUIRED" if spec.get("required") else "optional"
            print(f"\n{'=' * 72}\n{spec['filename']}  -- not present ({mark})")
            if spec.get("required"):
                problems += 1
            continue

        checked += 1
        header = read_header(path)
        block = colcfg[block_name]
        required = block.get("required", [])
        lower = {h.lower(): h for h in header}

        print(f"\n{'=' * 72}")
        print(f"{spec['filename']}   ({block_name} block, {len(header)} columns)")
        print("=" * 72)

        used: set[str] = set()
        for canonical, aliases in block.items():
            if canonical == "required":
                continue
            found = None
            via = None
            for alias in aliases:
                if alias.strip().lower() in lower:
                    found = lower[alias.strip().lower()]
                    via = alias
                    break
            is_required = canonical in required
            if found:
                used.add(found)
                note = "" if via == found else f"  (matched alias '{via}')"
                print(f"  ok       {canonical:20s} -> {found}{note}")
            else:
                flag = "MISSING " if is_required else "absent  "
                print(f"  {flag} {canonical:20s} -> none of {aliases}")
                if is_required:
                    problems += 1

        unmapped = [h for h in header if h not in used]
        if unmapped:
            print(f"\n  unmapped columns in the file ({len(unmapped)}):")
            for h in unmapped:
                print(f"    {h}")

        if args.show_sample:
            print(f"\n  first {args.show_sample} data row(s):")
            for row in read_sample(path, args.show_sample):
                pairs = ", ".join(
                    f"{h}={v!r}" for h, v in list(zip(header, row))[:8]
                )
                print(f"    {pairs}")

    print(f"\n{'=' * 72}")
    if problems:
        print(
            f"{problems} required column(s) unresolved.\n\n"
            "Fix configs/cpdb_columns.yaml by adding the real header name to the\n"
            "alias list for each MISSING canonical column, then re-run this check.\n"
            "The parser will refuse to run until they resolve, which is deliberate:\n"
            "a wrong mapping is a corrupt dataset, not a warning."
        )
        return 1

    print(
        f"All required columns resolved across {checked} file(s).\n\n"
        "Read the unmapped lists above before proceeding — a field you want\n"
        "usually appears there under a name the config did not anticipate.\n"
        "Then: python -m src.ingestion.parse_cpdb"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
