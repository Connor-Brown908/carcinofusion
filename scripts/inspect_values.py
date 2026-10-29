#!/usr/bin/env python3
"""Dump the distinct values of every vocabulary column in the raw CPDB files.

Why this exists
---------------
Every data failure in this project has the same shape: the pipeline assumed
what a column contains, the real file contained something else, and the
mismatch showed up far downstream as a dataset that came out small. The
column mapper fixed that for column *names*. This fixes it for column
*values*.

Run it once on the downloaded files, paste the output, and the label policy
gets written from the data instead of from a guess.

Deliberately stdlib-only: this runs on the raw tab-separated files before
pandas or RDKit are needed, so a missing dependency cannot block it.

Usage
-----
    python3 scripts/inspect_values.py data/raw/cpdb
    python3 scripts/inspect_values.py data/raw/cpdb --max-values 60
    python3 scripts/inspect_values.py data/raw/cpdb --column opinion
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

# Columns whose value set is a vocabulary the pipeline must agree with.
# Names are lowercased before matching, and every alias of a concept is
# listed, because CPDB's tables are not internally consistent.
VOCABULARY_COLUMNS: dict[str, tuple[str, ...]] = {
    "author assessment / opinion": (
        "opinion", "auop", "au_op", "authorop", "author_opinion",
        "assessment", "result", "call",
    ),
    "species": ("species", "spec", "sp"),
    "sex": ("sex", "gender"),
    "strain": ("strain",),
    "route": ("route", "rt", "xporoute", "exposure_route"),
    "target site / organ": ("tissue", "site", "target_site", "organ"),
    "tumour type": ("tumor", "tumour", "tumor_type", "histology"),
    "dosing duration": ("xpotime", "xpo", "dosing_duration"),
    "study duration": ("xprtime", "dur", "study_duration"),
    "inadequate flag": ("inad", "inadequate", "inadeq"),
    "significance / notes": ("notes", "note", "comment", "flag", "flags"),
}

# A column with more distinct values than this is free text (a chemical name,
# a reference), not a vocabulary. Reporting it would be noise.
FREE_TEXT_THRESHOLD = 400

# Never whitelist extensions. The first version of this script accepted
# .txt/.tsv/.csv/.dat and so could not read the .tab files that
# scripts/fetch_cpdb.sh writes — the same assume-then-fail-quietly mistake
# this script exists to prevent. A file is data if its first line parses as
# delimited text; that is decided by looking, not by its name.
SKIP_SUFFIXES = {
    ".pdf", ".zip", ".gz", ".tar", ".bz2", ".xz", ".7z", ".parquet",
    ".xlsx", ".xls", ".doc", ".docx", ".png", ".jpg", ".jpeg", ".gif",
    ".sqlite", ".db", ".pyc", ".so",
}


def looks_delimited(path: Path) -> tuple[bool, str]:
    """Decide from content whether this file is a delimited table.

    Returns (is_data, reason). The reason is printed for skipped files so a
    missing table is never a silent omission.
    """
    if path.suffix.lower() in SKIP_SUFFIXES:
        return False, f"binary or archive ({path.suffix})"
    try:
        with path.open("rb") as fh:
            head = fh.read(8192)
    except OSError as exc:
        return False, f"unreadable: {exc}"
    if not head.strip():
        return False, "empty"
    if b"\x00" in head:
        return False, "binary (contains null bytes)"

    text = head.decode("utf-8", errors="replace")
    first = text.splitlines()[0] if text.splitlines() else ""
    if first.lstrip().lower().startswith(("<!doctype", "<html", "<?xml")):
        return False, "HTML or XML, not data (a saved error page?)"
    if not any(d in first for d in ("\t", ",", ";", "|")):
        return False, "no delimiter on the first line"
    return True, "delimited text"


def sniff_delimiter(first_line: str) -> str:
    counts = {d: first_line.count(d) for d in ("\t", ",", ";", "|")}
    best = max(counts, key=lambda d: counts[d])
    return best if counts[best] else "\t"


def concept_for(column: str) -> str | None:
    key = column.strip().lower().replace(" ", "_")
    for concept, aliases in VOCABULARY_COLUMNS.items():
        if key in aliases:
            return concept
    return None


def inspect_file(
    path: Path,
    *,
    only_column: str | None,
    max_values: int,
) -> bool:
    """Print the vocabulary of one file. Returns False if it was unreadable."""
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            first = fh.readline()
            if not first.strip():
                print(f"  (empty file)")
                return False
            delim = sniff_delimiter(first)
            fh.seek(0)
            reader = csv.DictReader(fh, delimiter=delim)
            if not reader.fieldnames:
                print("  (no header row)")
                return False

            headers = [h for h in reader.fieldnames if h]
            counters: dict[str, Counter] = {h: Counter() for h in headers}
            n_rows = 0
            for row in reader:
                n_rows += 1
                for h in headers:
                    counters[h][(row.get(h) or "").strip()] += 1
    except OSError as exc:
        print(f"  (could not read: {exc})")
        return False

    print(f"  {n_rows:,} data rows, {len(headers)} columns, "
          f"delimiter={'tab' if delim == chr(9) else repr(delim)}")

    reported = 0
    for h in headers:
        if only_column and h.strip().lower() != only_column.strip().lower():
            continue
        concept = concept_for(h)
        if not only_column and concept is None:
            continue

        counter = counters[h]
        distinct = len(counter)
        if not only_column and distinct > FREE_TEXT_THRESHOLD:
            print(f"\n  {h}  ->  {concept}")
            print(f"    {distinct:,} distinct values: free text, not a "
                  f"vocabulary. Skipped.")
            reported += 1
            continue

        reported += 1
        label = f"  {h}" + (f"  ->  {concept}" if concept else "")
        print(f"\n{label}")
        print(f"    {distinct:,} distinct value(s)")
        blank = counter.get("", 0)
        for value, n in counter.most_common(max_values):
            shown = "(blank)" if value == "" else repr(value)
            pct = 100.0 * n / n_rows if n_rows else 0.0
            print(f"      {shown:<42} {n:>8,}  {pct:>5.1f}%")
        if distinct > max_values:
            print(f"      ... and {distinct - max_values:,} more "
                  f"(raise --max-values to see them)")
        if blank and blank == n_rows:
            print("      NOTE: this column is entirely blank in this file.")

    if reported == 0:
        if only_column:
            print(f"  (no column named {only_column!r} in this file)")
        else:
            print("  (no vocabulary columns in this file)")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Dump distinct values of CPDB vocabulary columns.",
    )
    ap.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("data/raw/cpdb"),
        help="directory holding the downloaded CPDB files "
             "(default: data/raw/cpdb)",
    )
    ap.add_argument(
        "--max-values",
        type=int,
        default=40,
        help="values listed per column, most frequent first (default 40)",
    )
    ap.add_argument(
        "--column",
        default=None,
        help="inspect only this exact column name, in every file, whether or "
             "not it is a known vocabulary column",
    )
    args = ap.parse_args(argv)

    directory: Path = args.directory
    if not directory.is_dir():
        print(f"error: {directory} is not a directory.", file=sys.stderr)
        print("Download the CPDB files first:  bash scripts/fetch_cpdb.sh",
              file=sys.stderr)
        return 2

    # Recurse: fetch_cpdb.sh writes flat, but an archive extracted by hand
    # leaves a subfolder, and a table one level down is still a table.
    candidates = sorted(p for p in directory.rglob("*") if p.is_file())
    files: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    for path in candidates:
        is_data, reason = looks_delimited(path)
        (files if is_data else skipped).append(
            path if is_data else (path, reason)  # type: ignore[arg-type]
        )

    if not files:
        print(f"error: no delimited data files under {directory}.",
              file=sys.stderr)
        if skipped:
            # Say what is actually there. "No data files" with no inventory
            # is the kind of dead-end message that wastes a round trip.
            print(f"\n{len(skipped)} file(s) present but not usable as data:",
                  file=sys.stderr)
            for path, reason in skipped[:25]:
                rel = path.relative_to(directory)
                print(f"  {str(rel):<40} {path.stat().st_size:>12,} bytes"
                      f"   {reason}", file=sys.stderr)
            if len(skipped) > 25:
                print(f"  ... and {len(skipped) - 25} more", file=sys.stderr)
        else:
            print(f"The directory is empty.", file=sys.stderr)
        print("\nDownload them with:  bash scripts/fetch_cpdb.sh",
              file=sys.stderr)
        print("Already downloaded elsewhere? Pass the directory:",
              file=sys.stderr)
        print("  python3 scripts/inspect_values.py <that directory>",
              file=sys.stderr)
        return 2

    print("=" * 72)
    print("CPDB VALUE INSPECTOR")
    print("=" * 72)
    print(f"Directory: {directory.resolve()}")
    print(f"Data files: {len(files)}")
    if skipped:
        print(f"Skipped:    {len(skipped)} "
              f"({', '.join(sorted({r for _, r in skipped}))})")
    print()
    print("Paste this whole output back. The label policy in")
    print("configs/labels.yaml will be written from these values rather than")
    print("assumed, which is what the pipeline now refuses to let me do.")

    ok = 0
    for path in files:
        print()
        print("-" * 72)
        print(f"{path.name}  ({path.stat().st_size:,} bytes)")
        print("-" * 72)
        if inspect_file(
            path, only_column=args.column, max_values=args.max_values
        ):
            ok += 1

    print()
    print("=" * 72)
    print(f"Read {ok} of {len(files)} files.")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
