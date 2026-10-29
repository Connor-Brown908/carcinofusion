#!/usr/bin/env python3
"""Build data/external/structure_resolution.tsv — CPDB chemicals to SMILES.

CPDB identifies chemicals by `chemcode` and CAS number but the main tables
carry no structures, and the pipeline deliberately performs no network lookup
of its own (so the structure a compound had at freeze time is recorded rather
than silently re-fetched later). This script is where that lookup happens,
once, explicitly, with its source and date written beside every row.

Three sources, tried in order per chemical:

  1. A local CPDB structure file, if you have one. CPDB shipped
     `chemical-structures.tab` (molecular weight, SMILES, InChI, sourced from
     EPA DSSTox). Its original toxnet.nlm.nih.gov URLs are dead since TOXNET
     was retired in 2019, so pass a local copy with --cpdb-structures if you
     can find one. It was last updated in 2007, so treat it as a starting
     point rather than authoritative.
  2. PubChem, by CAS number, via PUG REST.
  3. PubChem, by chemical name, for chemicals whose CAS lookup fails.

Nothing is guessed. A chemical that resolves nowhere is written to the
unresolved report with the reason, and the pipeline will hold it in
structure_review.parquet rather than give it a label.

Usage:
    python scripts/build_resolution_table.py
    python scripts/build_resolution_table.py --cpdb-structures chemical-structures.tab
    python scripts/build_resolution_table.py --limit 50        # try a few first
    python scripts/build_resolution_table.py --no-network      # local sources only
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

LOG = logging.getLogger("build_resolution")

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PROPS = "CanonicalSMILES,IsomericSMILES,InChI,InChIKey,MolecularFormula,MolecularWeight"
USER_AGENT = "carcinofusion-dataset-pipeline/1.0 (research use)"

# PubChem asks for no more than 5 requests a second and 400 a minute. Going
# faster gets you throttled or blocked, and this is a one-time job, so the
# polite rate is also the fast one overall.
MIN_INTERVAL = 0.25
_last_call = 0.0


def throttle() -> None:
    global _last_call
    wait = MIN_INTERVAL - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


def pubchem_get(path: str, *, timeout: int = 30, retries: int = 3) -> dict | None:
    """One PUG REST call. Returns parsed JSON, or None for a clean miss."""
    url = f"{PUBCHEM}/{path}"
    for attempt in range(retries):
        throttle()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None  # a genuine miss, not an error worth retrying
            if exc.code in (429, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            LOG.debug("HTTP %s for %s", exc.code, url)
            return None
        except Exception as exc:  # noqa: BLE001
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            LOG.debug("failed %s: %s", url, exc)
            return None
    return None


def first_property_row(payload: dict | None) -> dict | None:
    try:
        rows = payload["PropertyTable"]["Properties"]
        return rows[0] if rows else None
    except (KeyError, IndexError, TypeError):
        return None


def lookup_by_cas(cas: str) -> tuple[dict | None, str]:
    """CAS numbers are registered in PubChem as synonyms, so name search works."""
    enc = urllib.parse.quote(cas, safe="")
    row = first_property_row(pubchem_get(f"compound/name/{enc}/property/{PROPS}/JSON"))
    return (row, "pubchem_cas") if row else (None, "")


def lookup_by_name(name: str) -> tuple[dict | None, str]:
    enc = urllib.parse.quote(name, safe="")
    row = first_property_row(pubchem_get(f"compound/name/{enc}/property/{PROPS}/JSON"))
    return (row, "pubchem_name") if row else (None, "")


def read_chemname(path: Path) -> list[dict]:
    """Read cpdb.chemname.tab, tolerating the usual header spellings."""
    with path.open("r", encoding="latin-1", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        return []

    lower = {k.strip().lower(): k for k in rows[0]}

    def pick(*names):
        for n in names:
            if n in lower:
                return lower[n]
        return None

    code_col = pick("chemcode", "chem_code")
    cas_col = pick("casno", "cas", "cas_no")
    name_col = pick("chemname", "name", "chemical_name")
    if code_col is None:
        raise SystemExit(
            f"{path.name}: no chemcode column found. Headers were: {list(rows[0])}"
        )

    out, seen = [], set()
    for r in rows:
        code = (r.get(code_col) or "").strip()
        if not code or code in seen:
            continue
        seen.add(code)
        out.append(
            {
                "chemcode": code,
                "cas_number": (r.get(cas_col) or "").strip() if cas_col else "",
                "chemical_name": (r.get(name_col) or "").strip() if name_col else "",
            }
        )
    return out


def read_cpdb_structures(path: Path) -> dict[str, dict]:
    """Index CPDB's own structure file by CAS and by lower-cased name."""
    index: dict[str, dict] = {}
    with path.open("r", encoding="latin-1", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            lower = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
            smiles = lower.get("smiles") or lower.get("smile")
            if not smiles:
                continue
            rec = {
                "smiles": smiles,
                "inchi": lower.get("inchi", ""),
                "molecular_weight": lower.get("mw") or lower.get("molecular_weight", ""),
            }
            for key in ("casno", "cas", "cas_no"):
                if lower.get(key):
                    index[lower[key]] = rec
            for key in ("chemname", "name", "chemical_name"):
                if lower.get(key):
                    index[lower[key].lower()] = rec
    return index


def load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            LOG.warning("cache at %s is unreadable; starting fresh", path)
    return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chemname", default="data/raw/cpdb/cpdb.chemname.tab")
    ap.add_argument("--out", default="data/external/structure_resolution.tsv")
    ap.add_argument("--unresolved", default="data/external/structure_unresolved.tsv")
    ap.add_argument("--cpdb-structures", default=None, help="local chemical-structures.tab")
    ap.add_argument("--cache", default="data/external/.pubchem_cache.json")
    ap.add_argument("--limit", type=int, default=None, help="only the first N chemicals")
    ap.add_argument("--no-network", action="store_true", help="local sources only")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    chemname = Path(args.chemname)
    if not chemname.exists():
        LOG.error(
            "%s not found. Download CPDB first:\n  bash scripts/fetch_cpdb.sh",
            chemname,
        )
        return 2

    chemicals = read_chemname(chemname)
    if args.limit:
        chemicals = chemicals[: args.limit]
    LOG.info("chemicals to resolve: %d", len(chemicals))

    structures = {}
    if args.cpdb_structures:
        sp = Path(args.cpdb_structures)
        if sp.exists():
            structures = read_cpdb_structures(sp)
            LOG.info("CPDB structure file: %d index entries", len(structures))
        else:
            LOG.warning("structure file not found: %s", sp)

    cache_path = Path(args.cache)
    cache = load_cache(cache_path)
    today = time.strftime("%Y-%m-%d")

    resolved: list[dict] = []
    unresolved: list[dict] = []
    counts = {"cpdb_structure_file": 0, "pubchem_cas": 0, "pubchem_name": 0, "cache": 0}

    for i, chem in enumerate(chemicals, start=1):
        code, cas, name = chem["chemcode"], chem["cas_number"], chem["chemical_name"]
        smiles = source = inchi = ""

        hit = structures.get(cas) or (structures.get(name.lower()) if name else None)
        if hit:
            smiles, inchi, source = hit["smiles"], hit.get("inchi", ""), "cpdb_structure_file"
            counts["cpdb_structure_file"] += 1

        if not smiles:
            key = f"cas:{cas}" if cas else f"name:{name.lower()}"
            if key in cache:
                rec = cache[key]
                if rec.get("smiles"):
                    smiles, inchi = rec["smiles"], rec.get("inchi", "")
                    source = rec.get("source", "cache")
                    counts["cache"] += 1
            elif not args.no_network:
                row, src = (lookup_by_cas(cas) if cas else (None, ""))
                if not row and name:
                    row, src = lookup_by_name(name)
                if row:
                    smiles = row.get("IsomericSMILES") or row.get("CanonicalSMILES", "")
                    inchi = row.get("InChI", "")
                    source = src
                    counts[src] = counts.get(src, 0) + 1
                cache[key] = {"smiles": smiles, "inchi": inchi, "source": source}
                if i % 25 == 0:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(cache), encoding="utf-8")

        if smiles:
            resolved.append(
                {
                    "chemcode": code,
                    "cas_number": cas,
                    "chemical_name": name,
                    "smiles": smiles,
                    "inchi": inchi,
                    "source": source,
                    "retrieved_date": today,
                }
            )
        else:
            unresolved.append(
                {
                    "chemcode": code,
                    "cas_number": cas,
                    "chemical_name": name,
                    "reason": (
                        "local sources only and no match"
                        if args.no_network
                        else "no match by CAS or name in any source"
                    ),
                }
            )

        if i % 100 == 0:
            LOG.info("  %d/%d  resolved %d", i, len(chemicals), len(resolved))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["chemcode", "cas_number", "chemical_name", "smiles", "inchi", "source", "retrieved_date"]
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(resolved)

    up = Path(args.unresolved)
    with up.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["chemcode", "cas_number", "chemical_name", "reason"], delimiter="\t"
        )
        w.writeheader()
        w.writerows(unresolved)

    total = len(chemicals)
    pct = 100.0 * len(resolved) / total if total else 0.0
    print()
    print(f"Resolved   {len(resolved):5d} / {total}  ({pct:.1f}%)  -> {out}")
    print(f"Unresolved {len(unresolved):5d}              -> {up}")
    print("\nBy source:")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        if v:
            print(f"  {k:24s} {v}")
    print(
        "\nReview the unresolved list by hand before freezing. Those chemicals\n"
        "stay in structure_review.parquet and never receive a label, so a gap\n"
        "here quietly shrinks the dataset rather than corrupting it.\n"
        "\nSpot-check a sample of resolved rows too: a CAS number is not proof\n"
        "of identity, and PubChem name matches are the likeliest to be wrong.\n"
    )
    return 0 if resolved else 1


if __name__ == "__main__":
    sys.exit(main())
