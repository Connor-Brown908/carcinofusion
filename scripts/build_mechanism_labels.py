#!/usr/bin/env python3
"""Tag compounds genotoxic or non-genotoxic — Panel B of Figure 1.

Aggregate carcinogenicity accuracy is dominated by genotoxic carcinogens:
they are structurally recognisable and already caught by mutagenicity alerts,
so a model can score respectably while adding nothing on the cases that are
actually unsolved. Splitting the test set by mechanism is what makes a
performance claim mean something.

Three sources, best first. Which one supplied each call is recorded per
compound, because they are not equally strong evidence.

  1. CPDB's own Salmonella evaluations. Best, because it covers the same
     chemical set under a documented rule: mutagenic if Zeiger rated it
     "mutagenic" or "weakly mutagenic", or Gene-Tox rated it "positive";
     other evaluated chemicals are non-mutagens. CPDB reports these for 860
     of its 1,547 chemicals. Supply a local file with --cpdb-salmonella.

  2. The Hansen Ames benchmark, 6,512 compounds with CAS and SMILES, joined
     on CAS. Downloaded automatically unless --no-network.
     Hansen et al., J. Chem. Inf. Model. 2009, doi:10.1021/ci900161g

  3. Structural alerts, as a clearly-marked fallback. These INFER DNA
     reactivity from substructure; they are not a measurement, and a compound
     labelled this way should not be reported as experimentally genotoxic.
     Off unless --allow-alerts.

A compound with no mutagenicity evidence is left `unknown`, not assumed
non-genotoxic. Most non-genotoxic carcinogens would otherwise be
indistinguishable from untested ones, which is exactly the subset Panel B
exists to measure.

Usage:
    python3 scripts/build_mechanism_labels.py
    python3 scripts/build_mechanism_labels.py --allow-alerts
    python3 scripts/build_mechanism_labels.py --cpdb-salmonella salmonella.tab
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

from src.common import StageManifest, load_config, resolve_path, setup_logging, utc_now  # noqa: E402

LOG = logging.getLogger("mechanism")
STAGE = "14_mechanism_labels"
RDLogger.DisableLog("rdApp.*")

HANSEN_CSV = "https://doc.ml.tu-berlin.de/toxbenchmark/Mutagenicity_N6512.csv"
HANSEN_SMI = "https://doc.ml.tu-berlin.de/toxbenchmark/smiles_cas_N6512.smi"
USER_AGENT = "carcinofusion-dataset-pipeline/1.0 (research use)"

# DNA-reactivity alerts, in the spirit of the ICH M7 expert rule systems.
# Deliberately a short, well-understood list: a long one would look more
# thorough while being harder to defend, and these are a fallback anyway.
ALERTS = {
    "aromatic_nitro": "[a][N+](=O)[O-]",
    "aromatic_amine": "[NX3;H2,H1;!$(NC=O)][a]",
    "nitrosamine": "[NX3][NX2]=O",
    "epoxide": "C1OC1",
    "aziridine": "C1NC1",
    "azo_aromatic": "[a]N=N[a]",
    "hydrazine": "[NX3][NX3]",
    "alkyl_halide": "[CX4;H2,H3][Cl,Br,I]",
    "nitrogen_mustard": "N(CC[Cl,Br])CC[Cl,Br]",
    "michael_acceptor": "C=CC(=O)[#6,#7,#8]",
    "aromatic_hydroxylamine": "[a][NX3;H1][OX2;H1]",
    "acyl_halide": "C(=O)[Cl,Br]",
}


def fetch(url: str, timeout: int = 120) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.read()
    except Exception as exc:  # noqa: BLE001
        LOG.warning("could not fetch %s: %s", url, str(exc)[:120])
        return None


def normalise_cas(value) -> str | None:
    """CAS as digits-and-dashes, so 71-43-2 and 00071-43-2 match."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "-"}:
        return None
    s = "".join(ch for ch in s if ch.isdigit() or ch == "-")
    parts = s.split("-")
    if len(parts) == 3 and all(p.isdigit() and p for p in parts):
        return f"{int(parts[0])}-{parts[1]}-{parts[2]}"
    # Anything not CAS-shaped is no key at all. Returning the stripped string
    # would turn every malformed value into the same token ('not-a-cas' and
    # 'also--bad' both become '--'), which then match each other and attach
    # one compound's mutagenicity call to another's.
    return None


def load_hansen(cache: Path, *, no_network: bool) -> dict[str, int]:
    """CAS -> Ames call (1 mutagenic, 0 not) from the Hansen benchmark."""
    if not cache.exists() and not no_network:
        LOG.info("downloading the Hansen Ames benchmark (6,512 compounds)")
        data = fetch(HANSEN_CSV) or fetch(HANSEN_SMI)
        if data:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(data)
    if not cache.exists():
        LOG.warning("Hansen benchmark unavailable; skipping that source")
        return {}

    raw = cache.read_bytes().decode("utf-8", errors="replace")
    out: dict[str, int] = {}

    # The CSV and the .smi share fields but not a layout, so sniff rather
    # than assume which one landed in the cache.
    first = raw.splitlines()[0] if raw else ""
    if "," in first or ";" in first:
        try:
            df = pd.read_csv(io.StringIO(raw), sep=None, engine="python")
        except Exception as exc:  # noqa: BLE001
            LOG.warning("could not parse the Hansen file: %s", exc)
            return {}
        cols = {c.lower().strip(): c for c in df.columns}
        cas_col = next((cols[c] for c in ("cas_no", "cas", "casrn") if c in cols), None)
        act_col = next(
            (cols[c] for c in ("activity", "mutagenicity", "class", "label") if c in cols),
            None,
        )
        if not cas_col or not act_col:
            LOG.warning("Hansen file lacks CAS or activity columns: %s", list(df.columns))
            return {}
        for cas, act in zip(df[cas_col], df[act_col]):
            key = normalise_cas(cas)
            try:
                val = int(float(act))
            except (TypeError, ValueError):
                continue
            if key and val in (0, 1):
                out[key] = val
    else:
        # SMILES file: "<smiles>\t<cas>\t<activity>"
        for line in raw.splitlines():
            parts = line.replace("\t", " ").split()
            if len(parts) < 3:
                continue
            key = normalise_cas(parts[1])
            try:
                val = int(float(parts[2]))
            except ValueError:
                continue
            if key and val in (0, 1):
                out[key] = val

    LOG.info("Hansen benchmark: %d CAS-keyed Ames call(s)", len(out))
    return out


def load_cpdb_salmonella(path: Path) -> dict[str, int]:
    """CPDB's own Salmonella calls, keyed by chemcode and by CAS."""
    out: dict[str, int] = {}
    with path.open("r", encoding="latin-1", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            low = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            call = (
                low.get("salmonella")
                or low.get("mutagenicity")
                or low.get("ames")
                or low.get("sal")
            )
            if not call:
                continue
            c = call.strip().lower()
            # CPDB's documented rule: mutagenic or weakly mutagenic (Zeiger),
            # or positive (Gene-Tox) counts as mutagenic.
            if c in {"p", "+", "positive", "mutagenic", "weakly mutagenic", "wm", "1", "y", "yes"}:
                value = 1
            elif c in {"n", "-", "negative", "non-mutagenic", "nonmutagen", "0", "no"}:
                value = 0
            else:
                continue
            for key_field in ("chemcode", "chem_code"):
                if low.get(key_field):
                    out[low[key_field]] = value
            cas = normalise_cas(low.get("casno") or low.get("cas"))
            if cas:
                out[cas] = value
    LOG.info("CPDB Salmonella file: %d key(s)", len(out))
    return out


def alert_hits(smiles: str) -> list[str]:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return []
    hits = []
    for name, smarts in ALERTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            hits.append(name)
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpdb-salmonella", default=None)
    ap.add_argument("--allow-alerts", action="store_true",
                    help="infer from structural alerts where no measurement exists")
    ap.add_argument("--no-network", action="store_true")
    ap.add_argument("--cache", default="data/external/hansen_ames.csv")
    ap.add_argument("--out", default="data/external/mechanism_labels.tsv")
    args = ap.parse_args(argv)

    setup_logging()
    data_cfg = load_config("data")
    interim = resolve_path(data_cfg["paths"]["interim"])

    comp_path = interim / "compound_identity.parquet"
    exp_path = interim / "carcinogenicity_experiments.parquet"
    if not comp_path.exists():
        LOG.error("%s not found. Run the pipeline first.", comp_path)
        return 2

    man = StageManifest(stage=STAGE)
    man.add_input(comp_path)
    compounds = pd.read_parquet(comp_path)

    # chemcode -> CAS, so a CPDB-keyed source and a CAS-keyed one both land.
    cas_by_chemcode: dict[str, str] = {}
    if exp_path.exists():
        exp = pd.read_parquet(exp_path)
        for code, cas in zip(exp["chemcode"], exp["cas_number"]):
            key = normalise_cas(cas)
            if code and key:
                cas_by_chemcode.setdefault(str(code), key)

    cpdb_calls: dict[str, int] = {}
    if args.cpdb_salmonella:
        p = Path(args.cpdb_salmonella)
        if p.exists():
            cpdb_calls = load_cpdb_salmonella(p)
            man.add_input(p)
        else:
            man.warn(f"CPDB Salmonella file not found: {p}")

    hansen = load_hansen(Path(args.cache), no_network=args.no_network)

    rows = []
    counts = {"cpdb_salmonella": 0, "hansen_ames": 0, "structural_alerts": 0, "unknown": 0}

    for rec in compounds.to_dict("records"):
        cid = rec["compound_id"]
        codes = str(rec.get("source_chemcodes") or "").split("; ")
        cas_keys = {normalise_cas(cas_by_chemcode.get(c)) for c in codes if c}
        cas_keys.discard(None)

        ames = None
        source = None

        for c in codes:
            if c in cpdb_calls:
                ames, source = cpdb_calls[c], "cpdb_salmonella"
                break
        if ames is None:
            for k in cas_keys:
                if k in cpdb_calls:
                    ames, source = cpdb_calls[k], "cpdb_salmonella"
                    break
        if ames is None:
            for k in cas_keys:
                if k in hansen:
                    ames, source = hansen[k], "hansen_ames"
                    break

        hits = alert_hits(rec.get("canonical_smiles") or "")
        if ames is None and args.allow_alerts:
            ames = 1 if hits else 0
            source = "structural_alerts"

        if ames is None:
            mechanism, confidence = "unknown", "none"
            counts["unknown"] += 1
        else:
            mechanism = "genotoxic" if ames == 1 else "non_genotoxic"
            # An alert-derived call is an inference about structure, not a
            # measurement, and must not be reported as experimental evidence.
            confidence = "inferred" if source == "structural_alerts" else "measured"
            counts[source] = counts.get(source, 0) + 1

        rows.append(
            {
                "compound_id": cid,
                "mechanism": mechanism,
                "ames_call": ames,
                "mechanism_source": source,
                "mechanism_confidence": confidence,
                "structural_alerts": "; ".join(hits) or None,
                "n_structural_alerts": len(hits),
                "assigned_date": utc_now()[:10],
            }
        )

    out = pd.DataFrame(rows)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, sep="\t", index=False)
    man.add_output(path, rows=len(out))

    dist = out["mechanism"].value_counts().to_dict()
    man.counts = {"by_source": counts, "by_mechanism": {k: int(v) for k, v in dist.items()}}
    man.write()

    total = len(out)
    print(f"\nMechanism labels written: {path}  ({total:,} compounds)\n")
    print("  by source:")
    for k, v in counts.items():
        if v:
            print(f"    {k:<22} {v:>6,}")
    print("\n  by mechanism:")
    for k, v in dist.items():
        print(f"    {k:<22} {v:>6,}  ({100.0 * v / total:.1f}%)")

    known = total - counts["unknown"]
    ng = int((out["mechanism"] == "non_genotoxic").sum())
    print()
    if known < 50:
        print(
            "  Too few compounds have a mutagenicity call for a mechanism-split\n"
            "  figure. Supply CPDB's Salmonella file with --cpdb-salmonella, or\n"
            "  check that CAS numbers are reaching this script."
        )
        return 1
    if ng < 30:
        print(
            f"  Only {ng} non-genotoxic compounds. Panel B needs enough of them\n"
            f"  to carry a confidence interval; treat it as indicative below ~30."
        )
    if counts["structural_alerts"]:
        print(
            f"  {counts['structural_alerts']:,} call(s) came from structural alerts.\n"
            f"  Those are inferred, not measured. Report them separately or\n"
            f"  exclude them from Panel B; they cannot support a claim about\n"
            f"  experimentally confirmed non-genotoxic carcinogens."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
