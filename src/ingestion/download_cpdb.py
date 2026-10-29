"""Step 1 - acquire CPDB and record provenance.

Downloads the tab-separated CPDB distribution, or registers files you placed
in `data/raw/cpdb/` yourself, and records each file's SHA-256, size and
retrieval date in `manifests/`.

This repository never alters a raw file. `data/raw/` is read-only input for
every later stage.

Where the data comes from
-------------------------
CPDB is archival. NLM's catalogue entry states the dataset is no longer
updated, so what you download today should match what you download next year
— and the recorded checksums will prove whether it did.

Mirrors are tried in the order given in configs/data.yaml:

    nlm        ftp.nlm.nih.gov/projects/SISFTP/CPDB/Data Files/
    toxplanet  files.toxplanet.com/cpdb/tab/   (appends .txt to each filename)

The ToxPlanet suffix is normalised on save, so `cpdb.ncintp.tab.txt` lands as
`cpdb.ncintp.tab` and the parser sees one consistent set of names.

Usage:
    python -m src.ingestion.download_cpdb --fetch      # download, then register
    python -m src.ingestion.download_cpdb              # register files already present
    python -m src.ingestion.download_cpdb --fetch --mirror toxplanet
    python -m src.ingestion.download_cpdb --print-urls # just show where to get them
"""

from __future__ import annotations

import argparse
import logging
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

from src.common import (
    StageManifest,
    load_config,
    resolve_path,
    setup_logging,
    sha256_file,
    utc_now,
)

LOG = logging.getLogger("download_cpdb")
STAGE = "01_acquire_cpdb"

USER_AGENT = "carcinofusion-dataset-pipeline/1.0 (research use)"
# A tab file that comes back this small is an error page, not data.
MIN_PLAUSIBLE_BYTES = 512


def candidate_urls(filename: str, mirrors: list[dict]) -> list[tuple[str, str]]:
    """(mirror name, url) for one file, in mirror order."""
    out = []
    for m in mirrors:
        base = m["base_url"].rstrip("/") + "/"
        out.append((m.get("name", "mirror"), base + filename + m.get("suffix", "")))
    return out


def fetch(url: str, dest: Path, *, timeout: int = 120) -> tuple[bool, str]:
    """Download one file. Never raises; returns (ok, message)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = resp.read()
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code} from {url}"
    except urllib.error.URLError as exc:
        return False, f"unreachable: {url} ({exc.reason})"
    except (TimeoutError, ssl.SSLError) as exc:
        return False, f"timed out or TLS error: {url} ({exc})"
    except Exception as exc:  # noqa: BLE001
        return False, f"failed: {url} ({type(exc).__name__}: {exc})"

    if len(data) < MIN_PLAUSIBLE_BYTES:
        return False, (
            f"response from {url} was only {len(data)} bytes, which is too "
            f"small to be a CPDB table; treating as an error page"
        )
    head = data[:200].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        return False, f"response from {url} was HTML, not tab-separated data"

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return True, f"{len(data):,} bytes from {url}"


def print_urls(src: dict) -> None:
    """Where to get the files, for a manual download."""
    mirrors = src.get("mirrors", [])
    print("\nCPDB download locations\n" + "=" * 70)
    print(f"\nCatalogue entry: {src.get('catalogue', 'n/a')}")
    print("\nDocumentation (read before the first production run):")
    for d in src.get("documentation", []):
        print(f"  {d}")
    for m in mirrors:
        print(f"\n[{m.get('name')}]  {m['base_url']}")
        if m.get("suffix"):
            print(f"  note: filenames carry a '{m['suffix']}' suffix on this mirror")
        for spec in src["files"].values():
            mark = "required" if spec.get("required") else "optional"
            print(
                f"  {m['base_url'].rstrip('/')}/{spec['filename']}{m.get('suffix','')}"
                f"   ({mark})"
            )
    print(
        "\nSave them into data/raw/cpdb/ under their plain names "
        "(strip any .txt suffix), then run:\n"
        "  python -m src.ingestion.download_cpdb\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="download missing files")
    parser.add_argument("--mirror", default=None, help="use only this mirror by name")
    parser.add_argument("--force", action="store_true", help="re-download files already present")
    parser.add_argument("--print-urls", action="store_true", help="print sources and exit")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config("data")
    src = cfg["sources"]["cpdb"]

    if args.print_urls:
        print_urls(src)
        return 0

    raw_dir = resolve_path(src["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    mirrors = src.get("mirrors") or [{"name": "default", "base_url": src["base_url"], "suffix": ""}]
    if args.mirror:
        mirrors = [m for m in mirrors if m.get("name") == args.mirror]
        if not mirrors:
            LOG.error(
                "no mirror named %r; available: %s",
                args.mirror,
                [m.get("name") for m in src.get("mirrors", [])],
            )
            return 2

    man = StageManifest(stage=STAGE)
    man.config_versions = {
        "dataset_version": cfg.get("dataset_version"),
        "source_database": src.get("source_database"),
        "mirrors": [m.get("name") for m in mirrors],
    }

    present: list[str] = []
    missing_required: list[str] = []
    missing_optional: list[str] = []
    downloaded = 0

    for key, spec in src["files"].items():
        dest = raw_dir / spec["filename"]

        if args.fetch and (args.force or not dest.exists()):
            for mirror_name, url in candidate_urls(spec["filename"], mirrors):
                ok, msg = fetch(url, dest, timeout=args.timeout)
                if ok:
                    LOG.info("%s: %s", spec["filename"], msg)
                    downloaded += 1
                    break
                LOG.debug("%s: %s", mirror_name, msg)
            else:
                man.warn(f"{spec['filename']}: every mirror failed")

        if dest.exists():
            man.add_input(
                dest,
                file_key=key,
                role=spec["role"],
                required=spec.get("required", False),
            )
            present.append(spec["filename"])
            LOG.info(
                "present: %-24s %10s bytes  sha256 %s",
                spec["filename"],
                f"{dest.stat().st_size:,}",
                sha256_file(dest)[:16],
            )
        elif spec.get("required", False):
            missing_required.append(spec["filename"])
        else:
            missing_optional.append(spec["filename"])

    if missing_optional:
        man.warn(f"optional file(s) absent: {missing_optional}")

    man.counts = {
        "files_present": len(present),
        "files_downloaded": downloaded,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "retrieval_date": utc_now(),
    }

    if missing_required:
        man.warn(f"required files absent: {missing_required}")
        man.write()
        LOG.error(
            "Required CPDB files are missing from %s: %s\n"
            "Run with --fetch to download them, or see the sources with:\n"
            "  python -m src.ingestion.download_cpdb --print-urls",
            raw_dir,
            missing_required,
        )
        return 2

    man.write()
    LOG.info(
        "Registered %d CPDB file(s) in %s (%d downloaded this run). "
        "Checksums recorded in manifests/.",
        len(present),
        raw_dir,
        downloaded,
    )
    LOG.info(
        "Before the first production run, verify configs/cpdb_columns.yaml "
        "against the CPDB documentation: %s",
        (src.get("documentation") or ["see --print-urls"])[0],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
