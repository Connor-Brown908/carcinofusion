"""Step 1 - acquire CPDB and record provenance.

This repository never alters a raw file. This script's job is to verify that
the expected files are present in data/raw/cpdb/, record their checksums,
sizes and retrieval date, and refuse to proceed if a required file is absent.

Network access is attempted only when --fetch is passed. Many execution
environments cannot reach ftp.nlm.nih.gov; that is expected and is not an
error. The normal workflow is:

    1. Download the files yourself from the CPDB data page.
    2. Place them, unchanged, in data/raw/cpdb/.
    3. Run this script to register them.

Usage:
    python -m src.ingestion.download_cpdb              # register files already present
    python -m src.ingestion.download_cpdb --fetch      # also try to download them
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.common import (
    StageManifest,
    load_config,
    resolve_path,
    setup_logging,
    utc_now,
)

LOG = logging.getLogger("download_cpdb")
STAGE = "01_acquire_cpdb"


def fetch_file(url: str, dest: Path, timeout: int = 60) -> tuple[bool, str]:
    """Attempt a download. Returns (ok, message). Never raises."""
    import urllib.error
    import urllib.request

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            data = resp.read()
        if not data:
            return False, f"empty response from {url}"
        dest.write_bytes(data)
        return True, f"downloaded {len(data)} bytes from {url}"
    except urllib.error.URLError as exc:
        return False, f"network unreachable for {url}: {exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return False, f"download failed for {url}: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="attempt to download missing files over the network",
    )
    parser.add_argument(
        "--allow-missing-optional",
        action="store_true",
        default=True,
        help="proceed when optional files (dose tables) are absent",
    )
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config("data")
    src = cfg["sources"]["cpdb"]
    raw_dir = resolve_path(src["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    man = StageManifest(stage=STAGE)
    man.config_versions = {
        "dataset_version": cfg.get("dataset_version"),
        "source_database": src.get("source_database"),
        "base_url": src.get("base_url"),
    }

    present: list[str] = []
    missing_required: list[str] = []
    missing_optional: list[str] = []

    for key, spec in src["files"].items():
        dest = raw_dir / spec["filename"]

        if not dest.exists() and args.fetch:
            url = src["base_url"].rstrip("/") + "/" + spec["filename"]
            ok, msg = fetch_file(url, dest)
            LOG.info(msg) if ok else man.warn(msg)

        if dest.exists():
            man.add_input(
                dest,
                file_key=key,
                role=spec["role"],
                required=spec.get("required", False),
            )
            present.append(spec["filename"])
            LOG.info("present: %s (%d bytes)", spec["filename"], dest.stat().st_size)
        elif spec.get("required", False):
            missing_required.append(spec["filename"])
        else:
            missing_optional.append(spec["filename"])
            man.warn(
                f"optional file absent: {spec['filename']} "
                f"(dose/incidence detail will be unavailable)"
            )

    man.counts = {
        "files_present": len(present),
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "retrieval_date": utc_now(),
    }

    if missing_required:
        man.warn(f"required files absent: {missing_required}")
        man.write()
        LOG.error(
            "Required CPDB files are missing from %s: %s\n"
            "Download them from %s and place the untouched originals there, "
            "then re-run this step.",
            raw_dir,
            missing_required,
            src["base_url"],
        )
        return 2

    man.write()
    LOG.info(
        "Registered %d CPDB file(s) in %s. Checksums recorded in manifests/.",
        len(present),
        raw_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
