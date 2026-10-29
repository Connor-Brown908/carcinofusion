#!/usr/bin/env python3
"""Report which dependencies are present, and name the ones that are not.

Deliberately imports nothing beyond the standard library, so it runs in an
environment where everything else is missing — which is exactly when it is
needed.

    python3 scripts/check_env.py              # pipeline + benchmark
    python3 scripts/check_env.py --pipeline   # curation pipeline only
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

# (import name, pip name, what breaks without it)
PIPELINE = [
    ("yaml", "pyyaml", "reading configs/*.yaml"),
    ("numpy", "numpy", "numeric work throughout"),
    ("pandas", "pandas", "every dataset stage"),
    ("pyarrow", "pyarrow", "reading and writing parquet"),
    ("rdkit", "rdkit", "structure standardisation and fingerprints"),
]
BENCHMARK = [
    ("sklearn", "scikit-learn", "B1, B2 and every metric"),
    ("lightgbm", "lightgbm", "B3 gradient boosting"),
    ("torch", "torch", "B4 message-passing network"),
]
OPTIONAL = [
    ("xgboost", "xgboost", "B3 alternative backend"),
    ("transformers", "transformers", "B5 pretrained encoder"),
]


def present(module: str) -> str | None:
    """Version string if importable, else None. Does not import the module."""
    try:
        if importlib.util.find_spec(module) is None:
            return None
    except (ImportError, ValueError):
        return None
    try:
        return getattr(__import__(module), "__version__", "installed")
    except Exception:  # noqa: BLE001
        return "installed"


def report(title: str, group: list[tuple[str, str, str]]) -> list[str]:
    print(f"\n{title}")
    missing = []
    for mod, pip_name, why in group:
        version = present(mod)
        if version:
            print(f"  ok       {pip_name:<14} {version}")
        else:
            print(f"  MISSING  {pip_name:<14} needed for {why}")
            missing.append(pip_name)
    return missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pipeline", action="store_true", help="curation pipeline only")
    args = ap.parse_args(argv)

    print(f"Python {sys.version.split()[0]}  ({sys.executable})")

    missing = report("Curation pipeline", PIPELINE)
    blocking = list(missing)
    if not args.pipeline:
        missing_bench = report("Benchmark suite", BENCHMARK)
        blocking += missing_bench
        report("Optional", OPTIONAL)

    print()
    if not blocking:
        print("All required packages present.")
        return 0

    uniq = list(dict.fromkeys(blocking))
    print(f"{len(uniq)} package(s) missing. Install with ONE of:\n")
    print(f"  pip install {' '.join(uniq)}")
    print("\n  # or, if pip refuses on a system-managed Python (PEP 668):")
    print(f"  pip install --user {' '.join(uniq)}")
    print("\n  # or in a virtual environment, which avoids that problem entirely:")
    print("  python3 -m venv .venv && source .venv/bin/activate")
    print(f"  pip install {' '.join(uniq)}")
    print("\n  # or with conda, which is what environment.yml describes:")
    print("  conda env create -f environment.yml && conda activate carcinofusion")
    return 1


if __name__ == "__main__":
    sys.exit(main())
