"""Step 6b - generate the locked splits.

Splits are assigned over GROUPS, never over experimental records. Every record
belonging to a compound stays with that compound, and in the scaffold split
every compound sharing a scaffold group stays together.

Three strategies, each with a different role:
    random    diagnostic only, for comparison against baselines that quote it
    scaffold  primary development evaluation
    external  final validation on an independently curated set

After assignment the splits are VERIFIED, not assumed:
    * no InChIKey appears in two partitions
    * no scaffold group spans partitions in the scaffold split
    * nearest-neighbour Tanimoto similarity from test to train is measured and
      reported, because scaffold separation does not imply dissimilarity

Outputs:
    data/processed/splits_{endpoint}.parquet
    data/processed/split_report_{endpoint}.json

Usage:
    python -m src.datasets.split_dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import StageManifest, load_config, resolve_path, setup_logging, utc_now

LOG = logging.getLogger("split_dataset")
STAGE = "07_split_dataset"

PARTITIONS = ("train", "validation", "test")


class LeakageError(RuntimeError):
    """A split violated a hard leakage constraint."""


def assign_groups(
    groups: list[tuple[str, int]],
    fractions: dict[str, float],
    *,
    rng: np.random.Generator,
    largest_first: bool,
) -> dict[str, str]:
    """Assign whole groups to partitions, filling toward target sizes.

    `groups` is a list of (group_key, member_count). Returns group_key ->
    partition. Groups are never split.

    With largest_first, groups are ordered by descending size so that the big
    scaffold families land in training and the rare chemistry ends up in
    validation and test. That makes the evaluation harder, deliberately.
    """
    total = sum(n for _, n in groups)
    if total == 0:
        return {}

    if largest_first:
        # Descending size, ties broken by key so the order is reproducible.
        ordered = sorted(groups, key=lambda kv: (-kv[1], kv[0]))
    else:
        shuffled = sorted(groups, key=lambda kv: kv[0])
        idx = rng.permutation(len(shuffled))
        ordered = [shuffled[i] for i in idx]

    # Cumulative cutoffs. Groups fill train until it would overshoot its
    # target, then validation, then test. A group that would overshoot every
    # partition still has to land somewhere, so it goes to the one it was
    # being considered for; the realised fractions are reported either way.
    train_cutoff = fractions.get("train", 0.0) * total
    valid_cutoff = train_cutoff + fractions.get("validation", 0.0) * total

    assignment: dict[str, str] = {}
    filled = {p: 0 for p in PARTITIONS}

    for key, n in ordered:
        if filled["train"] + n <= train_cutoff:
            part = "train"
        elif filled["train"] + filled["validation"] + n <= valid_cutoff:
            part = "validation"
        elif filled["test"] == 0 or filled["train"] + filled["validation"] >= valid_cutoff:
            part = "test"
        else:
            part = "validation"
        assignment[key] = part
        filled[part] += n

    # A partition left completely empty makes the split unusable. Move the
    # smallest group out of the largest partition to populate it, so the
    # failure is visible as a skewed fraction rather than a missing set.
    for part in PARTITIONS:
        if filled[part] == 0 and len(assignment) > len(PARTITIONS):
            donor = max(PARTITIONS, key=lambda p: filled[p])
            movable = sorted(
                (k for k, v in assignment.items() if v == donor),
                key=lambda k: (dict(groups)[k], k),
            )
            if movable:
                moved = movable[0]
                filled[donor] -= dict(groups)[moved]
                assignment[moved] = part
                filled[part] += dict(groups)[moved]
    return assignment


def verify_no_leakage(
    df: pd.DataFrame,
    *,
    strategy: str,
    cfg: dict,
    man: StageManifest,
) -> dict:
    """Hard checks. Raises LeakageError rather than reporting a warning."""
    checks = cfg["leakage_checks"]
    result: dict = {}

    assigned = df[df["split_assignment"].isin(PARTITIONS)]

    if checks.get("forbid_inchikey_overlap", True):
        per_key = assigned.groupby("inchi_key")["split_assignment"].nunique()
        offenders = per_key[per_key > 1]
        result["inchikey_spanning_partitions"] = int(len(offenders))
        if len(offenders):
            raise LeakageError(
                f"{strategy}: {len(offenders)} InChIKey(s) appear in more than "
                f"one partition, e.g. {list(offenders.index[:5])}"
            )

    if strategy == "scaffold" and checks.get("forbid_scaffold_overlap_in_scaffold_split", True):
        per_group = assigned.groupby("scaffold_group")["split_assignment"].nunique()
        offenders = per_group[per_group > 1]
        result["scaffold_groups_spanning_partitions"] = int(len(offenders))
        if len(offenders):
            raise LeakageError(
                f"scaffold split: {len(offenders)} scaffold group(s) span more "
                f"than one partition, e.g. {list(offenders.index[:5])}"
            )

    return result


def measure_similarity(
    df: pd.DataFrame, *, cfg: dict, man: StageManifest, strategy: str
) -> dict:
    """Nearest-neighbour similarity from each held-out set to training."""
    from src.features.fingerprints import nearest_neighbour_similarities  # noqa: PLC0415

    sim_cfg = cfg["leakage_checks"]["similarity"]
    train = df[df["split_assignment"] == "train"]["canonical_smiles"].dropna().tolist()
    out: dict = {}

    for part in ("validation", "test"):
        query = df[df["split_assignment"] == part]["canonical_smiles"].dropna().tolist()
        if not query or not train:
            out[part] = {"n": 0}
            continue
        sims = nearest_neighbour_similarities(
            query,
            train,
            radius=sim_cfg.get("radius", 2),
            n_bits=sim_cfg.get("n_bits", 2048),
        )
        if sims.size == 0:
            out[part] = {"n": 0}
            continue
        median = float(np.median(sims))
        stats = {
            "n": int(sims.size),
            "median": round(median, 4),
            "mean": round(float(sims.mean()), 4),
            "p90": round(float(np.percentile(sims, 90)), 4),
            "max": round(float(sims.max()), 4),
            "fraction_above_0.7": round(float((sims >= 0.7).mean()), 4),
            "fraction_identical_fp": round(float((sims >= 0.999).mean()), 4),
        }
        threshold = sim_cfg.get("median_warn_threshold", 0.70)
        stats["median_warn_threshold"] = threshold
        stats["exceeds_threshold"] = bool(median >= threshold)
        if stats["exceeds_threshold"]:
            man.warn(
                f"{strategy}/{part}: median nearest-neighbour Tanimoto to train "
                f"is {median:.3f}, at or above the {threshold} threshold. This "
                f"partition is not measuring generalisation."
            )
        # A median below the threshold can still hide outright duplicates. A
        # held-out compound whose fingerprint is identical to a training
        # compound is memorisable whatever the rest of the distribution says,
        # so it is reported on its own.
        n_identical = int(round(stats["fraction_identical_fp"] * sims.size))
        if n_identical:
            man.warn(
                f"{strategy}/{part}: {n_identical} compound(s) have a "
                f"fingerprint identical to a training compound (Tanimoto 1.0). "
                f"These are effectively seen during training even though no "
                f"InChIKey is shared; usually stereoisomers or tautomers that "
                f"the fingerprint cannot distinguish."
            )
        out[part] = stats
    return out


def check_balance(df: pd.DataFrame, *, cfg: dict, man: StageManifest, strategy: str) -> dict:
    """Class counts per partition, flagged against the configured minimums."""
    bal = cfg["balance"]
    out: dict = {}
    for part in PARTITIONS:
        sub = df[df["split_assignment"] == part]
        n_pos = int((sub["label"] == 1).sum())
        n_neg = int((sub["label"] == 0).sum())
        out[part] = {"compounds": int(len(sub)), "positive": n_pos, "negative": n_neg}
        if n_pos < bal["min_positives_per_partition"]:
            man.warn(
                f"{strategy}/{part}: {n_pos} positive(s), below the configured "
                f"minimum of {bal['min_positives_per_partition']}"
            )
        if n_neg < bal["min_negatives_per_partition"]:
            man.warn(
                f"{strategy}/{part}: {n_neg} negative(s), below the configured "
                f"minimum of {bal['min_negatives_per_partition']}"
            )
    return out


def load_external(path: Path, man: StageManifest) -> set[str]:
    """InChIKeys of an external validation set, if one is supplied."""
    if not path.exists():
        return set()
    sep = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    ext = pd.read_csv(path, sep=sep, dtype=str)
    if "inchi_key" not in ext.columns:
        man.warn(f"{path.name}: no inchi_key column; external set ignored")
        return set()
    man.add_input(path)
    return set(ext["inchi_key"].dropna())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default=None)
    parser.add_argument("--endpoint", default=None, help="limit to one endpoint")
    args = parser.parse_args(argv)

    setup_logging()
    data_cfg = load_config("data")
    lab_cfg = load_config("labels")
    split_cfg = load_config("splits")
    processed = resolve_path(args.processed or data_cfg["paths"]["processed"])
    external_dir = resolve_path(data_cfg["paths"]["external"])

    man = StageManifest(stage=STAGE)
    man.config_versions = {
        "dataset_version": data_cfg.get("dataset_version"),
        "split_version": split_cfg["split_version"],
        "random_seed": split_cfg["random_seed"],
        "label_policy_version": lab_cfg["label_policy_version"],
    }

    external_keys = load_external(external_dir / "external_validation.csv", man)
    if external_keys:
        LOG.info("external validation set: %d InChIKey(s)", len(external_keys))

    endpoints = [args.endpoint] if args.endpoint else list(lab_cfg["endpoints"])
    exit_code = 0

    for name in endpoints:
        ds_path = processed / f"model_dataset_{name}.parquet"
        if not ds_path.exists():
            LOG.error("missing dataset: %s — run build_dataset first", ds_path)
            return 2
        man.add_input(ds_path)
        ds = pd.read_parquet(ds_path)

        # Only compounds eligible for the binary task are partitioned. The
        # rest are retained with an explicit 'excluded' assignment so that
        # nothing silently disappears between stages.
        eligible = ds[ds["training_eligibility"] == "eligible"].copy()
        LOG.info(
            "endpoint %s: %d compounds, %d eligible for splitting",
            name,
            len(ds),
            len(eligible),
        )
        if eligible.empty:
            man.warn(f"endpoint {name}: no eligible compounds; no split generated")
            continue

        # External compounds are removed from the internal pool entirely.
        in_external = eligible["inchi_key"].isin(external_keys)
        n_external = int(in_external.sum())
        internal = eligible[~in_external].copy()

        report: dict = {
            "generated_at": utc_now(),
            "endpoint": name,
            "dataset_version": data_cfg.get("dataset_version"),
            "split_version": split_cfg["split_version"],
            "label_policy_version": lab_cfg["label_policy_version"],
            "random_seed": split_cfg["random_seed"],
            "compounds_total": int(len(ds)),
            "compounds_eligible": int(len(eligible)),
            "compounds_in_external_set": n_external,
            "compounds_partitioned": int(len(internal)),
            "strategies": {},
        }

        frames: list[pd.DataFrame] = []
        for strategy, scfg in split_cfg["strategies"].items():
            if strategy == "external":
                continue

            rng = np.random.default_rng(split_cfg["random_seed"])
            group_col = scfg["group_by"]
            if group_col not in internal.columns:
                man.warn(
                    f"endpoint {name}: strategy {strategy} needs column "
                    f"{group_col}, which is absent; strategy skipped"
                )
                continue

            # groupby drops null keys by default, which would remove those
            # compounds from the split without a word. Refuse instead.
            n_null_keys = int(internal[group_col].isna().sum())
            if n_null_keys:
                msg = (
                    f"endpoint {name}, strategy {strategy}: {n_null_keys} "
                    f"compound(s) have a null {group_col}. Grouping would drop "
                    f"them silently; fix the upstream stage rather than "
                    f"splitting an incomplete set."
                )
                LOG.error("%s", msg)
                man.warn(msg)
                man.write()
                return 4

            sizes = internal.groupby(group_col).size()
            groups = [(str(k), int(v)) for k, v in sizes.items()]
            assert sum(n for _, n in groups) == len(internal), (
                "group sizes must account for every compound"
            )
            assignment = assign_groups(
                groups,
                split_cfg["fractions"],
                rng=rng,
                largest_first=scfg.get("assignment") == "largest_group_first",
            )

            part = internal.copy()
            part["split_strategy"] = strategy
            part["split_group"] = part[group_col].astype(str)
            part["split_assignment"] = part["split_group"].map(assignment)
            part["split_version"] = split_cfg["split_version"]

            try:
                leak = verify_no_leakage(part, strategy=strategy, cfg=split_cfg, man=man)
            except LeakageError as exc:
                LOG.error("%s", exc)
                man.warn(str(exc))
                man.write()
                return 4

            sim = measure_similarity(part, cfg=split_cfg, man=man, strategy=strategy)
            balance = check_balance(part, cfg=split_cfg, man=man, strategy=strategy)

            realised = (
                part["split_assignment"].value_counts(normalize=True).round(4).to_dict()
            )
            report["strategies"][strategy] = {
                "role": scfg["role"],
                "group_by": group_col,
                "n_groups": len(groups),
                "largest_group_size": int(max(n for _, n in groups)) if groups else 0,
                "target_fractions": split_cfg["fractions"],
                "realised_fractions": {str(k): float(v) for k, v in realised.items()},
                "partitions": balance,
                "leakage_checks": leak,
                "nearest_neighbour_similarity_to_train": sim,
            }

            # Re-attach external and excluded compounds with explicit labels.
            ext_rows = eligible[in_external].copy()
            if not ext_rows.empty:
                ext_rows["split_strategy"] = strategy
                ext_rows["split_group"] = ext_rows["inchi_key"]
                ext_rows["split_assignment"] = "external"
                ext_rows["split_version"] = split_cfg["split_version"]

            excl_rows = ds[ds["training_eligibility"] != "eligible"].copy()
            if not excl_rows.empty:
                excl_rows["split_strategy"] = strategy
                excl_rows["split_group"] = None
                excl_rows["split_assignment"] = "excluded"
                excl_rows["split_version"] = split_cfg["split_version"]

            frames.append(
                pd.concat(
                    [f for f in (part, ext_rows, excl_rows) if not f.empty],
                    ignore_index=True,
                )
            )
            LOG.info(
                "  %s split: %d groups -> %s",
                strategy,
                len(groups),
                {k: v["compounds"] for k, v in balance.items()},
            )

        if not frames:
            man.warn(f"endpoint {name}: no strategy produced a split")
            continue

        splits = pd.concat(frames, ignore_index=True)
        out_path = processed / f"splits_{name}.parquet"
        splits.to_parquet(out_path, index=False)
        man.add_output(out_path, rows=len(splits))

        rep_path = processed / f"split_report_{name}.json"
        with rep_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")
        man.add_output(rep_path)
        man.counts[f"endpoint_{name}"] = {
            "compounds_partitioned": report["compounds_partitioned"],
            "strategies": list(report["strategies"]),
        }

    man.write()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
