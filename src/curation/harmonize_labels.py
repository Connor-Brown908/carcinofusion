"""Step 4 - apply the label policy.

Turns experimental assessments into harmonised labels, then aggregates them to
compound level. The mapping lives entirely in configs/labels.yaml; this module
contains no hard-coded interpretation of any assessment string.

Three label states throughout:
    1     positive under the stated criteria
    0     negative under the stated criteria
    None  equivocal, insufficient, conflicting or otherwise unresolved

None is not negative. A chemical never tested in a sufficiently informative
bioassay is unresolved, not safe.

Compound-level aggregation never uses a majority vote. A compound holding both
a positive and a negative record is flagged conflicted and excluded from the
primary binary training set, while remaining in the database in full.

Outputs:
    data/interim/observations_labeled.parquet   record-level harmonised labels
    data/interim/compound_labels.parquet        compound-level labels per endpoint

Usage:
    python -m src.curation.harmonize_labels
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

from src.common import StageManifest, load_config, resolve_path, setup_logging

LOG = logging.getLogger("harmonize_labels")
STAGE = "04_harmonize_labels"

UNMAPPED = "unmapped_assessment"


# Missing values survive a parquet round-trip as float NaN rather than None.
# str(nan) is "nan", so without an explicit check a missing assessment would be
# reported as an unmapped value called "nan" and look like a gap in the policy.
_MISSING_STRINGS = {"nan", "none", "nat", "<na>"}


def normalise_assessment(
    value: object,
    *,
    missing_values: frozenset[str] | None = None,
) -> str | None:
    """Lower-case, collapse whitespace, strip punctuation noise.

    Returns None for anything that means "no assessment recorded", so that a
    missing value is counted as missing rather than as an unrecognised rule.

    `missing_values` carries the source's own sentinels for "no opinion
    stated" — CPDB writes a blank opinion as '0'. These must be declared in
    configs/labels.yaml rather than inferred: a sentinel mistaken for a code
    makes two thirds of a file look like a vocabulary gap, and a code
    mistaken for a sentinel silently discards real evidence.
    """
    if value is None or (isinstance(value, float) and value != value):
        return None
    if value is pd.NA or (not isinstance(value, str) and pd.isna(value)):
        return None
    s = str(value).strip().lower()
    if not s or s in _MISSING_STRINGS:
        return None
    if missing_values and s in missing_values:
        return None
    s = re.sub(r"\s+", " ", s)
    cleaned = s.strip(" .;:")
    if not cleaned:
        return None
    if missing_values and cleaned in missing_values:
        return None
    return cleaned


def missing_values_from(lab_cfg: dict) -> frozenset[str]:
    """The declared "no opinion stated" sentinels, normalised.

    Validated against the vocabulary: a token declared both a sentinel and a
    label code is a contradiction that would silently void that code, so it
    fails here rather than producing a quietly smaller dataset.
    """
    raw = lab_cfg.get("missing_assessment_values") or []
    sentinels = frozenset(
        str(v).strip().lower() for v in raw if str(v).strip()
    )
    for endpoint_name, cfg in (lab_cfg.get("endpoints") or {}).items():
        for bucket in ("positive", "negative", "unresolved"):
            for term in cfg.get(bucket, []):
                token = str(term).strip().lower()
                if token in sentinels:
                    raise ValueError(
                        f"configs/labels.yaml: {token!r} is listed both in "
                        f"missing_assessment_values and in the {bucket} "
                        f"vocabulary of endpoint {endpoint_name!r}. It cannot "
                        f"be both absent evidence and a label; remove it from "
                        f"one of the two."
                    )
    return sentinels


def build_assessment_map(endpoint_cfg: dict) -> dict[str, int | None]:
    """Flatten an endpoint's positive/negative/unresolved lists into a map."""
    mapping: dict[str, int | None] = {}
    for term in endpoint_cfg.get("positive", []):
        mapping[normalise_assessment(term)] = 1
    for term in endpoint_cfg.get("negative", []):
        mapping[normalise_assessment(term)] = 0
    for term in endpoint_cfg.get("unresolved", []):
        mapping[normalise_assessment(term)] = None
    return mapping


def label_records(
    experiments: pd.DataFrame,
    endpoint_name: str,
    endpoint_cfg: dict,
    *,
    man: StageManifest,
    missing_values: frozenset[str] | None = None,
) -> tuple[pd.Series, set[str]]:
    """Map each record's assessment to 1 / 0 / None for one endpoint."""
    mapping = build_assessment_map(endpoint_cfg)
    unmapped: set[str] = set()
    labels: list[int | None] = []

    for value in experiments["author_assessment"]:
        key = normalise_assessment(value, missing_values=missing_values)
        if key is None:
            labels.append(None)
            continue
        if key in mapping:
            labels.append(mapping[key])
        else:
            # Never defaults to negative. Recorded so the policy can be fixed.
            unmapped.add(key)
            labels.append(None)

    if unmapped:
        man.warn(
            f"endpoint '{endpoint_name}': {len(unmapped)} assessment value(s) "
            f"matched no rule and were set to null (not to 0): "
            f"{sorted(unmapped)[:20]}"
        )
    return pd.Series(labels, index=experiments.index, dtype="object"), unmapped


class VocabularyMismatch(RuntimeError):
    """Too many assessments matched no rule in the label policy."""


def check_unmapped(
    experiments: pd.DataFrame,
    labels: pd.Series,
    *,
    endpoint_name: str,
    policy: dict,
    missing_values: frozenset[str] | None = None,
) -> None:
    """Fail the run when the policy does not recognise the data's vocabulary.

    An unmapped assessment becomes null, which is the right per-record
    behaviour but a terrible failure mode in bulk: the compound loses its
    label, drops out of the training set, and the only symptom is a dataset
    that came out small. That is how a CPDB run reached the benchmark with
    eleven test compounds.

    The column mapper already refuses to guess and names the column it could
    not resolve. This does the same for values.
    """
    threshold = float(policy.get("fail_above_fraction", 1.0))
    if threshold >= 1.0:
        return

    assessed = experiments["author_assessment"].map(
        lambda v: normalise_assessment(v, missing_values=missing_values)
    )
    has_assessment = assessed.notna()
    n_assessed = int(has_assessment.sum())
    if n_assessed == 0:
        return

    # Only records that HAVE an assessment but got no label are a vocabulary
    # problem; a record with no assessment is missing data, not a mismatch.
    unmapped_mask = has_assessment & labels.isna()
    n_unmapped = int(unmapped_mask.sum())
    fraction = n_unmapped / n_assessed
    if fraction <= threshold:
        return

    counts = (
        assessed[unmapped_mask]
        .value_counts()
        .head(25)
    )
    min_count = int(policy.get("report_min_count", 1))
    lines = [
        f"    {str(value)[:48]:<48} {int(n):>7,}"
        for value, n in counts.items()
        if n >= min_count
    ]

    raise VocabularyMismatch(
        f"endpoint '{endpoint_name}': {n_unmapped:,} of {n_assessed:,} records "
        f"({fraction:.1%}) carry an assessment that matches no rule in "
        f"configs/labels.yaml, so they would be labelled null and their "
        f"compounds would drop out of the dataset.\n\n"
        f"  Unmapped values, most frequent first:\n\n"
        + "\n".join(lines)
        + "\n\n"
        f"  Add each to the positive, negative or unresolved list for this\n"
        f"  endpoint in configs/labels.yaml, then bump label_policy_version.\n"
        f"  CPDB's codes are defined in Tab_Separated_CPDB.pdf:\n"
        f"    https://ftp.nlm.nih.gov/projects/SISFTP/CPDB/Documentation/\n\n"
        f"  Do not guess. Mapping a code to the wrong class invents carcinogens\n"
        f"  or invents clean compounds, and nothing downstream will notice.\n\n"
        f"  To proceed anyway, set unmapped_assessment_policy.fail_above_fraction\n"
        f"  to 1.0 in configs/labels.yaml — but the dataset will be a fraction\n"
        f"  of its real size."
    )


def confidence_for(n_records: int, confidence_cfg: dict) -> str:
    """Label confidence band from the number of agreeing eligible records."""
    for band in ("high", "medium", "low"):
        spec = confidence_cfg.get(band)
        if spec and n_records >= spec["min_records"]:
            return band
    return "none"


def aggregate_to_compounds(
    labeled: pd.DataFrame,
    label_col: str,
    *,
    agg_cfg: dict,
    confidence_cfg: dict,
) -> pd.DataFrame:
    """Aggregate record labels to compound level without majority voting."""
    eligible_species = {s.lower() for s in agg_cfg.get("eligible_species", [])}
    min_neg = int(agg_cfg.get("min_negative_records", 1))
    conflict_resolution = agg_cfg.get("conflict_resolution", "exclude")

    rows: list[dict] = []
    for cid, grp in labeled.groupby("compound_id", dropna=True):
        species = grp["species"].fillna("").str.lower()
        eligible = grp[species.isin(eligible_species)] if eligible_species else grp

        labels = eligible[label_col]
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        n_null = int(labels.isna().sum())
        n_elig = len(eligible)

        conflicted = n_pos > 0 and n_neg > 0

        if conflicted:
            label = 1 if conflict_resolution == "positive" else None
            reason = f"conflicting evidence: {n_pos} positive, {n_neg} negative"
        elif n_pos > 0:
            label = 1
            reason = f"{n_pos} positive record(s), no negative record"
        elif n_neg >= min_neg:
            label = 0
            reason = f"{n_neg} negative record(s), no positive record"
        elif n_neg > 0:
            label = None
            reason = (
                f"only {n_neg} negative record(s); below min_negative_records="
                f"{min_neg}, left unresolved"
            )
        else:
            label = None
            reason = (
                "no eligible record carried a resolvable assessment"
                if n_elig
                else "no eligible record for this endpoint's species"
            )

        agreeing = n_pos if label == 1 else (n_neg if label == 0 else 0)
        rows.append(
            {
                "compound_id": cid,
                "label": label,
                "label_confidence": confidence_for(agreeing, confidence_cfg)
                if label is not None
                else "none",
                "is_conflicted": conflicted,
                "n_experiments": int(len(grp)),
                "n_eligible_experiments": n_elig,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "n_indeterminate": n_null,
                "n_species": int(eligible["species"].nunique(dropna=True)),
                "aggregation_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", default=None)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args(argv)

    setup_logging()
    data_cfg = load_config("data")
    lab_cfg = load_config("labels")
    interim = resolve_path(args.outdir or data_cfg["paths"]["interim"])
    interim.mkdir(parents=True, exist_ok=True)

    exp_path = Path(args.experiments) if args.experiments else interim / "experiments_linked.parquet"
    if not exp_path.exists():
        LOG.error("linked experiments not found: %s (run resolve_identity first)", exp_path)
        return 2

    man = StageManifest(stage=STAGE)
    man.config_versions = {
        "dataset_version": data_cfg.get("dataset_version"),
        "label_policy_version": lab_cfg["label_policy_version"],
        "endpoint_definition": lab_cfg["endpoint_definition"],
    }
    man.add_input(exp_path)

    experiments = pd.read_parquet(exp_path)
    LOG.info("records in: %d", len(experiments))

    labeled = experiments.copy()
    labeled["label_policy_version"] = lab_cfg["label_policy_version"]

    endpoint_names = list(lab_cfg["endpoints"].keys())
    missing_values = missing_values_from(lab_cfg)
    man.counts["missing_assessment_sentinels"] = sorted(missing_values)
    all_unmapped: set[str] = set()
    for name in endpoint_names:
        col = f"label_{name}"
        labeled[col], unmapped = label_records(
            experiments,
            name,
            lab_cfg["endpoints"][name],
            man=man,
            missing_values=missing_values,
        )
        all_unmapped |= unmapped
        # Refuse to build a dataset from a vocabulary the policy does not
        # recognise. Checked per endpoint, before anything is written.
        check_unmapped(
            experiments,
            labeled[col],
            endpoint_name=name,
            policy=lab_cfg.get("unmapped_assessment_policy", {}),
            missing_values=missing_values,
        )
        counts = labeled[col].value_counts(dropna=False).to_dict()
        LOG.info("endpoint %s record labels: %s", name, counts)
        man.counts[f"record_labels_{name}"] = {
            str(k): int(v) for k, v in counts.items()
        }

    obs_path = interim / "observations_labeled.parquet"
    labeled.to_parquet(obs_path, index=False)
    man.add_output(obs_path, rows=len(labeled))

    # Compound-level aggregation, one table per endpoint, joined side by side.
    linked = labeled[labeled["compound_id"].notna()]
    if len(linked) < len(labeled):
        man.warn(
            f"{len(labeled) - len(linked)} record(s) have no compound_id and are "
            f"excluded from compound-level labels; they remain in "
            f"observations_labeled.parquet"
        )

    compound_frames: list[pd.DataFrame] = []
    for name in endpoint_names:
        agg = aggregate_to_compounds(
            linked,
            f"label_{name}",
            agg_cfg=lab_cfg["aggregation"],
            confidence_cfg=lab_cfg["confidence"],
        )
        agg = agg.rename(
            columns={
                c: f"{c}_{name}"
                for c in (
                    "label",
                    "label_confidence",
                    "is_conflicted",
                    "n_positive",
                    "n_negative",
                    "n_indeterminate",
                    "aggregation_reason",
                )
            }
        )
        compound_frames.append(agg.set_index("compound_id"))

    if compound_frames:
        compounds = compound_frames[0]
        for extra in compound_frames[1:]:
            dupe = [c for c in extra.columns if c in compounds.columns]
            compounds = compounds.join(extra.drop(columns=dupe), how="outer")
        compounds = compounds.reset_index()
    else:
        compounds = pd.DataFrame()

    compounds["label_policy_version"] = lab_cfg["label_policy_version"]
    compounds["endpoint_definition"] = lab_cfg["endpoint_definition"]

    comp_path = interim / "compound_labels.parquet"
    compounds.to_parquet(comp_path, index=False)
    man.add_output(comp_path, rows=len(compounds))

    for name in endpoint_names:
        col = f"label_{name}"
        dist = compounds[col].value_counts(dropna=False).to_dict()
        man.counts[f"compound_labels_{name}"] = {
            str(k): int(v) for k, v in dist.items()
        }
        man.counts[f"conflicted_compounds_{name}"] = int(
            compounds[f"is_conflicted_{name}"].sum()
        )
        LOG.info("endpoint %s compound labels: %s", name, dist)

    man.counts["unmapped_assessment_values"] = sorted(all_unmapped)
    man.counts["compounds"] = int(len(compounds))
    man.write()

    LOG.info("compound labels written: %s (%d compounds)", comp_path, len(compounds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
