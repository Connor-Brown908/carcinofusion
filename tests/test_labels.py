"""Label policy tests.

The property under test throughout: missing evidence is never negative
evidence, and conflicting evidence is never resolved by majority vote.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import load_config  # noqa: E402
from src.curation.harmonize_labels import (  # noqa: E402
    aggregate_to_compounds,
    build_assessment_map,
    confidence_for,
    normalise_assessment,
)

LAB_CFG = load_config("labels")
AGG = LAB_CFG["aggregation"]
CONF = LAB_CFG["confidence"]


def frame(rows: list[tuple]) -> pd.DataFrame:
    """(compound_id, species, label) tuples as a labelled-observation frame."""
    return pd.DataFrame(
        [{"compound_id": c, "species": s, "label_primary": v} for c, s, v in rows]
    )


def agg(rows: list[tuple]) -> pd.DataFrame:
    return aggregate_to_compounds(
        frame(rows), "label_primary", agg_cfg=AGG, confidence_cfg=CONF
    )


class TestAssessmentNormalisation:
    def test_case_and_whitespace_are_ignored(self):
        assert normalise_assessment("  Clear   Evidence  ") == "clear evidence"

    def test_trailing_punctuation_is_stripped(self):
        assert normalise_assessment("No evidence.") == "no evidence"

    @pytest.mark.parametrize("value", [None, "", "   ", float("nan"), pd.NA])
    def test_missing_values_become_none(self, value):
        assert normalise_assessment(value) is None

    def test_the_string_nan_is_treated_as_missing(self):
        """A parquet round-trip turns None into NaN, whose str() is 'nan'.

        Without this, a missing assessment is reported as an unrecognised
        policy rule called 'nan' and a curator is sent chasing a phantom gap.
        """
        assert normalise_assessment("nan") is None
        assert normalise_assessment(float("nan")) is None


class TestEndpointMapping:
    def test_primary_leaves_some_evidence_unresolved(self):
        m = build_assessment_map(LAB_CFG["endpoints"]["primary"])
        assert m["clear evidence of carcinogenic activity"] == 1
        assert m["no evidence of carcinogenic activity"] == 0
        assert m["some evidence of carcinogenic activity"] is None

    def test_sensitivity_counts_some_evidence_as_positive(self):
        m = build_assessment_map(LAB_CFG["endpoints"]["sensitivity"])
        assert m["some evidence of carcinogenic activity"] == 1

    def test_equivocal_is_unresolved_under_both_endpoints(self):
        for name in ("primary", "sensitivity"):
            m = build_assessment_map(LAB_CFG["endpoints"][name])
            assert m["equivocal evidence of carcinogenic activity"] is None

    def test_no_endpoint_maps_anything_to_negative_by_default(self):
        """Only explicitly listed terms may produce a 0."""
        for name in LAB_CFG["endpoints"]:
            cfg = LAB_CFG["endpoints"][name]
            negatives = {normalise_assessment(t) for t in cfg["negative"]}
            m = build_assessment_map(cfg)
            assert {k for k, v in m.items() if v == 0} == negatives


class TestMissingIsNotNegative:
    """The central safety property of the whole label policy."""

    def test_unrecognised_assessment_is_null_not_negative(self):
        m = build_assessment_map(LAB_CFG["endpoints"]["primary"])
        assert "probably fine honestly" not in m

    def test_compound_with_only_null_records_is_unresolved(self):
        out = agg([("A", "rat", None), ("A", "rat", None)])
        assert out.loc[0, "label"] is None

    def test_untested_compound_is_not_labelled_safe(self):
        out = agg([("A", "rat", None)])
        assert out.loc[0, "label"] is None
        assert out.loc[0, "label"] != 0


class TestAggregationWithoutMajorityVote:
    def test_all_positive_gives_positive(self):
        out = agg([("A", "rat", 1), ("A", "rat", 1), ("A", "mouse", 1)])
        assert out.loc[0, "label"] == 1
        assert not out.loc[0, "is_conflicted"]

    def test_all_negative_gives_negative(self):
        out = agg([("A", "rat", 0), ("A", "mouse", 0)])
        assert out.loc[0, "label"] == 0
        assert not out.loc[0, "is_conflicted"]

    def test_a_single_negative_record_is_not_enough_for_a_negative_label(self):
        """Policy v1.1.0 raised min_negative_records to 2.

        One positive bioassay establishes that a compound CAN induce tumours.
        One negative establishes only that this study did not detect them,
        which is a weaker claim — and rodent bioassays reproduce at roughly
        81% balanced accuracy. Supporting a waiver means being trusted on the
        negatives, so a lone negative stays unresolved.
        """
        assert AGG["min_negative_records"] >= 2
        out = agg([("A", "rat", 0)])
        assert out.loc[0, "label"] is None
        assert not out.loc[0, "is_conflicted"]
        assert "min_negative_records" in out.loc[0, "aggregation_reason"]

    def test_a_single_positive_record_still_gives_a_positive_label(self):
        """The asymmetry is the point: positives are not held to the same bar."""
        out = agg([("A", "rat", 1)])
        assert out.loc[0, "label"] == 1

    def test_the_corroboration_bar_applies_per_compound_not_per_species(self):
        """Two negatives in one species corroborate each other."""
        out = agg([("A", "rat", 0), ("A", "rat", 0)])
        assert out.loc[0, "label"] == 0

    def test_one_positive_against_many_negatives_is_conflicted_not_outvoted(self):
        """A majority vote would call this negative. The policy must not."""
        out = agg([("A", "rat", 1)] + [("A", "mouse", 0)] * 9)
        assert bool(out.loc[0, "is_conflicted"])
        assert out.loc[0, "label"] is None

    def test_one_negative_against_many_positives_is_conflicted(self):
        out = agg([("A", "rat", 0)] + [("A", "mouse", 1)] * 9)
        assert bool(out.loc[0, "is_conflicted"])
        assert out.loc[0, "label"] is None

    def test_positives_with_nulls_still_resolve_positive(self):
        out = agg([("A", "rat", 1), ("A", "rat", None)])
        assert out.loc[0, "label"] == 1
        assert not out.loc[0, "is_conflicted"]

    def test_conflicted_compounds_keep_their_evidence_counts(self):
        out = agg([("A", "rat", 1), ("A", "mouse", 0), ("A", "rat", None)])
        row = out.loc[0]
        assert row["n_positive"] == 1
        assert row["n_negative"] == 1
        assert row["n_indeterminate"] == 1


class TestSpeciesEligibility:
    def test_non_rodent_evidence_does_not_create_a_label(self):
        out = agg([("A", "hamster", 0)])
        assert out.loc[0, "label"] is None
        assert out.loc[0, "n_eligible_experiments"] == 0

    def test_non_rodent_records_are_still_counted(self):
        out = agg([("A", "hamster", 0), ("A", "rat", 1)])
        assert out.loc[0, "n_experiments"] == 2
        assert out.loc[0, "n_eligible_experiments"] == 1
        assert out.loc[0, "label"] == 1

    def test_non_rodent_cannot_create_a_conflict(self):
        """A hamster negative must not conflict with a rat positive."""
        out = agg([("A", "hamster", 0), ("A", "rat", 1)])
        assert not bool(out.loc[0, "is_conflicted"])


class TestConfidence:
    def test_confidence_rises_with_agreeing_records(self):
        assert confidence_for(3, CONF) == "high"
        assert confidence_for(2, CONF) == "medium"
        assert confidence_for(1, CONF) == "low"
        assert confidence_for(0, CONF) == "none"

    def test_unresolved_compounds_have_no_confidence(self):
        out = agg([("A", "rat", 1), ("A", "rat", 0)])
        assert out.loc[0, "label_confidence"] == "none"

    def test_three_agreeing_records_are_high_confidence(self):
        out = agg([("A", "rat", 1)] * 3)
        assert out.loc[0, "label_confidence"] == "high"


class TestPolicyVersioning:
    def test_the_policy_declares_a_version(self):
        assert LAB_CFG["label_policy_version"]

    def test_the_endpoint_is_experimental_not_human(self):
        """The endpoint must not overclaim what rodent data can support."""
        text = LAB_CFG["endpoint_definition"].lower()
        assert "rodent" in text
        assert "not a claim about human" in text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
