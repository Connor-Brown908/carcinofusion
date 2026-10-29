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
from src.ingestion.parse_cpdb import (  # noqa: E402
    _null_tokens_resolver,
    check_null_tokens_against_labels,
)
from src.curation.harmonize_labels import (  # noqa: E402
    VocabularyMismatch,
    aggregate_to_compounds,
    build_assessment_map,
    check_unmapped,
    confidence_for,
    label_records,
    missing_values_from,
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


class _Manifest:
    """Minimal stand-in for the real manifest; collects warnings."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warn(self, message: str) -> None:
        self.warnings.append(message)


class TestVocabularyMismatch:
    """A vocabulary the policy does not recognise must stop the run.

    This is the regression guard for the failure mode that cost the most
    time on this project: unmapped assessments became null, the compounds
    carrying them lost their labels, and the only visible symptom was a
    benchmark built from eleven test compounds. Silence is the bug.
    """

    ENDPOINT = "primary"

    def _labels(self, values: list[object]):
        experiments = pd.DataFrame({"author_assessment": values})
        labels, _ = label_records(
            experiments,
            self.ENDPOINT,
            LAB_CFG["endpoints"][self.ENDPOINT],
            man=_Manifest(),
            missing_values=missing_values_from(LAB_CFG),
        )
        return experiments, labels

    def test_the_real_cpdb_distribution_is_fully_mapped(self):
        """The observed distribution of both CPDB experiment tables.

        Counts are from scripts/inspect_values.py on the real files:
        cpdb.lit.tab (14,207 rows) and cpdb.ncintp.tab (8,796 rows).
        Nothing in either file may be unmapped.
        """
        observed = (
            ["0"] * 5234 + ["-"] * 5042 + ["+"] * 3931          # cpdb.lit.tab
            + ["0"] * 5507 + ["c"] * 1720 + ["-"] * 994         # cpdb.ncintp.tab
            + ["e"] * 269 + ["p"] * 200 + ["a"] * 106
        )
        experiments, labels = self._labels(observed)
        check_unmapped(
            experiments,
            labels,
            endpoint_name=self.ENDPOINT,
            policy=self.policy,
            missing_values=missing_values_from(LAB_CFG),
        )
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        # 3,931 '+' and 1,720 'c' positive; 5,042 + 994 '-' negative.
        assert n_pos == 5651, n_pos
        assert n_neg == 6036, n_neg

    @property
    def policy(self) -> dict:
        policy = LAB_CFG.get("unmapped_assessment_policy")
        assert policy is not None, (
            "labels.yaml must define unmapped_assessment_policy; without it "
            "an unrecognised vocabulary silently empties the dataset."
        )
        return policy

    # The authoritative code list, quoted from CPDB's own appendix at
    # https://files.toxplanet.com/cpdb/pdfs/opinion.pdf
    #   +  literature: site evaluated positive
    #   -  site evaluated negative
    #   c  NCI "carcinogenic" / NTP "clear evidence"
    #   p  NTP "some evidence"  (weaker than clear)
    #   e  NTP "equivocal evidence"
    #   a  NCI "associated"; NTP-reassigned to equivocal
    #   i  NTP "inadequate study"
    EXPECTED = {
        "primary":     {"+": 1, "-": 0, "c": 1,
                        "p": None, "e": None, "a": None, "i": None},
        "sensitivity": {"+": 1, "-": 0, "c": 1, "p": 1,
                        "e": None, "a": None, "i": None},
    }

    def test_every_documented_opinion_code_maps_as_the_appendix_defines(self):
        """Lock the vocabulary to the appendix, code by code.

        'c' being absent from this mapping is what reduced a 1,500-chemical
        database to an 11-compound benchmark: it is the NCI/NTP file's
        primary positive code, 19.6% of its records.
        """
        for endpoint, expected in self.EXPECTED.items():
            mapping = build_assessment_map(LAB_CFG["endpoints"][endpoint])
            for code, want in expected.items():
                assert code in mapping, (
                    f"endpoint {endpoint}: opinion code {code!r} is documented "
                    f"in the CPDB appendix but appears in no vocabulary list, "
                    f"so every record carrying it would lose its label."
                )
                assert mapping[code] == want, (
                    f"endpoint {endpoint}: code {code!r} maps to "
                    f"{mapping[code]!r}, expected {want!r}."
                )

    def test_some_evidence_is_not_a_clear_positive(self):
        """'p' is NTP 'some evidence', explicitly weaker than clear evidence.

        The primary endpoint's own definition sends some-evidence findings to
        unresolved. Calling them positive there would overstate 200 records.
        """
        primary = build_assessment_map(LAB_CFG["endpoints"]["primary"])
        assert primary["p"] is None
        sensitivity = build_assessment_map(LAB_CFG["endpoints"]["sensitivity"])
        assert sensitivity["p"] == 1, (
            "the sensitivity endpoint exists precisely to count some-evidence "
            "as positive; if it does not, the two endpoints are identical."
        )

    def test_no_invented_codes_remain(self):
        """'n' was an invented negative code. No such opinion code exists."""
        documented = set("+-cpeai")
        for endpoint, cfg in LAB_CFG["endpoints"].items():
            mapping = build_assessment_map(cfg)
            for code in mapping:
                if code is None or len(code) > 1:
                    continue  # the NTP prose phrasings, kept for other sources
                assert code in documented, (
                    f"endpoint {endpoint}: {code!r} is a single-character "
                    f"opinion code that the CPDB appendix does not define. "
                    f"Mapping an invented code invents evidence."
                )

    def test_blank_opinion_is_missing_not_a_code(self):
        """CPDB writes 'no opinion stated' as '0' in the tab files.

        Treated as a code it looks like a vocabulary gap across two thirds of
        the NCI/NTP file; treated as negative it invents clean compounds.
        """
        missing = missing_values_from(LAB_CFG)
        assert "0" in missing
        assert normalise_assessment("0", missing_values=missing) is None
        # It must not also be a label, or the code would be silently voided.
        for cfg in LAB_CFG["endpoints"].values():
            for bucket in ("positive", "negative", "unresolved"):
                assert "0" not in [str(x) for x in cfg.get(bucket, [])]

    def test_a_sentinel_that_is_also_a_code_is_rejected(self):
        contradictory = {
            "missing_assessment_values": ["0", "c"],
            "endpoints": {"primary": {"positive": ["c"]}},
        }
        with pytest.raises(ValueError, match="cannot"):
            missing_values_from(contradictory)

    def test_unrecognised_vocabulary_raises(self):
        experiments, labels = self._labels(["NP"] * 30 + ["EE"] * 10)
        with pytest.raises(VocabularyMismatch) as exc:
            check_unmapped(
                experiments,
                labels,
                endpoint_name=self.ENDPOINT,
                policy=self.policy,
            )
        message = str(exc.value)
        # The error must name the offending values and their counts, or it
        # is no more actionable than the silent failure it replaces.
        assert "np" in message.lower()
        assert "30" in message
        assert "configs/labels.yaml" in message

    def test_a_small_tail_of_unknown_codes_does_not_abort(self):
        """An undocumented code appearing rarely must not stop a whole run.

        The check guards against a wholesale vocabulary mismatch, not against
        every stray value, or it would be unusable on real data.
        """
        experiments, labels = self._labels(
            ["+"] * 40 + ["-"] * 40 + ["?"] * 10 + ["Q"] * 5
        )
        check_unmapped(
            experiments,
            labels,
            endpoint_name=self.ENDPOINT,
            policy=self.policy,
            missing_values=missing_values_from(LAB_CFG),
        )

    def test_absent_assessment_is_not_a_mismatch(self):
        """No assessment is missing data, not a vocabulary problem."""
        experiments, labels = self._labels(["+"] * 5 + [None] * 95)
        check_unmapped(
            experiments,
            labels,
            endpoint_name=self.ENDPOINT,
            policy=self.policy,
        )

    def test_threshold_of_one_disables_the_check(self):
        """An explicit escape hatch must exist and must work."""
        experiments, labels = self._labels(["NP"] * 50)
        check_unmapped(
            experiments,
            labels,
            endpoint_name=self.ENDPOINT,
            policy={"fail_above_fraction": 1.0},
        )


class TestParserDoesNotPreemptThePolicy:
    """The parser must not null a value the label policy calls a code.

    This is the regression guard for the most expensive bug in the project.
    configs/cpdb_columns.yaml listed '-' as a global no-value token, and the
    parser runs before the label policy, so every negative author opinion in
    CPDB — 6,036 records, a quarter of the database — became null at stage 2.
    Nothing downstream could see it: the vocabulary check had no unmapped
    value to report, the audit found no conflicts, and the dataset emerged
    with 700 positives and zero negatives.
    """

    def test_the_assessment_column_keeps_every_label_code(self):
        colcfg = load_config("cpdb_columns")
        tokens = {
            str(t).strip().lower()
            for t in _null_tokens_resolver(colcfg)("author_assessment")
        }
        for endpoint, cfg in LAB_CFG["endpoints"].items():
            for bucket in ("positive", "negative", "unresolved"):
                for term in cfg.get(bucket, []):
                    token = str(term).strip().lower()
                    assert token not in tokens, (
                        f"{token!r} is a {bucket} code for endpoint "
                        f"{endpoint} but the parser nulls it before the "
                        f"label policy runs, so those records lose their "
                        f"label with no unmapped value reported."
                    )

    def test_the_minus_code_specifically_survives_parsing(self):
        """Named explicitly, because this one cost a positives-only dataset."""
        colcfg = load_config("cpdb_columns")
        tokens = _null_tokens_resolver(colcfg)("author_assessment")
        assert "-" not in tokens
        assert "" in tokens, "the empty string is still no-value"

    def test_other_columns_still_treat_a_dash_as_no_value(self):
        """The override is scoped: a bare '-' in a numeric column is blank."""
        colcfg = load_config("cpdb_columns")
        resolver = _null_tokens_resolver(colcfg)
        assert "-" in resolver("td50")
        assert "-" in resolver("target_organ")

    def test_the_cross_config_guard_fires_on_a_collision(self):
        colcfg = load_config("cpdb_columns")
        reintroduced = {
            k: v for k, v in colcfg.items() if k != "null_tokens_by_column"
        }
        with pytest.raises(ValueError, match="contradict each other"):
            check_null_tokens_against_labels(reintroduced, LAB_CFG)

    def test_the_guard_names_the_offending_token_and_the_remedy(self):
        colcfg = load_config("cpdb_columns")
        reintroduced = {
            k: v for k, v in colcfg.items() if k != "null_tokens_by_column"
        }
        with pytest.raises(ValueError) as exc:
            check_null_tokens_against_labels(reintroduced, LAB_CFG)
        message = str(exc.value)
        assert "'-'" in message
        assert "null_tokens_by_column" in message


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
