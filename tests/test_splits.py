"""Split tests.

The property under test: no compound, and in the scaffold split no scaffold
group, may appear in more than one partition — and a compound must never be
lost between stages.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import load_config  # noqa: E402
from src.datasets.split_dataset import (  # noqa: E402
    PARTITIONS,
    LeakageError,
    assign_groups,
    verify_no_leakage,
)
from src.features.fingerprints import nearest_neighbour_similarities  # noqa: E402

SPLIT_CFG = load_config("splits")
FRACTIONS = SPLIT_CFG["fractions"]


def rng():
    return np.random.default_rng(SPLIT_CFG["random_seed"])


class TestGroupAssignment:
    def test_every_group_is_assigned_exactly_one_partition(self):
        groups = [(f"g{i}", 1) for i in range(100)]
        a = assign_groups(groups, FRACTIONS, rng=rng(), largest_first=False)
        assert len(a) == len(groups)
        assert set(a.values()) <= set(PARTITIONS)

    def test_no_group_is_ever_split(self):
        """A group key maps to one partition, never two."""
        groups = [(f"g{i}", 5) for i in range(50)]
        a = assign_groups(groups, FRACTIONS, rng=rng(), largest_first=True)
        assert all(isinstance(v, str) for v in a.values())

    def test_train_receives_the_largest_share(self):
        groups = [(f"g{i}", 1) for i in range(100)]
        a = assign_groups(groups, FRACTIONS, rng=rng(), largest_first=False)
        counts = pd.Series(list(a.values())).value_counts()
        assert counts["train"] > counts.get("validation", 0)
        assert counts["train"] > counts.get("test", 0)

    def test_realised_fractions_track_the_targets(self):
        groups = [(f"g{i}", 1) for i in range(1000)]
        a = assign_groups(groups, FRACTIONS, rng=rng(), largest_first=False)
        frac = pd.Series(list(a.values())).value_counts(normalize=True)
        assert abs(frac["train"] - FRACTIONS["train"]) < 0.05

    def test_largest_group_first_puts_the_big_family_in_train(self):
        """Rare chemistry should land in validation and test, not train."""
        groups = [("huge", 70)] + [(f"g{i}", 1) for i in range(30)]
        a = assign_groups(groups, FRACTIONS, rng=rng(), largest_first=True)
        assert a["huge"] == "train"

    def test_assignment_is_reproducible_under_the_same_seed(self):
        groups = [(f"g{i}", (i % 7) + 1) for i in range(60)]
        a = assign_groups(groups, FRACTIONS, rng=rng(), largest_first=False)
        b = assign_groups(groups, FRACTIONS, rng=rng(), largest_first=False)
        assert a == b

    def test_no_partition_is_left_empty_when_groups_allow(self):
        groups = [(f"g{i}", 1) for i in range(20)]
        a = assign_groups(groups, FRACTIONS, rng=rng(), largest_first=True)
        assert set(a.values()) == set(PARTITIONS)

    def test_empty_input_is_handled(self):
        assert assign_groups([], FRACTIONS, rng=rng(), largest_first=True) == {}


class _Man:
    """Minimal manifest stand-in that records warnings."""

    def __init__(self):
        self.warnings: list[str] = []

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


class TestLeakageDetection:
    def _frame(self, rows):
        return pd.DataFrame(
            rows, columns=["inchi_key", "scaffold_group", "split_assignment"]
        )

    def test_clean_split_passes(self):
        df = self._frame(
            [
                ("K1", "S1", "train"),
                ("K2", "S1", "train"),
                ("K3", "S2", "test"),
            ]
        )
        out = verify_no_leakage(df, strategy="scaffold", cfg=SPLIT_CFG, man=_Man())
        assert out["inchikey_spanning_partitions"] == 0
        assert out["scaffold_groups_spanning_partitions"] == 0

    def test_duplicate_inchikey_across_partitions_raises(self):
        df = self._frame([("K1", "S1", "train"), ("K1", "S2", "test")])
        with pytest.raises(LeakageError, match="InChIKey"):
            verify_no_leakage(df, strategy="random", cfg=SPLIT_CFG, man=_Man())

    def test_scaffold_group_spanning_partitions_raises(self):
        df = self._frame([("K1", "S1", "train"), ("K2", "S1", "test")])
        with pytest.raises(LeakageError, match="scaffold group"):
            verify_no_leakage(df, strategy="scaffold", cfg=SPLIT_CFG, man=_Man())

    def test_scaffold_overlap_is_tolerated_in_the_random_split(self):
        """The random split is diagnostic; only the scaffold split forbids it."""
        df = self._frame([("K1", "S1", "train"), ("K2", "S1", "test")])
        out = verify_no_leakage(df, strategy="random", cfg=SPLIT_CFG, man=_Man())
        assert "scaffold_groups_spanning_partitions" not in out

    def test_excluded_rows_do_not_trigger_leakage(self):
        df = self._frame([("K1", "S1", "train"), ("K1", "S1", "excluded")])
        out = verify_no_leakage(df, strategy="scaffold", cfg=SPLIT_CFG, man=_Man())
        assert out["inchikey_spanning_partitions"] == 0


class TestSimilarityMeasurement:
    def test_identical_molecules_score_one(self):
        sims = nearest_neighbour_similarities(["CCO"], ["CCO"])
        assert sims[0] == pytest.approx(1.0)

    def test_dissimilar_molecules_score_low(self):
        sims = nearest_neighbour_similarities(["CCCCCCCCCC"], ["c1ccc2ccccc2c1"])
        assert sims[0] < 0.3

    def test_enantiomers_are_indistinguishable_to_the_fingerprint(self):
        """Why the scaffold split must keep stereoisomers together.

        Morgan fingerprints ignore chirality, so separating enantiomers across
        partitions leaks even though no InChIKey is shared.
        """
        sims = nearest_neighbour_similarities(
            ["C[C@H](N)C(=O)O"], ["C[C@@H](N)C(=O)O"]
        )
        assert sims[0] == pytest.approx(1.0)

    def test_unparseable_queries_are_dropped_not_scored(self):
        sims = nearest_neighbour_similarities(["not_a_smiles"], ["CCO"])
        assert sims.size == 0


@pytest.fixture(scope="module")
def built():
    """The splits produced by the end-to-end fixture run, if present."""
    processed = Path(__file__).resolve().parent.parent / "data" / "processed"
    path = processed / "splits_primary.parquet"
    if not path.exists():
        pytest.skip("run the pipeline on the fixture first (see tests/run_pipeline.sh)")
    return pd.read_parquet(path)


class TestEndToEndSplits:
    def test_no_compound_appears_twice_within_a_strategy(self, built):
        for strategy, grp in built.groupby("split_strategy"):
            assert not grp["compound_id"].duplicated().any(), strategy

    def test_no_inchikey_spans_partitions(self, built):
        for strategy, grp in built.groupby("split_strategy"):
            assigned = grp[grp["split_assignment"].isin(PARTITIONS)]
            spanning = assigned.groupby("inchi_key")["split_assignment"].nunique()
            assert (spanning <= 1).all(), strategy

    def test_scaffold_groups_do_not_span_partitions(self, built):
        grp = built[built["split_strategy"] == "scaffold"]
        assigned = grp[grp["split_assignment"].isin(PARTITIONS)]
        spanning = assigned.groupby("scaffold_group")["split_assignment"].nunique()
        assert (spanning <= 1).all()

    def test_every_compound_is_accounted_for(self, built):
        """Nothing may vanish between build and split."""
        processed = Path(__file__).resolve().parent.parent / "data" / "processed"
        ds = pd.read_parquet(processed / "model_dataset_primary.parquet")
        for strategy, grp in built.groupby("split_strategy"):
            assert set(grp["compound_id"]) == set(ds["compound_id"]), strategy

    def test_only_eligible_compounds_are_partitioned(self, built):
        assigned = built[built["split_assignment"].isin(PARTITIONS)]
        assert (assigned["training_eligibility"] == "eligible").all()

    def test_ineligible_compounds_are_marked_excluded(self, built):
        ineligible = built[built["training_eligibility"] != "eligible"]
        assert (ineligible["split_assignment"] == "excluded").all()

    def test_no_partitioned_compound_has_a_null_label(self, built):
        assigned = built[built["split_assignment"].isin(PARTITIONS)]
        assert assigned["label"].notna().all()

    def test_no_conflicted_compound_is_partitioned(self, built):
        assigned = built[built["split_assignment"].isin(PARTITIONS)]
        assert not assigned["is_conflicted"].any()

    def test_split_group_is_never_null_for_partitioned_rows(self, built):
        assigned = built[built["split_assignment"].isin(PARTITIONS)]
        assert assigned["split_group"].notna().all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
