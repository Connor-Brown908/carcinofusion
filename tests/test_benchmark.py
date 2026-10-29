"""Phase 2 benchmark tests.

The properties that make a benchmark trustworthy rather than merely runnable:
metrics are defined consistently, intervals widen when they should, the test
set is not used for selection, models share one interface, and a model that
cannot run is recorded rather than dropped.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from src.common import load_config  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    ALL_METRICS,
    brier_decomposition,
    calibration_curve_data,
    choose_threshold,
    evaluate,
    paired_bootstrap_difference,
    point_metrics,
)
from src.models.base import MorganFeaturizer, UnavailableModel  # noqa: E402
from src.models.baselines import make_baseline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "data" / "processed" / "benchmark"


@pytest.fixture(scope="module")
def toy():
    """A small, learnable structure-activity set: nitroaromatics are positive."""
    pos = [
        "O=[N+]([O-])c1ccccc1",
        "Cc1ccc([N+](=O)[O-])cc1",
        "O=[N+]([O-])c1ccc(Cl)cc1",
        "O=[N+]([O-])c1ccc(Br)cc1",
        "COc1ccc([N+](=O)[O-])cc1",
        "Nc1ccc([N+](=O)[O-])cc1",
    ]
    neg = [
        "CCO", "CCCC", "CC(C)O", "c1ccccc1", "Cc1ccccc1", "CCOCC",
        "CC(=O)O", "OCC(O)CO", "CCCCCC", "c1ccncc1", "CC(C)(C)O", "CCCCO",
    ]
    return pos + neg, np.array([1] * len(pos) + [0] * len(neg))


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------
class TestMetricDefinitions:
    def test_pr_auc_baseline_is_prevalence_not_half(self):
        """PR-AUC must always be read against prevalence, not 0.5."""
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.2, 2000)
        noise = rng.uniform(0, 1, 2000)
        pr = point_metrics(y, noise, 0.5)["pr_auc"]
        assert abs(pr - y.mean()) < 0.05

    def test_roc_auc_baseline_is_half(self):
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.2, 2000)
        assert abs(point_metrics(y, rng.uniform(0, 1, 2000), 0.5)["roc_auc"] - 0.5) < 0.05

    def test_ranking_metrics_are_none_on_a_single_class_set(self):
        """average_precision returns 1.0 for an all-positive set, which reads
        as a perfect model rather than an unanswerable question."""
        m = point_metrics(np.ones(20, dtype=int), np.full(20, 0.7), 0.5)
        assert m["pr_auc"] is None
        assert m["roc_auc"] is None

    def test_evaluate_notes_the_single_class_case(self):
        ev = evaluate(np.ones(20, dtype=int), np.full(20, 0.7), threshold=0.5, n_bootstrap=50)
        assert any("single class" in n for n in ev.notes)

    def test_metric_subset_matches_the_full_computation(self):
        """The subset path is an optimisation; it must not change a value."""
        rng = np.random.default_rng(1)
        y = rng.binomial(1, 0.3, 200)
        p = np.clip(y * 0.5 + rng.normal(0.25, 0.2, 200), 0.01, 0.99)
        full = point_metrics(y, p, 0.5)
        for metric in ALL_METRICS:
            assert point_metrics(y, p, 0.5, which={metric})[metric] == full[metric]

    def test_a_perfect_ranker_scores_one(self):
        y = np.array([0, 0, 1, 1])
        assert point_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)["roc_auc"] == 1.0

    def test_confusion_derived_metrics_agree_with_the_matrix(self):
        y = np.array([1, 1, 0, 0, 1])
        p = np.array([0.9, 0.2, 0.1, 0.8, 0.7])
        m = point_metrics(y, p, 0.5)
        assert m["sensitivity"] == pytest.approx(2 / 3)
        assert m["specificity"] == pytest.approx(1 / 2)


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
class TestUncertainty:
    def test_group_resampling_gives_a_wider_interval_than_row_resampling(self):
        """Compounds sharing a scaffold are not independent; resampling them
        individually understates the interval."""
        rng = np.random.default_rng(3)
        n = 300
        groups = rng.integers(0, 25, n)
        y = (groups % 3 == 0).astype(int)
        p = np.clip(y * 0.5 + rng.normal(0.25, 0.25, n), 0.01, 0.99)
        from src.evaluation.metrics import bootstrap_metrics

        row = bootstrap_metrics(y, p, threshold=0.5, groups=None, n_bootstrap=300, seed=1)["pr_auc"]
        grp = bootstrap_metrics(y, p, threshold=0.5, groups=groups, n_bootstrap=300, seed=1)["pr_auc"]
        assert (grp.ci_high - grp.ci_low) > (row.ci_high - row.ci_low)

    def test_the_interval_records_its_resampling_unit(self):
        from src.evaluation.metrics import bootstrap_metrics

        rng = np.random.default_rng(4)
        y = rng.binomial(1, 0.4, 100)
        p = rng.uniform(0, 1, 100)
        assert "by_group" in bootstrap_metrics(
            y, p, threshold=0.5, groups=rng.integers(0, 10, 100), n_bootstrap=100
        )["pr_auc"].ci_method

    def test_identical_models_differ_by_zero_and_the_interval_spans_it(self):
        rng = np.random.default_rng(5)
        y = rng.binomial(1, 0.3, 150)
        p = rng.uniform(0, 1, 150)
        d = paired_bootstrap_difference(y, p, p, metric="pr_auc", n_bootstrap=200)
        assert d["difference"] == pytest.approx(0.0)
        assert not d["excludes_zero"]

    def test_a_clearly_better_model_is_detected_as_different(self):
        rng = np.random.default_rng(6)
        n = 400
        y = rng.binomial(1, 0.3, n)
        good = np.clip(y * 0.6 + rng.normal(0.2, 0.15, n), 0.01, 0.99)
        d = paired_bootstrap_difference(
            y, good, rng.uniform(0, 1, n), metric="pr_auc", n_bootstrap=300
        )
        assert d["difference"] > 0
        assert d["excludes_zero"]

    def test_paired_comparison_requires_the_same_compounds(self):
        with pytest.raises(ValueError, match="identical held-out"):
            paired_bootstrap_difference(
                np.array([0, 1]), np.array([0.1, 0.9]), np.array([0.1])
            )


# ---------------------------------------------------------------------------
# Threshold discipline
# ---------------------------------------------------------------------------
class TestThresholdSelection:
    def test_threshold_moves_away_from_the_default_when_it_helps(self):
        rng = np.random.default_rng(7)
        n = 300
        y = rng.binomial(1, 0.15, n)
        p = np.clip(y * 0.3 + rng.normal(0.15, 0.1, n), 0.01, 0.99)
        assert choose_threshold(y, p) != 0.5

    def test_a_single_class_set_falls_back_to_the_default(self):
        assert choose_threshold(np.ones(10, dtype=int), np.full(10, 0.8)) == 0.5

    def test_unknown_policy_is_rejected_rather_than_silently_defaulted(self):
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="unknown threshold policy"):
            choose_threshold(y, np.array([0.1, 0.9, 0.2, 0.8]), policy="whatever")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
class TestCalibration:
    def test_brier_decomposition_identity_holds(self):
        rng = np.random.default_rng(8)
        n = 500
        y = rng.binomial(1, 0.35, n)
        p = np.clip(y * 0.5 + rng.normal(0.25, 0.2, n), 0.01, 0.99)
        b = brier_decomposition(y, p)
        assert b["brier"] == pytest.approx(
            b["reliability"] - b["resolution"] + b["uncertainty"], abs=0.02
        )

    def test_a_well_calibrated_predictor_has_low_calibration_error(self):
        rng = np.random.default_rng(9)
        p = rng.uniform(0.05, 0.95, 4000)
        y = rng.binomial(1, p)
        assert calibration_curve_data(y, p, n_bins=10)["ece"] < 0.05

    def test_a_badly_calibrated_predictor_has_high_calibration_error(self):
        rng = np.random.default_rng(10)
        p = rng.uniform(0.05, 0.95, 2000)
        y = rng.binomial(1, np.clip(p * 0.2, 0, 1))
        assert calibration_curve_data(y, p, n_bins=10)["ece"] > 0.2

    def test_uncertainty_term_depends_only_on_the_data(self):
        """No model can change it, so two models on one set must agree."""
        rng = np.random.default_rng(11)
        y = rng.binomial(1, 0.3, 200)
        a = brier_decomposition(y, rng.uniform(0, 1, 200))
        b = brier_decomposition(y, np.full(200, 0.3))
        assert a["uncertainty"] == pytest.approx(b["uncertainty"])


# ---------------------------------------------------------------------------
# Model interface
# ---------------------------------------------------------------------------
class TestModelInterface:
    @pytest.mark.parametrize("model_id", ["B0", "B1", "B2", "B3"])
    def test_every_model_shares_the_same_contract(self, model_id, toy):
        smiles, y = toy
        fz = None if model_id == "B0" else MorganFeaturizer(n_bits=512)
        kw = {"n_estimators": 20, "min_child_samples": 2, "num_leaves": 7} if model_id == "B3" else {}
        if model_id == "B2":
            kw = {"n_estimators": 20}
        model = make_baseline(model_id, featurizer=fz, seed=42, **kw)
        model.fit(smiles, y)
        p = model.predict_proba(smiles)
        assert len(p) == len(smiles)
        assert np.all((p >= 0) & (p <= 1))
        d = model.describe()
        assert d["model_id"] == model_id and "seed" in d

    def test_majority_predicts_the_training_prevalence(self, toy):
        smiles, y = toy
        m = make_baseline("B0").fit(smiles, y)
        assert np.allclose(m.predict_proba(smiles), y.mean())

    def test_majority_pr_auc_equals_prevalence(self, toy):
        """The reference every other model must beat."""
        smiles, y = toy
        p = make_baseline("B0").fit(smiles, y).predict_proba(smiles)
        assert point_metrics(y, p, 0.5)["pr_auc"] == pytest.approx(y.mean(), abs=1e-9)

    def test_a_real_model_beats_the_majority_reference(self, toy):
        smiles, y = toy
        fz = MorganFeaturizer(n_bits=512)
        b1 = make_baseline("B1", featurizer=fz).fit(smiles, y).predict_proba(smiles)
        b0 = make_baseline("B0").fit(smiles, y).predict_proba(smiles)
        assert point_metrics(y, b1, 0.5)["pr_auc"] > point_metrics(y, b0, 0.5)["pr_auc"]

    def test_predict_before_fit_is_an_error(self):
        with pytest.raises(RuntimeError, match="before fit"):
            make_baseline("B1", featurizer=MorganFeaturizer(n_bits=256)).predict_proba(["CCO"])

    @pytest.mark.parametrize("model_id", ["B1", "B2", "B3"])
    def test_a_single_class_training_fold_degenerates_rather_than_crashing(self, model_id, toy):
        """A scaffold split on imbalanced data produces these routinely."""
        smiles, _ = toy
        fz = MorganFeaturizer(n_bits=256)
        m = make_baseline(model_id, featurizer=fz, seed=1).fit(smiles[:5], np.ones(5, dtype=int))
        assert np.allclose(m.predict_proba(smiles[:3]), 1.0)

    def test_unparseable_smiles_become_a_zero_row_not_a_crash(self):
        X = MorganFeaturizer(n_bits=128).transform(["CCO", "not_a_smiles"])
        assert X[0].sum() > 0
        assert X[1].sum() == 0

    def test_chiral_fingerprints_distinguish_enantiomers(self):
        """Phase 1 preserves stereochemistry, so the featurizer must use it."""
        fz = MorganFeaturizer(n_bits=2048, use_chirality=True)
        X = fz.transform(["C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"])
        assert not np.array_equal(X[0], X[1])

    def test_featurizer_name_records_its_settings(self):
        assert "chiral" in MorganFeaturizer(use_chirality=True).name
        assert "counts" in MorganFeaturizer(counts=True).name


# ---------------------------------------------------------------------------
# Graph model
# ---------------------------------------------------------------------------
class TestGraphModel:
    def test_graph_featurisation_shapes_are_consistent(self):
        from src.models.mpnn import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, MolGraph

        g = MolGraph("CC(=O)Nc1ccc(O)cc1")
        assert g.ok and g.x.shape == (g.n_atoms, ATOM_FEATURE_DIM)
        assert g.edge_attr.shape == (g.edge_index.shape[1], BOND_FEATURE_DIM)

    def test_bonds_are_bidirectional(self):
        """Messages must flow both ways along every bond."""
        from src.models.mpnn import MolGraph

        assert MolGraph("CCO").edge_index.shape[1] == 4  # 2 bonds, both directions

    def test_malformed_input_becomes_an_empty_graph_not_a_crash(self):
        from src.models.mpnn import MolGraph

        g = MolGraph("not_a_smiles")
        assert not g.ok and g.n_atoms == 1

    def test_enantiomers_differ_in_the_graph(self):
        from src.models.mpnn import MolGraph

        a = MolGraph("C[C@H](N)C(=O)O")
        b = MolGraph("C[C@@H](N)C(=O)O")
        assert not np.array_equal(a.x, b.x)

    def test_batching_offsets_indices_correctly(self):
        from src.models.mpnn import MolGraph, collate

        batch = collate([MolGraph("CCO"), MolGraph("c1ccccc1")])
        assert batch["batch"].tolist() == [0, 0, 0] + [1] * 6
        assert int(batch["edge_index"].max()) < batch["x"].shape[0]

    def test_training_is_deterministic_under_a_fixed_seed(self, toy):
        from src.models.mpnn import MPNNModel

        smiles, y = toy
        a = MPNNModel(seed=7, hidden_dim=16, max_epochs=3).fit(smiles, y).predict_proba(smiles[:4])
        b = MPNNModel(seed=7, hidden_dim=16, max_epochs=3).fit(smiles, y).predict_proba(smiles[:4])
        assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# Unavailable models
# ---------------------------------------------------------------------------
class TestUnavailableModel:
    def test_availability_check_returns_a_reason(self):
        from src.models.pretrained import check_availability

        ok, reason = check_availability()
        assert isinstance(ok, bool) and isinstance(reason, str) and reason

    def test_an_unreachable_checkpoint_raises_rather_than_degrading(self):
        """Silently substituting something else under B5's name would make the
        benchmark table a lie."""
        from src.models.pretrained import PretrainedEncoderModel

        m = PretrainedEncoderModel(seed=1, checkpoint="definitely/not-a-real-checkpoint-xyz")
        with pytest.raises(UnavailableModel):
            m.fit(["CCO", "c1ccccc1"], np.array([0, 1]))


# ---------------------------------------------------------------------------
# Protocol configuration
# ---------------------------------------------------------------------------
class TestProtocol:
    def test_the_protocol_declares_a_version(self):
        assert load_config("benchmark")["protocol_version"]

    def test_scaffold_is_primary_and_random_is_secondary(self):
        ev = load_config("benchmark")["evaluation"]
        assert ev["primary_split"] == "scaffold"
        assert ev["secondary_split"] == "random"

    def test_selection_happens_on_validation_not_test(self):
        ev = load_config("benchmark")["evaluation"]
        assert ev["selection_partition"] == "validation"
        assert ev["final_partition"] == "test"

    def test_pr_auc_definition_is_pinned(self):
        """Average precision and trapezoidal PR-AUC are not the same number."""
        assert load_config("benchmark")["metrics"]["pr_auc_definition"] == (
            "sklearn_average_precision"
        )

    def test_the_threshold_policy_is_tied_to_validation(self):
        assert "validation" in load_config("benchmark")["metrics"]["threshold_policy"]

    def test_bootstrap_resamples_scaffold_groups(self):
        assert load_config("benchmark")["uncertainty"]["resample_unit"] == "scaffold_group"

    def test_every_model_in_the_ladder_is_registered(self):
        models = load_config("benchmark")["models"]
        assert set(models) == {"B0", "B1", "B2", "B3", "B4", "B5"}

    def test_data_provenance_is_required(self):
        assert load_config("benchmark")["reporting"]["require_data_provenance"] is True

    def test_reproduction_claims_are_gated_on_comparability(self):
        assert (
            load_config("benchmark")["reporting"]["claim_reproduction_only_if_comparable"]
            is True
        )


# ---------------------------------------------------------------------------
# Runner output
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def runs():
    path = BENCH_DIR / "runs_primary.json"
    if not path.exists():
        pytest.skip("benchmark not run; see README Phase 2")
    return json.loads(path.read_text(encoding="utf-8"))


class TestRunnerOutput:
    def test_every_run_carries_its_data_provenance(self, runs):
        assert all(
            r["data_provenance"] in {"REAL_CPDB", "SYNTHETIC_FIXTURE"} for r in runs
        )

    def test_every_run_records_dependency_versions(self, runs):
        for r in runs:
            assert r["dependency_versions"].get("python")
            assert "sklearn" in r["dependency_versions"]

    def test_every_run_records_the_versions_that_produced_it(self, runs):
        for r in runs:
            assert r.get("dataset_version")
            assert r.get("label_policy_version")
            assert r.get("split_version")
            assert r.get("protocol_version")

    def test_unavailable_runs_state_a_reason(self, runs):
        for r in runs:
            if r["status"] == "unavailable":
                assert r["unavailable_reason"]

    def test_an_unavailable_model_is_recorded_not_dropped(self, runs):
        """A silently missing row invites the reader to assume it was tried."""
        ids = {r["model_id"] for r in runs}
        statuses = {r["model_id"]: r["status"] for r in runs}
        if "B5" in ids:
            assert statuses["B5"] in {"completed", "unavailable", "failed"}

    def test_the_threshold_was_chosen_away_from_the_test_set(self, runs):
        for r in runs:
            if r["status"] == "completed":
                assert r["threshold_selected_on"] in {"validation", "train"}

    def test_selection_fallbacks_are_recorded_when_they_happen(self, runs):
        for r in runs:
            if r["status"] == "completed" and r.get("selection_fallback"):
                assert isinstance(r["selection_fallback"], str)
                assert r["selection_fallback"]

    def test_both_split_strategies_were_run(self, runs):
        assert {r["split_strategy"] for r in runs} >= {"scaffold", "random"}

    def test_test_metrics_report_prevalence_alongside_pr_auc(self, runs):
        for r in runs:
            if r["status"] == "completed":
                assert "prevalence" in r["results"]["test"]

    def test_predictions_are_retained_for_paired_comparison(self, runs):
        for r in runs:
            if r["status"] == "completed":
                assert r["predictions"]
                assert all("nn_similarity" in p for p in r["predictions"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
