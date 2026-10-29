"""Phase 1 acceptance criteria, as executable tests.

One class per criterion. These assert the properties against the artefacts the
pipeline actually produced, so the claim that Phase 1 is complete is checked
rather than asserted.

Run the fixture pipeline first:  bash tests/run_pipeline.sh
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
MANIFESTS = ROOT / "manifests"
PARTITIONS = ("train", "validation", "test")


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"{path.name} absent; run bash tests/run_pipeline.sh first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def experiments():
    return _load(INTERIM / "carcinogenicity_experiments.parquet")


@pytest.fixture(scope="module")
def compounds():
    return _load(INTERIM / "compound_identity.parquet")


@pytest.fixture(scope="module")
def review():
    return _load(INTERIM / "structure_review.parquet")


@pytest.fixture(scope="module")
def observations():
    return _load(INTERIM / "observations_labeled.parquet")


@pytest.fixture(scope="module")
def dataset():
    return _load(PROCESSED / "model_dataset_primary.parquet")


@pytest.fixture(scope="module")
def splits():
    return _load(PROCESSED / "splits_primary.parquet")


@pytest.fixture(scope="module")
def ledger():
    return _load(PROCESSED / "curation_ledger.parquet")


@pytest.fixture(scope="module")
def audit():
    path = PROCESSED / "audit_report.json"
    if not path.exists():
        pytest.skip("audit report absent; run bash tests/run_pipeline.sh first")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Traceability
# ---------------------------------------------------------------------------
class TestTraceability:
    """Every included label can be traced to its source evidence."""

    def test_every_labelled_compound_has_source_records(self, dataset, observations):
        labelled = dataset[dataset["label"].notna()]["compound_id"]
        have_evidence = set(observations["compound_id"].dropna())
        assert set(labelled) <= have_evidence

    def test_every_record_names_its_source_file_and_line(self, experiments):
        assert experiments["source_record_id"].notna().all()
        assert experiments["source_record_id"].str.contains(":").all()

    def test_every_record_names_its_source_database(self, experiments):
        assert experiments["source_database"].notna().all()

    def test_the_chain_from_label_to_raw_line_is_unbroken(self, dataset, observations):
        """compound -> experiment record -> source file and line number."""
        cid = dataset[dataset["label"].notna()].iloc[0]["compound_id"]
        recs = observations[observations["compound_id"] == cid]
        assert len(recs) >= 1
        assert recs["experiment_id"].notna().all()
        assert recs["source_record_id"].notna().all()
        assert recs["author_assessment_raw"].notna().any()

    def test_every_compound_label_states_its_reason(self, dataset):
        assert dataset["aggregation_reason"].notna().all()

    def test_label_policy_version_is_stamped_on_every_row(self, dataset):
        assert dataset["label_policy_version"].notna().all()

    def test_raw_file_checksums_are_recorded_at_parse_time(self):
        path = MANIFESTS / "02_parse_cpdb_latest.json"
        if not path.exists():
            pytest.skip("parse manifest absent")
        rec = json.loads(path.read_text(encoding="utf-8"))
        inputs = [i for i in rec["inputs"] if i.get("file_key")]
        assert inputs
        assert all(i.get("sha256") for i in inputs)


# ---------------------------------------------------------------------------
# 2. Structure integrity
# ---------------------------------------------------------------------------
class TestStructureIntegrity:
    """Every included compound has a valid, documented molecular representation."""

    def test_no_included_compound_lacks_a_structure(self, dataset):
        assert dataset["canonical_smiles"].notna().all()
        assert dataset["inchi_key"].notna().all()

    def test_included_structures_all_parse(self, dataset):
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        assert all(
            Chem.MolFromSmiles(s) is not None for s in dataset["canonical_smiles"]
        )

    def test_only_successfully_standardized_statuses_are_included(self, dataset):
        assert set(dataset["structure_status"]) <= {
            "standardized",
            "standardized_from_mixture",
        }

    def test_the_standardization_version_is_documented(self, dataset):
        assert dataset["standardization_version"].notna().all()

    def test_every_compound_records_the_structure_it_came_from(self, compounds):
        assert compounds["input_smiles"].notna().all()

    def test_every_structure_names_its_provenance(self, compounds):
        assert compounds["resolution_source"].notna().all()

    def test_failed_structures_are_retained_with_a_reason(self, review):
        if review.empty:
            pytest.skip("no failed structures in this run")
        assert review["structure_status"].notna().all()
        assert review["reason"].notna().all()

    def test_failed_structures_never_reach_the_dataset(self, dataset, review):
        if review.empty:
            pytest.skip("no failed structures in this run")
        assert not set(review["chemcode"]) & set(
            dataset["compound_id"]
        ), "a reviewed chemcode must not appear as a compound id"


# ---------------------------------------------------------------------------
# 3. Duplicate control
# ---------------------------------------------------------------------------
class TestDuplicateControl:
    """Duplicate chemical records are identified and handled consistently."""

    def test_exact_identity_is_unique(self, compounds):
        assert not compounds["inchi_key"].duplicated().any()

    def test_compound_ids_are_unique(self, compounds):
        assert not compounds["compound_id"].duplicated().any()

    def test_merges_are_recorded_not_hidden(self, compounds):
        merged = compounds[compounds["n_source_chemicals"] > 1]
        if merged.empty:
            pytest.skip("no merges in this run")
        assert merged["source_chemcodes"].notna().all()
        assert all(";" in s for s in merged["source_chemcodes"])

    def test_every_merge_appears_in_the_ledger(self, compounds, ledger):
        merged = set(
            compounds[compounds["n_source_chemicals"] > 1]["compound_id"]
        )
        if not merged:
            pytest.skip("no merges in this run")
        logged = set(
            ledger[ledger["reason_code"] == "compounds_merged"]["entity_id"]
        )
        assert merged <= logged

    def test_a_compound_appears_once_per_split_strategy(self, splits):
        for strategy, grp in splits.groupby("split_strategy"):
            assert not grp["compound_id"].duplicated().any(), strategy

    def test_stereoisomers_are_not_merged(self, compounds):
        """Distinct stereoisomers must stay distinct compounds."""
        dupes = compounds[compounds["connectivity_key"].duplicated(keep=False)]
        if dupes.empty:
            pytest.skip("no stereoisomer pairs in this run")
        assert not dupes["inchi_key"].duplicated().any()


# ---------------------------------------------------------------------------
# 4. Label consistency
# ---------------------------------------------------------------------------
class TestLabelConsistency:
    """Every training label follows a predefined policy."""

    def test_labels_take_only_the_permitted_values(self, dataset):
        assert set(dataset["label"].dropna().unique()) <= {0, 1, 0.0, 1.0}

    def test_every_partitioned_compound_carries_a_label(self, splits):
        assigned = splits[splits["split_assignment"].isin(PARTITIONS)]
        assert assigned["label"].notna().all()

    def test_the_policy_version_is_declared_and_stamped(self, dataset):
        cfg = load_config("labels")
        assert cfg["label_policy_version"]
        assert set(dataset["label_policy_version"].dropna()) == {
            cfg["label_policy_version"]
        }

    def test_the_endpoint_definition_travels_with_the_data(self, dataset):
        assert dataset["endpoint_definition"].notna().all()

    def test_both_endpoints_are_produced(self):
        cfg = load_config("labels")
        for name in cfg["endpoints"]:
            assert (PROCESSED / f"model_dataset_{name}.parquet").exists()

    def test_labels_are_reproducible_from_the_policy(self, observations):
        """Re-applying the policy to the records reproduces the stored labels."""
        from src.curation.harmonize_labels import build_assessment_map, normalise_assessment

        cfg = load_config("labels")
        mapping = build_assessment_map(cfg["endpoints"]["primary"])
        for _, row in observations.head(200).iterrows():
            key = normalise_assessment(row["author_assessment"])
            expected = mapping.get(key) if key is not None else None
            actual = row["label_primary"]
            actual = None if pd.isna(actual) else int(actual)
            assert actual == expected, row["experiment_id"]


# ---------------------------------------------------------------------------
# 5. Conflict handling
# ---------------------------------------------------------------------------
class TestConflictHandling:
    """Contradictory and equivocal findings are preserved and handled explicitly."""

    def test_conflicted_compounds_are_flagged(self, dataset):
        assert "is_conflicted" in dataset.columns

    def test_conflicted_compounds_are_excluded_from_partitions(self, splits):
        assigned = splits[splits["split_assignment"].isin(PARTITIONS)]
        assert not assigned["is_conflicted"].any()

    def test_conflicted_compounds_are_retained_not_dropped(self, dataset):
        conflicted = dataset[dataset["is_conflicted"]]
        if conflicted.empty:
            pytest.skip("no conflicts in this run")
        assert (
            conflicted["training_eligibility"] == "excluded_conflicting_evidence"
        ).all()

    def test_conflicted_compounds_keep_their_evidence_counts(self, dataset):
        conflicted = dataset[dataset["is_conflicted"]]
        if conflicted.empty:
            pytest.skip("no conflicts in this run")
        assert (conflicted["n_positive"] > 0).all()
        assert (conflicted["n_negative"] > 0).all()

    def test_conflicts_are_broken_down_by_axis(self):
        path = PROCESSED / "conflicts_detail.parquet"
        detail = _load(path)
        if detail.empty:
            pytest.skip("no conflicts in this run")
        assert "conflict_type" in detail.columns
        assert detail["conflict_type"].notna().all()
        assert detail["conflicting_axes"].notna().all()

    def test_equivocal_records_are_null_not_negative(self, observations):
        equivocal = observations[
            observations["author_assessment"]
            .fillna("")
            .str.contains("quivocal", regex=False)
        ]
        if equivocal.empty:
            pytest.skip("no equivocal records in this run")
        assert equivocal["label_primary"].isna().all()
        assert equivocal["label_sensitivity"].isna().all()

    def test_unresolved_is_never_recorded_as_negative(self, dataset):
        unresolved = dataset[dataset["training_eligibility"] == "excluded_unresolved_label"]
        if unresolved.empty:
            pytest.skip("no unresolved compounds in this run")
        assert unresolved["label"].isna().all()

    def test_null_survives_into_postgres_as_null(self):
        """Exported labels must be integers or NULL, never 0 standing in."""
        path = PROCESSED / "pg" / "observations.csv"
        if not path.exists():
            pytest.skip("postgres export not run")
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        values = set(raw["harmonized_label"])
        assert values <= {"0", "1", r"\N"}, values
        assert r"\N" in values, "unresolved labels must survive as NULL markers"


# ---------------------------------------------------------------------------
# 6. Leakage control
# ---------------------------------------------------------------------------
class TestLeakageControl:
    """All records for a compound remain in the same partition."""

    def test_all_records_of_a_compound_share_one_partition(self, observations, splits):
        """The criterion as worded, checked at record level."""
        for strategy, grp in splits.groupby("split_strategy"):
            assignment = grp.set_index("compound_id")["split_assignment"]
            linked = observations[observations["compound_id"].notna()].copy()
            linked["split_assignment"] = linked["compound_id"].map(assignment)
            linked = linked[linked["split_assignment"].isin(PARTITIONS)]
            per_compound = linked.groupby("compound_id")["split_assignment"].nunique()
            assert (per_compound <= 1).all(), strategy

    def test_no_inchikey_spans_partitions(self, splits):
        for strategy, grp in splits.groupby("split_strategy"):
            assigned = grp[grp["split_assignment"].isin(PARTITIONS)]
            spanning = assigned.groupby("inchi_key")["split_assignment"].nunique()
            assert (spanning <= 1).all(), strategy

    def test_scaffold_groups_do_not_span_partitions(self, splits):
        grp = splits[splits["split_strategy"] == "scaffold"]
        assigned = grp[grp["split_assignment"].isin(PARTITIONS)]
        spanning = assigned.groupby("scaffold_group")["split_assignment"].nunique()
        assert (spanning <= 1).all()

    def test_split_group_is_never_null_for_partitioned_rows(self, splits):
        assigned = splits[splits["split_assignment"].isin(PARTITIONS)]
        assert assigned["split_group"].notna().all()

    def test_similarity_to_training_is_measured_and_reported(self):
        path = PROCESSED / "split_report_primary.json"
        if not path.exists():
            pytest.skip("split report absent")
        rep = json.loads(path.read_text(encoding="utf-8"))
        for strategy, info in rep["strategies"].items():
            sim = info["nearest_neighbour_similarity_to_train"]
            for part in ("validation", "test"):
                assert part in sim, f"{strategy}/{part}"
                if sim[part].get("n"):
                    assert "median" in sim[part]
                    assert "fraction_identical_fp" in sim[part]

    def test_leakage_checks_are_recorded_as_zero(self):
        path = PROCESSED / "split_report_primary.json"
        if not path.exists():
            pytest.skip("split report absent")
        rep = json.loads(path.read_text(encoding="utf-8"))
        for strategy, info in rep["strategies"].items():
            for key, value in info["leakage_checks"].items():
                assert value == 0, f"{strategy}.{key} = {value}"

    def test_no_compound_is_lost_between_build_and_split(self, dataset, splits):
        for strategy, grp in splits.groupby("split_strategy"):
            assert set(grp["compound_id"]) == set(dataset["compound_id"]), strategy


# ---------------------------------------------------------------------------
# 7. Reproducibility
# ---------------------------------------------------------------------------
class TestReproducibility:
    """Dataset generation can be rerun from versioned inputs and configuration."""

    def test_every_config_declares_a_version(self):
        assert load_config("data")["dataset_version"]
        assert load_config("labels")["label_policy_version"]
        assert load_config("splits")["split_version"]

    def test_the_split_seed_is_fixed(self):
        assert isinstance(load_config("splits")["random_seed"], int)

    def test_group_assignment_is_deterministic(self):
        import numpy as np

        from src.datasets.split_dataset import assign_groups

        cfg = load_config("splits")
        groups = [(f"g{i}", (i % 5) + 1) for i in range(80)]
        a = assign_groups(
            groups,
            cfg["fractions"],
            rng=np.random.default_rng(cfg["random_seed"]),
            largest_first=False,
        )
        b = assign_groups(
            groups,
            cfg["fractions"],
            rng=np.random.default_rng(cfg["random_seed"]),
            largest_first=False,
        )
        assert a == b

    def test_standardization_is_deterministic(self):
        from src.chemistry.standardize import standardize_smiles

        for smi in ("CC(=O)Nc1ccc(O)cc1", "C[C@H](N)C(=O)O", "CC(=O)[O-].[Na+]"):
            assert (
                standardize_smiles(smi).inchi_key == standardize_smiles(smi).inchi_key
            )

    def test_rerunning_a_stage_reproduces_it_byte_for_byte(self, tmp_path):
        """Identical inputs must give an identical file, not merely equal data.

        The curation ledger is the stage most at risk, because an obvious
        implementation stamps a wall-clock time on every row.
        """
        import subprocess

        src = PROCESSED / "curation_ledger.parquet"
        if not src.exists():
            pytest.skip("ledger absent; run bash tests/run_pipeline.sh first")
        from src.common import sha256_file

        before = sha256_file(src)
        result = subprocess.run(
            [sys.executable, "-m", "src.curation.build_ledger"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert sha256_file(src) == before

    def test_the_ledger_carries_no_wall_clock_column(self, ledger):
        """A per-row timestamp would break byte-level reproducibility."""
        assert "recorded_at" not in ledger.columns

    def test_every_stage_wrote_a_manifest(self):
        expected = [
            "02_parse_cpdb",
            "03_resolve_identity",
            "04_harmonize_labels",
            "05_audit",
            "06_build_dataset",
            "07_split_dataset",
            "08_ledger",
        ]
        for stage in expected:
            assert (MANIFESTS / f"{stage}_latest.json").exists(), stage

    def test_manifests_record_inputs_outputs_and_versions(self):
        path = MANIFESTS / "07_split_dataset_latest.json"
        if not path.exists():
            pytest.skip("split manifest absent")
        rec = json.loads(path.read_text(encoding="utf-8"))
        assert rec["inputs"] and rec["outputs"]
        assert rec["config_versions"]["random_seed"] is not None
        assert rec["environment"]["python"]

    def test_the_freeze_lock_pins_every_published_file(self):
        lock = PROCESSED / "frozen" / load_config("data")["dataset_version"] / "DATASET_LOCK.json"
        if not lock.exists():
            pytest.skip("dataset not frozen in this run")
        rec = json.loads(lock.read_text(encoding="utf-8"))
        assert rec["files"]
        assert all(f.get("sha256") for f in rec["files"])
        assert rec["random_seed"] is not None
        assert rec["label_policy_version"]

    def test_the_frozen_dataset_still_matches_its_lock(self):
        version = load_config("data")["dataset_version"]
        frozen = PROCESSED / "frozen" / version
        lock_path = frozen / "DATASET_LOCK.json"
        if not lock_path.exists():
            pytest.skip("dataset not frozen in this run")
        from src.common import sha256_file

        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        for entry in lock["files"]:
            p = frozen / entry["name"]
            assert p.exists(), entry["name"]
            assert sha256_file(p) == entry["sha256"], entry["name"]


# ---------------------------------------------------------------------------
# 8. Auditability
# ---------------------------------------------------------------------------
class TestAuditability:
    """A machine-readable curation report documents exclusions and transformations."""

    def test_the_audit_report_is_machine_readable(self, audit):
        assert audit["dataset_version"]
        assert audit["label_policy_version"]
        assert "structure_resolution" in audit
        assert "endpoints" in audit

    def test_the_ledger_records_both_event_types(self, ledger):
        assert set(ledger["event_type"]) == {"exclusion", "transformation"}

    def test_every_ledger_event_names_its_entity_stage_and_reason(self, ledger):
        for col in ("entity_type", "entity_id", "stage", "reason_code"):
            assert ledger[col].notna().all(), col

    def test_reason_codes_come_from_a_closed_vocabulary(self, ledger):
        from src.curation.build_ledger import EXCLUSION_REASONS, TRANSFORMATION_REASONS

        assert set(ledger["reason_code"]) <= EXCLUSION_REASONS | TRANSFORMATION_REASONS

    def test_every_excluded_structure_is_in_the_ledger(self, review, ledger):
        if review.empty:
            pytest.skip("no failed structures in this run")
        logged = set(
            ledger[ledger["entity_type"] == "source_chemical"]["entity_id"]
        )
        assert set(review["chemcode"]) <= logged

    def test_every_excluded_compound_is_in_the_ledger(self, dataset, ledger):
        excluded = set(
            dataset[dataset["training_eligibility"] != "eligible"]["compound_id"]
        )
        if not excluded:
            pytest.skip("nothing excluded in this run")
        logged = set(
            ledger[
                (ledger["event_type"] == "exclusion")
                & (ledger["entity_type"] == "compound")
                & (ledger["endpoint_name"] == "primary")
            ]["entity_id"]
        )
        assert excluded <= logged

    def test_transformations_are_enumerated_not_only_counted(self, ledger):
        transforms = ledger[ledger["event_type"] == "transformation"]
        assert len(transforms) > 0
        assert transforms["detail"].notna().all()

    def test_the_summary_counts_match_the_ledger(self, ledger):
        path = PROCESSED / "curation_ledger_summary.json"
        if not path.exists():
            pytest.skip("ledger summary absent")
        s = json.loads(path.read_text(encoding="utf-8"))
        assert s["total_events"] == len(ledger)
        assert s["exclusions"]["total"] == int(
            (ledger["event_type"] == "exclusion").sum()
        )
        assert s["transformations"]["total"] == int(
            (ledger["event_type"] == "transformation").sum()
        )

    def test_any_compound_absence_can_be_explained_from_the_ledger(
        self, dataset, splits, ledger
    ):
        """The question the ledger exists to answer, in one filter."""
        partitioned = set(
            splits[
                (splits["split_strategy"] == "scaffold")
                & (splits["split_assignment"].isin(PARTITIONS))
            ]["compound_id"]
        )
        absent = set(dataset["compound_id"]) - partitioned
        if not absent:
            pytest.skip("every compound was partitioned")
        explained = set(
            ledger[
                (ledger["event_type"] == "exclusion")
                & (ledger["endpoint_name"] == "primary")
            ]["entity_id"]
        )
        assert absent <= explained

    def test_warnings_are_captured_in_manifests(self):
        path = MANIFESTS / "03_resolve_identity_latest.json"
        if not path.exists():
            pytest.skip("manifest absent")
        rec = json.loads(path.read_text(encoding="utf-8"))
        assert "warnings" in rec


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
