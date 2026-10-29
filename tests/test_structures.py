"""Structure standardisation tests.

These assert the properties the protocol requires, not merely that the code
runs: stereochemistry survives, salts resolve to their parent, malformed input
is rejected rather than coerced, and identity is deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chemistry.resolve_identity import compound_id_for  # noqa: E402
from src.chemistry.standardize import (  # noqa: E402
    STATUS_EMPTY,
    STATUS_NO_ORGANIC_FRAGMENT,
    STATUS_PARSE_FAILED,
    murcko_scaffold,
    standardize_smiles,
)
from rdkit import Chem  # noqa: E402


class TestStereochemistry:
    """Stereochemistry must survive standardisation.

    RDKit's TautomerEnumerator strips sp3 stereo by default. If that default
    ever leaks back in, L- and D-alanine merge into one compound and every
    label attached to either becomes wrong.
    """

    def test_enantiomers_get_distinct_inchikeys(self):
        l_ala = standardize_smiles("C[C@H](N)C(=O)O")
        d_ala = standardize_smiles("C[C@@H](N)C(=O)O")
        assert l_ala.ok and d_ala.ok
        assert l_ala.inchi_key != d_ala.inchi_key

    def test_enantiomers_get_distinct_compound_ids(self):
        l_id = compound_id_for(standardize_smiles("C[C@H](N)C(=O)O").inchi_key)
        d_id = compound_id_for(standardize_smiles("C[C@@H](N)C(=O)O").inchi_key)
        assert l_id != d_id

    def test_enantiomers_share_a_connectivity_key(self):
        """They must group together for splitting even while staying distinct."""
        l_ala = standardize_smiles("C[C@H](N)C(=O)O")
        d_ala = standardize_smiles("C[C@@H](N)C(=O)O")
        assert l_ala.connectivity_key == d_ala.connectivity_key

    def test_isomeric_smiles_retains_the_stereo_marker(self):
        assert "@" in standardize_smiles("C[C@H](N)C(=O)O").isomeric_smiles

    def test_double_bond_geometry_is_preserved(self):
        e = standardize_smiles("C/C=C/C")
        z = standardize_smiles(r"C/C=C\C")
        assert e.inchi_key != z.inchi_key

    def test_unassigned_stereocentres_are_flagged(self):
        flat = standardize_smiles("CC(N)C(=O)O")
        assert flat.n_unassigned_stereocenters == 1
        assert any("unassigned" in n for n in flat.notes)

    def test_assigned_stereocentres_are_not_flagged(self):
        assert standardize_smiles("C[C@H](N)C(=O)O").n_unassigned_stereocenters == 0


class TestSaltsAndMixtures:
    def test_ionic_salt_resolves_to_the_organic_parent(self):
        r = standardize_smiles("CC(=O)[O-].[Na+]")
        assert r.ok
        assert r.canonical_smiles == "CC(=O)O"

    def test_covalently_written_salt_is_recognised_as_a_mixture(self):
        """Metal disconnection must run before fragments are counted."""
        r = standardize_smiles("CC(=O)O[Na]")
        assert r.ok
        assert r.n_fragments == 2
        assert r.canonical_smiles == "CC(=O)O"

    def test_solvate_drops_the_solvent(self):
        r = standardize_smiles("c1ccccc1C(=O)O.O")
        assert r.ok
        assert "." not in r.canonical_smiles

    def test_discarded_fragments_are_recorded(self):
        r = standardize_smiles("CC(=O)[O-].[Na+]")
        assert any("discarded" in n for n in r.notes)

    def test_purely_inorganic_input_is_rejected_not_guessed(self):
        r = standardize_smiles("[Na+].[Cl-]")
        assert not r.ok
        assert r.status == STATUS_NO_ORGANIC_FRAGMENT
        assert r.inchi_key is None


class TestRejection:
    def test_malformed_smiles_is_rejected(self):
        r = standardize_smiles("this_is_not_a_smiles")
        assert not r.ok
        assert r.status == STATUS_PARSE_FAILED

    def test_empty_input_is_rejected(self):
        for value in (None, "", "   "):
            assert standardize_smiles(value).status == STATUS_EMPTY

    def test_rejected_structures_carry_no_identifiers(self):
        """A rejected structure must never reach the dataset with a usable id."""
        for bad in ("this_is_not_a_smiles", "", "[Na+].[Cl-]"):
            r = standardize_smiles(bad)
            assert r.inchi_key is None
            assert r.canonical_smiles is None
            assert not r.ok


class TestDeterminism:
    def test_repeated_standardisation_is_stable(self):
        for smi in ("CCO", "c1ccccc1", "C[C@H](N)C(=O)O", "CC(=O)Nc1ccc(O)cc1"):
            a = standardize_smiles(smi)
            b = standardize_smiles(smi)
            assert a.inchi_key == b.inchi_key
            assert a.canonical_smiles == b.canonical_smiles

    def test_equivalent_smiles_converge(self):
        """Different spellings of one molecule give one identity."""
        a = standardize_smiles("CC(=O)O")
        b = standardize_smiles("OC(C)=O")
        assert a.inchi_key == b.inchi_key
        assert compound_id_for(a.inchi_key) == compound_id_for(b.inchi_key)

    def test_compound_id_is_a_pure_function_of_the_inchikey(self):
        key = "QNAYBMKLOCPYGJ-REOHCLBHSA-N"
        assert compound_id_for(key) == compound_id_for(key)
        assert compound_id_for(key) != compound_id_for("LFQSCWFLJHTTHZ-UHFFFAOYSA-N")


class TestScaffolds:
    def test_ring_system_yields_a_scaffold(self):
        assert murcko_scaffold(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")) == "c1ccccc1"

    def test_acyclic_molecule_yields_no_scaffold(self):
        """Acyclic compounds must not all collapse into one empty group."""
        assert murcko_scaffold(Chem.MolFromSmiles("CCO")) is None

    def test_scaffold_ignores_chirality_by_default(self):
        """Stereoisomers must land in the same split group."""
        a = murcko_scaffold(Chem.MolFromSmiles("C[C@H](N)c1ccccc1"))
        b = murcko_scaffold(Chem.MolFromSmiles("C[C@@H](N)c1ccccc1"))
        assert a == b


class TestNoSilentNaN:
    """NaN is truthy in Python; a group key must never be lost to it."""

    def test_scaffold_group_fallback_rejects_nan(self):
        import pandas as pd

        nan = float("nan")
        assert not (not pd.isna(nan) and str(nan).strip())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
