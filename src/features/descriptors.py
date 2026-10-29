"""Physicochemical descriptors.

A small, interpretable set. Interpretability matters here: a carcinogenicity
prediction that cannot be traced to a structural feature or a set of similar
compounds cannot take part in a weight-of-evidence assessment, so descriptors
that a toxicologist recognises are preferred over a larger opaque set.
"""

from __future__ import annotations

import logging

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

LOG = logging.getLogger("descriptors")

DESCRIPTORS = {
    "mol_weight": Descriptors.MolWt,
    "heavy_atoms": lambda m: float(m.GetNumHeavyAtoms()),
    "logp": Crippen.MolLogP,
    "tpsa": rdMolDescriptors.CalcTPSA,
    "h_donors": rdMolDescriptors.CalcNumHBD,
    "h_acceptors": rdMolDescriptors.CalcNumHBA,
    "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds,
    "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings,
    "aliphatic_rings": rdMolDescriptors.CalcNumAliphaticRings,
    "ring_count": rdMolDescriptors.CalcNumRings,
    "fraction_csp3": rdMolDescriptors.CalcFractionCSP3,
    "heteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
    "formal_charge": lambda m: float(Chem.GetFormalCharge(m)),
}


def compute(smiles: str) -> dict[str, float | None]:
    """Descriptors for one SMILES. Unparseable input gives all-None."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return {name: None for name in DESCRIPTORS}
    out: dict[str, float | None] = {}
    for name, fn in DESCRIPTORS.items():
        try:
            out[name] = float(fn(mol))
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


def compute_frame(smiles_list: list[str]) -> pd.DataFrame:
    """Descriptor table, one row per input, aligned with the input order."""
    return pd.DataFrame([compute(s) for s in smiles_list])


__all__ = ["DESCRIPTORS", "compute", "compute_frame"]
