"""Molecular fingerprints.

Used for two distinct purposes, which must not be confused:

  model features     fingerprints fed to a classifier
  leakage measurement nearest-neighbour similarity between split partitions

The second is the one that matters in Phase 1. Scaffold grouping reduces
leakage but does not guarantee chemical dissimilarity, so the similarity
between train and test is measured explicitly rather than assumed.
"""

from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

LOG = logging.getLogger("fingerprints")


def morgan_generator(radius: int = 2, n_bits: int = 2048):
    """Morgan fingerprint generator (ECFP-like)."""
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def fingerprint(smiles: str, *, radius: int = 2, n_bits: int = 2048):
    """RDKit explicit bit vector for one SMILES, or None if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return morgan_generator(radius, n_bits).GetFingerprint(mol)


def fingerprint_many(
    smiles_list: list[str], *, radius: int = 2, n_bits: int = 2048
) -> tuple[list, list[int]]:
    """Fingerprint a list of SMILES.

    Returns (fingerprints, indices) where indices are the positions in the
    input that parsed successfully. Failures are dropped from the output but
    their positions are recoverable, so a caller never silently misaligns
    fingerprints with labels.
    """
    gen = morgan_generator(radius, n_bits)
    fps, idx = [], []
    for i, smi in enumerate(smiles_list):
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(gen.GetFingerprint(mol))
        idx.append(i)
    if len(fps) < len(smiles_list):
        LOG.warning(
            "%d of %d structures did not parse for fingerprinting",
            len(smiles_list) - len(fps),
            len(smiles_list),
        )
    return fps, idx


def to_numpy(fps: list) -> np.ndarray:
    """Stack fingerprints into a dense uint8 array."""
    if not fps:
        return np.zeros((0, 0), dtype=np.uint8)
    arrs = []
    for fp in fps:
        arr = np.zeros((fp.GetNumBits(),), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        arrs.append(arr)
    return np.vstack(arrs)


def max_similarity_to_set(query_fp, reference_fps: list) -> float:
    """Highest Tanimoto similarity from one molecule to a reference set."""
    if not reference_fps:
        return 0.0
    return float(max(DataStructs.BulkTanimotoSimilarity(query_fp, reference_fps)))


def nearest_neighbour_similarities(
    query_smiles: list[str],
    reference_smiles: list[str],
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """For each query, the Tanimoto similarity to its nearest reference.

    This is the leakage measurement. A test set whose median nearest-neighbour
    similarity to training is high is not measuring generalisation, however
    cleanly the scaffolds were separated.
    """
    q_fps, _ = fingerprint_many(query_smiles, radius=radius, n_bits=n_bits)
    r_fps, _ = fingerprint_many(reference_smiles, radius=radius, n_bits=n_bits)
    if not q_fps or not r_fps:
        return np.zeros((0,), dtype=float)
    return np.array([max_similarity_to_set(fp, r_fps) for fp in q_fps], dtype=float)


__all__ = [
    "fingerprint",
    "fingerprint_many",
    "max_similarity_to_set",
    "morgan_generator",
    "nearest_neighbour_similarities",
    "to_numpy",
]
