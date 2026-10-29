"""Common model interface and featurizers.

Every benchmark model implements the same three methods, so the runner is
identical for a majority-class classifier and a graph neural network. A
comparison between models that each had their own training and evaluation path
would be comparing the paths as much as the models.

    fit(smiles, y, smiles_val, y_val)   -> None
    predict_proba(smiles)               -> np.ndarray of P(positive)
    describe()                          -> dict for the provenance record

Featurizers are separate from models so that B1 and B2 provably see identical
inputs and the only difference is the estimator.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from src.features.descriptors import DESCRIPTORS, compute

LOG = logging.getLogger("models")
RDLogger.DisableLog("rdApp.*")


class UnavailableModel(RuntimeError):
    """A model that cannot run in this environment, with a stated reason.

    Raised rather than returning a degenerate model, so the suite records the
    model as `unavailable` with its reason instead of silently dropping it or
    reporting a meaningless score.
    """


# ---------------------------------------------------------------------------
# Featurizers
# ---------------------------------------------------------------------------


@dataclass
class MorganFeaturizer:
    """Morgan (ECFP-style) fingerprints.

    `use_chirality` is on by default: Phase 1 preserves stereochemistry, so
    discarding it here would throw away information the dataset went to some
    trouble to keep. It also means enantiomers are distinguishable to the
    model, which plain Morgan bits cannot do.
    """

    radius: int = 2
    n_bits: int = 2048
    use_chirality: bool = True
    counts: bool = False

    def __post_init__(self) -> None:
        self._gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            fpSize=self.n_bits,
            includeChirality=self.use_chirality,
        )

    @property
    def name(self) -> str:
        kind = "counts" if self.counts else "bits"
        chi = "chiral" if self.use_chirality else "achiral"
        return f"morgan_r{self.radius}_{self.n_bits}_{kind}_{chi}"

    def transform(self, smiles: list[str]) -> np.ndarray:
        rows = np.zeros((len(smiles), self.n_bits), dtype=np.float32)
        failed = 0
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                failed += 1
                continue
            arr = np.zeros((self.n_bits,), dtype=np.int32)
            if self.counts:
                fp = self._gen.GetCountFingerprint(mol)
                for idx, val in fp.GetNonzeroElements().items():
                    arr[idx] = val
            else:
                DataStructs.ConvertToNumpyArray(self._gen.GetFingerprint(mol), arr)
            rows[i] = arr
        if failed:
            # An all-zero row is a silent, plausible-looking feature vector, so
            # the count is surfaced rather than left to be discovered later.
            LOG.warning(
                "%d of %d structures did not parse; their feature rows are all-zero",
                failed,
                len(smiles),
            )
        return rows

    def describe(self) -> dict:
        return {
            "kind": "morgan",
            "radius": self.radius,
            "n_bits": self.n_bits,
            "use_chirality": self.use_chirality,
            "counts": self.counts,
        }


@dataclass
class DescriptorFeaturizer:
    """A small, interpretable physicochemical set.

    Deliberately short. Hundreds of redundant descriptors make a baseline
    harder to reproduce and no more accurate, and they obscure which property
    is carrying a prediction.
    """

    names: list[str] = field(default_factory=lambda: list(DESCRIPTORS.keys()))

    @property
    def name(self) -> str:
        return f"descriptors_{len(self.names)}"

    def transform(self, smiles: list[str]) -> np.ndarray:
        rows = np.zeros((len(smiles), len(self.names)), dtype=np.float32)
        for i, smi in enumerate(smiles):
            d = compute(smi)
            for j, key in enumerate(self.names):
                v = d.get(key)
                rows[i, j] = 0.0 if v is None else float(v)
        return rows

    def describe(self) -> dict:
        return {"kind": "descriptors", "names": self.names}


@dataclass
class ConcatFeaturizer:
    """Fingerprints plus descriptors, standardised on the training fold."""

    parts: list
    _mean: np.ndarray | None = None
    _std: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "+".join(p.name for p in self.parts)

    def transform(self, smiles: list[str]) -> np.ndarray:
        return np.hstack([p.transform(smiles) for p in self.parts])

    def describe(self) -> dict:
        return {"kind": "concat", "parts": [p.describe() for p in self.parts]}


# ---------------------------------------------------------------------------
# Model interface
# ---------------------------------------------------------------------------


class BenchmarkModel(ABC):
    """One interface for every model in the ladder."""

    model_id: str = "B?"
    model_name: str = "unnamed"
    representation: str = "none"

    def __init__(self, seed: int = 42, **hyperparameters):
        self.seed = seed
        self.hyperparameters = hyperparameters
        self._fitted = False

    @abstractmethod
    def fit(
        self,
        smiles: list[str],
        y: np.ndarray,
        smiles_val: list[str] | None = None,
        y_val: np.ndarray | None = None,
    ) -> "BenchmarkModel":
        """Train. Validation data is for early stopping only, never for the
        final score."""

    @abstractmethod
    def predict_proba(self, smiles: list[str]) -> np.ndarray:
        """P(positive) for each input, aligned with the input order."""

    def describe(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "representation": self.representation,
            "seed": self.seed,
            "hyperparameters": self.hyperparameters,
        }


class SklearnModel(BenchmarkModel):
    """Shared plumbing for the scikit-learn-style estimators (B1, B2, B3)."""

    def __init__(self, featurizer, seed: int = 42, **hyperparameters):
        super().__init__(seed=seed, **hyperparameters)
        self.featurizer = featurizer
        self.estimator = None

    @abstractmethod
    def _build(self):
        """Construct the underlying estimator."""

    def fit(self, smiles, y, smiles_val=None, y_val=None):
        X = self.featurizer.transform(list(smiles))
        y = np.asarray(y).astype(int)

        # A scaffold split on a small or imbalanced dataset can hand a fold a
        # single class. Several estimators raise on that. Rather than let the
        # run die, the model degenerates to the constant it was shown and says
        # so, which is what it has actually learned.
        self.single_class_ = None
        if len(np.unique(y)) < 2:
            self.single_class_ = int(y[0]) if len(y) else 0
            self.estimator = None
            self._fitted = True
            LOG.warning(
                "%s: training fold holds only class %d; model degenerates to a "
                "constant predictor",
                self.model_id,
                self.single_class_,
            )
            return self

        self.estimator = self._build()
        self.estimator.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, smiles):
        if not self._fitted:
            raise RuntimeError("predict_proba called before fit")
        if getattr(self, "single_class_", None) is not None:
            # Returning the class it saw is honest; 0.5 would invent a
            # confidence the model does not have.
            return np.full(len(smiles), float(self.single_class_))
        X = self.featurizer.transform(list(smiles))
        proba = self.estimator.predict_proba(X)
        classes = list(getattr(self.estimator, "classes_", [0, 1]))
        if 1 not in classes:
            return np.full(len(smiles), float(classes[0]))
        return proba[:, classes.index(1)]

    def describe(self) -> dict:
        d = super().describe()
        d["featurizer"] = self.featurizer.describe()
        return d


def build_featurizer(spec: dict, descriptors: list[str] | None = None):
    """Construct a featurizer from a config block."""
    kind = spec.get("kind", "morgan")
    if kind == "morgan":
        fp = MorganFeaturizer(
            radius=spec.get("radius", 2),
            n_bits=spec.get("n_bits", 2048),
            use_chirality=spec.get("use_chirality", True),
            counts=spec.get("counts", False),
        )
        if descriptors:
            return ConcatFeaturizer(parts=[fp, DescriptorFeaturizer(names=descriptors)])
        return fp
    if kind == "descriptors":
        return DescriptorFeaturizer(names=descriptors or list(DESCRIPTORS.keys()))
    raise ValueError(f"unknown featurizer kind: {kind}")


__all__ = [
    "BenchmarkModel",
    "ConcatFeaturizer",
    "DescriptorFeaturizer",
    "MorganFeaturizer",
    "SklearnModel",
    "UnavailableModel",
    "build_featurizer",
]
