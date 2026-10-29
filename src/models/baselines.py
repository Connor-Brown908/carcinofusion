"""B0-B3: the inexpensive half of the benchmark ladder.

B0  majority class        sanity reference
B1  logistic regression   simple linear baseline on Morgan fingerprints
B2  random forest         nonlinear, same fingerprints as B1
B3  gradient boosting     fingerprints, optionally plus descriptors

B1 and B2 share a featurizer instance type and settings so the comparison
isolates the estimator. B3 is run both with and without descriptors, because
"do descriptors help" is a question worth answering rather than assuming.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.base import BenchmarkModel, MorganFeaturizer, SklearnModel

LOG = logging.getLogger("baselines")


class MajorityClassModel(BenchmarkModel):
    """B0 - predicts the training set's majority class for everything.

    Its job is to expose misleading accuracy under class imbalance: at 80%
    negatives it scores 0.80 accuracy while being useless, and its PR-AUC sits
    at the positive prevalence. Any model that cannot beat it has learned
    nothing.
    """

    model_id = "B0"
    model_name = "majority_class"
    representation = "none"

    def fit(self, smiles, y, smiles_val=None, y_val=None):
        y = np.asarray(y).astype(int)
        self.prevalence_ = float(y.mean()) if len(y) else 0.0
        self.majority_ = int(round(self.prevalence_))
        self._fitted = True
        return self

    def predict_proba(self, smiles):
        if not self._fitted:
            raise RuntimeError("predict_proba called before fit")
        # The training prevalence, constant for every compound. This is the
        # best-calibrated constant predictor, so its Brier score is a fair
        # reference; emitting a hard 0 or 1 would make the Brier meaningless.
        return np.full(len(smiles), self.prevalence_, dtype=float)

    def describe(self) -> dict:
        d = super().describe()
        d["learned"] = {
            "train_prevalence": getattr(self, "prevalence_", None),
            "majority_class": getattr(self, "majority_", None),
        }
        return d


class LogisticRegressionModel(SklearnModel):
    """B1 - logistic regression on Morgan fingerprints."""

    model_id = "B1"
    model_name = "logistic_regression"
    representation = "morgan_fp"

    def _build(self):
        return LogisticRegression(
            C=self.hyperparameters.get("C", 1.0),
            class_weight=self.hyperparameters.get("class_weight"),
            max_iter=self.hyperparameters.get("max_iter", 5000),
            solver=self.hyperparameters.get("solver", "liblinear"),
            random_state=self.seed,
        )


class RandomForestModel(SklearnModel):
    """B2 - random forest on the same fingerprints as B1."""

    model_id = "B2"
    model_name = "random_forest"
    representation = "morgan_fp"

    def _build(self):
        return RandomForestClassifier(
            n_estimators=self.hyperparameters.get("n_estimators", 300),
            max_depth=self.hyperparameters.get("max_depth"),
            min_samples_leaf=self.hyperparameters.get("min_samples_leaf", 1),
            class_weight=self.hyperparameters.get("class_weight"),
            n_jobs=self.hyperparameters.get("n_jobs", -1),
            random_state=self.seed,
        )


class GradientBoostingModel(SklearnModel):
    """B3 - gradient-boosted trees, LightGBM or XGBoost.

    The backend is recorded with the result: the two libraries differ enough
    in defaults that a number from one is not interchangeable with the other.
    """

    model_id = "B3"
    model_name = "gradient_boosting"
    representation = "morgan_fp_plus_descriptors"

    def __init__(self, featurizer, seed: int = 42, backend: str = "lightgbm", **hp):
        super().__init__(featurizer=featurizer, seed=seed, **hp)
        self.backend = backend

    def _build(self):
        if self.backend == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMClassifier(
                num_leaves=self.hyperparameters.get("num_leaves", 31),
                learning_rate=self.hyperparameters.get("learning_rate", 0.1),
                n_estimators=self.hyperparameters.get("n_estimators", 200),
                min_child_samples=self.hyperparameters.get("min_child_samples", 20),
                class_weight=self.hyperparameters.get("class_weight"),
                random_state=self.seed,
                verbosity=self.hyperparameters.get("verbosity", -1),
                n_jobs=-1,
            )
        if self.backend == "xgboost":
            import xgboost as xgb

            return xgb.XGBClassifier(
                max_leaves=self.hyperparameters.get("num_leaves", 31),
                learning_rate=self.hyperparameters.get("learning_rate", 0.1),
                n_estimators=self.hyperparameters.get("n_estimators", 200),
                min_child_weight=self.hyperparameters.get("min_child_samples", 1),
                random_state=self.seed,
                verbosity=0,
                n_jobs=-1,
                eval_metric="logloss",
            )
        raise ValueError(f"unknown gradient boosting backend: {self.backend}")

    def describe(self) -> dict:
        d = super().describe()
        d["backend"] = self.backend
        return d


def make_baseline(model_id: str, *, featurizer=None, seed: int = 42, **hp) -> BenchmarkModel:
    """Construct one of B0-B3."""
    if model_id == "B0":
        return MajorityClassModel(seed=seed)
    if featurizer is None:
        featurizer = MorganFeaturizer()
    if model_id == "B1":
        return LogisticRegressionModel(featurizer=featurizer, seed=seed, **hp)
    if model_id == "B2":
        return RandomForestModel(featurizer=featurizer, seed=seed, **hp)
    if model_id == "B3":
        backend = hp.pop("backend", "lightgbm")
        return GradientBoostingModel(
            featurizer=featurizer, seed=seed, backend=backend, **hp
        )
    raise ValueError(f"unknown baseline id: {model_id}")


__all__ = [
    "GradientBoostingModel",
    "LogisticRegressionModel",
    "MajorityClassModel",
    "RandomForestModel",
    "make_baseline",
]
