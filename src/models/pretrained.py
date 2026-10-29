"""B5 - pretrained molecular encoder (ChemBERTa-style).

Two modes:
    frozen_embeddings  encode SMILES once, fit a logistic head on the vectors
    finetune           update the encoder along with the head

The adapter is written in full so it runs wherever the weights are reachable.
Where they are not, it raises `UnavailableModel` with the reason, and the
runner records the model as `unavailable` rather than dropping it from the
suite or substituting something else under its name. A benchmark table with a
silently missing row invites the reader to assume the model was tried and lost.

Pretraining is not assumed to help. On a dataset of a few thousand compounds a
pretrained encoder frequently loses to gradient boosting on fingerprints, and
reporting that honestly is the point of running it.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from src.models.base import BenchmarkModel, UnavailableModel

LOG = logging.getLogger("pretrained")

DEFAULT_CHECKPOINT = "seyonec/ChemBERTa-zinc-base-v1"


def check_availability(checkpoint: str = DEFAULT_CHECKPOINT) -> tuple[bool, str]:
    """Can this model actually run here? Returns (ok, reason).

    Checked before training rather than discovered mid-run, so the suite can
    record the reason alongside the other results.
    """
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False, "the transformers package is not installed"

    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "pytorch is not installed"

    # A local copy, if one has been placed there, needs no network.
    local = os.environ.get("CARCINOFUSION_PRETRAINED_DIR")
    if local and os.path.isdir(local):
        return True, f"using local checkpoint directory {local}"

    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(checkpoint)
        return True, "checkpoint reachable"
    except Exception as exc:  # noqa: BLE001
        detail = str(exc).strip().splitlines()[0][:200] if str(exc) else type(exc).__name__
        return False, (
            f"pretrained weights for '{checkpoint}' could not be fetched: {detail}"
        )


class PretrainedEncoderModel(BenchmarkModel):
    """B5 adapter."""

    model_id = "B5"
    model_name = "pretrained_encoder"
    representation = "smiles"

    def __init__(self, seed: int = 42, **hp):
        super().__init__(seed=seed, **hp)
        self.checkpoint = hp.get("checkpoint", DEFAULT_CHECKPOINT)
        self.mode = hp.get("mode", "frozen_embeddings")
        self.max_length = hp.get("max_length", 256)
        self.batch_size = hp.get("batch_size", 32)
        self.C = hp.get("C", 1.0)
        self.lr = hp.get("learning_rate", 2e-5)
        self.max_epochs = hp.get("max_epochs", 10)
        self._tokenizer = None
        self._encoder = None
        self._head = None

    def _load(self):
        ok, reason = check_availability(self.checkpoint)
        if not ok:
            raise UnavailableModel(reason)

        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.manual_seed(self.seed)
        source = os.environ.get("CARCINOFUSION_PRETRAINED_DIR") or self.checkpoint
        self._tokenizer = AutoTokenizer.from_pretrained(source)
        self._encoder = AutoModel.from_pretrained(source)
        self._encoder.eval()

    def _embed(self, smiles: list[str]) -> np.ndarray:
        """Mean-pooled last hidden state, masked over real tokens."""
        import torch

        vectors: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(smiles), self.batch_size):
                chunk = [s if s else "" for s in smiles[start : start + self.batch_size]]
                enc = self._tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                out = self._encoder(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                # Padding tokens must not contribute to the mean, or the
                # embedding would depend on batch composition.
                pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
                vectors.append(pooled.cpu().numpy())
        return np.vstack(vectors) if vectors else np.zeros((0, 1))

    def fit(self, smiles, y, smiles_val=None, y_val=None):
        from sklearn.linear_model import LogisticRegression

        self._load()
        y = np.asarray(y).astype(int)

        self.single_class_ = None
        if len(np.unique(y)) < 2:
            self.single_class_ = int(y[0]) if len(y) else 0
            self._fitted = True
            LOG.warning("B5: training fold holds a single class; constant predictor")
            return self

        if self.mode == "finetune":
            self._finetune(list(smiles), y, smiles_val, y_val)
        else:
            X = self._embed(list(smiles))
            self._head = LogisticRegression(
                C=self.C,
                max_iter=5000,
                class_weight=self.hyperparameters.get("class_weight"),
                random_state=self.seed,
            )
            self._head.fit(X, y)
        self._fitted = True
        return self

    def _finetune(self, smiles, y, smiles_val, y_val):
        import torch
        import torch.nn as nn

        device = torch.device("cpu")
        hidden = self._encoder.config.hidden_size
        self._head = nn.Linear(hidden, 1)
        self._encoder.train()

        params = list(self._encoder.parameters()) + list(self._head.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr)
        n_pos = float(y.sum())
        pos_weight = torch.tensor(
            [(len(y) - n_pos) / n_pos if n_pos else 1.0], dtype=torch.float32
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(smiles))

        for _ in range(self.max_epochs):
            rng.shuffle(idx)
            for start in range(0, len(idx), self.batch_size):
                chunk = idx[start : start + self.batch_size]
                enc = self._tokenizer(
                    [smiles[i] for i in chunk],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(device)
                out = self._encoder(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
                logits = self._head(pooled).squeeze(-1)
                opt.zero_grad()
                loss_fn(logits, torch.tensor(y[chunk], dtype=torch.float32)).backward()
                opt.step()
        self._encoder.eval()

    def predict_proba(self, smiles):
        if not self._fitted:
            raise RuntimeError("predict_proba called before fit")
        if getattr(self, "single_class_", None) is not None:
            return np.full(len(smiles), float(self.single_class_))

        if self.mode == "finetune":
            import torch

            probs: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(smiles), self.batch_size):
                    chunk = list(smiles[start : start + self.batch_size])
                    enc = self._tokenizer(
                        chunk,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    out = self._encoder(**enc).last_hidden_state
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
                    probs.append(
                        torch.sigmoid(self._head(pooled).squeeze(-1)).cpu().numpy()
                    )
            return np.concatenate(probs) if probs else np.zeros(0)

        X = self._embed(list(smiles))
        return self._head.predict_proba(X)[:, list(self._head.classes_).index(1)]

    def describe(self) -> dict:
        d = super().describe()
        d["checkpoint"] = self.checkpoint
        d["mode"] = self.mode
        d["embedding_dim"] = (
            int(self._encoder.config.hidden_size) if self._encoder else None
        )
        return d


__all__ = ["DEFAULT_CHECKPOINT", "PretrainedEncoderModel", "check_availability"]
