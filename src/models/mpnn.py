"""B4 - message-passing neural network over the molecular graph.

Written directly in PyTorch rather than pulling in a graph library, so the
atom and bond featurisation is visible and auditable: what the model sees is
the point of comparing it against fingerprints.

Molecules are batched by concatenating their graphs with index offsets, which
avoids padding and keeps the message passing a single scatter-add.

Chirality is included in the atom features. Phase 1 preserves stereochemistry
and B1-B3 use chiral fingerprints, so dropping it here would make the
comparison about the information each model receives rather than about the
representation.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger

from src.models.base import BenchmarkModel

LOG = logging.getLogger("mpnn")
RDLogger.DisableLog("rdApp.*")

# Elements common in drug-like and industrial chemistry. Anything else falls
# into a shared "other" bucket rather than being silently mapped to carbon.
ATOM_TYPES = [6, 7, 8, 9, 15, 16, 17, 35, 53, 5, 14, 34, 33, 11, 19, 20, 30, 26]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]

ATOM_FEATURE_DIM = (
    len(ATOM_TYPES) + 1 + 6 + 5 + len(HYBRIDIZATIONS) + 1 + 1 + 1 + len(CHIRAL_TAGS)
)
BOND_FEATURE_DIM = len(BOND_TYPES) + 1 + 1 + 1 + len(BOND_STEREO)


def _one_hot(value, choices: list) -> list[float]:
    """One-hot with a trailing 'other' slot, so an unseen value is marked
    rather than collapsed onto the first category."""
    vec = [0.0] * (len(choices) + 1)
    try:
        vec[choices.index(value)] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec


def atom_features(atom: Chem.Atom) -> list[float]:
    feats: list[float] = []
    feats += _one_hot(atom.GetAtomicNum(), ATOM_TYPES)
    feats += _one_hot(atom.GetTotalDegree(), [0, 1, 2, 3, 4])
    feats += _one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1])
    feats += _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)[:-1]
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    feats.append(1.0 if atom.IsInRing() else 0.0)
    feats.append(atom.GetTotalNumHs() / 4.0)
    feats += _one_hot(atom.GetChiralTag(), CHIRAL_TAGS)[:-1]
    return feats


def bond_features(bond: Chem.Bond) -> list[float]:
    feats: list[float] = []
    feats += _one_hot(bond.GetBondType(), BOND_TYPES)
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    feats.append(1.0 if bond.IsInRing() else 0.0)
    feats += _one_hot(bond.GetStereo(), BOND_STEREO)[:-1]
    return feats


class MolGraph:
    """One molecule as flat arrays."""

    __slots__ = ("x", "edge_index", "edge_attr", "n_atoms", "ok")

    def __init__(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None or mol.GetNumAtoms() == 0:
            self.ok = False
            self.n_atoms = 1
            self.x = np.zeros((1, ATOM_FEATURE_DIM), dtype=np.float32)
            self.edge_index = np.zeros((2, 0), dtype=np.int64)
            self.edge_attr = np.zeros((0, BOND_FEATURE_DIM), dtype=np.float32)
            return

        self.ok = True
        self.n_atoms = mol.GetNumAtoms()
        self.x = np.array(
            [atom_features(a) for a in mol.GetAtoms()], dtype=np.float32
        )

        src, dst, attrs = [], [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = bond_features(bond)
            # Both directions: messages must flow either way.
            src += [i, j]
            dst += [j, i]
            attrs += [bf, bf]
        self.edge_index = (
            np.array([src, dst], dtype=np.int64)
            if src
            else np.zeros((2, 0), dtype=np.int64)
        )
        self.edge_attr = (
            np.array(attrs, dtype=np.float32)
            if attrs
            else np.zeros((0, BOND_FEATURE_DIM), dtype=np.float32)
        )


def collate(graphs: list[MolGraph]) -> dict:
    """Concatenate graphs with index offsets into one disjoint batch."""
    xs, eis, eas, batch_idx = [], [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        xs.append(g.x)
        if g.edge_index.shape[1]:
            eis.append(g.edge_index + offset)
            eas.append(g.edge_attr)
        batch_idx.append(np.full(g.n_atoms, i, dtype=np.int64))
        offset += g.n_atoms
    return {
        "x": torch.from_numpy(np.vstack(xs)),
        "edge_index": torch.from_numpy(
            np.hstack(eis) if eis else np.zeros((2, 0), dtype=np.int64)
        ),
        "edge_attr": torch.from_numpy(
            np.vstack(eas) if eas else np.zeros((0, BOND_FEATURE_DIM), dtype=np.float32)
        ),
        "batch": torch.from_numpy(np.concatenate(batch_idx)),
        "n_graphs": len(graphs),
    }


class MPNN(nn.Module):
    """Edge-conditioned message passing with mean+max readout."""

    def __init__(self, hidden_dim: int = 128, n_steps: int = 3, dropout: float = 0.15):
        super().__init__()
        self.n_steps = n_steps
        self.atom_encoder = nn.Sequential(
            nn.Linear(ATOM_FEATURE_DIM, hidden_dim), nn.ReLU()
        )
        self.message = nn.Sequential(
            nn.Linear(hidden_dim + BOND_FEATURE_DIM, hidden_dim), nn.ReLU()
        )
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        x, ei, ea = batch["x"], batch["edge_index"], batch["edge_attr"]
        bidx, n_graphs = batch["batch"], batch["n_graphs"]

        h = self.atom_encoder(x)
        for _ in range(self.n_steps):
            if ei.shape[1] > 0:
                src, dst = ei[0], ei[1]
                msg = self.message(torch.cat([h[src], ea], dim=1))
                agg = torch.zeros_like(h).index_add_(0, dst, msg)
            else:
                agg = torch.zeros_like(h)
            h = self.update(agg, h)

        # Mean and max pooling per graph, concatenated.
        dim = h.shape[1]
        mean = torch.zeros(n_graphs, dim, device=h.device).index_add_(0, bidx, h)
        counts = torch.zeros(n_graphs, 1, device=h.device).index_add_(
            0, bidx, torch.ones(h.shape[0], 1, device=h.device)
        )
        mean = mean / counts.clamp(min=1)

        mx = torch.full((n_graphs, dim), -1e9, device=h.device)
        mx = mx.index_reduce_(0, bidx, h, reduce="amax", include_self=True)
        mx = torch.where(torch.isinf(mx) | (mx < -1e8), torch.zeros_like(mx), mx)

        return self.readout(torch.cat([mean, mx], dim=1)).squeeze(-1)


class MPNNModel(BenchmarkModel):
    """B4 adapter conforming to the benchmark model interface."""

    model_id = "B4"
    model_name = "message_passing_nn"
    representation = "molecular_graph"

    def __init__(self, seed: int = 42, **hp):
        super().__init__(seed=seed, **hp)
        self.hidden_dim = hp.get("hidden_dim", 128)
        self.n_steps = hp.get("n_message_passing_steps", 3)
        self.dropout = hp.get("dropout", 0.15)
        self.lr = hp.get("learning_rate", 1e-3)
        self.batch_size = hp.get("batch_size", 32)
        self.max_epochs = hp.get("max_epochs", 120)
        self.patience = hp.get("early_stopping_patience", 20)
        self.weight_decay = hp.get("weight_decay", 0.0)
        self.net: MPNN | None = None
        self.best_epoch_: int | None = None

    def _graphs(self, smiles: list[str]) -> list[MolGraph]:
        graphs = [MolGraph(s) for s in smiles]
        bad = sum(1 for g in graphs if not g.ok)
        if bad:
            LOG.warning(
                "%d of %d structures did not parse; represented as empty graphs",
                bad,
                len(graphs),
            )
        return graphs

    def fit(self, smiles, y, smiles_val=None, y_val=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        y = np.asarray(y).astype(np.float32)
        graphs = self._graphs(list(smiles))

        self.single_class_ = None
        if len(np.unique(y)) < 2:
            self.single_class_ = float(y[0]) if len(y) else 0.0
            self._fitted = True
            LOG.warning("B4: training fold holds a single class; constant predictor")
            return self

        self.net = MPNN(self.hidden_dim, self.n_steps, self.dropout)
        opt = torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # Class imbalance handled in the loss rather than by resampling, so
        # every compound is seen exactly once per epoch.
        n_pos, n_neg = float(y.sum()), float(len(y) - y.sum())
        pos_weight = torch.tensor(
            [n_neg / n_pos if n_pos > 0 else 1.0], dtype=torch.float32
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        val_graphs = self._graphs(list(smiles_val)) if smiles_val is not None else None
        y_val_t = (
            torch.tensor(np.asarray(y_val).astype(np.float32))
            if y_val is not None
            else None
        )

        best_val, best_state, bad_epochs = np.inf, None, 0
        idx = np.arange(len(graphs))
        rng = np.random.default_rng(self.seed)

        for epoch in range(self.max_epochs):
            self.net.train()
            rng.shuffle(idx)
            for start in range(0, len(idx), self.batch_size):
                chunk = idx[start : start + self.batch_size]
                batch = collate([graphs[i] for i in chunk])
                target = torch.tensor(y[chunk])
                opt.zero_grad()
                loss = loss_fn(self.net(batch), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()

            if val_graphs is None or y_val_t is None or len(val_graphs) == 0:
                continue

            # Early stopping on validation loss only. The test set is never
            # consulted during training.
            self.net.eval()
            with torch.no_grad():
                vloss = float(
                    loss_fn(self.net(collate(val_graphs)), y_val_t).item()
                )
            if vloss < best_val - 1e-5:
                best_val, bad_epochs = vloss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                self.best_epoch_ = epoch
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    LOG.info("B4: early stop at epoch %d", epoch)
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self._fitted = True
        return self

    def predict_proba(self, smiles):
        if not self._fitted:
            raise RuntimeError("predict_proba called before fit")
        if getattr(self, "single_class_", None) is not None:
            return np.full(len(smiles), self.single_class_)
        self.net.eval()
        graphs = self._graphs(list(smiles))
        out: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(graphs), 256):
                batch = collate(graphs[start : start + 256])
                out.append(torch.sigmoid(self.net(batch)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def describe(self) -> dict:
        d = super().describe()
        d["architecture"] = {
            "hidden_dim": self.hidden_dim,
            "n_message_passing_steps": self.n_steps,
            "dropout": self.dropout,
            "atom_feature_dim": ATOM_FEATURE_DIM,
            "bond_feature_dim": BOND_FEATURE_DIM,
            "includes_chirality": True,
            "parameters": (
                int(sum(p.numel() for p in self.net.parameters())) if self.net else None
            ),
        }
        d["training"] = {
            "learning_rate": self.lr,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.patience,
            "best_epoch": self.best_epoch_,
        }
        return d


__all__ = ["ATOM_FEATURE_DIM", "BOND_FEATURE_DIM", "MPNN", "MPNNModel", "MolGraph", "collate"]
