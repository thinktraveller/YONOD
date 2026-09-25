"""FISD descriptor (50-d, two parallel GCNs + 2-in-1 MLP).

Reproduces the inference pipeline from
``化学描述符相关项目/FISD/test_MLMS/reproduce_mlms.py``:

    SMILES -> molecular graph (45-d atom features)
          -> GNN_MSE   (5 GCN layers, 50-d)
          -> GNN_COS   (5 GCN layers, 50-d)
          -> concat(MSE, COS) -> TwoInOne MLP -> 50-d FISD vector

Three .pth weights load from a single directory (default:
``化学描述符相关项目/FISD/model/``). Model architecture is inlined here rather
than imported from the upstream project, because the upstream ``code/`` only
ships Jupyter notebooks -- no pip-installable module.

Requires ``torch_geometric`` for ``GCNConv`` and the pooling ops.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

from .base import BaseDescriptor


# Default location for the three pretrained checkpoints (bundled with the
# repo at WEIGHTS/FISD/). The legacy location 化学描述符相关项目/FISD/model/
# is checked as a fallback in case someone still has that layout.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = _PROJECT_ROOT / "WEIGHTS" / "FISD"
_LEGACY_MODEL_DIR = _PROJECT_ROOT / "化学描述符相关项目" / "FISD" / "model"
MSE_CKPT = "qm_9_mse_model.pth"
COS_CKPT = "qm_9_cos_model.pth"
TWO_IN_ONE_CKPT = "qm_9_2in1_model.pth"

ATOM_FEATURE_DIM = 45


# --- Model architectures (matched bit-for-bit to upstream notebooks) ---


class _GNN(nn.Module):
    def __init__(self, hidden_channels: int = 512) -> None:
        super().__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(ATOM_FEATURE_DIM, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels, 50)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = self.conv5(x, edge_index).relu()
        x = torch.cat(
            (global_mean_pool(x, batch), global_max_pool(x, batch)), dim=1
        )
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        return self.out(x)


class _TwoInOne(nn.Module):
    def __init__(self, hidden_channels: int = 256) -> None:
        super().__init__()
        torch.manual_seed(42)
        self.nn1 = nn.Linear(100, hidden_channels * 2)
        self.nn2 = nn.Linear(hidden_channels * 2, hidden_channels * 2)
        self.nn3 = nn.Linear(hidden_channels * 2, hidden_channels * 2)
        self.nn4 = nn.Linear(hidden_channels * 2, hidden_channels * 2)
        self.out = nn.Linear(hidden_channels * 2, 50)

    def forward(self, x):
        x = self.nn1(x).relu()
        x = self.nn2(x).relu()
        x = self.nn3(x).relu()
        x = self.nn4(x).relu()
        return self.out(x)


# --- Atom featurization (copied from reproduce_mlms.py) ---


def _one_hot(x, permitted):
    if x not in permitted:
        x = permitted[-1]
    return [int(x == p) for p in permitted]


def _atom_features(atom) -> np.ndarray:
    syms = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "Unknown"]
    feats = (
        _one_hot(atom.GetSymbol(), syms)
        + _one_hot(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
        + _one_hot(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
        + _one_hot(
            str(atom.GetHybridization()),
            ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"],
        )
        + [int(atom.IsInRing())]
        + [int(atom.GetIsAromatic())]
        + [(atom.GetMass() - 10.812) / 116.092]
        + [
            (Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6
        ]
        + [
            (Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)
            / 0.76
        ]
        + _one_hot(
            str(atom.GetChiralTag()),
            ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"],
        )
        + _one_hot(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
    )
    return np.asarray(feats, dtype=np.float32)


def _smiles_to_graph(smi: str) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    n = mol.GetNumAtoms()
    if n == 0:
        return None
    X = np.zeros((n, ATOM_FEATURE_DIM), dtype=np.float32)
    for atom in mol.GetAtoms():
        X[atom.GetIdx()] = _atom_features(atom)
    rows, cols = np.nonzero(GetAdjacencyMatrix(mol))
    if rows.size == 0:
        # Single-atom (or disconnected) molecule -- add self-loop so GCN doesn't break.
        rows = np.arange(n, dtype=np.int64)
        cols = np.arange(n, dtype=np.int64)
    edge_index = torch.from_numpy(np.stack([rows, cols]).astype(np.int64))
    return Data(x=torch.from_numpy(X), edge_index=edge_index)


class FISDDescriptor(BaseDescriptor):
    """50-d FISD descriptor via dual GCN + 2-in-1 MLP."""

    name = "fisd"
    output_dim = 50

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        if model_dir is not None:
            self.model_dir = Path(model_dir)
        elif DEFAULT_MODEL_DIR.exists():
            self.model_dir = DEFAULT_MODEL_DIR
        elif _LEGACY_MODEL_DIR.exists():
            self.model_dir = _LEGACY_MODEL_DIR
        else:
            self.model_dir = DEFAULT_MODEL_DIR  # will raise on _load if missing
        self.device = device
        self.batch_size = batch_size
        self._mse = None
        self._cos = None
        self._two = None

    # --- Lazy load -------------------------------------------------------

    def _resolve_device(self) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self) -> None:
        if self._mse is not None:
            return
        import sys

        self.device = self._resolve_device()
        mse = _GNN(hidden_channels=512)
        cos = _GNN(hidden_channels=512)
        two = _TwoInOne(hidden_channels=256)
        mse.load_state_dict(
            torch.load(self.model_dir / MSE_CKPT, map_location=self.device)
        )
        cos.load_state_dict(
            torch.load(self.model_dir / COS_CKPT, map_location=self.device)
        )
        two.load_state_dict(
            torch.load(
                self.model_dir / TWO_IN_ONE_CKPT, map_location=self.device
            )
        )
        self._mse = mse.to(self.device).eval()
        self._cos = cos.to(self.device).eval()
        self._two = two.to(self.device).eval()
        print(
            f"[FISDDescriptor] loaded 3 checkpoints from {self.model_dir} "
            f"on device={self.device}",
            file=sys.stderr,
        )

    # --- Forward ---------------------------------------------------------

    @torch.no_grad()
    def _encode_batch(self, graphs: List[Data]) -> np.ndarray:
        batch = Batch.from_data_list(graphs).to(self.device)
        mse_out = self._mse(batch.x, batch.edge_index, batch.batch)
        cos_out = self._cos(batch.x, batch.edge_index, batch.batch)
        combined = torch.cat([mse_out, cos_out], dim=1)
        fisd = self._two(combined)
        return fisd.detach().cpu().numpy().astype(np.float32)

    def featurize(
        self, smiles_list: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._load()
        n = len(smiles_list)
        features, mask = self._empty_outputs(n, dtype=np.float32)

        # Build graphs; skip invalid SMILES (mask stays False).
        graphs: List[Data] = []
        idx_map: List[int] = []
        for i, smi in enumerate(smiles_list):
            if not smi:
                continue
            g = _smiles_to_graph(smi)
            if g is None:
                continue
            graphs.append(g)
            idx_map.append(i)

        if not graphs:
            return features, mask

        bs = self.batch_size
        i = 0
        while i < len(graphs):
            vecs = self._encode_batch(graphs[i : i + bs])
            for j, vec in enumerate(vecs):
                k = idx_map[i + j]
                features[k] = vec
                mask[k] = True
            i += bs

        return features, mask
