"""Stereo-aware D-MPNN, GINE and AttentiveFP implemented with RDKit/PyTorch.

Only the molecules supplied to ``fit`` are used for epoch selection.  A grouped
holdout inside that subset chooses an epoch; a fresh network is then fitted to
the whole supplied subset for that many epochs.  This interface deliberately
does not accept an outer validation or test set.  See docs/GRAPH_MODELS.md for
the exact architectures, featurization, search space and provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem, rdBase
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
import torch
from torch import nn
from torch.nn import functional as F

from .search_space import graph_candidate_configurations


GRAPH_SCHEMA_VERSION = "nephrotox-graph-v4-searched-architectures-canonical-stereo"
GRAPH_MODEL_NAMES = ("D-MPNN", "GINE", "AttentiveFP")
PARAMETER_DESIGN_PATH = Path(__file__).resolve().parents[2] / "configs" / "original_parameter_design.json"
_PARAMETER_DESIGN = json.loads(PARAMETER_DESIGN_PATH.read_text(encoding="utf-8"))
GRAPH_SELECTION_MODE = _PARAMETER_DESIGN["current_graph_selection_mode"]
GRAPH_REFERENCE_CONFIGURATION_IDS = dict(_PARAMETER_DESIGN["graph_reference_configuration_ids"])
_TRAINING_DEFAULTS = dict(_PARAMETER_DESIGN["implementation_training_defaults"]["graph"])

# Supplementary Table S3 supplies a reference candidate, not a selected winner.
# The search-space module adds reproducibly sampled candidates. Neither old
# performance nor old selected epochs enter candidate generation or selection.
GRAPH_REFERENCE_CONFIGS = {
    name: dict(params)
    for name, params in _PARAMETER_DESIGN["graph_reference_configurations"].items()
}
GRAPH_CANDIDATES = {
    name: graph_candidate_configurations(name)
    for name in GRAPH_MODEL_NAMES
}

_ATOM_CATEGORIES = {
    "atomic_number": list(range(1, 119)),
    "total_degree": list(range(11)),
    "formal_charge": list(range(-5, 6)),
    "hydrogen_count": list(range(9)),
    "hybridization": ["UNSPECIFIED", "S", "SP", "SP2", "SP3", "SP2D",
                      "SP3D", "SP3D2", "OTHER"],
    "radical_electrons": list(range(5)),
    "chiral_tag": list(range(9)),
    "chiral_permutation": list(range(31)),
    "CIP_code": ["R", "S", "r", "s", "unspecified"],
}
_BOND_CATEGORIES = {
    "bond_type": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "DATIVE", "ZERO",
                  "QUADRUPLE", "QUINTUPLE", "HEXTUPLE", "IONIC", "HYDROGEN",
                  "THREECENTER", "OTHER"],
    "bond_stereo": list(range(8)),
    "bond_direction": list(range(7)),
}


def _one_hot(value, choices):
    """Every categorical field has an additional explicit unknown channel."""
    result = [float(value == choice) for choice in choices]
    return result + [float(value not in choices)]


ATOM_FEATURE_NAMES = tuple(
    [f"{key}={value}" for key, values in _ATOM_CATEGORIES.items()
     for value in (*values, "unknown")]
    + ["aromatic", "in_ring", "mass_div_100", "explicit_isotope_div_300"]
)
BOND_FEATURE_NAMES = tuple(
    [f"{key}={value}" for key, values in _BOND_CATEGORIES.items()
     for value in (*values, "unknown")]
    + ["conjugated", "in_ring"]
)
ATOM_FDIM = len(ATOM_FEATURE_NAMES)
BOND_FDIM = len(BOND_FEATURE_NAMES)


def graph_feature_schema():
    """Return a JSON-serializable, complete feature/order definition."""
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "rdkit_version": rdBase.rdkitVersion,
        "atom_features": list(ATOM_FEATURE_NAMES),
        "bond_features": list(BOND_FEATURE_NAMES),
        "atom_dimension": ATOM_FDIM,
        "bond_dimension": BOND_FDIM,
        "directed_edges_per_bond": 2,
        "hydrogens": "RDKit implicit/explicit H counts; no AddHs",
        "stereochemistry": "canonical-isomeric traversal chiral tag/permutation, CIP and bond stereo/direction mapped back to original indices; no completion",
        "stereo_atom_index_mapping": "RDKit canonical SMILES _smilesAtomOutputOrder; original input atom/bond indices retained",
        "mixtures": "multiple disconnected input components are rejected",
        "occlusion": "remove both directed edges; retain original atoms and atom features",
    }


def _atom_features(atom):
    values = (
        atom.GetAtomicNum(), atom.GetTotalDegree(), atom.GetFormalCharge(),
        atom.GetTotalNumHs(includeNeighbors=True), str(atom.GetHybridization()),
        atom.GetNumRadicalElectrons(), int(atom.GetChiralTag()),
        atom.GetIntProp("_chiralPermutation") if atom.HasProp("_chiralPermutation") else 0,
        atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else "unspecified",
    )
    features = []
    for value, choices in zip(values, _ATOM_CATEGORIES.values()):
        features.extend(_one_hot(value, choices))
    features.extend((float(atom.GetIsAromatic()), float(atom.IsInRing()),
                     atom.GetMass() / 100.0, atom.GetIsotope() / 300.0))
    return features


def _bond_features(bond):
    values = (str(bond.GetBondType()), int(bond.GetStereo()), int(bond.GetBondDir()))
    features = []
    for value, choices in zip(values, _BOND_CATEGORIES.values()):
        features.extend(_one_hot(value, choices))
    features.extend((float(bond.GetIsConjugated()), float(bond.IsInRing())))
    return features


@dataclass(frozen=True)
class MolecularGraph:
    """One molecule. edge_index uses source row 0 and destination row 1."""
    smiles: str
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    reverse_edge: np.ndarray
    bond_index: np.ndarray
    n_bonds: int


@lru_cache(maxsize=8192)
def _cached_graph(smiles: str, removed_bonds: tuple[int, ...]):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid or empty molecular SMILES: {smiles!r}")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Graph input must contain exactly one molecular component")
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        raise ValueError("Unspecified/dummy atoms cannot enter graph modeling")
    # This assigns labels only for stereochemistry actually present in SMILES.
    # It does not enumerate or fill an unspecified stereocentre.
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    # CW/CCW chiral tags, non-tetrahedral permutations and slash directions
    # depend on SMILES traversal. Read them from one canonical isomeric
    # traversal, then map features back to the *original* atom/bond indices.
    # This preserves occlusion numbering while removing representation-order
    # artifacts. CIP alone would not resolve non-tetrahedral stereochemistry.
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    canonical_order = json.loads(mol.GetProp("_smilesAtomOutputOrder"))
    canonical = Chem.MolFromSmiles(canonical_smiles)
    if (canonical is None or canonical.GetNumAtoms() != mol.GetNumAtoms()
            or canonical.GetNumBonds() != mol.GetNumBonds()
            or sorted(canonical_order) != list(range(mol.GetNumAtoms()))):
        raise ValueError("Canonical stereo feature/source index mapping failed")
    Chem.AssignStereochemistry(canonical, cleanIt=True, force=True)
    original_to_canonical = np.empty(mol.GetNumAtoms(), dtype=np.int64)
    original_to_canonical[canonical_order] = np.arange(mol.GetNumAtoms())
    n_bonds = mol.GetNumBonds()
    if any(index < 0 or index >= n_bonds for index in removed_bonds):
        raise ValueError(f"Bond index outside 0..{n_bonds - 1}")
    removed = set(removed_bonds)
    x = np.asarray([_atom_features(canonical.GetAtomWithIdx(int(original_to_canonical[i])))
                    for i in range(mol.GetNumAtoms())], dtype=np.float32)
    source, destination, edges, reverse, indices = [], [], [], [], []
    for bond in mol.GetBonds():
        if bond.GetIdx() in removed:
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        offset = len(source)
        source.extend((a, b))
        destination.extend((b, a))
        canonical_bond = canonical.GetBondBetweenAtoms(int(original_to_canonical[a]),
                                                       int(original_to_canonical[b]))
        if canonical_bond is None:
            raise ValueError("Canonical stereo bond/source index mapping failed")
        features = _bond_features(canonical_bond)
        edges.extend((features, features))
        reverse.extend((offset + 1, offset))
        indices.extend((bond.GetIdx(), bond.GetIdx()))
    edge_index = np.asarray([source, destination], dtype=np.int64).reshape(2, -1)
    edge_attr = np.asarray(edges, dtype=np.float32).reshape(-1, BOND_FDIM)
    reverse_edge = np.asarray(reverse, dtype=np.int64)
    bond_index = np.asarray(indices, dtype=np.int64)
    for array in (x, edge_index, edge_attr, reverse_edge, bond_index):
        array.setflags(write=False)
    return MolecularGraph(smiles, x, edge_index, edge_attr, reverse_edge,
                          bond_index, n_bonds)


def molecular_graph(smiles: str, removed_bonds: Iterable[int] = ()):
    """Build a stereo-preserving graph; optionally mask RDKit bond indices.

    Bond masking changes connectivity presented to the network without changing
    molecular valence, H counts, aromaticity or the chemical structure itself.
    This is a feature perturbation, not a simulated reaction/metabolite.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a nonempty string")
    indices = tuple(removed_bonds)
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
           for i in indices):
        raise ValueError("Removed bond indices must be integers")
    return _cached_graph(smiles, tuple(sorted(set(int(i) for i in indices))))


@dataclass
class GraphBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    reverse_edge: torch.Tensor
    batch: torch.Tensor
    num_graphs: int


def collate_graphs(graphs: Sequence[MolecularGraph]):
    """Concatenate molecules with separate node/edge offsets and no cross edges."""
    if not graphs:
        raise ValueError("Cannot collate an empty molecular batch")
    xs, edge_indices, edge_features, reverses, membership = [], [], [], [], []
    node_offset = edge_offset = 0
    for i, graph in enumerate(graphs):
        xs.append(graph.x)
        edge_indices.append(graph.edge_index + node_offset)
        edge_features.append(graph.edge_attr)
        reverses.append(graph.reverse_edge + edge_offset)
        membership.append(np.full(len(graph.x), i, dtype=np.int64))
        node_offset += len(graph.x)
        edge_offset += graph.edge_index.shape[1]
    return GraphBatch(
        x=torch.from_numpy(np.concatenate(xs, axis=0)),
        edge_index=torch.from_numpy(np.concatenate(edge_indices, axis=1)),
        edge_attr=torch.from_numpy(np.concatenate(edge_features, axis=0)),
        reverse_edge=torch.from_numpy(np.concatenate(reverses)),
        batch=torch.from_numpy(np.concatenate(membership)), num_graphs=len(graphs),
    )


def segment_sum(values, index, count):
    result = values.new_zeros((count,) + tuple(values.shape[1:]))
    return result.index_add_(0, index, values)


def segment_softmax(logits, index, count):
    """Numerically stable softmax separately for each destination node/graph."""
    if logits.numel() == 0:
        return logits
    maximum = logits.new_full((count,), -torch.inf)
    maximum.scatter_reduce_(0, index, logits.detach(), reduce="amax", include_self=True)
    weights = torch.exp(logits - maximum[index])
    return weights / segment_sum(weights, index, count)[index].clamp_min(1e-20)


def nonbacktracking_messages(hidden_edges, edge_index, reverse_edge, num_atoms):
    """For v->w, sum incoming u->v except its immediate reverse w->v."""
    incoming = segment_sum(hidden_edges, edge_index[1], num_atoms)
    return incoming[edge_index[0]] - hidden_edges[reverse_edge]


def _mean_pool(x, graph_batch):
    pooled = segment_sum(x, graph_batch.batch, graph_batch.num_graphs)
    counts = torch.bincount(graph_batch.batch, minlength=graph_batch.num_graphs)
    return pooled / counts.to(x.dtype).clamp_min(1).unsqueeze(1)


def _prediction_head(input_dim, hidden_dim, num_layers, dropout):
    """Chemprop v2 convention: num_layers hidden layers plus one output map."""
    layers = []
    for layer in range(num_layers + 1):
        last = layer == num_layers
        layers.append(nn.Linear(input_dim if layer == 0 else hidden_dim,
                                1 if last else hidden_dim))
        if not last:
            layers.extend((nn.ReLU(), nn.Dropout(dropout)))
    return nn.Sequential(*layers)


class DMPNN(nn.Module):
    """Directed nonbacktracking messages, sum aggregation and separate FFN."""
    def __init__(self, hidden_dim=256, depth=5, dropout=0.0,
                 message_hidden_dim=None, message_depth=None,
                 message_dropout=None, aggregation="sum", ffn_hidden_dim=300,
                 ffn_num_layers=3, ffn_dropout=0.25, **_):
        super().__init__()
        hidden_dim = hidden_dim if message_hidden_dim is None else message_hidden_dim
        depth = depth if message_depth is None else message_depth
        dropout = dropout if message_dropout is None else message_dropout
        self.depth, self.dropout = depth, dropout
        if aggregation not in {"sum", "mean"}:
            raise ValueError("D-MPNN aggregation must be sum or mean")
        self.aggregation = aggregation
        self.input = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_dim, bias=False)
        self.message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.atom_output = nn.Linear(ATOM_FDIM + hidden_dim, hidden_dim)
        self.predictor = _prediction_head(hidden_dim, ffn_hidden_dim,
                                          ffn_num_layers, ffn_dropout)

    def forward(self, graph):
        source = graph.edge_index[0]
        # The residual is the preactivation input; depth includes this first
        # edge embedding and depth-1 nonbacktracking updates (Chemprop convention).
        initial = self.input(torch.cat((graph.x[source], graph.edge_attr), dim=1))
        hidden = F.relu(initial)
        for _ in range(self.depth - 1):
            message = nonbacktracking_messages(hidden, graph.edge_index,
                                               graph.reverse_edge, len(graph.x))
            hidden = F.dropout(F.relu(initial + self.message(message)),
                               self.dropout, self.training)
        incoming = segment_sum(hidden, graph.edge_index[1], len(graph.x))
        atoms = F.dropout(F.relu(self.atom_output(torch.cat((graph.x, incoming), dim=1))),
                          self.dropout, self.training)
        pooled = (segment_sum(atoms, graph.batch, graph.num_graphs)
                  if self.aggregation == "sum" else _mean_pool(atoms, graph))
        return self.predictor(pooled).squeeze(-1)


class SafeBatchNorm1d(nn.BatchNorm1d):
    """Use running statistics for an exceptional single-atom training batch."""
    def forward(self, x):
        if self.training and len(x) < 2:
            return F.batch_norm(x, self.running_mean, self.running_var,
                                self.weight, self.bias, training=False,
                                momentum=self.momentum, eps=self.eps)
        return super().forward(x)


class GINELayer(nn.Module):
    """MLP((1+epsilon)*x_i + sum_j ReLU(x_j + W_edge*e_ji))."""
    def __init__(self, hidden_dim, mlp_num_layers=2,
                 normalization="batch_norm", train_epsilon=True):
        super().__init__()
        if train_epsilon:
            self.epsilon = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("epsilon", torch.zeros(()))
        self.edge_projection = nn.Linear(BOND_FDIM, hidden_dim)
        update = []
        for layer in range(mlp_num_layers):
            update.append(nn.Linear(hidden_dim, hidden_dim))
            if layer < mlp_num_layers - 1:
                update.append(nn.ReLU())
        self.update = nn.Sequential(*update)
        normalizers = {"batch_norm": SafeBatchNorm1d, "layer_norm": nn.LayerNorm,
                       "none": lambda _: nn.Identity()}
        if normalization not in normalizers:
            raise ValueError("Unknown GINE normalization")
        self.normalization = normalizers[normalization](hidden_dim)

    def forward(self, x, graph):
        source, destination = graph.edge_index
        message = F.relu(x[source] + self.edge_projection(graph.edge_attr))
        total = segment_sum(message, destination, len(x))
        return self.normalization(self.update((1.0 + self.epsilon) * x + total))


class GINE(nn.Module):
    def __init__(self, hidden_dim=256, depth=4, dropout=0.2,
                 num_layers=None, mlp_num_layers=2, normalization="batch_norm",
                 pooling="gated_attention", train_epsilon=True, **_):
        super().__init__()
        depth = depth if num_layers is None else num_layers
        self.dropout = dropout
        if pooling not in {"gated_attention", "mean", "sum"}:
            raise ValueError("Unknown GINE pooling")
        self.pooling = pooling
        self.input = nn.Linear(ATOM_FDIM, hidden_dim)
        self.layers = nn.ModuleList([GINELayer(hidden_dim, mlp_num_layers,
                                              normalization, train_epsilon)
                                     for _ in range(depth)])
        if pooling == "gated_attention":
            self.pool_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                          nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.predictor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                       nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, graph):
        atoms = F.relu(self.input(graph.x))
        for layer in self.layers:
            update = F.dropout(F.relu(layer(atoms, graph)), self.dropout, self.training)
            atoms = atoms + update
        if self.pooling == "gated_attention":
            weights = segment_softmax(self.pool_gate(atoms).squeeze(-1),
                                      graph.batch, graph.num_graphs)
            pooled = segment_sum(atoms * weights[:, None], graph.batch, graph.num_graphs)
        elif self.pooling == "sum":
            pooled = segment_sum(atoms, graph.batch, graph.num_graphs)
        else:
            pooled = _mean_pool(atoms, graph)
        return self.predictor(pooled).squeeze(-1)


class AtomAttention(nn.Module):
    """Single-head neighbor attention; first layer also conditions on bonds."""
    def __init__(self, hidden_dim, dropout, use_bonds=False):
        super().__init__()
        self.dropout, self.use_bonds = dropout, use_bonds
        self.source_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.source_attention = nn.Parameter(torch.empty(hidden_dim))
        self.destination_attention = nn.Parameter(torch.empty(hidden_dim))
        if use_bonds:
            self.edge_projection = nn.Linear(hidden_dim + BOND_FDIM, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.xavier_uniform_(self.source_attention.unsqueeze(0))
        nn.init.xavier_uniform_(self.destination_attention.unsqueeze(0))

    def forward(self, x, graph):
        source, destination = graph.edge_index
        projected = self.source_projection(x)
        if self.use_bonds:
            key = F.leaky_relu(self.edge_projection(torch.cat((x[source], graph.edge_attr), dim=1)),
                               negative_slope=0.01)
            query = x[destination]
        else:
            key, query = projected[source], projected[destination]
        logits = F.leaky_relu((key * self.source_attention).sum(1)
                               + (query * self.destination_attention).sum(1),
                               negative_slope=0.01)
        attention = segment_softmax(logits, destination, len(x))
        attention = F.dropout(attention, self.dropout, self.training)
        return segment_sum(projected[source] * attention[:, None], destination, len(x)) + self.bias


class MoleculeAttention(nn.Module):
    """Bipartite atom->molecule attention used at each GRU readout timestep."""
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.dropout = dropout
        self.projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.atom_attention = nn.Parameter(torch.empty(hidden_dim))
        self.molecule_attention = nn.Parameter(torch.empty(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.xavier_uniform_(self.atom_attention.unsqueeze(0))
        nn.init.xavier_uniform_(self.molecule_attention.unsqueeze(0))

    def forward(self, atoms, molecules, graph):
        keys, queries = self.projection(atoms), self.projection(molecules)
        logits = F.leaky_relu((keys * self.atom_attention).sum(1)
                               + (queries[graph.batch] * self.molecule_attention).sum(1),
                               negative_slope=0.01)
        attention = segment_softmax(logits, graph.batch, graph.num_graphs)
        attention = F.dropout(attention, self.dropout, self.training)
        return segment_sum(keys * attention[:, None], graph.batch, graph.num_graphs) + self.bias


class AttentiveFP(nn.Module):
    """Bond-conditioned first attention, atom GRUs and iterative molecular GRU."""
    def __init__(self, hidden_dim=256, depth=3, dropout=0.2, readout_steps=2,
                 hidden_channels=None, num_layers=None, num_timesteps=None, **_):
        super().__init__()
        hidden_dim = hidden_dim if hidden_channels is None else hidden_channels
        depth = depth if num_layers is None else num_layers
        readout_steps = readout_steps if num_timesteps is None else num_timesteps
        self.dropout, self.readout_steps = dropout, readout_steps
        self.input = nn.Linear(ATOM_FDIM, hidden_dim)
        self.attention_layers = nn.ModuleList([
            AtomAttention(hidden_dim, dropout, use_bonds=(i == 0))
            for i in range(depth)
        ])
        self.atom_grus = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for _ in range(depth)])
        self.molecular_attention = MoleculeAttention(hidden_dim, dropout)
        self.molecular_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, graph):
        atoms = F.leaky_relu(self.input(graph.x), negative_slope=0.01)
        for attention, gru in zip(self.attention_layers, self.atom_grus):
            message = F.dropout(F.elu(attention(atoms, graph)), self.dropout, self.training)
            atoms = F.relu(gru(message, atoms))
        molecules = F.relu(segment_sum(atoms, graph.batch, graph.num_graphs))
        for _ in range(self.readout_steps):
            message = F.dropout(F.elu(self.molecular_attention(atoms, molecules, graph)),
                                self.dropout, self.training)
            molecules = F.relu(self.molecular_gru(message, molecules))
        return self.output(F.dropout(molecules, self.dropout, self.training)).squeeze(-1)


def _normalize_name(name):
    aliases = {"dmpnn": "D-MPNN", "gine": "GINE", "attentivefp": "AttentiveFP"}
    key = str(name).lower().replace("-", "").replace("_", "").replace(" ", "")
    if key not in aliases:
        raise ValueError(f"Unknown graph architecture: {name!r}")
    return aliases[key]


def _new_network(name, params):
    return {"D-MPNN": DMPNN, "GINE": GINE, "AttentiveFP": AttentiveFP}[name](**params)


def _validate_params(params, model_name):
    """Resolve documented aliases and reject unused/misspelled parameters."""
    aliases = {
        "D-MPNN": {"hidden_dim": "message_hidden_dim", "depth": "message_depth",
                   "dropout": "message_dropout"},
        "GINE": {"depth": "num_layers"},
        "AttentiveFP": {"hidden_dim": "hidden_channels", "depth": "num_layers",
                        "readout_steps": "num_timesteps"},
    }[model_name]
    params = dict(params)
    for alias, canonical in aliases.items():
        if alias in params:
            if canonical in params and params[alias] != params[canonical]:
                raise ValueError(f"Conflicting {alias} and {canonical}")
            params[canonical] = params.pop(alias)
    allowed = set(_TRAINING_DEFAULTS) | set(GRAPH_REFERENCE_CONFIGS[model_name])
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"Unknown graph parameters: {sorted(unknown)}")
    result = dict(_TRAINING_DEFAULTS, **GRAPH_REFERENCE_CONFIGS[model_name])
    result.update(params)
    integer_keys = {"hidden_dim", "message_hidden_dim", "message_depth", "ffn_hidden_dim",
                    "ffn_num_layers", "mlp_num_layers", "num_layers", "hidden_channels",
                    "num_timesteps", "batch_size", "max_epochs", "patience"} & set(result)
    for key in integer_keys:
        if isinstance(result[key], (bool, np.bool_)) or int(result[key]) != result[key] or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
        result[key] = int(result[key])
    for key in ("learning_rate", "gradient_clip"):
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
        result[key] = float(result[key])
    for key in ("weight_decay", "min_delta"):
        if not np.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
        result[key] = float(result[key])
    for key in {"dropout", "message_dropout", "ffn_dropout"} & set(result):
        if not np.isfinite(result[key]) or not 0 <= result[key] < 1:
            raise ValueError(f"{key} must be in [0,1)")
        result[key] = float(result[key])
    if not np.isfinite(result["early_stop_fraction"]) or not 0 < result["early_stop_fraction"] < 0.5:
        raise ValueError("early_stop_fraction must be in (0,0.5)")
    result["early_stop_fraction"] = float(result["early_stop_fraction"])
    if "aggregation" in result and result["aggregation"] not in {"sum", "mean"}:
        raise ValueError("aggregation must be sum or mean")
    if "pooling" in result and result["pooling"] not in {"gated_attention", "mean", "sum"}:
        raise ValueError("Unsupported GINE pooling")
    if "normalization" in result and result["normalization"] not in {"batch_norm", "layer_norm", "none"}:
        raise ValueError("Unsupported GINE normalization")
    if "train_epsilon" in result and not isinstance(result["train_epsilon"], bool):
        raise ValueError("train_epsilon must be a bool")
    if result["learning_rate_schedule"] != "constant":
        raise ValueError("Only a constant graph learning-rate schedule is supported")
    return result


def grouped_epoch_split(y, groups, seed, fraction=0.2):
    """Choose a two-class early-stop subset with strictly disjoint groups.

    Stratified group folds are attempted first. Group shuffle candidates handle
    awkward small/large groups. If no defensible two-class split exists, fail
    explicitly instead of reverting to an ungrouped random split.
    """
    y, groups = np.asarray(y, dtype=int), np.asarray(groups, dtype=str)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("At least three structure groups are needed for grouped epoch selection")
    n_splits = min(max(2, round(1 / fraction)), len(unique_groups))
    splits = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                      random_state=int(seed)).split(np.zeros(len(y)), y, groups))
    splits.extend(GroupShuffleSplit(n_splits=64, test_size=fraction,
                                     random_state=int(seed)).split(np.zeros(len(y)), y, groups))
    valid = []
    for train, validation in splits:
        if len(np.unique(y[train])) != 2 or len(np.unique(y[validation])) != 2:
            continue
        if set(groups[train]) & set(groups[validation]):
            raise AssertionError("Structure group crossed an epoch-selection boundary")
        score = abs(len(validation) / len(y) - fraction) + abs(y[validation].mean() - y.mean())
        valid.append((float(score), tuple(int(i) for i in validation), train, validation))
    if not valid:
        raise ValueError("No two-class group-disjoint epoch-selection split is possible")
    _, _, train, validation = min(valid, key=lambda item: (item[0], item[1]))
    return np.asarray(train, dtype=int), np.asarray(validation, dtype=int)


def _seed_everything(seed, n_jobs):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(n_jobs)
    torch.use_deterministic_algorithms(True)


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def _positive_weight(y):
    y = np.asarray(y, dtype=int)
    if set(y.tolist()) != {0, 1}:
        raise ValueError("Both classes are required in every graph fitting subset")
    return float((y == 0).sum() / (y == 1).sum())


def _fit_epoch(model, graphs, targets, indices, optimizer, params, rng, pos_weight):
    model.train()
    order = rng.permutation(indices)
    total, count = 0.0, 0
    for start in range(0, len(order), params["batch_size"]):
        batch_ids = order[start:start + params["batch_size"]]
        graph = collate_graphs([graphs[int(i)] for i in batch_ids])
        truth = torch.as_tensor(targets[batch_ids], dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits = model(graph)
        loss = F.binary_cross_entropy_with_logits(logits, truth,
                                                  pos_weight=logits.new_tensor(pos_weight))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite graph training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), params["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach()) * len(batch_ids)
        count += len(batch_ids)
    return total / count


def _predict_network(model, graphs, batch_size):
    model.eval()
    logits = []
    with torch.inference_mode():
        for start in range(0, len(graphs), batch_size):
            batch = collate_graphs(graphs[start:start + batch_size])
            logits.append(model(batch).detach().cpu().numpy().astype(np.float64))
    if not logits:
        return np.empty(0, dtype=np.float64)
    result = np.concatenate(logits)
    if not np.isfinite(result).all():
        raise FloatingPointError("Graph prediction contains nonfinite logits")
    return result


class GraphEstimator:
    """A CPU graph classifier; seeds average probabilities, never logits.

    Candidate selection is performed by the grouped inner-CV runner, outside
    this fit. The caller passes the same three seeds to OOF and final fits.
    ``joblib.dump`` works directly; ``save_checkpoint`` also writes portable
    CPU state dictionaries and architecture metadata.
    """
    def __init__(self, model_name, params, seed=20260810, seeds=None, n_jobs=2):
        self.model_name = _normalize_name(model_name)
        self.params = _validate_params(dict(params), self.model_name)
        self.seed = int(seed)
        self.seeds = tuple(int(s) for s in (seeds if seeds is not None else [seed]))
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must contain at least one unique integer")
        if any(s < 0 or s >= 2 ** 32 for s in self.seeds):
            raise ValueError("Each seed must be in [0,2**32)")
        if isinstance(n_jobs, bool) or int(n_jobs) != n_jobs or n_jobs < 1:
            raise ValueError("n_jobs must be a positive CPU thread count")
        self.n_jobs = int(n_jobs)

    def fit(self, smiles, y, groups, sample_ids=None, log_dir=None):
        smiles = np.asarray(smiles, dtype=str)
        y_original = np.asarray(y)
        y = np.asarray(y, dtype=np.int64)
        groups = np.asarray(groups, dtype=str)
        if smiles.ndim != 1 or y.ndim != 1 or groups.ndim != 1:
            raise ValueError("smiles, y and groups must be one-dimensional")
        if len(smiles) != len(y) or len(y) != len(groups) or not len(y):
            raise ValueError("smiles/y/groups lengths differ or are empty")
        if not np.array_equal(y_original, y) or set(y.tolist()) != {0, 1}:
            raise ValueError("Graph labels must be exactly binary 0/1")
        if any(not s.strip() for s in groups):
            raise ValueError("Structure group identifiers cannot be empty")
        sample_ids = np.asarray(sample_ids if sample_ids is not None else
                                [f"fit_row_{i}" for i in range(len(y))], dtype=str)
        if (sample_ids.ndim != 1 or len(sample_ids) != len(y)
                or len(set(sample_ids)) != len(y) or any(not s.strip() for s in sample_ids)):
            raise ValueError("sample_ids must be unique and match the fitting rows")
        started = time.perf_counter()
        graphs = [molecular_graph(s) for s in smiles]
        graph_build_seconds = time.perf_counter() - started
        self.models_ = []
        records = []
        input_hash = hashlib.sha256(json.dumps(
            list(zip(sample_ids.tolist(), smiles.tolist(), y.tolist(), groups.tolist())),
            ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        for seed in self.seeds:
            seed_started = time.perf_counter()
            inner_train, inner_validation = grouped_epoch_split(
                y, groups, seed, self.params["early_stop_fraction"])
            _seed_everything(seed, self.n_jobs)
            model = _new_network(self.model_name, self.params)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.params["learning_rate"],
                                          weight_decay=self.params["weight_decay"])
            rng = np.random.default_rng(seed)
            stopping_curve, best_loss, best_epoch, stale = [], float("inf"), 0, 0
            early_weight = _positive_weight(y[inner_train])
            for epoch in range(1, self.params["max_epochs"] + 1):
                epoch_started = time.perf_counter()
                loss = _fit_epoch(model, graphs, y, inner_train, optimizer,
                                  self.params, rng, early_weight)
                validation_logits = _predict_network(
                    model, [graphs[int(i)] for i in inner_validation], self.params["batch_size"])
                # stable unweighted BCE, evaluated only on this fit's grouped holdout.
                validation_loss = float(np.mean(np.logaddexp(0.0, validation_logits)
                                                 - y[inner_validation] * validation_logits))
                stopping_curve.append(dict(epoch=epoch, training_weighted_bce=loss,
                                            validation_bce=validation_loss,
                                            learning_rate=self.params["learning_rate"],
                                            seconds=time.perf_counter() - epoch_started))
                if validation_loss < best_loss - self.params["min_delta"]:
                    best_loss, best_epoch, stale = validation_loss, epoch, 0
                else:
                    stale += 1
                if log_dir is not None:
                    _atomic_json(Path(log_dir) / f"seed_{seed}_epoch_selection.json", {
                        "phase": "epoch_selection", "seed": seed, "curve": stopping_curve,
                        "selected_epoch_so_far": best_epoch,
                        "training_ids": sample_ids[inner_train].tolist(),
                        "validation_ids": sample_ids[inner_validation].tolist(),
                    })
                if stale >= self.params["patience"]:
                    break
            # Retrain from a new initialization for the chosen fixed epoch on
            # every row supplied to fit. No outer validation or test is visible.
            _seed_everything(seed, self.n_jobs)
            model = _new_network(self.model_name, self.params)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.params["learning_rate"],
                                          weight_decay=self.params["weight_decay"])
            rng = np.random.default_rng(seed)
            all_rows, refit_curve = np.arange(len(y)), []
            refit_weight = _positive_weight(y)
            for epoch in range(1, best_epoch + 1):
                epoch_started = time.perf_counter()
                loss = _fit_epoch(model, graphs, y, all_rows, optimizer,
                                  self.params, rng, refit_weight)
                refit_curve.append(dict(epoch=epoch, training_weighted_bce=loss,
                                         learning_rate=self.params["learning_rate"],
                                         seconds=time.perf_counter() - epoch_started))
                if log_dir is not None:
                    _atomic_json(Path(log_dir) / f"seed_{seed}_refit.json", {
                        "phase": "fixed_epoch_refit", "seed": seed,
                        "selected_epoch": best_epoch, "curve": refit_curve,
                        "training_ids": sample_ids.tolist(),
                    })
            model.eval()
            self.models_.append(model)
            record = {
                "seed": seed, "epoch_selection_training_ids": sample_ids[inner_train].tolist(),
                "epoch_selection_validation_ids": sample_ids[inner_validation].tolist(),
                "epoch_selection_training_groups": sorted(str(g) for g in set(groups[inner_train])),
                "epoch_selection_validation_groups": sorted(str(g) for g in set(groups[inner_validation])),
                "epoch_selection_group_overlap": 0,
                "early_stop_positive_weight": early_weight,
                "epoch_selection_curve": stopping_curve,
                "selected_epoch": best_epoch,
                "selected_validation_bce": best_loss,
                "maximum_epoch_reached": len(stopping_curve) == self.params["max_epochs"],
                "refit_training_ids": sample_ids.tolist(), "refit_positive_weight": refit_weight,
                "refit_curve": refit_curve, "seconds": time.perf_counter() - seed_started,
                "trainable_parameters": sum(p.numel() for p in model.parameters()),
                "model_definition": str(model),
            }
            records.append(record)
            if log_dir is not None:
                checkpoint = Path(log_dir) / f"seed_{seed}_state.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"schema_version": GRAPH_SCHEMA_VERSION, "model_name": self.model_name,
                            "params": self.params, "seed": seed, "selected_epoch": best_epoch,
                            "state_dict": model.state_dict(), "fit_input_sha256": input_hash}, checkpoint)
                record["checkpoint"] = checkpoint.name
                record["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.fit_metadata_ = {
            "model_name": self.model_name, "implementation": "pure PyTorch + RDKit",
            "selection_mode": GRAPH_SELECTION_MODE,
            "reference_configuration_id": GRAPH_REFERENCE_CONFIGURATION_IDS[self.model_name],
            "configuration_sha256": hashlib.sha256(json.dumps(
                self.params, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "schema_version": GRAPH_SCHEMA_VERSION, "feature_schema": graph_feature_schema(),
            "params": dict(self.params), "seeds": list(self.seeds), "threads": self.n_jobs,
            "torch_version": str(torch.__version__), "device": "cpu",
            "fit_input_sha256": input_hash, "fit_sample_ids": sample_ids.tolist(),
            "fit_group_count": len(set(groups)), "graph_build_seconds": graph_build_seconds,
            "epoch_selection_metric": "unweighted grouped-holdout BCE within the fitting subset",
            "refit_policy": "fresh initialization at same seed; all fit rows; fixed selected epoch; fixed learning rate",
            "probability_aggregation": "arithmetic mean of member sigmoid probabilities",
            "members": records, "fit_seconds": time.perf_counter() - started,
        }
        if log_dir is not None:
            _atomic_json(Path(log_dir) / "fit_metadata.json", self.fit_metadata_)
        return self

    def _seed_probabilities_graphs(self, graphs):
        if not hasattr(self, "models_") or not self.models_:
            raise RuntimeError("Graph estimator has not been fitted")
        torch.set_num_threads(self.n_jobs)
        if not graphs:
            return np.empty((len(self.models_), 0), dtype=np.float64)
        # torch.sigmoid avoids overflow at extreme logits and preserves [0,1].
        probabilities = [torch.sigmoid(torch.from_numpy(_predict_network(
            model, graphs, self.params["batch_size"]))).numpy() for model in self.models_]
        return np.asarray(probabilities, dtype=np.float64)

    def _predict_graphs(self, graphs):
        positive = self._seed_probabilities_graphs(graphs).mean(axis=0)
        return np.column_stack((1.0 - positive, positive))

    def predict_seed_probabilities(self, smiles):
        """Positive probability per member, in declared seed order: seeds x n."""
        if isinstance(smiles, str):
            raise ValueError("predict_seed_probabilities expects a sequence of SMILES")
        return self._seed_probabilities_graphs([molecular_graph(str(s)) for s in smiles])

    def predict_proba(self, smiles):
        if isinstance(smiles, str):
            raise ValueError("predict_proba expects a sequence of SMILES, not one string")
        return self._predict_graphs([molecular_graph(str(s)) for s in smiles])

    def predict(self, smiles):
        return (self.predict_proba(smiles)[:, 1] >= 0.5).astype(int)

    def predict_with_removed_bonds(self, smiles, bond_indices):
        """Probability after edge masking, not chemical bond deletion/sanitizing.

        A string + integer sequence masks those bonds in one molecule and returns
        shape (1,2). A sequence of strings requires one integer sequence per
        molecule and returns (n,2). RDKit bond numbering is that of the exact
        input SMILES, retained in the graph's ``bond_index`` field.
        """
        if isinstance(smiles, str):
            graphs = [molecular_graph(smiles, bond_indices)]
        else:
            smiles, bond_indices = list(smiles), list(bond_indices)
            if len(smiles) != len(bond_indices):
                raise ValueError("Each molecule needs its own removed-bond index sequence")
            graphs = [molecular_graph(str(s), indices) for s, indices in zip(smiles, bond_indices)]
        return self._predict_graphs(graphs)

    def save_checkpoint(self, path):
        if not hasattr(self, "models_"):
            raise RuntimeError("Graph estimator has not been fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": GRAPH_SCHEMA_VERSION, "model_name": self.model_name,
                    "params": self.params, "seed": self.seed, "seeds": list(self.seeds),
                    "n_jobs": self.n_jobs,
                    "states": [model.state_dict() for model in self.models_],
                    "fit_metadata": self.fit_metadata_}, path)
        return path


def make_graph_estimator(model_name, params, seed=20260810, seeds=None, n_jobs=2):
    return GraphEstimator(model_name, params, seed=seed, seeds=seeds, n_jobs=n_jobs)


def load_graph_estimator(path):
    """Restore an own-produced CPU checkpoint using Torch's restricted loader."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("schema_version") != GRAPH_SCHEMA_VERSION:
        raise ValueError("Graph checkpoint schema version differs")
    estimator = make_graph_estimator(payload["model_name"], payload["params"],
                                      payload["seed"], payload["seeds"], payload["n_jobs"])
    estimator.models_ = []
    for state in payload["states"]:
        model = _new_network(estimator.model_name, estimator.params)
        model.load_state_dict(state, strict=True)
        model.eval()
        estimator.models_.append(model)
    if len(estimator.models_) != len(estimator.seeds):
        raise ValueError("Graph checkpoint seed/state counts differ")
    estimator.classes_ = np.array([0, 1], dtype=np.int64)
    estimator.fit_metadata_ = payload["fit_metadata"]
    return estimator
