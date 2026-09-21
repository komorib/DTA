import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, to_dense_batch


ATOM_ID = {
    "C": 1,
    "N": 2,
    "Br": 3,
    "O": 4,
    "S": 5,
    "Cl": 6,
    "F": 7,
    "I": 8,
    "As": 8,
    "P": 8,
    "K": 8,
    "Na": 8,
    "Ge": 8,
    "Se": 8,
    "Au": 8,
    "Pt": 8,
    "Rh": 8,
    "B": 8,
    "H": 8,
    "*": 8,
    "Fe": 8,
    "unknown": 8,
}

ATOM_ID_CHOICES = list(range(1, 9))
CHIRALITY_CHOICES = [
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_OTHER",
    "misc",
]
DEGREE_CHOICES = list(range(0, 11)) + ["misc"]
FORMAL_CHARGE_CHOICES = list(range(-5, 6)) + ["misc"]
NUM_H_CHOICES = list(range(0, 9)) + ["misc"]
HYBRIDIZATION_CHOICES = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"]
BOND_TYPE_CHOICES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"]
BOND_STEREO_CHOICES = [
    "STEREONONE",
    "STEREOZ",
    "STEREOE",
    "STEREOCIS",
    "STEREOTRANS",
    "STEREOANY",
]

LIGAND_2D_NODE_DIM = 55
LIGAND_2D_EDGE_DIM = 13


def _one_hot(value, choices):
    if value not in choices:
        value = choices[-1]
    return [1.0 if value == choice else 0.0 for choice in choices]


def _atom_symbol(atom):
    return Chem.GetPeriodicTable().GetElementSymbol(atom.GetAtomicNum())


def _atom_features(atom):
    atom_type = ATOM_ID.get(_atom_symbol(atom), ATOM_ID["unknown"])
    return np.asarray(
        _one_hot(atom_type, ATOM_ID_CHOICES)
        + _one_hot(str(atom.GetChiralTag()), CHIRALITY_CHOICES)
        + _one_hot(atom.GetTotalDegree(), DEGREE_CHOICES)
        + _one_hot(atom.GetFormalCharge(), FORMAL_CHARGE_CHOICES)
        + _one_hot(int(atom.GetTotalNumHs()), NUM_H_CHOICES)
        + _one_hot(str(atom.GetHybridization()), HYBRIDIZATION_CHOICES)
        + [1.0 if atom.GetIsAromatic() else 0.0]
        + [1.0 if atom.IsInRing() else 0.0],
        dtype=np.float32,
    )


def _bond_features(bond):
    return np.asarray(
        [0.0, 1.0 if bond.GetIsConjugated() else 0.0]
        + _one_hot(str(bond.GetBondType()), BOND_TYPE_CHOICES)
        + _one_hot(str(bond.GetStereo()), BOND_STEREO_CHOICES),
        dtype=np.float32,
    )


def _ligand_las_index(mol):
    adj = torch.from_numpy(Chem.GetAdjacencyMatrix(mol)).long()
    eye = torch.eye(adj.size(0), dtype=torch.long)
    one_hop = torch.where(adj + eye > 0, torch.ones_like(adj), torch.zeros_like(adj))
    two_hop = torch.where(one_hop @ one_hop > 0, torch.ones_like(adj), torch.zeros_like(adj))
    extended = two_hop - eye
    for ring in Chem.GetSymmSSSR(mol):
        ring_atoms = list(ring)
        for i in ring_atoms:
            for j in ring_atoms:
                if i != j:
                    extended[i, j] += 1
    row, col = torch.where(extended > 0)
    return torch.stack([row, col], dim=0).long()


def _ligand_2d_graph_from_mol(mol):
    if mol is None:
        raise ValueError("Cannot build 1D2D ligand graph from None mol")
    if mol.GetNumConformers() == 0:
        raise ValueError("Cannot build 1D2D ligand graph without 3D conformer")

    atom_features = [_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.as_tensor(np.stack(atom_features, axis=0), dtype=torch.float32)

    edges = []
    edge_features = []
    for bond in mol.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        feat = _bond_features(bond)
        edges.append((left, right))
        edge_features.append(feat)
        edges.append((right, left))
        edge_features.append(feat)
    if edges:
        edge_index = torch.tensor(np.asarray(edges).T, dtype=torch.long)
        edge_attr = torch.as_tensor(np.stack(edge_features, axis=0), dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, LIGAND_2D_EDGE_DIM), dtype=torch.float32)

    conf = mol.GetConformer()
    coords = torch.tensor(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=torch.float32,
    )
    return x, edge_index, edge_attr, coords, _ligand_las_index(mol)


def _as_sequence(sequence):
    if isinstance(sequence, str):
        return sequence
    return "".join(str(aa) for aa in list(sequence))


def _select_esm_embedding(chain_data, esm_key):
    if esm_key in chain_data:
        return chain_data[esm_key]
    for fallback_key in ("esm2_embedding", "esm1b_embedding", "esm_embedding", "seq_emb"):
        if fallback_key in chain_data:
            return chain_data[fallback_key]
    return None


def _protein_edges(ca_coords, chain_lengths, cutoff=8.0):
    ca_coords = np.asarray(ca_coords, dtype=np.float32).reshape(-1, 3)
    n_residue = len(ca_coords)
    edges = set()
    offset = 0
    for chain_length in chain_lengths:
        for index in range(max(int(chain_length) - 1, 0)):
            left = offset + index
            right = left + 1
            edges.add((left, right))
            edges.add((right, left))
        offset += int(chain_length)
    if n_residue > 1:
        distances = np.linalg.norm(ca_coords[:, None, :] - ca_coords[None, :, :], axis=-1)
        row, col = np.where((distances <= float(cutoff)) & (distances > 0))
        for left, right in zip(row.tolist(), col.tolist()):
            edges.add((left, right))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(np.asarray(sorted(edges)).T, dtype=torch.long)


def _protein_1d2d_inputs(
    protein_features,
    esm_key="esm2_embedding",
    prot_emb_dim=1280,
    allow_zero_embedding=False,
    protein_edge_cutoff=8.0,
    ligand_coords=None,
    max_protein_residues=128,
    name=None,
):
    if not isinstance(protein_features, dict):
        raise ValueError("protein_features must be a dict of chain_id -> features")

    emb_parts = []
    ca_parts = []
    chain_lengths = []
    for chain_id in sorted(protein_features.keys()):
        chain_data = protein_features[chain_id]
        if not isinstance(chain_data, dict):
            continue
        if "sequence" not in chain_data or "backbone_coords" not in chain_data:
            continue

        sequence = _as_sequence(chain_data["sequence"])
        coords = np.asarray(chain_data["backbone_coords"], dtype=np.float32)
        if len(sequence) == 0:
            continue
        if coords.shape != (len(sequence), 4, 3):
            raise ValueError(
                "Backbone shape mismatch for {} chain {}: sequence {}, coords {}".format(
                    name, chain_id, len(sequence), tuple(coords.shape)
                )
            )
        esm = _select_esm_embedding(chain_data, esm_key)
        if esm is None:
            if not allow_zero_embedding:
                raise ValueError("Missing ESM embedding for {} chain {}".format(name, chain_id))
            esm = np.zeros((len(sequence), int(prot_emb_dim)), dtype=np.float32)
        else:
            esm = np.asarray(esm, dtype=np.float32)
            if esm.ndim != 2 or esm.shape != (len(sequence), int(prot_emb_dim)):
                raise ValueError(
                    "ESM shape mismatch for {} chain {}: expected ({}, {}), got {}".format(
                        name, chain_id, len(sequence), int(prot_emb_dim), tuple(esm.shape)
                    )
                )
        emb_parts.append(esm)
        ca_parts.append(coords[:, 1, :])
        chain_lengths.append(len(sequence))

    if not emb_parts:
        raise ValueError("No usable protein chains found for 1D2D branch: {}".format(name))

    protein_x = torch.as_tensor(np.concatenate(emb_parts, axis=0), dtype=torch.float32)
    protein_coords = torch.as_tensor(np.concatenate(ca_parts, axis=0), dtype=torch.float32)
    if max_protein_residues is not None and int(max_protein_residues) > 0:
        max_protein_residues = int(max_protein_residues)
        if protein_coords.size(0) > max_protein_residues:
            if ligand_coords is None:
                keep = np.arange(max_protein_residues, dtype=np.int64)
            else:
                ligand_coords = np.asarray(ligand_coords, dtype=np.float32).reshape(-1, 3)
                ca_coords = protein_coords.numpy()
                distances = np.linalg.norm(
                    ca_coords[:, None, :] - ligand_coords[None, :, :],
                    axis=-1,
                ).min(axis=1)
                keep = np.argsort(distances, kind="mergesort")[:max_protein_residues]
                keep = np.sort(keep).astype(np.int64)
            protein_x = protein_x[torch.as_tensor(keep, dtype=torch.long)]
            protein_coords = protein_coords[torch.as_tensor(keep, dtype=torch.long)]
            chain_lengths = []
    protein_edge_index = _protein_edges(
        protein_coords.numpy(),
        chain_lengths,
        cutoff=protein_edge_cutoff,
    )
    return protein_x, protein_coords, protein_edge_index


class OneDTwoDData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "protein_edge_index":
            return int(self.protein_x.size(0))
        if key == "ligand_las_index":
            return int(self.x.size(0))
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in {"protein_edge_index", "ligand_las_index"}:
            return -1
        return super().__cat_dim__(key, value, *args, **kwargs)


def featurize_one_d_two_d(
    protein_features,
    mol=None,
    sdf_path=None,
    name=None,
    esm_key="esm2_embedding",
    prot_emb_dim=1280,
    allow_zero_embedding=False,
    protein_edge_cutoff=8.0,
    max_protein_residues=128,
):
    if mol is None:
        if sdf_path is None:
            raise ValueError("1D2D branch requires either an RDKit mol or an SDF path")
        mol = Chem.MolFromMolFile(str(sdf_path), removeHs=True)
    if mol is None:
        raise ValueError("Failed to parse ligand for 1D2D branch: {}".format(sdf_path))

    x, edge_index, edge_attr, ligand_coords, ligand_las_index = _ligand_2d_graph_from_mol(mol)
    protein_x, protein_coords, protein_edge_index = _protein_1d2d_inputs(
        protein_features,
        esm_key=esm_key,
        prot_emb_dim=prot_emb_dim,
        allow_zero_embedding=allow_zero_embedding,
        protein_edge_cutoff=protein_edge_cutoff,
        ligand_coords=ligand_coords.numpy(),
        max_protein_residues=max_protein_residues,
        name=name,
    )
    return OneDTwoDData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        ligand_coords=ligand_coords,
        ligand_las_index=ligand_las_index,
        ligand_len=torch.tensor([x.size(0)], dtype=torch.long),
        protein_x=protein_x,
        protein_coords=protein_coords,
        protein_edge_index=protein_edge_index,
        protein_len=torch.tensor([protein_x.size(0)], dtype=torch.long),
        name=name,
    )


class EdgeGINConv(MessagePassing):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, edge_attr):
        if edge_attr.numel() == 0:
            agg = torch.zeros_like(x)
        else:
            agg = self.propagate(edge_index, x=x, edge_attr=self.edge_encoder(edge_attr.float()))
        return self.mlp((1.0 + self.eps) * x + agg)

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)


class Ligand2DGNN(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, num_layers=3, dropout=0.25):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.node_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [EdgeGINConv(hidden_dim, edge_dim) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = float(dropout)

    def forward(self, data):
        h = self.node_proj(data.x.float())
        for i, conv in enumerate(self.convs):
            h_next = conv(h, data.edge_index, data.edge_attr)
            h_next = self.norms[i](h_next)
            if i != len(self.convs) - 1:
                h_next = F.relu(h_next)
            h_next = F.dropout(h_next, p=self.dropout, training=self.training)
            h = h_next
        return h


class Protein2DGNN(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, dropout=0.25):
        super().__init__()
        self.convs = nn.ModuleList(
            [torch_geometric.nn.GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = float(dropout)

    def forward(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.convs):
            h_next = conv(h, edge_index)
            h_next = self.norms[i](h_next)
            if i != len(self.convs) - 1:
                h_next = F.relu(h_next)
            h_next = F.dropout(h_next, p=self.dropout, training=self.training)
            h = h_next
        return h


class TriangleProteinToCompound(nn.Module):
    def __init__(self, embedding_channels, c):
        super().__init__()
        self.layernorm = nn.LayerNorm(embedding_channels)
        self.layernorm_c = nn.LayerNorm(c)
        self.gate_linear1 = nn.Linear(embedding_channels, c)
        self.gate_linear2 = nn.Linear(embedding_channels, c)
        self.linear1 = nn.Linear(embedding_channels, c)
        self.linear2 = nn.Linear(embedding_channels, c)
        self.ending_gate_linear = nn.Linear(embedding_channels, embedding_channels)
        self.linear_after_sum = nn.Linear(c, embedding_channels)

    def forward(self, z, protein_pair, compound_pair, z_mask):
        z = self.layernorm(z)
        protein_pair = self.layernorm(protein_pair)
        compound_pair = self.layernorm(compound_pair)
        ab1 = self.gate_linear1(z).sigmoid() * self.linear1(z) * z_mask
        ab2 = self.gate_linear2(z).sigmoid() * self.linear2(z) * z_mask
        protein_pair = self.gate_linear2(protein_pair).sigmoid() * self.linear2(protein_pair)
        compound_pair = self.gate_linear1(compound_pair).sigmoid() * self.linear1(compound_pair)
        gate = self.ending_gate_linear(z).sigmoid()
        block1 = torch.einsum("bikc,bkjc->bijc", protein_pair, ab1)
        block2 = torch.einsum("bikc,bjkc->bijc", ab2, compound_pair)
        return gate * self.linear_after_sum(self.layernorm_c(block1 + block2)) * z_mask


class TriangleSelfAttentionRowWise(nn.Module):
    def __init__(self, embedding_channels, c=16, num_attention_heads=4):
        super().__init__()
        self.num_attention_heads = int(num_attention_heads)
        self.attention_head_size = int(c)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.layernorm = nn.LayerNorm(embedding_channels)
        self.linear_q = nn.Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_k = nn.Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_v = nn.Linear(embedding_channels, self.all_head_size, bias=False)
        self.g = nn.Linear(embedding_channels, self.all_head_size)
        self.final_linear = nn.Linear(self.all_head_size, embedding_channels)

    def _reshape_heads(self, x):
        return x.view(*x.size()[:-1], self.num_attention_heads, self.attention_head_size)

    def forward(self, z, z_mask):
        z = self.layernorm(z)
        batch_size, protein_len = z.shape[:2]
        attention_mask = 1e9 * (z_mask.float().view(batch_size, protein_len, 1, 1, -1) - 1.0)
        q = self._reshape_heads(self.linear_q(z))
        k = self._reshape_heads(self.linear_k(z))
        v = self._reshape_heads(self.linear_v(z))
        logits = torch.einsum("biqhc,bikhc->bihqk", q, k) + attention_mask
        weights = F.softmax(logits, dim=-1)
        output = torch.einsum("bihqk,bikhc->biqhc", weights, v)
        output = self._reshape_heads(self.g(z)).sigmoid() * output
        output = output.reshape(*output.size()[:-2], self.all_head_size)
        return self.final_linear(output) * z_mask.unsqueeze(-1)


class Transition(nn.Module):
    def __init__(self, embedding_channels, expansion=4):
        super().__init__()
        self.layernorm = nn.LayerNorm(embedding_channels)
        self.linear1 = nn.Linear(embedding_channels, expansion * embedding_channels)
        self.linear2 = nn.Linear(expansion * embedding_channels, embedding_channels)

    def forward(self, z):
        return self.linear2(F.relu(self.linear1(self.layernorm(z))))


def _dense_by_lengths(values, lengths):
    lengths = lengths.long().view(-1)
    batch_size = int(lengths.numel())
    max_len = int(lengths.max().item())
    out = values.new_zeros((batch_size, max_len, values.size(-1)))
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=values.device)
    start = 0
    for i, length in enumerate(lengths.tolist()):
        end = start + int(length)
        out[i, :length] = values[start:end]
        mask[i, :length] = True
        start = end
    return out, mask


def _distance_one_hot(coords, bin_size=2.0, bin_min=-1.0, bin_max=30.0):
    distances = torch.cdist(coords, coords, compute_mode="donot_use_mm_for_euclid_dist")
    distances = distances.clamp(max=bin_max)
    indices = torch.div(distances - bin_min, bin_size, rounding_mode="floor").long()
    return F.one_hot(indices, num_classes=16).float()


def _mol_pair_distribution(coords, las_mask):
    distances = torch.cdist(coords, coords, p=2)
    distances = distances.clone()
    distances[las_mask <= 0] = 15.0
    distances = distances.clamp(max=15.0)
    indices = torch.div(distances + 0.5, 1.0, rounding_mode="floor").long()
    return F.one_hot(indices, num_classes=16).float()


class OneDTwoDBranch(nn.Module):
    def __init__(
        self,
        prot_emb_dim=1280,
        hidden_dim=64,
        output_dim=128,
        gnn_layers=3,
        triangle_layers=2,
        dropout=0.25,
        use_protein_2d=True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_protein_2d = bool(use_protein_2d)
        self.ligand_gnn = Ligand2DGNN(
            LIGAND_2D_NODE_DIM,
            LIGAND_2D_EDGE_DIM,
            self.hidden_dim,
            num_layers=gnn_layers,
            dropout=dropout,
        )
        self.protein_1d = nn.Sequential(
            nn.LayerNorm(int(prot_emb_dim)),
            nn.Linear(int(prot_emb_dim), self.hidden_dim),
            nn.LeakyReLU(0.1),
        )
        if self.use_protein_2d:
            self.protein_2d = Protein2DGNN(
                self.hidden_dim,
                num_layers=max(1, int(gnn_layers) - 1),
                dropout=dropout,
            )
        pair_mid_dim = max(16, self.hidden_dim // 2)
        attention_head_dim = max(8, self.hidden_dim // 4)
        self.protein_pair_proj = nn.Linear(16, self.hidden_dim)
        self.ligand_pair_proj = nn.Linear(16, self.hidden_dim)
        self.triangle_layers = nn.ModuleList(
            [TriangleProteinToCompound(self.hidden_dim, pair_mid_dim) for _ in range(int(triangle_layers))]
        )
        self.row_attention_layers = nn.ModuleList(
            [
                TriangleSelfAttentionRowWise(
                    self.hidden_dim,
                    c=attention_head_dim,
                    num_attention_heads=4,
                )
                for _ in range(int(triangle_layers))
            ]
        )
        self.transitions = nn.ModuleList(
            [Transition(self.hidden_dim) for _ in range(int(triangle_layers))]
        )
        self.dropout = nn.Dropout(float(dropout))
        self.protein_norm = nn.LayerNorm(self.hidden_dim)
        self.ligand_norm = nn.LayerNorm(self.hidden_dim)
        self.pairwise_linear = nn.Linear(self.hidden_dim, 1)
        self.protein_final = nn.Linear(self.hidden_dim, int(output_dim))
        self.ligand_final = nn.Linear(self.hidden_dim, int(output_dim))
        self.out = nn.Sequential(
            nn.LayerNorm(int(output_dim) * 3),
            nn.Linear(int(output_dim) * 3, int(output_dim)),
            nn.LeakyReLU(0.1),
        )

    def _ligand_las_dense(self, data, ligand_mask):
        batch_size, max_atoms = ligand_mask.shape
        las = torch.zeros(
            (batch_size, max_atoms, max_atoms),
            dtype=torch.float32,
            device=ligand_mask.device,
        )
        if not hasattr(data, "ligand_las_index") or data.ligand_las_index.numel() == 0:
            return las
        src, dst = data.ligand_las_index
        graph_ids = data.batch[src]
        local_src = src - data.ptr[graph_ids]
        local_dst = dst - data.ptr[graph_ids]
        las[graph_ids, local_src, local_dst] = 1.0
        return las

    def forward(self, data):
        ligand_h = self.ligand_gnn(data)
        protein_h_1d = self.protein_1d(data.protein_x.float())
        if self.use_protein_2d:
            protein_h_2d = self.protein_2d(protein_h_1d, data.protein_edge_index)
            protein_h = 0.5 * protein_h_2d + 0.5 * protein_h_1d
        else:
            protein_h = protein_h_1d

        ligand_h, ligand_mask = to_dense_batch(ligand_h, data.batch)
        ligand_coords, _ = to_dense_batch(data.ligand_coords.float(), data.batch)
        protein_h, protein_mask = _dense_by_lengths(protein_h, data.protein_len)
        protein_coords, _ = _dense_by_lengths(data.protein_coords.float(), data.protein_len)

        z_mask = protein_mask.unsqueeze(2) & ligand_mask.unsqueeze(1)
        z_mask_f = z_mask.float().unsqueeze(-1)

        protein_pair = self.protein_pair_proj(
            _distance_one_hot(protein_coords, bin_size=2.0, bin_min=-1.0, bin_max=30.0)
        )
        ligand_pair = self.ligand_pair_proj(
            _mol_pair_distribution(ligand_coords, self._ligand_las_dense(data, ligand_mask))
        )
        z = self.protein_norm(protein_h).unsqueeze(2) * self.ligand_norm(ligand_h).unsqueeze(1)
        z = z * z_mask_f

        for triangle, row_attention, transition in zip(
            self.triangle_layers,
            self.row_attention_layers,
            self.transitions,
        ):
            z = z + self.dropout(triangle(z, protein_pair, ligand_pair, z_mask_f))
            z = z + self.dropout(row_attention(z, z_mask))
            z = transition(z) * z_mask_f

        pairwise = torch.sigmoid(self.pairwise_linear(z)).squeeze(-1) * z_mask.float()
        protein_weights = pairwise.sum(dim=2)
        ligand_weights = pairwise.sum(dim=1)
        protein_weights = protein_weights / (protein_weights.sum(dim=1, keepdim=True) + 1e-6)
        ligand_weights = ligand_weights / (ligand_weights.sum(dim=1, keepdim=True) + 1e-6)

        protein_summary = (
            self.protein_final(protein_h) * protein_weights.unsqueeze(-1)
        ).sum(dim=1)
        ligand_summary = (
            self.ligand_final(ligand_h) * ligand_weights.unsqueeze(-1)
        ).sum(dim=1)
        return self.out(
            torch.cat(
                [protein_summary, ligand_summary, protein_summary * ligand_summary],
                dim=-1,
            )
        )
