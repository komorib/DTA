from dataclasses import dataclass
import math

import torch
from torch import nn


LIGAND_COVALENT_EDGE = 0
LIGAND_SPATIAL_EDGE = 1
PROTEIN_SPATIAL_EDGE = 2
CROSS_EDGE = 3
EDGE_TYPE_COUNT = 4


@dataclass
class ScalarVectorState:
    scalar: torch.Tensor
    vector: torch.Tensor


@dataclass
class TypedInterfaceEdges:
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    distance: torch.Tensor
    direction: torch.Tensor
    bond_attr: torch.Tensor

    def to(self, device):
        return TypedInterfaceEdges(
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            distance=self.distance.to(device),
            direction=self.direction.to(device),
            bond_attr=self.bond_attr.to(device),
        )


class RadialBasis(nn.Module):
    def __init__(self, channels=16, cutoff=6.0):
        super().__init__()
        if int(channels) != channels or channels <= 0:
            raise ValueError("channels must be a positive integer")
        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        self.channels = int(channels)
        self.cutoff = float(cutoff)
        self.register_buffer(
            "centers",
            torch.linspace(0.0, self.cutoff, self.channels),
        )
        self.gamma = float(self.channels) / (
            self.cutoff * self.cutoff
        )

    def forward(self, distance):
        if distance.ndim != 1:
            raise ValueError("distance must have shape [E]")
        return torch.exp(
            -self.gamma
            * (
                distance[:, None]
                - self.centers[None, :].to(distance.dtype)
            )
            ** 2
        )


def cosine_cutoff(distance, cutoff=6.0):
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    scaled = distance / float(cutoff)
    envelope = 0.5 * (torch.cos(math.pi * scaled) + 1.0)
    return torch.where(
        (distance >= 0.0) & (distance < float(cutoff)),
        envelope,
        torch.zeros_like(distance),
    )


def scatter_softmax(values, index, dim_size=None):
    if values.ndim == 0:
        raise ValueError("values must have an edge dimension")
    if index.ndim != 1 or index.shape[0] != values.shape[0]:
        raise ValueError("index must have shape [E]")
    if index.dtype != torch.long:
        index = index.long()
    if dim_size is None:
        dim_size = (
            int(index.max().item()) + 1 if index.numel() else 0
        )
    dim_size = int(dim_size)
    if dim_size < 0:
        raise ValueError("dim_size must be non-negative")
    if index.numel() and (
        int(index.min()) < 0 or int(index.max()) >= dim_size
    ):
        raise ValueError("scatter index is outside dim_size")
    if values.shape[0] == 0:
        return values.clone()

    expanded_index = index.reshape(
        (-1,) + (1,) * (values.ndim - 1)
    ).expand_as(values)
    maxima = torch.full(
        (dim_size,) + tuple(values.shape[1:]),
        -torch.inf,
        dtype=values.dtype,
        device=values.device,
    )
    maxima.scatter_reduce_(
        0,
        expanded_index,
        values,
        reduce="amax",
        include_self=True,
    )
    exponentials = torch.exp(values - maxima[index])
    denominators = torch.zeros_like(maxima)
    denominators.index_add_(0, index, exponentials)
    return exponentials / denominators[index].clamp_min(
        torch.finfo(values.dtype).tiny
    )


def _validate_coordinates(coordinates, name):
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 3
        or not torch.is_floating_point(coordinates)
    ):
        raise ValueError(f"{name} must have floating shape [N,3]")
    if not torch.isfinite(coordinates).all():
        raise ValueError(f"{name} must contain finite coordinates")


def _radius_edges(coordinates, cutoff, chunk_size):
    count = coordinates.shape[0]
    if count == 0:
        return torch.empty(
            (2, 0), dtype=torch.long, device=coordinates.device
        )
    sources = []
    destinations = []
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        displacement = (
            coordinates[start:stop, None, :]
            - coordinates[None, :, :]
        )
        distance_squared = torch.sum(displacement * displacement, dim=-1)
        mask = (
            (distance_squared > 0.0)
            & (distance_squared <= float(cutoff) ** 2)
        )
        local_source, destination = torch.nonzero(
            mask, as_tuple=True
        )
        sources.append(local_source + start)
        destinations.append(destination)
    if not sources:
        return torch.empty(
            (2, 0), dtype=torch.long, device=coordinates.device
        )
    return torch.stack(
        [torch.cat(sources), torch.cat(destinations)], dim=0
    )


def _cross_radius_edges(
    ligand_coordinates,
    protein_coordinates,
    cutoff,
    chunk_size,
):
    ligand_count = ligand_coordinates.shape[0]
    protein_count = protein_coordinates.shape[0]
    if ligand_count == 0 or protein_count == 0:
        return torch.empty(
            (2, 0),
            dtype=torch.long,
            device=ligand_coordinates.device,
        )
    ligand_sources = []
    protein_destinations = []
    for start in range(0, ligand_count, chunk_size):
        stop = min(start + chunk_size, ligand_count)
        displacement = (
            ligand_coordinates[start:stop, None, :]
            - protein_coordinates[None, :, :]
        )
        distance_squared = torch.sum(displacement * displacement, dim=-1)
        mask = distance_squared <= float(cutoff) ** 2
        local_ligand, protein = torch.nonzero(
            mask, as_tuple=True
        )
        ligand_sources.append(local_ligand + start)
        protein_destinations.append(protein)
    ligand = torch.cat(ligand_sources)
    protein = torch.cat(protein_destinations) + ligand_count
    forward = torch.stack([ligand, protein], dim=0)
    reverse = torch.stack([protein, ligand], dim=0)
    return torch.cat([forward, reverse], dim=1)


def _bidirectional_unique_bonds(
    bond_index,
    bond_attr,
    ligand_count,
    coordinate_dtype,
):
    if bond_index.ndim != 2 or bond_index.shape[0] != 2:
        raise ValueError("ligand_bond_index must have shape [2,E]")
    bond_index = bond_index.to(dtype=torch.long)
    edge_count = bond_index.shape[1]
    if bond_attr is None:
        bond_attr = torch.empty(
            (edge_count, 0),
            dtype=coordinate_dtype,
            device=bond_index.device,
        )
    else:
        if bond_attr.ndim == 1:
            bond_attr = bond_attr[:, None]
        if bond_attr.ndim != 2 or bond_attr.shape[0] != edge_count:
            raise ValueError(
                "ligand_bond_attr must align with ligand bonds"
            )
        bond_attr = bond_attr.to(dtype=coordinate_dtype)
    if edge_count == 0:
        return bond_index, bond_attr
    if (
        int(bond_index.min()) < 0
        or int(bond_index.max()) >= ligand_count
    ):
        raise ValueError("ligand bond index is out of range")

    bidirectional = torch.cat(
        [bond_index, bond_index.flip(0)], dim=1
    )
    bidirectional_attr = torch.cat(
        [bond_attr, bond_attr], dim=0
    )
    not_self = bidirectional[0] != bidirectional[1]
    bidirectional = bidirectional[:, not_self]
    bidirectional_attr = bidirectional_attr[not_self]
    code = bidirectional[0] * ligand_count + bidirectional[1]
    order = torch.argsort(code, stable=True)
    code = code[order]
    bidirectional = bidirectional[:, order]
    bidirectional_attr = bidirectional_attr[order]
    keep = torch.ones(
        code.shape[0], dtype=torch.bool, device=code.device
    )
    keep[1:] = code[1:] != code[:-1]
    return bidirectional[:, keep], bidirectional_attr[keep]


def build_typed_interface_edges(
    ligand_coords,
    protein_coords,
    ligand_bond_index,
    ligand_bond_attr=None,
    ligand_spatial_cutoff=4.5,
    protein_spatial_cutoff=4.5,
    cross_cutoff=6.0,
    chunk_size=256,
    ligand_graph=None,
    protein_graph=None,
):
    _validate_coordinates(ligand_coords, "ligand_coords")
    _validate_coordinates(protein_coords, "protein_coords")
    if ligand_coords.device != protein_coords.device:
        raise ValueError(
            "ligand and protein coordinates must share a device"
        )
    if ligand_coords.dtype != protein_coords.dtype:
        raise ValueError(
            "ligand and protein coordinates must share a dtype"
        )
    for name, cutoff in (
        ("ligand_spatial_cutoff", ligand_spatial_cutoff),
        ("protein_spatial_cutoff", protein_spatial_cutoff),
        ("cross_cutoff", cross_cutoff),
    ):
        if cutoff <= 0:
            raise ValueError(f"{name} must be positive")
    if int(chunk_size) != chunk_size or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    chunk_size = int(chunk_size)
    ligand_count = ligand_coords.shape[0]
    if (ligand_graph is None) != (protein_graph is None):
        raise ValueError("ligand_graph and protein_graph must be provided together")
    node_graph = None
    if ligand_graph is not None:
        ligand_graph = ligand_graph.to(device=ligand_coords.device, dtype=torch.long)
        protein_graph = protein_graph.to(device=protein_coords.device, dtype=torch.long)
        if ligand_graph.shape != (ligand_count,) or protein_graph.shape != (protein_coords.shape[0],):
            raise ValueError("graph membership must align with node coordinates")
        node_graph = torch.cat([ligand_graph, protein_graph], dim=0)

    covalent, covalent_attr = _bidirectional_unique_bonds(
        ligand_bond_index.to(ligand_coords.device),
        (
            None
            if ligand_bond_attr is None
            else ligand_bond_attr.to(ligand_coords.device)
        ),
        ligand_count,
        ligand_coords.dtype,
    )
    ligand_spatial = _radius_edges(
        ligand_coords, ligand_spatial_cutoff, chunk_size
    )
    if covalent.shape[1] and ligand_spatial.shape[1]:
        covalent_code = (
            covalent[0] * ligand_count + covalent[1]
        )
        spatial_code = (
            ligand_spatial[0] * ligand_count
            + ligand_spatial[1]
        )
        ligand_spatial = ligand_spatial[
            :, ~torch.isin(spatial_code, covalent_code)
        ]

    protein_spatial = _radius_edges(
        protein_coords, protein_spatial_cutoff, chunk_size
    )
    protein_spatial = protein_spatial + ligand_count
    cross = _cross_radius_edges(
        ligand_coords,
        protein_coords,
        cross_cutoff,
        chunk_size,
    )

    edge_parts = [
        covalent,
        ligand_spatial,
        protein_spatial,
        cross,
    ]
    edge_index = torch.cat(edge_parts, dim=1)
    edge_type = torch.cat(
        [
            torch.full(
                (edges.shape[1],),
                edge_type_id,
                dtype=torch.long,
                device=ligand_coords.device,
            )
            for edge_type_id, edges in enumerate(edge_parts)
        ]
    )
    bond_dimension = covalent_attr.shape[1]
    bond_attr = torch.cat(
        [
            covalent_attr,
            *[
                torch.zeros(
                    (edges.shape[1], bond_dimension),
                    dtype=ligand_coords.dtype,
                    device=ligand_coords.device,
                )
                for edges in edge_parts[1:]
            ],
        ],
        dim=0,
    )

    if node_graph is not None:
        same_graph = node_graph[edge_index[0]] == node_graph[edge_index[1]]
        edge_index = edge_index[:, same_graph]
        edge_type = edge_type[same_graph]
        bond_attr = bond_attr[same_graph]

    coordinates = torch.cat(
        [ligand_coords, protein_coords], dim=0
    )
    displacement = (
        coordinates[edge_index[1]]
        - coordinates[edge_index[0]]
    )
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    direction = displacement / distance[:, None].clamp_min(1e-12)
    return TypedInterfaceEdges(
        edge_index=edge_index,
        edge_type=edge_type,
        distance=distance,
        direction=direction,
        bond_attr=bond_attr,
    )


def ligand_chiral_pseudoscalar(
    ligand_coords,
    ligand_bond_index,
    canonical_rank,
    chiral_tag,
):
    _validate_coordinates(ligand_coords, "ligand_coords")
    atom_count = ligand_coords.shape[0]
    if canonical_rank.shape != (atom_count,):
        raise ValueError("canonical_rank must have shape [L]")
    if chiral_tag.shape != (atom_count,):
        raise ValueError("chiral_tag must have shape [L]")
    if (
        ligand_bond_index.ndim != 2
        or ligand_bond_index.shape[0] != 2
    ):
        raise ValueError("ligand_bond_index must have shape [2,E]")
    if ligand_bond_index.numel() and (
        int(ligand_bond_index.min()) < 0
        or int(ligand_bond_index.max()) >= atom_count
    ):
        raise ValueError("ligand bond index is out of range")

    values = ligand_coords.new_zeros((atom_count, 1))
    for center in range(atom_count):
        tag = int(chiral_tag[center])
        if tag == 0:
            continue
        neighbors = ligand_bond_index[
            1, ligand_bond_index[0] == center
        ].unique()
        if neighbors.numel() < 3:
            continue
        ordered = sorted(
            (int(index) for index in neighbors),
            key=lambda index: (
                int(canonical_rank[index]),
                index,
            ),
        )
        vectors = (
            ligand_coords[
                torch.tensor(
                    ordered[:3],
                    dtype=torch.long,
                    device=ligand_coords.device,
                )
            ]
            - ligand_coords[center]
        )
        triple = torch.dot(
            vectors[0], torch.cross(vectors[1], vectors[2], dim=0)
        )
        normalization = torch.prod(
            torch.linalg.vector_norm(vectors, dim=-1)
        ).clamp_min(1e-12)
        values[center, 0] = triple / normalization
    return values


class TypedScalarVectorBlock(nn.Module):
    def __init__(
        self,
        scalar_dim=96,
        vector_channels=16,
        radial_channels=16,
        bond_feature_dim=6,
        cutoff=6.0,
    ):
        super().__init__()
        if scalar_dim <= 0 or vector_channels <= 0:
            raise ValueError(
                "scalar_dim and vector_channels must be positive"
            )
        if bond_feature_dim < 0:
            raise ValueError("bond_feature_dim must be non-negative")
        self.scalar_dim = int(scalar_dim)
        self.vector_channels = int(vector_channels)
        self.bond_feature_dim = int(bond_feature_dim)
        self.cutoff = float(cutoff)
        self.scalar_norm = nn.LayerNorm(self.scalar_dim)
        self.chirality_projection = nn.Linear(
            1, self.scalar_dim, bias=False
        )
        type_embedding_dim = min(16, self.scalar_dim)
        self.edge_type_embedding = nn.Embedding(
            EDGE_TYPE_COUNT, type_embedding_dim
        )
        self.radial_basis = RadialBasis(
            channels=radial_channels, cutoff=cutoff
        )
        message_input_dim = (
            self.scalar_dim
            + int(radial_channels)
            + self.bond_feature_dim
            + type_embedding_dim
        )
        message_output_dim = (
            self.scalar_dim + 2 * self.vector_channels
        )
        hidden_dim = max(self.scalar_dim, message_output_dim)
        self.type_message_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(message_input_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, message_output_dim),
                )
                for _ in range(EDGE_TYPE_COUNT)
            ]
        )
        self.scalar_update = nn.Sequential(
            nn.Linear(
                2 * self.scalar_dim + self.vector_channels,
                2 * self.scalar_dim,
            ),
            nn.SiLU(),
            nn.Linear(2 * self.scalar_dim, self.scalar_dim),
        )
        self.vector_mix = nn.Linear(
            self.vector_channels,
            self.vector_channels,
            bias=False,
        )
        self.vector_gate = nn.Linear(
            2 * self.scalar_dim, self.vector_channels
        )

    def _validate(self, state, edges, pseudoscalar):
        scalar = state.scalar
        vector = state.vector
        if scalar.ndim != 2 or scalar.shape[1] != self.scalar_dim:
            raise ValueError("scalar state has an invalid shape")
        if vector.shape != (
            scalar.shape[0],
            self.vector_channels,
            3,
        ):
            raise ValueError("vector state has an invalid shape")
        if edges.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        edge_count = edges.edge_index.shape[1]
        if (
            edges.edge_type.shape != (edge_count,)
            or edges.distance.shape != (edge_count,)
            or edges.direction.shape != (edge_count, 3)
            or edges.bond_attr.shape
            != (edge_count, self.bond_feature_dim)
        ):
            raise ValueError("typed edge fields are not aligned")
        if edges.edge_index.numel() and (
            int(edges.edge_index.min()) < 0
            or int(edges.edge_index.max()) >= scalar.shape[0]
        ):
            raise ValueError("edge index is out of range")
        if edges.edge_type.numel() and (
            int(edges.edge_type.min()) < 0
            or int(edges.edge_type.max()) >= EDGE_TYPE_COUNT
        ):
            raise ValueError("edge type is out of range")
        if pseudoscalar is None:
            return scalar.new_zeros((scalar.shape[0], 1))
        if pseudoscalar.shape != (scalar.shape[0], 1):
            raise ValueError("pseudoscalar must have shape [N,1]")
        return pseudoscalar.to(
            device=scalar.device, dtype=scalar.dtype
        )

    def forward(self, state, edges, pseudoscalar=None):
        pseudoscalar = self._validate(
            state, edges, pseudoscalar
        )
        scalar = state.scalar
        vector = state.vector
        scalar_pre = self.scalar_norm(scalar)
        scalar_pre = (
            scalar_pre
            + self.chirality_projection(pseudoscalar)
        )
        vector_scale = torch.sqrt(
            torch.mean(
                torch.sum(vector * vector, dim=-1),
                dim=-1,
                keepdim=True,
            )
            + 1e-8
        )[:, :, None]
        vector_pre = vector / vector_scale

        aggregate_scalar = torch.zeros_like(scalar)
        aggregate_vector = torch.zeros_like(vector)
        radial = self.radial_basis(edges.distance)
        envelope = cosine_cutoff(
            edges.distance, cutoff=self.cutoff
        )
        source = edges.edge_index[0]
        destination = edges.edge_index[1]
        for edge_type_id, message_mlp in enumerate(
            self.type_message_mlps
        ):
            mask = edges.edge_type == edge_type_id
            if not torch.any(mask):
                continue
            typed_source = source[mask]
            typed_destination = destination[mask]
            message_input = torch.cat(
                [
                    scalar_pre[typed_source],
                    radial[mask],
                    edges.bond_attr[mask].to(scalar.dtype),
                    self.edge_type_embedding(edges.edge_type[mask]),
                ],
                dim=-1,
            )
            coefficients = message_mlp(message_input)
            scalar_message = coefficients[
                :, : self.scalar_dim
            ]
            vector_gate = coefficients[
                :,
                self.scalar_dim : (
                    self.scalar_dim + self.vector_channels
                ),
            ]
            radial_gate = coefficients[
                :, self.scalar_dim + self.vector_channels :
            ]
            typed_envelope = envelope[mask, None]
            scalar_message = scalar_message * typed_envelope
            vector_message = (
                vector_gate[:, :, None]
                * vector_pre[typed_source]
                + radial_gate[:, :, None]
                * edges.direction[mask, None, :]
            ) * typed_envelope[:, :, None]
            scalar_message = scalar_message.to(
                dtype=aggregate_scalar.dtype
            )
            vector_message = vector_message.to(
                dtype=aggregate_vector.dtype
            )
            aggregate_scalar.index_add_(
                0, typed_destination, scalar_message
            )
            aggregate_vector.index_add_(
                0, typed_destination, vector_message
            )

        aggregate_vector_norm = torch.sqrt(
            torch.sum(
                aggregate_vector * aggregate_vector, dim=-1
            )
            + 1e-8
        )
        scalar_delta = self.scalar_update(
            torch.cat(
                [
                    scalar_pre,
                    aggregate_scalar,
                    aggregate_vector_norm,
                ],
                dim=-1,
            )
        )
        mixed_vector = self.vector_mix(
            aggregate_vector.transpose(1, 2)
        ).transpose(1, 2)
        vector_delta = mixed_vector * torch.sigmoid(
            self.vector_gate(
                torch.cat(
                    [scalar_pre, aggregate_scalar], dim=-1
                )
            )
        )[:, :, None]
        return ScalarVectorState(
            scalar=scalar + scalar_delta,
            vector=vector + vector_delta,
        )


class LigandCentricInterfacePool(nn.Module):
    def __init__(self, scalar_dim=96, latent_dim=128):
        super().__init__()
        if scalar_dim <= 0 or latent_dim <= 0:
            raise ValueError(
                "scalar_dim and latent_dim must be positive"
            )
        self.scalar_dim = int(scalar_dim)
        self.query = nn.Linear(self.scalar_dim, self.scalar_dim)
        self.key = nn.Linear(self.scalar_dim, self.scalar_dim)
        self.value = nn.Linear(self.scalar_dim, self.scalar_dim)
        self.context_projection = nn.Linear(
            self.scalar_dim, self.scalar_dim
        )
        self.ligand_norm = nn.LayerNorm(self.scalar_dim)
        self.pool_score = nn.Linear(self.scalar_dim, 1)
        self.latent_projection = nn.Sequential(
            nn.Linear(2 * self.scalar_dim, latent_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        ligand_scalar,
        protein_scalar,
        protein_to_ligand_edge_index,
    ):
        if (
            ligand_scalar.ndim != 2
            or ligand_scalar.shape[1] != self.scalar_dim
            or protein_scalar.ndim != 2
            or protein_scalar.shape[1] != self.scalar_dim
        ):
            raise ValueError(
                "ligand and protein scalar states have invalid shapes"
            )
        edges = protein_to_ligand_edge_index
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError(
                "protein_to_ligand_edge_index must have shape [2,E]"
            )
        if edges.numel() and (
            int(edges[0].min()) < 0
            or int(edges[0].max()) >= protein_scalar.shape[0]
            or int(edges[1].min()) < 0
            or int(edges[1].max()) >= ligand_scalar.shape[0]
        ):
            raise ValueError(
                "protein-to-ligand edge index is out of range"
            )

        context = torch.zeros_like(ligand_scalar)
        if edges.shape[1]:
            protein_index = edges[0].long()
            ligand_index = edges[1].long()
            score = torch.sum(
                self.query(ligand_scalar)[ligand_index]
                * self.key(protein_scalar)[protein_index],
                dim=-1,
            ) / math.sqrt(self.scalar_dim)
            attention = scatter_softmax(
                score,
                ligand_index,
                dim_size=ligand_scalar.shape[0],
            )
            context.index_add_(
                0,
                ligand_index,
                self.value(protein_scalar)[protein_index]
                * attention[:, None],
            )
        interface_ligand = self.ligand_norm(
            ligand_scalar + self.context_projection(context)
        )
        ligand_attention = torch.softmax(
            self.pool_score(interface_ligand).squeeze(-1),
            dim=0,
        )
        pooled = torch.sum(
            interface_ligand * ligand_attention[:, None], dim=0
        )
        mean_pooled = torch.mean(interface_ligand, dim=0)
        summary = torch.cat([pooled, mean_pooled], dim=-1)
        latent = self.latent_projection(summary).unsqueeze(0)
        return latent, interface_ligand
