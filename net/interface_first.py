from copy import deepcopy
from dataclasses import dataclass
import math
from types import SimpleNamespace

import torch
from torch import nn

from net.GIN import GNN_node
from net.interface_blocks import (
    CROSS_EDGE,
    LigandCentricInterfacePool,
    ScalarVectorState,
    TypedScalarVectorBlock,
    build_typed_interface_edges,
    ligand_chiral_pseudoscalar,
)


LIGAND_ATOM_FEATURE_DIM = 55
LIGAND_BOND_FEATURE_DIM = 13
PROTEIN_ELEMENT_CLASSES = 4
PROTEIN_ATOM_NAME_CLASSES = 38
PROTEIN_RESIDUE_CLASSES = 20


@dataclass
class LocalInterfaceOutput:
    ligand_scalar: torch.Tensor
    protein_scalar: torch.Tensor
    interface_ligand: torch.Tensor
    interface_latent: torch.Tensor
    interface_score: torch.Tensor
    projection: torch.Tensor


@dataclass
class LocalInterfaceStaticState:
    ligand_scalar: torch.Tensor
    protein_scalar: torch.Tensor


def _require_tensor(item, name, ndim=None):
    value = item.get(name)
    if not torch.is_tensor(value):
        raise ValueError(f"{name} must be a tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    return value


def _validate_class_tokens(tokens, class_count, name):
    if tokens.ndim != 1:
        raise ValueError(f"{name} must have shape [N]")
    if tokens.numel() and (
        int(tokens.min()) < 0
        or int(tokens.max()) >= int(class_count)
    ):
        raise ValueError(f"{name} contains an out-of-range class")
    return tokens.long()


class ProteinAtomChemicalEncoder(nn.Module):
    def __init__(self, scalar_dim=96):
        super().__init__()
        self.scalar_dim = int(scalar_dim)
        if self.scalar_dim <= 0:
            raise ValueError("scalar_dim must be positive")
        self.element_embedding = nn.Embedding(
            PROTEIN_ELEMENT_CLASSES, self.scalar_dim
        )
        self.atom_name_embedding = nn.Embedding(
            PROTEIN_ATOM_NAME_CLASSES, self.scalar_dim
        )
        self.residue_embedding = nn.Embedding(
            PROTEIN_RESIDUE_CLASSES, self.scalar_dim
        )
        self.backbone_embedding = nn.Embedding(2, self.scalar_dim)
        self.output_norm = nn.LayerNorm(self.scalar_dim)

    def forward(
        self,
        element,
        atom_name_class,
        residue_type,
        is_backbone,
    ):
        element = _validate_class_tokens(
            element, PROTEIN_ELEMENT_CLASSES, "protein_element"
        )
        atom_name_class = _validate_class_tokens(
            atom_name_class,
            PROTEIN_ATOM_NAME_CLASSES,
            "protein_atom_name_class",
        )
        residue_type = _validate_class_tokens(
            residue_type,
            PROTEIN_RESIDUE_CLASSES,
            "protein_residue_type",
        )
        is_backbone = _validate_class_tokens(
            is_backbone.long(), 2, "protein_is_backbone"
        )
        atom_count = element.shape[0]
        if (
            atom_name_class.shape[0] != atom_count
            or residue_type.shape[0] != atom_count
            or is_backbone.shape[0] != atom_count
        ):
            raise ValueError(
                "protein chemical features must align by atom"
            )
        state = (
            self.element_embedding(element)
            + self.atom_name_embedding(atom_name_class)
            + self.residue_embedding(residue_type)
            + self.backbone_embedding(is_backbone)
        )
        return self.output_norm(state)


class ResidueToAtomAdapter(nn.Module):
    MODES = {"gated_residual", "concat", "no_esm"}

    def __init__(self, esm_dim=640, scalar_dim=96):
        super().__init__()
        self.esm_dim = int(esm_dim)
        self.scalar_dim = int(scalar_dim)
        self.local_projection = nn.Sequential(
            nn.LayerNorm(self.esm_dim),
            nn.Linear(self.esm_dim, 128),
            nn.SiLU(),
            nn.Linear(128, self.scalar_dim),
        )
        self.gate = nn.Linear(2 * self.scalar_dim, self.scalar_dim)
        self.concat_projection = nn.Linear(
            2 * self.scalar_dim, self.scalar_dim
        )

    def forward(
        self,
        atom_chemical_state,
        residue_esm,
        parent_residue,
        mode="gated_residual",
    ):
        if mode not in self.MODES:
            raise ValueError(
                "residue-to-atom mode must be one of "
                + ", ".join(sorted(self.MODES))
            )
        if (
            atom_chemical_state.ndim != 2
            or atom_chemical_state.shape[1] != self.scalar_dim
        ):
            raise ValueError(
                "atom_chemical_state has an invalid shape"
            )
        if (
            residue_esm.ndim != 2
            or residue_esm.shape[1] != self.esm_dim
        ):
            raise ValueError("residue_esm has an invalid shape")
        if parent_residue.shape != (
            atom_chemical_state.shape[0],
        ):
            raise ValueError(
                "parent_residue must align with protein atoms"
            )
        parent_residue = parent_residue.long()
        if parent_residue.numel() and (
            int(parent_residue.min()) < 0
            or int(parent_residue.max()) >= residue_esm.shape[0]
        ):
            raise ValueError("parent_residue is out of range")

        residue_state = self.local_projection(residue_esm)
        atom_residue_state = residue_state[parent_residue]
        if mode == "no_esm":
            return atom_chemical_state
        combined = torch.cat(
            [atom_chemical_state, atom_residue_state], dim=-1
        )
        if mode == "concat":
            return self.concat_projection(combined)
        gate = torch.sigmoid(self.gate(combined))
        return atom_chemical_state + gate * atom_residue_state


class CompleteLocalInterfacePath(nn.Module):
    def __init__(
        self,
        ligand_gnn_depth=3,
        ligand_dropout=0.1,
        scalar_dim=96,
        vector_channels=16,
        interface_dim=128,
        projection_dim=64,
        residue_to_atom_mode="gated_residual",
    ):
        super().__init__()
        if residue_to_atom_mode not in ResidueToAtomAdapter.MODES:
            raise ValueError("invalid residue_to_atom_mode")
        self.scalar_dim = int(scalar_dim)
        self.vector_channels = int(vector_channels)
        self.residue_to_atom_mode = residue_to_atom_mode
        self.ligand_gnn = GNN_node(
            LIGAND_ATOM_FEATURE_DIM,
            int(ligand_gnn_depth),
            self.scalar_dim,
            edge_attr_option=True,
            drop_ratio=float(ligand_dropout),
        )
        self.protein_chemical_encoder = ProteinAtomChemicalEncoder(
            scalar_dim=self.scalar_dim
        )
        self.esm_atom_adapter = ResidueToAtomAdapter(
            esm_dim=640, scalar_dim=self.scalar_dim
        )
        self.interface_blocks = nn.ModuleList(
            [
                TypedScalarVectorBlock(
                    scalar_dim=self.scalar_dim,
                    vector_channels=self.vector_channels,
                    radial_channels=16,
                    bond_feature_dim=LIGAND_BOND_FEATURE_DIM,
                    cutoff=6.0,
                )
                for _ in range(4)
            ]
        )
        self.interface_pool = LigandCentricInterfacePool(
            scalar_dim=self.scalar_dim,
            latent_dim=interface_dim,
        )
        self.projection_head = nn.Sequential(
            nn.Linear(interface_dim, interface_dim),
            nn.SiLU(),
            nn.Linear(interface_dim, projection_dim),
        )
        self.score_head = nn.Linear(interface_dim, 1)

    def _ligand_states(self, item):
        atom_features = _require_tensor(
            item, "ligand_atom_features", ndim=2
        )
        if atom_features.shape[1] != LIGAND_ATOM_FEATURE_DIM:
            raise ValueError(
                "ligand_atom_features must have shape [L,55]"
            )
        edge_index = _require_tensor(
            item, "ligand_edge_index", ndim=2
        ).long()
        edge_attr = _require_tensor(
            item, "ligand_edge_attr", ndim=2
        )
        if (
            edge_index.shape[0] != 2
            or edge_attr.shape
            != (edge_index.shape[1], LIGAND_BOND_FEATURE_DIM)
        ):
            raise ValueError(
                "ligand edges and 13-D bond features are not aligned"
            )
        graph = SimpleNamespace(
            x=atom_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=torch.zeros(
                atom_features.shape[0],
                dtype=torch.long,
                device=atom_features.device,
            ),
        )
        return self.ligand_gnn(graph)

    def encode_static(self, item):
        native_ligand_coords = _require_tensor(
            item, "ligand_coords", ndim=2
        )
        if native_ligand_coords.shape[1] != 3:
            raise ValueError(
                "ligand_coords must have shape [L,3]"
            )
        protein_coords = _require_tensor(
            item, "protein_coords", ndim=2
        )
        residue_esm = _require_tensor(
            item, "residue_esm", ndim=2
        )
        if protein_coords.shape[1] != 3:
            raise ValueError("protein_coords must have shape [P,3]")
        if (
            native_ligand_coords.device != protein_coords.device
            or native_ligand_coords.dtype != protein_coords.dtype
        ):
            raise ValueError(
                "ligand and protein coordinates must share device/dtype"
            )
        ligand_scalar = self._ligand_states(item)
        if (
            ligand_scalar.device != native_ligand_coords.device
            or ligand_scalar.shape[0] != native_ligand_coords.shape[0]
        ):
            raise ValueError(
                "ligand graph states do not align with coordinates"
            )

        protein_chemical = self.protein_chemical_encoder(
            _require_tensor(item, "protein_element"),
            _require_tensor(item, "protein_atom_name_class"),
            _require_tensor(item, "protein_residue_type"),
            _require_tensor(item, "protein_is_backbone"),
        )
        protein_scalar = self.esm_atom_adapter(
            protein_chemical,
            residue_esm,
            _require_tensor(item, "protein_parent_residue"),
            mode=self.residue_to_atom_mode,
        )
        if protein_scalar.shape[0] != protein_coords.shape[0]:
            raise ValueError(
                "protein atom states do not align with coordinates"
            )

        return LocalInterfaceStaticState(
            ligand_scalar=ligand_scalar,
            protein_scalar=protein_scalar,
        )

    def forward_geometry(
        self,
        item,
        static_state,
        ligand_coords=None,
    ):
        if not isinstance(
            static_state, LocalInterfaceStaticState
        ):
            raise ValueError(
                "static_state must come from encode_static"
            )
        native_ligand_coords = _require_tensor(
            item, "ligand_coords", ndim=2
        )
        if ligand_coords is None:
            ligand_coords = native_ligand_coords
        if (
            ligand_coords.shape != native_ligand_coords.shape
            or ligand_coords.shape[1] != 3
        ):
            raise ValueError(
                "ligand coordinate override must have shape [L,3]"
            )
        protein_coords = _require_tensor(
            item, "protein_coords", ndim=2
        )
        ligand_scalar = static_state.ligand_scalar
        protein_scalar = static_state.protein_scalar
        if (
            ligand_scalar.shape[0] != ligand_coords.shape[0]
            or protein_scalar.shape[0] != protein_coords.shape[0]
            or ligand_scalar.device != ligand_coords.device
            or protein_scalar.device != protein_coords.device
        ):
            raise ValueError(
                "static states do not align with pose geometry"
            )

        ligand_edge_index = _require_tensor(
            item, "ligand_edge_index", ndim=2
        ).long()
        ligand_edge_attr = _require_tensor(
            item, "ligand_edge_attr", ndim=2
        )
        edges = build_typed_interface_edges(
            ligand_coords,
            protein_coords,
            ligand_edge_index,
            ligand_edge_attr,
            ligand_spatial_cutoff=4.5,
            protein_spatial_cutoff=4.5,
            cross_cutoff=6.0,
        )
        scalar = torch.cat(
            [ligand_scalar, protein_scalar], dim=0
        )
        state = ScalarVectorState(
            scalar=scalar,
            vector=scalar.new_zeros(
                (
                    scalar.shape[0],
                    self.vector_channels,
                    3,
                )
            ),
        )
        ligand_pseudoscalar = ligand_chiral_pseudoscalar(
            ligand_coords,
            ligand_edge_index,
            _require_tensor(item, "ligand_canonical_rank"),
            _require_tensor(item, "ligand_chiral_tag"),
        )
        pseudoscalar = scalar.new_zeros((scalar.shape[0], 1))
        pseudoscalar[: ligand_scalar.shape[0]] = (
            ligand_pseudoscalar
        )
        for block in self.interface_blocks:
            state = block(state, edges, pseudoscalar)

        ligand_count = ligand_scalar.shape[0]
        ligand_scalar = state.scalar[:ligand_count]
        protein_scalar = state.scalar[ligand_count:]
        protein_to_ligand = (
            (edges.edge_type == CROSS_EDGE)
            & (edges.edge_index[0] >= ligand_count)
            & (edges.edge_index[1] < ligand_count)
        )
        cross_index = torch.stack(
            [
                edges.edge_index[0, protein_to_ligand]
                - ligand_count,
                edges.edge_index[1, protein_to_ligand],
            ],
            dim=0,
        )
        interface_latent, interface_ligand = (
            self.interface_pool(
                ligand_scalar,
                protein_scalar,
                cross_index,
            )
        )
        return LocalInterfaceOutput(
            ligand_scalar=ligand_scalar,
            protein_scalar=protein_scalar,
            interface_ligand=interface_ligand,
            interface_latent=interface_latent,
            interface_score=self.score_head(interface_latent),
            projection=self.projection_head(interface_latent),
        )

    def forward(self, item, ligand_coords=None):
        static_state = self.encode_static(item)
        return self.forward_geometry(
            item,
            static_state,
            ligand_coords=ligand_coords,
        )


class ProteinGlobalEncoder(nn.Module):
    def __init__(
        self,
        esm_dim=640,
        scalar_dim=128,
        vector_channels=16,
        cutoff=8.0,
    ):
        super().__init__()
        self.esm_dim = int(esm_dim)
        self.scalar_dim = int(scalar_dim)
        self.vector_channels = int(vector_channels)
        self.cutoff = float(cutoff)
        self.esm_projection = nn.Sequential(
            nn.LayerNorm(self.esm_dim),
            nn.Linear(self.esm_dim, self.scalar_dim),
        )
        self.blocks = nn.ModuleList(
            [
                TypedScalarVectorBlock(
                    scalar_dim=self.scalar_dim,
                    vector_channels=self.vector_channels,
                    radial_channels=16,
                    bond_feature_dim=0,
                    cutoff=self.cutoff,
                )
                for _ in range(2)
            ]
        )

    def forward(self, residue_esm, residue_ca_coords):
        if (
            residue_esm.ndim != 2
            or residue_esm.shape[1] != self.esm_dim
        ):
            raise ValueError("residue_esm must have shape [R,640]")
        if residue_ca_coords.shape != (
            residue_esm.shape[0],
            3,
        ):
            raise ValueError(
                "residue_ca_coords must align with residue_esm"
            )
        scalar = self.esm_projection(residue_esm)
        empty_coords = residue_ca_coords.new_empty((0, 3))
        empty_bonds = torch.empty(
            (2, 0),
            dtype=torch.long,
            device=residue_ca_coords.device,
        )
        edges = build_typed_interface_edges(
            empty_coords,
            residue_ca_coords,
            empty_bonds,
            None,
            ligand_spatial_cutoff=self.cutoff,
            protein_spatial_cutoff=self.cutoff,
            cross_cutoff=self.cutoff,
        )
        state = ScalarVectorState(
            scalar=scalar,
            vector=scalar.new_zeros(
                (
                    scalar.shape[0],
                    self.vector_channels,
                    3,
                )
            ),
        )
        for block in self.blocks:
            state = block(state, edges)
        return state.scalar


class ConditionedAttentionPool(nn.Module):
    def __init__(self, input_dim, condition_dim, output_dim):
        super().__init__()
        self.input_dim = int(input_dim)
        self.condition_dim = int(condition_dim)
        self.output_dim = int(output_dim)
        if min(
            self.input_dim,
            self.condition_dim,
            self.output_dim,
        ) <= 0:
            raise ValueError("pool dimensions must be positive")
        self.query = nn.Linear(
            self.condition_dim, self.input_dim, bias=False
        )
        self.key = nn.Linear(
            self.input_dim, self.input_dim, bias=False
        )
        self.value = nn.Linear(
            self.input_dim, self.output_dim
        )
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(self, states, condition, mask=None):
        if states.ndim != 2 or states.shape[1] != self.input_dim:
            raise ValueError("states have an invalid shape")
        if condition.shape != (1, self.condition_dim):
            raise ValueError("condition must have shape [1,C]")
        if mask is None:
            mask = torch.ones(
                states.shape[0],
                dtype=torch.bool,
                device=states.device,
            )
        if mask.shape != (states.shape[0],):
            raise ValueError("mask must have shape [N]")
        mask = mask.to(device=states.device, dtype=torch.bool)
        if not torch.any(mask):
            raise ValueError("conditioned pooling needs at least one state")
        valid_states = states[mask]
        query = self.query(condition).squeeze(0)
        score = torch.sum(
            self.key(valid_states) * query[None, :],
            dim=-1,
        ) / math.sqrt(self.input_dim)
        attention = torch.softmax(score, dim=0)
        pooled = torch.sum(
            self.value(valid_states) * attention[:, None],
            dim=0,
        )
        return self.output_norm(pooled).unsqueeze(0)


class InterfaceFirstGerNADTA(nn.Module):
    def __init__(
        self,
        ligand_gnn_depth=3,
        ligand_dropout=0.1,
        affinity_dropout=0.2,
        residue_to_atom_mode="gated_residual",
    ):
        super().__init__()
        self.local_path = CompleteLocalInterfacePath(
            ligand_gnn_depth=ligand_gnn_depth,
            ligand_dropout=ligand_dropout,
            residue_to_atom_mode=residue_to_atom_mode,
        )
        self.protein_global = ProteinGlobalEncoder()
        self.protein_conditioned_pool = ConditionedAttentionPool(
            input_dim=128,
            condition_dim=128,
            output_dim=128,
        )
        self.ligand_conditioned_pool = ConditionedAttentionPool(
            input_dim=96,
            condition_dim=128,
            output_dim=96,
        )
        self.affinity_head = nn.Sequential(
            nn.Linear(352, 128),
            nn.LeakyReLU(),
            nn.Dropout(float(affinity_dropout)),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        item,
        ligand_coords=None,
        return_features=False,
        return_multiscale_features=False,
    ):
        if return_features and return_multiscale_features:
            raise ValueError(
                "return_features and return_multiscale_features are mutually exclusive"
            )
        local = self.local_path(
            item, ligand_coords=ligand_coords
        )
        protein_states = self.protein_global(
            _require_tensor(item, "residue_esm", ndim=2),
            _require_tensor(item, "residue_ca_coords", ndim=2),
        )
        protein_summary = self.protein_conditioned_pool(
            protein_states,
            local.interface_latent,
            mask=item.get("residue_mask"),
        )
        ligand_summary = self.ligand_conditioned_pool(
            local.ligand_scalar,
            local.interface_latent,
            mask=item.get("ligand_mask"),
        )
        affinity_input = torch.cat(
            [
                protein_summary,
                ligand_summary,
                local.interface_latent,
            ],
            dim=-1,
        )
        captured_head = []
        handle = None
        feature_capture_requested = (
            bool(return_features) or bool(return_multiscale_features)
        )
        if feature_capture_requested:
            handle = self.affinity_head[-1].register_forward_pre_hook(
                lambda module, inputs: captured_head.append(inputs[0])
            )
        try:
            prediction = self.affinity_head(affinity_input)
        finally:
            if handle is not None:
                handle.remove()
        if feature_capture_requested and len(captured_head) != 1:
            raise RuntimeError(
                "failed to capture interface affinity head"
            )
        if return_multiscale_features:
            effective_ligand_coords = (
                ligand_coords
                if ligand_coords is not None
                else _require_tensor(item, "ligand_coords", ndim=2)
            )
            return prediction, {
                "interface_ligand_nodes": local.ligand_scalar,
                "interface_protein_nodes": local.protein_scalar,
                "interface_residue_nodes": protein_states,
                "interface_ligand_coords": effective_ligand_coords,
                "interface_protein_coords": _require_tensor(
                    item, "protein_coords", ndim=2
                ),
                "interface_residue_coords": _require_tensor(
                    item, "residue_ca_coords", ndim=2
                ),
                "interface_ligand_summary": ligand_summary,
                "interface_protein_summary": protein_summary,
                "interface_local_summary": local.interface_latent,
                "interface_head": captured_head[0],
                "interface_score": local.interface_score,
                "interface_projection": local.projection,
            }
        if return_features:
            return prediction, {
                "interface_protein": protein_summary,
                "interface_ligand": ligand_summary,
                "interface_local": local.interface_latent,
                "interface_head": captured_head[0],
            }
        return (
            prediction,
            local.interface_latent,
            local.interface_score,
            local.projection,
        )

    def forward_packed(self, packed):
        """Return padded multiscale features for a packed Interface batch."""
        required = ("batch_size", "ligand_offsets", "protein_offsets", "residue_offsets")
        if any(not hasattr(packed, name) for name in required):
            raise ValueError("packed interface batch has an invalid contract")
        predictions = []
        per_item_features = []
        for index in range(int(packed.batch_size)):
            ligand_start = int(packed.ligand_offsets[index])
            ligand_stop = int(packed.ligand_offsets[index + 1])
            protein_start = int(packed.protein_offsets[index])
            protein_stop = int(packed.protein_offsets[index + 1])
            residue_start = int(packed.residue_offsets[index])
            residue_stop = int(packed.residue_offsets[index + 1])
            edge_mask = ((packed.ligand_edge_index[0] >= ligand_start) & (packed.ligand_edge_index[0] < ligand_stop))
            item = {
                "key": packed.keys[index],
                "ligand_atom_features": packed.ligand_atom_features[ligand_start:ligand_stop],
                "ligand_coords": packed.ligand_coords[ligand_start:ligand_stop],
                "ligand_edge_index": packed.ligand_edge_index[:, edge_mask] - ligand_start,
                "ligand_edge_attr": packed.ligand_edge_attr[edge_mask],
                "ligand_canonical_rank": packed.ligand_canonical_rank[ligand_start:ligand_stop],
                "ligand_chiral_tag": packed.ligand_chiral_tag[ligand_start:ligand_stop],
                "protein_coords": packed.protein_coords[protein_start:protein_stop],
                "protein_element": packed.protein_element[protein_start:protein_stop],
                "protein_atom_name_class": packed.protein_atom_name_class[protein_start:protein_stop],
                "protein_residue_type": packed.protein_residue_type[protein_start:protein_stop],
                "protein_is_backbone": packed.protein_is_backbone[protein_start:protein_stop],
                "protein_parent_residue": packed.protein_parent_residue[protein_start:protein_stop] - residue_start,
                "residue_esm": packed.residue_esm[residue_start:residue_stop],
                "residue_ca_coords": packed.residue_ca_coords[residue_start:residue_stop],
            }
            prediction, features = self(item, return_multiscale_features=True)
            predictions.append(prediction)
            per_item_features.append(features)
        variable = ("interface_ligand_nodes", "interface_protein_nodes", "interface_residue_nodes", "interface_ligand_coords", "interface_protein_coords", "interface_residue_coords")
        result = {}
        for name in variable:
            values = [features[name] for features in per_item_features]
            maximum = max(value.shape[0] for value in values)
            padded = values[0].new_zeros((len(values), maximum, values[0].shape[1]))
            mask = torch.zeros((len(values), maximum), dtype=torch.bool, device=padded.device)
            for item_index, value in enumerate(values):
                padded[item_index, : value.shape[0]] = value
                mask[item_index, : value.shape[0]] = True
            result[name] = padded
            if name.endswith("_nodes"):
                result[name.replace("_nodes", "_mask")] = mask
        for name in ("interface_ligand_summary", "interface_protein_summary", "interface_local_summary", "interface_head", "interface_score", "interface_projection"):
            result[name] = torch.cat([features[name] for features in per_item_features], dim=0)
        return torch.cat(predictions, dim=0), result

    @staticmethod
    def _set_trainable(module, trainable):
        for parameter in module.parameters():
            parameter.requires_grad = bool(trainable)

    def configure_stage1(self):
        self._set_trainable(self, False)
        self._set_trainable(self.local_path, True)

    def configure_stage2_warmup(self):
        self._set_trainable(self, False)
        self._set_trainable(self.local_path, True)
        self._set_trainable(self.affinity_head, True)

    def configure_stage2_full(self):
        self._set_trainable(self, True)

    def stage1_state_dict(self):
        return {
            "local_path": deepcopy(self.local_path.state_dict())
        }

    def load_stage1_state_dict(self, state_dict, strict=True):
        if not isinstance(state_dict, dict):
            raise RuntimeError(
                "Stage-1 state must be a dictionary"
            )
        if "local_path" not in state_dict:
            raise RuntimeError(
                "Stage-1 state is missing local_path"
            )
        if strict and set(state_dict) != {"local_path"}:
            raise RuntimeError(
                "Stage-1 state contains unexpected top-level keys"
            )
        return self.local_path.load_state_dict(
            state_dict["local_path"], strict=bool(strict)
        )

    @staticmethod
    def _trainable_parameters(modules):
        return [
            parameter
            for module in modules
            for parameter in module.parameters()
            if parameter.requires_grad
        ]

    def optimizer_parameter_groups(
        self,
        local_lr,
        global_lr,
        affinity_lr=None,
    ):
        local_lr = float(local_lr)
        global_lr = float(global_lr)
        affinity_lr = (
            global_lr
            if affinity_lr is None
            else float(affinity_lr)
        )
        if min(local_lr, global_lr, affinity_lr) <= 0:
            raise ValueError("optimizer learning rates must be positive")
        groups = []
        definitions = (
            ("local", [self.local_path], local_lr),
            (
                "global",
                [
                    self.protein_global,
                    self.protein_conditioned_pool,
                    self.ligand_conditioned_pool,
                ],
                global_lr,
            ),
            ("affinity", [self.affinity_head], affinity_lr),
        )
        seen = set()
        for name, modules, learning_rate in definitions:
            parameters = self._trainable_parameters(modules)
            duplicate = seen.intersection(
                id(parameter) for parameter in parameters
            )
            if duplicate:
                raise RuntimeError(
                    "optimizer parameter groups overlap"
                )
            seen.update(id(parameter) for parameter in parameters)
            if parameters:
                groups.append(
                    {
                        "name": name,
                        "params": parameters,
                        "lr": learning_rate,
                    }
                )
        return groups
