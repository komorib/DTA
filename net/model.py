import torch
import pickle
import sys
#sys.path.append("net/")
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from equiformer_pytorch import Equiformer
from net.boltz_interface import BoltzInterfaceHead
from torch_geometric.data import Batch
from net.GIN import GNN_node,GNN_node_test
from net.MLP import MLP
from net.Trigonometry import Transition,TriangleProteinToCompound_v2,TriangleSelfAttentionRowWise,TriangleProteinToCompound_v3
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
import sys
import os
#sys.path.append("../")
from utils.net_utils import *
import time
from edl_pytorch import Dirichlet, NormalInvGamma, evidential_classification


def _ensure_pbcnet_import_path(pbcnet_root=None):
    if not pbcnet_root:
        return
    pbcnet_root = os.path.abspath(pbcnet_root)
    model_code = os.path.join(pbcnet_root, "model_code")
    for entry in (pbcnet_root, model_code):
        if os.path.isdir(entry) and entry not in sys.path:
            sys.path.insert(0, entry)



def _torch_load_cpu_checkpoint(checkpoint_path, pbcnet_root=None):
    _ensure_pbcnet_import_path(pbcnet_root)
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _normalize_state_dict_keys(state_dict):
    if not state_dict:
        raise ValueError("empty state_dict for PBCNet initialization")
    if all(isinstance(key, str) and key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def load_pbcnet_state_dict(checkpoint_path, pbcnet_root=None):
    payload = _torch_load_cpu_checkpoint(checkpoint_path, pbcnet_root=pbcnet_root)
    if isinstance(payload, nn.Module):
        return _normalize_state_dict_keys(payload.state_dict())
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, nn.Module):
                return _normalize_state_dict_keys(candidate.state_dict())
            if isinstance(candidate, dict) and candidate:
                return _normalize_state_dict_keys(candidate)
        if payload and all(isinstance(key, str) for key in payload.keys()) and all(torch.is_tensor(value) for value in payload.values()):
            return _normalize_state_dict_keys(payload)
    raise ValueError(f"Unsupported PBCNet checkpoint format: {checkpoint_path}")


def infer_pbcnet_hidden_dim_from_state_dict(state_dict):
    norm_weight = state_dict.get("norm.weight")
    if norm_weight is not None:
        return int(norm_weight.numel())
    first_ffn = state_dict.get("FNN.0.weight")
    if first_ffn is not None and getattr(first_ffn, "ndim", 0) == 2:
        return int(first_ffn.shape[1])
    return None


def infer_pbcnet_num_layers_from_state_dict(state_dict):
    layer_ids = set()
    prefix = "encoder.layers."
    for key in state_dict.keys():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        layer_id = suffix.split('.', 1)[0]
        if layer_id.isdigit():
            layer_ids.add(int(layer_id))
    if not layer_ids:
        return None
    return max(layer_ids) + 1


# GerNA 主模型：把 RNA 1D/2D/3D 与分子 2D/3D 融合，输出 binding evidence 和 pairwise interaction。
class GerNA(nn.Module):
    def __init__(self, params, input_dim_rna = 640, input_dim_mol = 55, trigonometry = True, mol_graph = True, coors = True, rna_repre = True, rna_graph = True, coors_3_bead = True, uncertainty = False, num_classes = 2, use_other_GNN = False, GNN_type = "gcn", rna_num_tokens = None, rna_3d_input_dim = None, disable_protein_2d = False, dti_front_fusion = "none", dti_front_input_dim = None, dti_late_fusion = "none", dti_late_input_dim = None, dti_late_hidden_dim = None, ligand_unimol_fusion = "none", ligand_unimol_input_dim = None, ligand_unimol2_3d_fusion = "none", ligand_unimol2_input_dim = None, ligand_unimol2_3d_fusion_stage = "post", pbcnet_late_fusion = "none", pbcnet_input_dim = None, pbcnet_hidden_dim = None, pbcnet_trainable = False, pbcnet_root = None, pbcnet_model_hidden_dim = 96, pbcnet_init_checkpoint = None, boltz_interface_fusion = False, boltz_interface_input_dim = 384, boltz_interface_hidden_dim = 128):
        super(GerNA, self).__init__()
        
        self.dropout_layer = nn.Dropout(0.4)
        """hyper part"""
        # GNN_depth, inner_CNN_depth, DMA_depth, k_head, kernel_size, hidden_size1, hidden_size2 = params
        GNN_depth, DMA_depth, hidden_size1, hidden_size2 = params
        self.GNN_depth = GNN_depth
        #self.inner_CNN_depth = inner_CNN_depth
        self.DMA_depth = DMA_depth
        #self.k_head = k_head
        #self.kernel_size = kernel_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.input_dim_rna = input_dim_rna
        #if not embedding
        self.input_dim_mol = input_dim_mol
        self.trigonometry = trigonometry
        self.mol_graph = mol_graph
        self.coors = coors
        self.rna_graph = rna_graph
        self.disable_protein_2d = bool(disable_protein_2d)
        if self.disable_protein_2d:
            self.rna_graph = False
        self.dti_front_fusion = str(dti_front_fusion or "none")
        self.dti_late_fusion = str(dti_late_fusion or "none")
        self.ligand_unimol_fusion = str(ligand_unimol_fusion or "none")
        self.ligand_unimol2_3d_fusion = str(ligand_unimol2_3d_fusion or "none")
        self.ligand_unimol2_3d_fusion_stage = str(ligand_unimol2_3d_fusion_stage or "post")
        self.pbcnet_late_fusion = str(pbcnet_late_fusion or "none")
        self.pbcnet_trainable = bool(pbcnet_trainable)
        self.pbcnet_root = pbcnet_root
        self.pbcnet_init_checkpoint = pbcnet_init_checkpoint
        self.boltz_interface_fusion = bool(boltz_interface_fusion)
        self.boltz_interface_input_dim = int(boltz_interface_input_dim)
        self.boltz_interface_hidden_dim = int(boltz_interface_hidden_dim)
        self.pbcnet_init_state_dict = None
        self.pbcnet_model_hidden_dim = int(pbcnet_model_hidden_dim)
        self.pbcnet_model_num_layer = 5
        if self.pbcnet_trainable and self.pbcnet_init_checkpoint:
            self.pbcnet_init_state_dict = load_pbcnet_state_dict(
                self.pbcnet_init_checkpoint,
                pbcnet_root=self.pbcnet_root,
            )
            inferred_hidden_dim = infer_pbcnet_hidden_dim_from_state_dict(self.pbcnet_init_state_dict)
            if inferred_hidden_dim is None:
                raise ValueError(
                    f"Could not infer PBCNet hidden dimension from checkpoint: {self.pbcnet_init_checkpoint}"
                )
            inferred_num_layer = infer_pbcnet_num_layers_from_state_dict(self.pbcnet_init_state_dict)
            if inferred_num_layer is None:
                raise ValueError(
                    f"Could not infer PBCNet layer count from checkpoint: {self.pbcnet_init_checkpoint}"
                )
            self.pbcnet_model_hidden_dim = inferred_hidden_dim
            self.pbcnet_model_num_layer = inferred_num_layer
        if self.dti_front_fusion not in {"none", "add", "concat"}:
            raise ValueError("dti_front_fusion must be one of: none, add, concat")
        if self.dti_late_fusion not in {"none", "concat"}:
            raise ValueError("dti_late_fusion must be one of: none, concat")
        if self.ligand_unimol_fusion not in {"none", "add", "concat"}:
            raise ValueError("ligand_unimol_fusion must be one of: none, add, concat")
        if self.ligand_unimol2_3d_fusion not in {"none", "add", "concat"}:
            raise ValueError("ligand_unimol2_3d_fusion must be one of: none, add, concat")
        if self.ligand_unimol2_3d_fusion_stage not in {"pre", "post"}:
            raise ValueError("ligand_unimol2_3d_fusion_stage must be one of: pre, post")
        if self.pbcnet_late_fusion not in {"none", "concat"}:
            raise ValueError("pbcnet_late_fusion must be one of: none, concat")
        if self.ligand_unimol2_3d_fusion == "none":
            self.ligand_unimol2_3d_fusion_stage = "post"
        if self.dti_front_fusion != "none" and (dti_front_input_dim is None or int(dti_front_input_dim) <= 0):
            raise ValueError("dti_front_input_dim must be positive when dti_front_fusion is enabled")
        if self.dti_late_fusion != "none" and (dti_late_input_dim is None or int(dti_late_input_dim) <= 0):
            raise ValueError("dti_late_input_dim must be positive when dti_late_fusion is enabled")
        if self.ligand_unimol_fusion != "none" and (ligand_unimol_input_dim is None or int(ligand_unimol_input_dim) <= 0):
            raise ValueError("ligand_unimol_input_dim must be positive when ligand_unimol_fusion is enabled")
        if self.ligand_unimol2_3d_fusion != "none" and (ligand_unimol2_input_dim is None or int(ligand_unimol2_input_dim) <= 0):
            raise ValueError("ligand_unimol2_input_dim must be positive when ligand_unimol2_3d_fusion is enabled")
        if self.pbcnet_late_fusion != "none" and (pbcnet_input_dim is None or int(pbcnet_input_dim) <= 0):
            raise ValueError("pbcnet_input_dim must be positive when pbcnet_late_fusion is enabled")
        self.dti_front_input_dim = int(dti_front_input_dim) if dti_front_input_dim else None
        self.dti_late_input_dim = int(dti_late_input_dim) if dti_late_input_dim else None
        self.ligand_unimol_input_dim = int(ligand_unimol_input_dim) if ligand_unimol_input_dim else None
        self.ligand_unimol2_input_dim = int(ligand_unimol2_input_dim) if ligand_unimol2_input_dim else None
        if self.pbcnet_trainable and pbcnet_input_dim is None:
            pbcnet_input_dim = self.pbcnet_model_hidden_dim
        self.pbcnet_input_dim = int(pbcnet_input_dim) if pbcnet_input_dim else None
        if self.pbcnet_trainable and self.pbcnet_input_dim is not None and self.pbcnet_input_dim != self.pbcnet_model_hidden_dim:
            raise ValueError(
                "Trainable PBCNet output dim must match encoder hidden dim: "
                f"got pbcnet_input_dim={self.pbcnet_input_dim}, expected {self.pbcnet_model_hidden_dim}"
            )
        self.dti_late_hidden_dim = int(dti_late_hidden_dim) if dti_late_hidden_dim else self.hidden_size2
        self.pbcnet_hidden_dim = int(pbcnet_hidden_dim) if pbcnet_hidden_dim else self.hidden_size2
        self.coors_3_bead = coors_3_bead
        self.rna_repre = rna_repre
        self.rna_3d_input_dim = rna_3d_input_dim
        self.rna_3d_input_norm = nn.LayerNorm(rna_3d_input_dim) if rna_3d_input_dim is not None else None
        self.uncertainty = uncertainty
        self.num_classes = num_classes
        self.use_other_GNN = use_other_GNN
        self.GNN_type = GNN_type
        # RNA 原版用 1..4/1..8 的原子 token；PDBbind protein 版用 1..20 的氨基酸 token。
        if rna_num_tokens is None:
            rna_num_tokens = 5 if self.coors_3_bead else 9
        self.rna_num_tokens = rna_num_tokens
        
        
        # 2D 图分支：RNA 二级结构图和分子图分别用 GNN 编码。
        """GCN module for molecule and RNA"""
        if self.rna_graph:
            self.GCN_rna = GNN_node(self.input_dim_rna, self.GNN_depth, self.hidden_size1, edge_attr_option = False, gnn_type = 'gcn')  #RNA edge has no attribute features
        
        if self.mol_graph:
            if use_other_GNN:
                self.GCN_mol = GNN_node_test(self.input_dim_mol, self.GNN_depth, self.hidden_size1, gnn_type = GNN_type)
            else:
                self.GCN_mol = GNN_node(self.input_dim_mol, self.GNN_depth, self.hidden_size1, gnn_type = 'gcn')

        """MLP Module"""
        if self.rna_repre:
            self.mlp_rna = MLP(self.input_dim_rna, self.hidden_size1)
        if self.dti_front_fusion != "none":
            if not self.rna_repre:
                raise ValueError("DTI front fusion requires RNA/protein 1D representations")
            self.dti_front_norm = nn.LayerNorm(self.dti_front_input_dim)
            self.dti_front_proj = nn.Linear(self.dti_front_input_dim, self.hidden_size1)
            if self.dti_front_fusion == "concat":
                self.dti_front_merge = nn.Sequential(
                    nn.LayerNorm(self.hidden_size1 * 2),
                    nn.Linear(self.hidden_size1 * 2, self.hidden_size1),
                    nn.LeakyReLU(0.1),
                )
        if self.ligand_unimol_fusion != "none":
            self.ligand_unimol_norm = nn.LayerNorm(self.ligand_unimol_input_dim)
            self.ligand_unimol_2d_proj = nn.Linear(self.ligand_unimol_input_dim, self.hidden_size1)
            if self.ligand_unimol_fusion == "concat":
                self.ligand_unimol_2d_merge = nn.Sequential(
                    nn.LayerNorm(self.hidden_size1 * 2),
                    nn.Linear(self.hidden_size1 * 2, self.hidden_size1),
                    nn.LeakyReLU(0.1),
                )
        if self.ligand_unimol2_3d_fusion != "none":
            self.ligand_unimol2_norm = nn.LayerNorm(self.ligand_unimol2_input_dim)
            if self.ligand_unimol2_3d_fusion_stage == "pre":
                self.mol_token_emb = nn.Embedding(10, 16)
                self.ligand_unimol2_pre_proj = nn.Linear(self.ligand_unimol2_input_dim, 16)
                if self.ligand_unimol2_3d_fusion == "concat":
                    self.ligand_unimol2_pre_merge = nn.Sequential(
                        nn.LayerNorm(32),
                        nn.Linear(32, 16),
                        nn.LeakyReLU(0.1),
                    )
            else:
                self.ligand_unimol2_3d_proj = nn.Linear(self.ligand_unimol2_input_dim, 16)
                if self.ligand_unimol2_3d_fusion == "concat":
                    self.ligand_unimol2_3d_merge = nn.Sequential(
                        nn.LayerNorm(32),
                        nn.Linear(32, 16),
                        nn.LeakyReLU(0.1),
                    )
        
        # 3D 坐标分支：Equiformer 用 token + 坐标编码空间构象。
        """EquiFormer Block"""
        if self.coors:
            if self.ligand_unimol2_3d_fusion != "none" and self.ligand_unimol2_3d_fusion_stage == "pre":
                self.equi_mol = Equiformer( num_tokens = None, dim_in = 16, dim = (16, 4, 2), dim_head = (10, 10, 10), heads = (2, 2, 2), num_linear_attn_heads = 0,
                                            num_degrees = 3, depth = 2, attend_self = True, l2_dist_attention = False, reversible = True)
            else:
                self.equi_mol = Equiformer( num_tokens = 10, dim = (16, 4, 2), dim_head = (10, 10, 10), heads = (2, 2, 2), num_linear_attn_heads = 0, 
                                            num_degrees = 3, depth = 2, attend_self = True, l2_dist_attention = False, reversible = True)
            if self.rna_3d_input_dim is not None:
                self.equi_rna = Equiformer( num_tokens = None, dim_in = self.rna_3d_input_dim, dim = (16, 4, 2), dim_head = (10, 10, 10), heads = (2, 2, 2), num_linear_attn_heads = 0,
                                           num_degrees = 3, depth = 2, attend_self = True, l2_dist_attention = False, reversible = True)
            elif self.coors_3_bead:
                self.equi_rna = Equiformer( num_tokens = self.rna_num_tokens, dim = (16, 4, 2), dim_head = (10, 10, 10), heads = (2, 2, 2), num_linear_attn_heads = 0, 
                                           num_degrees = 3, depth = 2, attend_self = True, l2_dist_attention = False, reversible = True)
            else:
                self.equi_rna = Equiformer( num_tokens = self.rna_num_tokens, dim = (16, 4, 2), dim_head = (10, 10, 10), heads = (2, 2, 2), num_linear_attn_heads = 0, 
                                           num_degrees = 3, depth = 2, attend_self = True, l2_dist_attention = False, reversible = True)

        """pairwise_prediction_module"""
        if self.mol_graph:
            self.pairwise_mol_2d = nn.Linear(self.hidden_size1, self.hidden_size1)
        if self.rna_graph:
            self.pairwise_rna_2d = nn.Linear(self.hidden_size1, self.hidden_size1)
        if self.rna_repre:
            self.pairwise_rna_1d = nn.Linear(self.hidden_size1, self.hidden_size1)
        if self.coors:
            self.pairwise_rna_3d = nn.Linear(16, 16)
            self.pairwise_mol_3d = nn.Linear(16, 16)
            self.layernorm_3d = torch.nn.LayerNorm(16)
            
        # 论文中的 geometric/triangular constraint 主要在这里初始化，借鉴 TankBind 的三角更新。
        if trigonometry:
            """Tankbind gengxin pair"""
            self.n_trigonometry_module_stack = 2 # n_trigonometry_module_stack
            if self.mol_graph:

                self.rna_to_mol_list_2d = []
                self.rna_to_mol_list_2d = nn.ModuleList([TriangleProteinToCompound_v3(embedding_channels=self.hidden_size1, c=self.hidden_size1) for _ in range(self.n_trigonometry_module_stack)])
                self.triangle_self_attention_list_2d = nn.ModuleList([TriangleSelfAttentionRowWise(embedding_channels=self.hidden_size1) for _ in range(self.n_trigonometry_module_stack)])
                self.tranistion_2d = Transition(embedding_channels=self.hidden_size1, n=4)
                self.layernorm_2d = torch.nn.LayerNorm(self.hidden_size1)
                self.dropout_2d = nn.Dropout2d(p=0.25)
                self.linear_2d = nn.Linear(self.hidden_size1, 1)
                self.rna_pair_embedding_2d = nn.Linear(16,self.hidden_size1)
                self.mol_pair_embedding_2d = nn.Linear(16,self.hidden_size1)
            #Origin, which use feat to represent the coords.
                # self.rna_to_mol_list = []
                # self.rna_to_mol_list = nn.ModuleList([TriangleProteinToCompound_v2(embedding_channels=self.hidden_size1, c=self.hidden_size1) for _ in range(self.n_trigonometry_module_stack)])
                # self.triangle_self_attention_list = nn.ModuleList([TriangleSelfAttentionRowWise(embedding_channels=self.hidden_size1) for _ in range(self.n_trigonometry_module_stack)])
                # self.tranistion = Transition(embedding_channels=self.hidden_size1, n=4)
                # self.layernorm = torch.nn.LayerNorm(self.hidden_size1)
                # self.dropout = nn.Dropout2d(p=0.25)
                # self.linear = nn.Linear(self.hidden_size1, 1)

            if self.coors:
                self.rna_to_mol_list_3d = []
                self.rna_to_mol_list_3d = nn.ModuleList([TriangleProteinToCompound_v3(embedding_channels=16, c=16) for _ in range(self.n_trigonometry_module_stack)])
                self.triangle_self_attention_list_3d = nn.ModuleList([TriangleSelfAttentionRowWise(embedding_channels=16) for _ in range(self.n_trigonometry_module_stack)])
                self.tranistion_3d = Transition(embedding_channels=16, n=4)
                self.layernorm_3d = torch.nn.LayerNorm(16)
                self.dropout_3d = nn.Dropout2d(p=0.25)
                self.linear_3d = nn.Linear(16, 1)
                self.rna_pair_embedding = nn.Linear(16,16)
                self.mol_pair_embedding = nn.Linear(16,16)
            
        # Affinity 模块先把各模态节点特征映射到统一维度，再由 DMA 汇聚成 pair-level 表征。
        """Affinity Prediction Module"""
        if self.mol_graph:
            self.mol_2d_final = nn.Linear( self.hidden_size1, self.hidden_size2 )
            if self.rna_repre:
                self.rna_1d_final = nn.Linear( self.hidden_size1, self.hidden_size2 )
            if self.rna_graph:
                self.rna_2d_final = nn.Linear( self.hidden_size1, self.hidden_size2 )
        if self.coors:
            self.mol_3d_final = nn.Linear( 16, self.hidden_size2 )
            self.rna_3d_final = nn.Linear( 16, self.hidden_size2 )
        #Output layer
        base_affinity_input_dim = self.hidden_size2 * self.hidden_size2
        if self.mol_graph and self.coors:
            base_affinity_input_dim *= 2
        self.affinity_input_dim = base_affinity_input_dim
        if self.dti_late_fusion == "concat":
            self.dti_late_norm = nn.LayerNorm(self.dti_late_input_dim)
            self.dti_late_proj = nn.Sequential(
                nn.Linear(self.dti_late_input_dim, self.dti_late_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            self.affinity_input_dim += self.dti_late_hidden_dim
        if self.pbcnet_late_fusion == "concat":
            self.pbcnet_late_norm = nn.LayerNorm(self.pbcnet_input_dim)
            self.pbcnet_late_proj = nn.Sequential(
                nn.Linear(self.pbcnet_input_dim, self.pbcnet_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            self.affinity_input_dim += self.pbcnet_hidden_dim
        if self.boltz_interface_fusion:
            self.boltz_interface_head = BoltzInterfaceHead(
                emb_dim=self.boltz_interface_input_dim,
                hidden_dim=self.boltz_interface_hidden_dim,
                dropout=0.3,
            )
            self.affinity_input_dim += self.boltz_interface_head.output_dim
        if self.pbcnet_trainable:
            self.pbcnet_encoder = self._init_trainable_pbcnet()
        if uncertainty:
            self.W_out = nn.Sequential(
                nn.Linear(self.affinity_input_dim, 1024),
                nn.ReLU(),nn.Dropout(0.4),
                nn.Linear(1024, 128),nn.ReLU(),nn.Dropout(0.4), Dirichlet(128, self.num_classes))
        else:
            self.W_out = nn.Sequential(
                nn.Linear(self.affinity_input_dim, 1024),
                nn.ReLU(),nn.Dropout(0.4),
                nn.Linear(1024, 128),nn.ReLU(),nn.Dropout(0.4), nn.Linear(128, 1))

        #DMA parameters
        if self.mol_graph:
#             self.mc0 = nn.Linear(hidden_size2, hidden_size2)
#             self.mp0 = nn.Linear(hidden_size2, hidden_size2)
            self.mc1 = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.mp1 = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.hc0 = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.hp0 = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.hc1 = nn.ModuleList([nn.Linear(self.hidden_size2, 1) for i in range(self.DMA_depth)])
            self.hp1 = nn.ModuleList([nn.Linear(self.hidden_size2, 1) for i in range(self.DMA_depth)])
            self.m_to_r_transform = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.r_to_m_transform = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.GRU_dma = nn.GRUCell(self.hidden_size2, self.hidden_size2)
            
        if self.coors:
#             self.mc0_3d = nn.Linear(hidden_size2, hidden_size2)
#             self.mp0_3d = nn.Linear(hidden_size2, hidden_size2)
            self.mc1_3d = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.mp1_3d = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.hc0_3d = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.hp0_3d = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.hc1_3d = nn.ModuleList([nn.Linear(self.hidden_size2, 1) for i in range(self.DMA_depth)])
            self.hp1_3d = nn.ModuleList([nn.Linear(self.hidden_size2, 1) for i in range(self.DMA_depth)])
            self.m_to_r_transform_3d = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.r_to_m_transform_3d = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.GRU_dma_3d = nn.GRUCell(self.hidden_size2, self.hidden_size2)

#         """Pairwise Interaction Prediction Module"""
#         self.pairwise_molecule = nn.Linear(self.hidden_size1, self.hidden_size1)
#         self.pairwise_rna = nn.Linear(self.hidden_size1, self.hidden_size1)
    
    def _init_trainable_pbcnet(self):
        if not self.pbcnet_root:
            raise ValueError("pbcnet_root must be provided when pbcnet_trainable is enabled")
        _ensure_pbcnet_import_path(self.pbcnet_root)
        from models.readout import PBCNetv2

        encoder = PBCNetv2(
            hidden_channels=self.pbcnet_model_hidden_dim,
            num_layer=self.pbcnet_model_num_layer,
        )
        if self.pbcnet_init_state_dict is not None:
            encoder.load_state_dict(self.pbcnet_init_state_dict)
            print(
                "pbcnet_init_checkpoint:",
                self.pbcnet_init_checkpoint,
                "pbcnet_model_hidden_dim:",
                self.pbcnet_model_hidden_dim,
                "pbcnet_model_num_layer:",
                self.pbcnet_model_num_layer,
            )
        return encoder

    def compute_trainable_pbcnet_global(self, PBCNet_graph, output_device=None):
        if not self.pbcnet_trainable:
            return None
        if PBCNet_graph is None:
            raise ValueError("Trainable PBCNet is enabled but PBCNet_graph was not provided")
        pbcnet_device = next(self.pbcnet_encoder.parameters()).device
        PBCNet_graph = PBCNet_graph.to(pbcnet_device)
        PBCNet_graph.edges['int'].data['bond_scalar'] = torch.clip(
            PBCNet_graph.edges['int'].data['bond_scalar'].long(),
            min=0,
            max=4,
        )
        pbcnet_global, _ = self.pbcnet_encoder._readout(PBCNet_graph, False)
        if output_device is not None:
            pbcnet_global = pbcnet_global.to(output_device)
        return pbcnet_global


    def apply_boltz_interface_fusion(self, kroneck, boltz_interface=None):
        if not self.boltz_interface_fusion:
            return kroneck
        if boltz_interface is None:
            raise ValueError("Boltz interface fusion is enabled but Boltz interface features were not provided")
        boltz_latent = self.boltz_interface_head(boltz_interface)
        if boltz_latent.shape[0] != kroneck.shape[0]:
            raise ValueError(
                f"Boltz interface batch size {boltz_latent.shape[0]} does not match GerNA batch size {kroneck.shape[0]}"
            )
        return torch.cat((kroneck, boltz_latent), dim=-1)

    def apply_dti_front_fusion(self, RNA_1d_fea, Seq_mask=None, DTI_protein_tokens=None, DTI_protein_mask=None):
        # Front fusion operates before GerNA pair construction. Shape flow:
        #   GerNA chain ESM: RNA_repre [B, L, 640] -> mlp_rna -> RNA_1d_fea [B, L, hidden_size1]
        #   DTI protein token: DTI_protein_tokens [B, L, dti_front_input_dim=256]
        #       -> LayerNorm + Linear(256, hidden_size1) -> dti_tokens [B, L, hidden_size1]
        #   add: RNA_1d_fea + dti_tokens -> [B, L, hidden_size1]
        #   concat: cat -> [B, L, 2*hidden_size1] -> Linear -> [B, L, hidden_size1]
        if self.dti_front_fusion == "none":
            return RNA_1d_fea
        if DTI_protein_tokens is None:
            raise ValueError("DTI front fusion is enabled but DTI_protein_tokens was not provided")
        if DTI_protein_tokens.dim() != 3:
            raise ValueError(f"DTI_protein_tokens must have shape [B, L, D], got {tuple(DTI_protein_tokens.shape)}")
        if DTI_protein_tokens.size(0) != RNA_1d_fea.size(0) or DTI_protein_tokens.size(1) != RNA_1d_fea.size(1):
            raise ValueError(
                "DTI protein token shape must align with GerNA protein 1D features: "
                f"got {tuple(DTI_protein_tokens.shape[:2])}, expected {tuple(RNA_1d_fea.shape[:2])}"
            )
        dti_tokens = self.dti_front_norm(DTI_protein_tokens.float())
        dti_tokens = F.leaky_relu(self.dti_front_proj(dti_tokens), 0.1)
        if DTI_protein_mask is not None:
            dti_tokens = dti_tokens * DTI_protein_mask.to(dti_tokens.device).float().unsqueeze(-1)
        if Seq_mask is not None:
            dti_tokens = dti_tokens * Seq_mask.to(dti_tokens.device).float().unsqueeze(-1)
        if self.dti_front_fusion == "add":
            return RNA_1d_fea + dti_tokens
        return self.dti_front_merge(torch.cat([RNA_1d_fea, dti_tokens], dim=-1))

    def apply_dti_late_fusion(self, kroneck, DTI_global=None):
        # Late fusion uses a pooled DTI pair/global feature after GerNA pooling. Shape flow:
        #   kroneck [B, hidden_size2*2] from GerNA affinity module
        #   DTI_global [B, dti_late_input_dim=512] -> Linear -> [B, dti_late_hidden_dim]
        #   output is concatenated [B, hidden_size2*2 + dti_late_hidden_dim]
        if self.dti_late_fusion == "none":
            return kroneck
        if DTI_global is None:
            raise ValueError("DTI late fusion is enabled but DTI_global was not provided")
        if DTI_global.dim() != 2:
            raise ValueError(f"DTI_global must have shape [B, D], got {tuple(DTI_global.shape)}")
        if DTI_global.size(0) != kroneck.size(0):
            raise ValueError("DTI_global batch size must match GerNA affinity batch size")
        dti_global = self.dti_late_norm(DTI_global.float())
        dti_global = self.dti_late_proj(dti_global)
        return torch.cat([kroneck, dti_global], dim=-1)


    def apply_pbcnet_late_fusion(self, kroneck, PBCNet_global=None):
        # PBCNet2.0 _readout is a single-complex pocket-ligand representation.
        # Shape flow: PBCNet_global [B, pbcnet_input_dim=128] -> Linear -> [B, pbcnet_hidden_dim],
        # then concatenate after GerNA DMA pooling so it cannot disturb atom/residue encoders.
        if self.pbcnet_late_fusion == "none":
            return kroneck
        if PBCNet_global is None:
            raise ValueError("PBCNet late fusion is enabled but PBCNet_global was not provided")
        if PBCNet_global.dim() != 2:
            raise ValueError(f"PBCNet_global must have shape [B, D], got {tuple(PBCNet_global.shape)}")
        if PBCNet_global.size(0) != kroneck.size(0):
            raise ValueError("PBCNet_global batch size must match GerNA affinity batch size")
        pbcnet_global = self.pbcnet_late_norm(PBCNet_global.float())
        pbcnet_global = self.pbcnet_late_proj(pbcnet_global)
        return torch.cat([kroneck, pbcnet_global], dim=-1)


    def _project_ligand_unimol_tokens(self, ligand_unimol_tokens, ligand_unimol_mask, target_mask, target_len, out_dim, proj):
        # Shared atom-level projection for old ligand Uni-Mol features. Shape flow:
        #   ligand_unimol_tokens [B, A, ligand_unimol_input_dim=512]
        #       -> LayerNorm + Linear(512, out_dim) -> [B, A, out_dim]
        # Here out_dim is hidden_size1 for the ligand 2D branch.
        if self.ligand_unimol_fusion == "none":
            return None
        if ligand_unimol_tokens is None:
            raise ValueError("Ligand Uni-Mol fusion is enabled but ligand_unimol_tokens was not provided")
        if ligand_unimol_tokens.dim() != 3:
            raise ValueError(f"ligand_unimol_tokens must have shape [B, A, D], got {tuple(ligand_unimol_tokens.shape)}")
        if ligand_unimol_tokens.size(1) != target_len:
            raise ValueError(
                "Ligand Uni-Mol token length must align with GerNA ligand atom features: "
                f"got {ligand_unimol_tokens.size(1)}, expected {target_len}"
            )
        tokens = self.ligand_unimol_norm(ligand_unimol_tokens.float())
        tokens = F.leaky_relu(proj(tokens), 0.1)
        if ligand_unimol_mask is not None:
            tokens = tokens * ligand_unimol_mask.to(tokens.device).float().unsqueeze(-1)
        if target_mask is not None:
            tokens = tokens * target_mask.to(tokens.device).float().unsqueeze(-1)
        return tokens

    def apply_ligand_unimol_fusion_2d(self, Mol_2d_fea, vertex_mask=None, ligand_unimol_tokens=None, ligand_unimol_mask=None):
        # Old Uni-Mol token features are fused into the ligand 2D graph branch
        # after the RDKit atom graph has been encoded by GNN_node. Shape flow:
        #   RDKit ligand graph x [sum_A, 55] -> GNN_node -> Mol_2d_fea [B, A, hidden_size1]
        #   old Uni-Mol [B, A, 512] -> Linear -> [B, A, hidden_size1]
        #   add keeps [B, A, hidden_size1]; concat uses [B, A, 2*hidden_size1] -> [B, A, hidden_size1].
        if self.ligand_unimol_fusion == "none":
            return Mol_2d_fea
        tokens = self._project_ligand_unimol_tokens(
            ligand_unimol_tokens,
            ligand_unimol_mask,
            vertex_mask,
            Mol_2d_fea.size(1),
            self.hidden_size1,
            self.ligand_unimol_2d_proj,
        )
        if self.ligand_unimol_fusion == "add":
            return Mol_2d_fea + tokens
        return self.ligand_unimol_2d_merge(torch.cat([Mol_2d_fea, tokens], dim=-1))

    def apply_ligand_unimol_fusion_3d(self, Mol_feats, Mol_mask=None, ligand_unimol_tokens=None, ligand_unimol_mask=None):
        # Current ligand Uni-Mol tokens are non-PDB-conformer atom features; keep them out of the true 3D branch.
        return Mol_feats

    def apply_ligand_unimol2_fusion_pre_equiformer(self, Mol_feats, Mol_mask=None, ligand_unimol2_tokens=None, ligand_unimol2_mask=None):
        # Pre-Equiformer Uni-Mol2 fusion. Shape flow:
        #   GerNA ligand atom ids [B, A] -> Embedding(10, 16) -> atom_emb [B, A, 16]
        #   Uni-Mol2 token [B, A, 768] -> Linear(768, 16) -> unimol2_emb [B, A, 16]
        #   add/concat -> fused atom state [B, A, 16], then Equiformer(dim_in=16) sees both chemistry and coordinates.
        if self.ligand_unimol2_3d_fusion == "none" or self.ligand_unimol2_3d_fusion_stage != "pre":
            return Mol_feats
        if ligand_unimol2_tokens is None:
            raise ValueError("Ligand Uni-Mol2 pre-Equiformer fusion is enabled but ligand_unimol2_tokens was not provided")
        if ligand_unimol2_tokens.dim() != 3:
            raise ValueError(f"ligand_unimol2_tokens must have shape [B, A, D], got {tuple(ligand_unimol2_tokens.shape)}")
        if ligand_unimol2_tokens.size(1) != Mol_feats.size(1):
            raise ValueError(
                "Ligand Uni-Mol2 token length must align with GerNA ligand atom features: "
                f"got {ligand_unimol2_tokens.size(1)}, expected {Mol_feats.size(1)}"
            )
        atom_tokens = self.mol_token_emb(Mol_feats.long())
        tokens = self.ligand_unimol2_norm(ligand_unimol2_tokens.float())
        tokens = F.leaky_relu(self.ligand_unimol2_pre_proj(tokens), 0.1)
        if ligand_unimol2_mask is not None:
            tokens = tokens * ligand_unimol2_mask.to(tokens.device).float().unsqueeze(-1)
        if Mol_mask is not None:
            mask = Mol_mask.to(tokens.device).float().unsqueeze(-1)
            tokens = tokens * mask
            atom_tokens = atom_tokens * mask
        if self.ligand_unimol2_3d_fusion == "add":
            return atom_tokens + tokens
        return self.ligand_unimol2_pre_merge(torch.cat([atom_tokens, tokens], dim=-1))

    def apply_ligand_unimol2_fusion_3d(self, Mol_feats, Mol_mask=None, ligand_unimol2_tokens=None, ligand_unimol2_mask=None):
        if self.ligand_unimol2_3d_fusion_stage != "post":
            return Mol_feats
        # Uni-Mol2 tokens were extracted in GerNA ligand atom order, so they can
        # be fused into the ligand 3D branch after Equiformer produces atom states. Shape flow:
        #   ligand 3D input atom ids/coors -> Equiformer -> Mol_feats [B, A, 16]
        #   Uni-Mol2 token [B, A, ligand_unimol2_input_dim=768] -> Linear(768, 16) -> [B, A, 16]
        #   add keeps [B, A, 16]; concat uses [B, A, 32] -> [B, A, 16].
        if self.ligand_unimol2_3d_fusion == "none":
            return Mol_feats
        if ligand_unimol2_tokens is None:
            raise ValueError("Ligand Uni-Mol2 3D fusion is enabled but ligand_unimol2_tokens was not provided")
        if ligand_unimol2_tokens.dim() != 3:
            raise ValueError(f"ligand_unimol2_tokens must have shape [B, A, D], got {tuple(ligand_unimol2_tokens.shape)}")
        if ligand_unimol2_tokens.size(1) != Mol_feats.size(1):
            raise ValueError(
                "Ligand Uni-Mol2 token length must align with GerNA ligand 3D atom features: "
                f"got {ligand_unimol2_tokens.size(1)}, expected {Mol_feats.size(1)}"
            )
        tokens = self.ligand_unimol2_norm(ligand_unimol2_tokens.float())
        tokens = F.leaky_relu(self.ligand_unimol2_3d_proj(tokens), 0.1)
        if ligand_unimol2_mask is not None:
            tokens = tokens * ligand_unimol2_mask.to(tokens.device).float().unsqueeze(-1)
        if Mol_mask is not None:
            tokens = tokens * Mol_mask.to(tokens.device).float().unsqueeze(-1)
        if self.ligand_unimol2_3d_fusion == "add":
            return Mol_feats + tokens
        return self.ligand_unimol2_3d_merge(torch.cat([Mol_feats, tokens], dim=-1))

    def mask_softmax(self, a, mask, dim=-1):
        a_max = torch.max(a, dim, keepdim=True)[0]
        a_exp = torch.exp(a-a_max)
        a_exp = a_exp*mask
        a_softmax = a_exp/(torch.sum(a_exp,dim,keepdim=True)+1e-6)
        return a_softmax
    
    # 关键模块：显式预测 RNA base/token 与 molecule atom 的 pairwise interaction matrix。
    def Pairwise_pred_module(self, batch_size, RNA_1d_fea=None, RNA_2d_fea=None, Mol_2d_fea=None, seq_mask=None, vertex_mask=None, RNA_feats=None, 
                             Mol_feats=None, RNA_C4_coors= None, RNA_coors=None, Mol_coors=None, RNA_mask=None, Mol_mask=None, Mol_LAS_dis=None):

        if self.mol_graph:
            pairwise_mol_2d_fea = F.leaky_relu(self.pairwise_mol_2d(Mol_2d_fea), 0.1)
            if self.rna_repre:
                pairwise_rna_2d_fea = F.leaky_relu(self.pairwise_rna_1d(RNA_1d_fea), 0.1)
                if self.rna_graph:
                    self.fusion_grad_frac = 0.5
                    pairwise_rna_2d_fea = self.fusion_grad_frac * F.leaky_relu(self.pairwise_rna_2d(RNA_2d_fea), 0.1) + (1-self.fusion_grad_frac) * pairwise_rna_2d_fea 
            else:
                pairwise_rna_2d_fea = F.leaky_relu(self.pairwise_rna_2d(RNA_2d_fea), 0.1)
        if self.coors:
            pairwise_rna_3d_fea = F.leaky_relu(self.pairwise_rna_3d( RNA_feats ), 0.1)
            pairwise_mol_3d_fea = F.leaky_relu(self.pairwise_mol_3d( Mol_feats ), 0.1)
        
        #revised in 20240902
        #special_pair_2d = pairwise_rna_2d_fea

        if not self.trigonometry:
            if self.coors: #计算3d_interaction matrix
                pairwise_pred_3d = torch.sigmoid(torch.matmul(pairwise_rna_3d_fea, pairwise_mol_3d_fea.transpose(1,2)))
                pairwise_mask_3d = torch.matmul( RNA_mask.view(batch_size,-1,1), Mol_mask.view(batch_size,1,-1) )
                pairwise_pred_3d = pairwise_pred_3d * pairwise_mask_3d
            if self.mol_graph:
                #2d_interaction_matrix  pairwise_pred_2d
                pairwise_pred_2d = torch.sigmoid(torch.matmul(pairwise_rna_2d_fea, pairwise_mol_2d_fea.transpose(1,2)))
                pairwise_mask_2d = torch.matmul( seq_mask.view(batch_size,-1,1), vertex_mask.view(batch_size,1,-1) )
                pairwise_pred_2d = pairwise_pred_2d * pairwise_mask_2d
                if self.coors:
                    return torch.sigmoid(pairwise_pred_2d), torch.sigmoid(pairwise_pred_3d)
                else:
                    return torch.sigmoid(pairwise_pred_2d)
            else:
                return torch.sigmoid(pairwise_pred_3d)
        # trigonometry=True 时，不直接点积，而是加入 RNA 内部距离、分子内部距离和三角注意力更新。
        else: #The pairwise interaction prediction which have trigonometry-aware constraint.
            if self.mol_graph:
                z_mask_2d = seq_mask.unsqueeze(2) * vertex_mask.unsqueeze(1)
                if self.disable_protein_2d:
                    # No protein 2D graph or CA-CA triangle geometry: predict 2D pairwise from chain ESM 1D and ligand 2D only.
                    mol_out_batched_2d = self.layernorm_2d(Mol_2d_fea)
                    if not self.rna_repre:
                        raise ValueError("disable_protein_2d requires RNA/protein 1D representations")
                    rna_out_batched_2d = self.layernorm_2d(RNA_1d_fea)
                    z_2d = rna_out_batched_2d.unsqueeze(2) * mol_out_batched_2d.unsqueeze(1)
                    pairwise_pred_2d = self.linear_2d(z_2d).squeeze(-1)
                    pairwise_pred_2d = pairwise_pred_2d * z_mask_2d
                else:
                    # shape: RNA_C4_coors [B, Lmax, 3] -> rna_pair_2d [B, Lmax, Lmax, 16]。
                    rna_pair_2d = get_pair_dis_one_hot(RNA_C4_coors, bin_size=2, bin_min=-1, bin_max=30)
                    # shape: Mol_coors [B, Amax, 3] + LAS [B, Amax, Amax] -> mol_pair_2d [B, Amax, Amax, 16]。
                    mol_pair_2d = get_mol_pair_dis_distribution(Mol_coors, Mol_LAS_dis)
                    
                    rna_pair_2d = self.rna_pair_embedding_2d(rna_pair_2d.float())
                    mol_pair_2d = self.mol_pair_embedding_2d(mol_pair_2d.float())  #rna_pair_embedding_2d and mol_pair_embedding_2d need re-define.
                    
                    mol_out_batched_2d = self.layernorm_2d(Mol_2d_fea)
                    
                    if self.rna_repre:
                        rna_out_batched_1d = self.layernorm_2d(RNA_1d_fea)
                        if self.rna_graph:
                            rna_out_batched_2d = self.layernorm_2d(RNA_2d_fea)
                            self.fusion_grad_frac = 0.5
                            rna_out_batched_2d = self.fusion_grad_frac * rna_out_batched_2d + (1-self.fusion_grad_frac) * rna_out_batched_1d
                        else:
                            rna_out_batched_2d = rna_out_batched_1d
                    else:
                        rna_out_batched_2d = self.layernorm_2d(RNA_2d_fea)

                    # z_2d 的形状是 [B, RNA_len, Mol_atoms, hidden]，每个格子代表一个 base-atom pair。
                    z_2d = rna_out_batched_2d.unsqueeze(2) * mol_out_batched_2d.unsqueeze(1)

                    for _ in range(1):
                        for i_module in range(self.n_trigonometry_module_stack):
                            z_2d = z_2d + self.dropout_2d(self.rna_to_mol_list_2d[i_module](z_2d, rna_pair_2d, mol_pair_2d, z_mask_2d.unsqueeze(-1)))
                            z_2d = z_2d + self.dropout_2d(self.triangle_self_attention_list_2d[i_module](z_2d, z_mask_2d))
                            z_2d = self.tranistion_2d(z_2d)
                    # shape: z_2d [B, Lmax, Amax, H1] -> pairwise_pred_2d [B, Lmax, Amax]。
                    pairwise_pred_2d = self.linear_2d(z_2d).squeeze(-1)  #linear 2d also need to re-define
                    pairwise_pred_2d = pairwise_pred_2d * z_mask_2d #[z_mask.long()]

            if self.coors:
                # shape: RNA_coors [B, R3max, 3] -> rna_pair [B, R3max, R3max, 16]。
                rna_pair = get_pair_dis_one_hot(RNA_coors, bin_size=2, bin_min=-1, bin_max=30)
                mol_pair = get_mol_pair_dis_distribution(Mol_coors, Mol_LAS_dis)
                                                   
                rna_pair = self.rna_pair_embedding(rna_pair.float())
                mol_pair = self.mol_pair_embedding(mol_pair.float())
                
                mol_out_batched_3d = self.layernorm_3d(Mol_feats)
                rna_out_batched_3d = self.layernorm_3d(RNA_feats)
                # z of shape, b, protein_length, compound_length, channels.
                # z = torch.einsum("bik,bjk->bijk", rna_out_batched_3d, mol_out_batched_3d)
                # z_mask = torch.einsum("bi,bj->bij", RNA_mask, Mol_mask)

                # 3D 分支同样构造 base/token-atom pair 表征，只是特征来自 Equiformer。
                z = rna_out_batched_3d.unsqueeze(2) * mol_out_batched_3d.unsqueeze(1)
                z_mask = RNA_mask.unsqueeze(2) * Mol_mask.unsqueeze(1)

                for _ in range(1):
                    for i_module in range(self.n_trigonometry_module_stack):
                        z = z + self.dropout_3d(self.rna_to_mol_list_3d[i_module](z, rna_pair, mol_pair, z_mask.unsqueeze(-1)))  #这边用到坐标上可能会报错
                        z = z + self.dropout_3d(self.triangle_self_attention_list_3d[i_module](z, z_mask))
                        z = self.tranistion_3d(z)
                # shape: z [B, R3max, Amax, 16] -> pairwise_pred_3d [B, R3max, Amax]。
                pairwise_pred_3d = self.linear_3d(z).squeeze(-1)
                pairwise_pred_3d = pairwise_pred_3d * z_mask #[z_mask.long()]
                if self.mol_graph:
                    return torch.sigmoid(pairwise_pred_2d), torch.sigmoid(pairwise_pred_3d)
                else:
                    return torch.sigmoid(pairwise_pred_3d)
            else:
                return torch.sigmoid(pairwise_pred_2d)
                                                   

    # Affinity 模块：用 pairwise matrix 作为跨模态注意力，引导 RNA 和分子 summary 向量生成。
    def Affinity_pred_module(self, batch_size, RNA_1d_fea=None, RNA_2d_fea=None, Mol_2d_fea=None, seq_mask=None, vertex_mask=None, RNA_feats=None, Mol_feats=None, RNA_mask=None, Mol_mask=None, pairwise_pred_2d=None, pairwise_pred_3d=None, dti_global=None, pbcnet_global=None, boltz_interface=None ): 
        if self.mol_graph:
            Mol_2d_fea = F.leaky_relu(self.mol_2d_final(Mol_2d_fea), 0.1)
            if self.rna_repre:
                RNA_1d_fea = F.leaky_relu(self.rna_1d_final(RNA_1d_fea), 0.1)
                if self.rna_graph:
                    RNA_2d_fea = F.leaky_relu(self.rna_2d_final(RNA_2d_fea), 0.1)
                    self.fusion_grad_frac = 0.5
                    RNA_2d_fea = self.fusion_grad_frac * RNA_2d_fea + ( 1 - self.fusion_grad_frac ) * RNA_1d_fea
                else:
                    RNA_2d_fea = RNA_1d_fea
            else:
                RNA_2d_fea = F.leaky_relu(self.rna_2d_final(RNA_2d_fea), 0.1)
        if self.coors:
            RNA_feats = F.leaky_relu(self.rna_3d_final(RNA_feats), 0.1)
            Mol_feats = F.leaky_relu(self.mol_3d_final(Mol_feats), 0.1)
        
        if self.coors:
            # shape: DMA_3D 汇聚后 mf_3d/rf_3d 都是 [B, H2]。
            mf_3d, rf_3d = self.dma_gru_3d( batch_size, RNA_feats, Mol_feats, RNA_mask, Mol_mask, pairwise_pred_3d )  # RNA_length * Mol_length
            # shape: 外积 [B, H2, H2] -> flatten kroneck_3d [B, H2*H2]。
            kroneck_3d = F.leaky_relu(torch.matmul(mf_3d.unsqueeze(-1), rf_3d.unsqueeze(-2)).view(-1,self.hidden_size2*self.hidden_size2), 0.1)
        if self.mol_graph:
            # shape: DMA_2D 汇聚后 mf_2d/rf_2d 都是 [B, H2]。
            mf_2d, rf_2d = self.dma_gru_2d( batch_size, RNA_2d_fea, Mol_2d_fea, seq_mask, vertex_mask, pairwise_pred_2d )   # RNA_length * Mol_length
            kroneck_2d = F.leaky_relu(torch.matmul(mf_2d.view(batch_size,-1,1), rf_2d.view(batch_size,1,-1)).view(batch_size,-1), 0.1)
        #print(kroneck_2d.shape, kroneck_3d.shape)
        if self.mol_graph and self.coors:
        # shape: 同时使用 2D+3D 时 kroneck [B, 2*H2*H2]；单分支时 [B, H2*H2]。
            kroneck = torch.cat((kroneck_2d,kroneck_3d),dim=1)
        elif self.mol_graph:
            kroneck = kroneck_2d
        elif self.coors:
            kroneck = kroneck_3d
        kroneck = self.apply_dti_late_fusion(kroneck, dti_global)
        kroneck = self.apply_pbcnet_late_fusion(kroneck, pbcnet_global)
        kroneck = self.apply_boltz_interface_fusion(kroneck, boltz_interface)
        if self.uncertainty:
            # shape: W_out 输出 [B, 2]，两类 Dirichlet alpha/evidence。
            affinity_pred = self.W_out(kroneck)
            #affinity_pred = torch.sigmoid( self.W_out(kroneck) )
        else:
            affinity_pred = torch.sigmoid( self.W_out(kroneck) )
        return affinity_pred

    # DMA/记忆汇聚：pairwise matrix 决定 RNA->分子、分子->RNA 的注意力信息流。
    def dma_gru_2d(self, batch_size, RNA_2d_fea, Mol_2d_fea, seq_mask, vertex_mask, pairwise_pred_2d ):
        
        # shape: 输入 RNA_2d_fea [B, Lmax, H2]、Mol_2d_fea [B, Amax, H2]、pairwise [B, Lmax, Amax]。
        vertex_mask = vertex_mask.view(batch_size,-1,1)
        seq_mask = seq_mask.view(batch_size,-1,1)
        #平均特征
        m0 = torch.sum(Mol_2d_fea * vertex_mask, dim=1) / torch.sum(vertex_mask, dim=1)
        r0 = torch.sum(RNA_2d_fea * seq_mask, dim=1) / torch.sum(seq_mask, dim=1)
        m = m0*r0  #初始全局记忆
        for DMA_iter in range(self.DMA_depth):
            #全局特征
            # shape: pairwise [B, L, A] @ Mol [B, A, H2] -> mol_to_rna [B, L, H2]。
            mol_to_rna = torch.matmul(pairwise_pred_2d, F.tanh(self.m_to_r_transform[DMA_iter](Mol_2d_fea)))  # batch * n_base * hidden
            # shape: pairwise^T [B, A, L] @ RNA [B, L, H2] -> rna_to_mol [B, A, H2]。
            rna_to_mol = torch.matmul(pairwise_pred_2d.transpose(1,2), F.tanh(self.r_to_m_transform[DMA_iter](RNA_2d_fea)))  # batch * n_vertex * hidden
            mol_tmp = F.tanh(self.hc0[DMA_iter](Mol_2d_fea))*F.tanh(self.mc1[DMA_iter](m)).view(batch_size,1,-1)*rna_to_mol #batch * n_vertex *hidden
            rna_tmp = F.tanh(self.hp0[DMA_iter](RNA_2d_fea))*F.tanh(self.mp1[DMA_iter](m)).view(batch_size,1,-1)*mol_to_rna #batch * n_base *hidden
            mol_att = self.mask_softmax(self.hc1[DMA_iter](mol_tmp).view(batch_size,-1), vertex_mask.view(batch_size,-1)) 
            rna_att = self.mask_softmax(self.hp1[DMA_iter](rna_tmp).view(batch_size,-1), seq_mask.view(batch_size,-1))
            mf = torch.sum(Mol_2d_fea*mol_att.view(batch_size,-1,1), dim=1)  #mol_att is processed by softmax
            rf = torch.sum(RNA_2d_fea*rna_att.view(batch_size,-1,1), dim=1)
            m = self.GRU_dma(m, mf*rf)
        return mf, rf
    
    def dma_gru_3d(self, batch_size, RNA_feats, Mol_feats, RNA_mask, Mol_mask, pairwise_pred_3d ):
        vertex_mask = Mol_mask.unsqueeze(-1)#.view(batch_size,-1,1)
        seq_mask = RNA_mask.unsqueeze(-1)#.view(batch_size,-1,1)
        m0 = torch.sum(Mol_feats * vertex_mask, dim=1) / torch.sum(vertex_mask, dim=1)
        r0 = torch.sum(RNA_feats * seq_mask, dim=1) / torch.sum(seq_mask, dim=1)
        m = m0 * r0
        for DMA_iter in range(self.DMA_depth):
            mol_to_rna = torch.matmul(pairwise_pred_3d, F.tanh(self.m_to_r_transform_3d[DMA_iter](Mol_feats)))  # batch * n_residue * hidden
            rna_to_mol = torch.matmul(pairwise_pred_3d.transpose(1,2), F.tanh(self.r_to_m_transform_3d[DMA_iter](RNA_feats)))  # batch * n_vertex * hidden
            
            mol_tmp = F.tanh(self.hc0_3d[DMA_iter](Mol_feats))*F.tanh(self.mc1_3d[DMA_iter](m)).unsqueeze(-2)*rna_to_mol
            rna_tmp = F.tanh(self.hp0_3d[DMA_iter](RNA_feats))*F.tanh(self.mp1_3d[DMA_iter](m)).unsqueeze(-2)*mol_to_rna
            mol_att = self.mask_softmax(self.hc1_3d[DMA_iter](mol_tmp).squeeze(-1), vertex_mask.squeeze(-1)) 
            rna_att = self.mask_softmax(self.hp1_3d[DMA_iter](rna_tmp).squeeze(-1), seq_mask.squeeze(-1))

            mf = torch.sum(Mol_feats*mol_att.unsqueeze(-1), dim=1)
            rf = torch.sum(RNA_feats*rna_att.unsqueeze(-1), dim=1)
            m = self.GRU_dma_3d(m, mf*rf)
        return mf, rf

    # forward 阅读主线：先分别编码 1D/2D/3D 模态，再预测 pairwise matrix，最后做 affinity 汇聚。
    def forward(self, RNA_repre = None, Seq_mask = None, RNA_Graph = None, Mol_Graph = None, RNA_feats = None, RNA_C4_coors = None, RNA_coors = None, RNA_mask = None, Mol_feats = None, Mol_coors = None, Mol_mask = None, Mol_LAS_dis=None, DTI_protein_tokens=None, DTI_protein_mask=None, DTI_global=None, Ligand_unimol_tokens=None, Ligand_unimol_mask=None, Ligand_unimol2_tokens=None, Ligand_unimol2_mask=None, PBCNet_global=None, PBCNet_graph=None, Boltz_interface=None): #改动，增加Las_dis，即几何距离约束
        # 2D 图编码后用 to_dense_batch 从 PyG 稀疏节点表征变回 [B, L, C]。
        if self.rna_graph:
            RNA_2d_fea = self.GCN_rna( RNA_Graph )
            # shape: GNN 输出从 [sum_L, H1] -> dense RNA_2d_fea [B, Lmax, H1]，seq_mask [B, Lmax]。
            RNA_2d_fea, seq_mask = to_dense_batch(RNA_2d_fea, RNA_Graph.batch)
            seq_mask = seq_mask.float().to(RNA_2d_fea.device)#cuda(5)
        if self.mol_graph:
            Mol_2d_fea = self.GCN_mol( Mol_Graph )
            # shape: GNN 输出从 [sum_A, H1] -> dense Mol_2d_fea [B, Amax, H1]，vertex_mask [B, Amax]。
            Mol_2d_fea, vertex_mask = to_dense_batch(Mol_2d_fea, Mol_Graph.batch)
            vertex_mask = vertex_mask.float().to(Mol_2d_fea.device)
            Mol_2d_fea = self.apply_ligand_unimol_fusion_2d(Mol_2d_fea, vertex_mask, Ligand_unimol_tokens, Ligand_unimol_mask)
        if self.rna_repre:
            # shape: RNA_repre [B, Lmax, 640] -> RNA_1d_fea [B, Lmax, H1]。
            RNA_1d_fea = self.mlp_rna( RNA_repre )
            seq_mask = Seq_mask.float().to(RNA_1d_fea.device)
            RNA_1d_fea = self.apply_dti_front_fusion(RNA_1d_fea, seq_mask, DTI_protein_tokens, DTI_protein_mask)
        # Equiformer 返回多阶特征，这里只取 type0 标量特征进入后续 pairwise/DMA。
        if self.coors:
            if self.ligand_unimol2_3d_fusion != "none" and self.ligand_unimol2_3d_fusion_stage == "pre":
                # shape: atom ids [B, Amax] + Uni-Mol2 [B, Amax, 768] -> fused continuous input [B, Amax, 16] -> Equiformer.
                Mol_feats = self.apply_ligand_unimol2_fusion_pre_equiformer(Mol_feats, Mol_mask, Ligand_unimol2_tokens, Ligand_unimol2_mask)
                Mol_feats = self.equi_mol(Mol_feats.float(), Mol_coors.float(), Mol_mask).type0
            else:
                # shape: Mol_feats token [B, Amax] + coords [B, Amax, 3] -> [B, Amax, 16]。
                Mol_feats = self.equi_mol(Mol_feats.long(), Mol_coors.float(), Mol_mask).type0  # batch * max(length) * 16 (mini_size for equibind)
                Mol_feats = self.apply_ligand_unimol2_fusion_3d(Mol_feats, Mol_mask, Ligand_unimol2_tokens, Ligand_unimol2_mask)
            # shape: RNA_feats token [B, R3max] or continuous [B, R3max, Din] + coords -> [B, R3max, 16].
            rna_continuous_3d = self.rna_3d_input_norm is not None
            if rna_continuous_3d:
                RNA_feats = self.rna_3d_input_norm(RNA_feats.float())
            RNA_feats = self.equi_rna(RNA_feats.float() if rna_continuous_3d else RNA_feats.long(), RNA_coors.float(), RNA_mask).type0

            RNA_mask = RNA_mask.float()
            Mol_mask = Mol_mask.float()
        if self.pbcnet_trainable:
            pbcnet_output_device = RNA_feats.device if self.coors else RNA_1d_fea.device
            PBCNet_global = self.compute_trainable_pbcnet_global(PBCNet_graph, output_device=pbcnet_output_device)
        #print(RNA_1d_fea.shape, RNA_feats.shape, Mol_feats.shape )
        assert self.mol_graph or self.coors
        if self.mol_graph:
            batch_size = seq_mask.size(0)
        elif self.coors:
            batch_size = RNA_feats.size(0)
                                   
        if self.mol_graph:
            if self.coors:
                if self.rna_graph:
                    if self.rna_repre:
                        pairwise_pred_2d, pairwise_pred_3d = self.Pairwise_pred_module( batch_size, RNA_1d_fea, RNA_2d_fea, Mol_2d_fea, seq_mask, vertex_mask, RNA_feats, Mol_feats, RNA_C4_coors, RNA_coors, Mol_coors, RNA_mask, Mol_mask, Mol_LAS_dis)
                    else:
                        pairwise_pred_2d, pairwise_pred_3d = self.Pairwise_pred_module( batch_size, None, RNA_2d_fea, Mol_2d_fea, seq_mask, vertex_mask, RNA_feats, Mol_feats, RNA_C4_coors, RNA_coors, Mol_coors, RNA_mask, Mol_mask, Mol_LAS_dis)
                else:
                    pairwise_pred_2d, pairwise_pred_3d = self.Pairwise_pred_module( batch_size, RNA_1d_fea, None, Mol_2d_fea, seq_mask, vertex_mask, RNA_feats, Mol_feats, RNA_C4_coors, RNA_coors, Mol_coors, RNA_mask, Mol_mask, Mol_LAS_dis)
            else:
                if self.rna_graph:
                    if self.rna_repre:
                        pairwise_pred_2d = self.Pairwise_pred_module( batch_size, RNA_1d_fea, RNA_2d_fea, Mol_2d_fea, seq_mask, vertex_mask, None, None, RNA_C4_coors, None, Mol_coors, None, None, Mol_LAS_dis)
                    else:
                        pairwise_pred_2d = self.Pairwise_pred_module( batch_size, None, RNA_2d_fea, Mol_2d_fea, seq_mask, vertex_mask, None, None, RNA_C4_coors, None, Mol_coors, None, None, Mol_LAS_dis)
                else: 
                    pairwise_pred_2d = self.Pairwise_pred_module( batch_size, RNA_1d_fea, None, Mol_2d_fea, seq_mask, vertex_mask, None, None, RNA_C4_coors, None, Mol_coors, None, None, Mol_LAS_dis)
        else:
            pairwise_pred_3d = self.Pairwise_pred_module( batch_size, None, None, None, None, None, RNA_feats, Mol_feats, RNA_C4_coors, RNA_coors, Mol_coors, RNA_mask, Mol_mask, Mol_LAS_dis)
        
        if self.mol_graph:
            if self.coors:
                if self.rna_repre:
                    if self.rna_graph:
                        affinity_pred = self.Affinity_pred_module( batch_size, RNA_1d_fea, RNA_2d_fea, Mol_2d_fea, seq_mask, vertex_mask, RNA_feats, Mol_feats, RNA_mask, Mol_mask, pairwise_pred_2d, pairwise_pred_3d, DTI_global, PBCNet_global, Boltz_interface )
                    else:
                        affinity_pred = self.Affinity_pred_module( batch_size, RNA_1d_fea, None, Mol_2d_fea, seq_mask, vertex_mask, RNA_feats, Mol_feats, RNA_mask, Mol_mask, pairwise_pred_2d, pairwise_pred_3d, DTI_global, PBCNet_global, Boltz_interface )
                else:
                    affinity_pred = self.Affinity_pred_module( batch_size, None, RNA_2d_fea, Mol_2d_fea, seq_mask, vertex_mask, RNA_feats, Mol_feats, RNA_mask, Mol_mask, pairwise_pred_2d, pairwise_pred_3d, DTI_global, PBCNet_global, Boltz_interface )
            else:
                if self.rna_repre:
                    if self.rna_graph:
                        affinity_pred = self.Affinity_pred_module( batch_size, RNA_1d_fea, RNA_2d_fea, Mol_2d_fea, seq_mask, vertex_mask, None, None, None, None, pairwise_pred_2d, None, DTI_global, PBCNet_global, Boltz_interface )
                    else:
                        affinity_pred = self.Affinity_pred_module( batch_size, RNA_1d_fea, None, Mol_2d_fea, seq_mask, vertex_mask, None, None, None, None, pairwise_pred_2d, None, DTI_global, PBCNet_global, Boltz_interface )
                else:
                    affinity_pred = self.Affinity_pred_module( batch_size, None, RNA_2d_fea, Mol_2d_fea, seq_mask, vertex_mask, None, None, None, None, pairwise_pred_2d, None, DTI_global, PBCNet_global, Boltz_interface )
        else:
            affinity_pred = self.Affinity_pred_module( batch_size, None, None, None, None, None, RNA_feats, Mol_feats, RNA_mask, Mol_mask, None, pairwise_pred_3d, DTI_global, PBCNet_global, Boltz_interface )
            return affinity_pred, pairwise_pred_3d

        # 注意完整 2D+3D 模式下这里只返回 2D pairwise；3D pairwise 只参与 affinity，不暴露给推理脚本。
        return affinity_pred, pairwise_pred_2d

