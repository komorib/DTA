import pickle
import json
import torch
from torch.utils import data
from torch_geometric.data import Batch
import numpy as np
import dgl


import numpy as np
import sys
import os

from torch_geometric.data import Data
from torch.autograd import Variable
import torch.optim as optim
import random
from utils.net_utils import *
from utils.metrics import *

torch.multiprocessing.set_sharing_strategy('file_system')


# 训练数据 pkl 比推理 pkl 多最后一项 Label_list；字段顺序是整个项目的数据契约。
class GerNA_dataset(data.Dataset):
    def __init__(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            self.data = pickle.load(f)
        self.RNA_repre, self.Mol_graph, self.RNA_Graph, self.RNA_feats, self.RNA_C4_coors, self.RNA_coors, self.Mol_feats, self.Mol_coors, self.LAS_input, self.Label_list = self.data
        
    def __len__(self):
        return len(self.RNA_repre)

    def __getitem__(self, index):
        return self.RNA_repre[index], self.Mol_graph[index], self.RNA_Graph[index], self.RNA_feats[index], self.RNA_C4_coors[index], self.RNA_coors[index], self.Mol_feats[index], self.Mol_coors[index], self.LAS_input[index], self.Label_list[index]


class GerNAChunkedDataset(data.Dataset):
    def __init__(self, manifest_path):
        self.manifest_path = manifest_path
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
        checkpoint_dir = manifest['checkpoint_dir']
        if not os.path.isabs(checkpoint_dir):
            checkpoint_dir = os.path.join(manifest_dir, checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir
        self.samples = manifest['samples']
        self._cached_chunk_path = None
        self._cached_data = None

    def __len__(self):
        return len(self.samples)

    def _load_chunk(self, chunk_name):
        chunk_path = chunk_name
        if not os.path.isabs(chunk_path):
            chunk_path = os.path.join(self.checkpoint_dir, chunk_name)
        if chunk_path != self._cached_chunk_path:
            with open(chunk_path, 'rb') as f:
                payload = pickle.load(f)
            self._cached_chunk_path = chunk_path
            self._cached_data = payload['data']
        return self._cached_data

    def __getitem__(self, index):
        sample = self.samples[index]
        chunk_data = self._load_chunk(sample['chunk'])
        local_index = sample['local_index']
        return tuple(field[local_index] for field in chunk_data)


# 这里把变长 RNA/分子样本 padding 成 dense tensor，同时保留 PyG 图 batch 和 mask。
def _pad_dti_protein_tokens(dti_features):
    arrays = [feature.get("protein_tokens") for feature in dti_features]
    if all(array is None for array in arrays):
        return None, None
    if any(array is None for array in arrays):
        raise ValueError("Every DTI fusion payload in a batch must contain protein_tokens when one sample has it")
    arrays = [np.asarray(array, dtype=np.float32) for array in arrays]
    dims = {array.shape[1] for array in arrays if array.ndim == 2}
    if len(dims) != 1 or any(array.ndim != 2 for array in arrays):
        raise ValueError("DTI protein_tokens must have consistent shape [L, D] inside a batch")
    masks = []
    for feature, array in zip(dti_features, arrays):
        mask = feature.get("protein_mask")
        if mask is None:
            mask = np.ones((array.shape[0],), dtype=bool)
        mask = np.asarray(mask).astype(bool).reshape(-1)
        if mask.shape[0] != array.shape[0]:
            raise ValueError("DTI protein_mask length must match protein_tokens length")
        masks.append(mask)
    return torch.tensor(np.array(pack2D(arrays))).float(), torch.tensor(np.array(pack1D(masks))).bool()


def _stack_dti_global(dti_features):
    arrays = [feature.get("global") for feature in dti_features]
    if all(array is None for array in arrays):
        return None
    if any(array is None for array in arrays):
        raise ValueError("Every DTI fusion payload in a batch must contain global when one sample has it")
    arrays = [np.asarray(array, dtype=np.float32).reshape(-1) for array in arrays]
    dims = {array.shape[0] for array in arrays}
    if len(dims) != 1:
        raise ValueError("DTI global features must have the same dimension inside a batch")
    return torch.tensor(np.stack(arrays, axis=0)).float()



def _stack_pbcnet_global(pbcnet_features):
    arrays = [feature.get("global") for feature in pbcnet_features]
    if any(array is None for array in arrays):
        raise ValueError("Every PBCNet fusion payload in a batch must contain global")
    arrays = [np.asarray(array, dtype=np.float32).reshape(-1) for array in arrays]
    dims = {array.shape[0] for array in arrays}
    if len(dims) != 1:
        raise ValueError("PBCNet global features must have the same dimension inside a batch")
    return torch.tensor(np.stack(arrays, axis=0)).float()


def _batch_pbcnet_graph(pbcnet_features):
    graphs = [feature.get("graph") for feature in pbcnet_features]
    if any(graph is None for graph in graphs):
        raise ValueError("Every trainable PBCNet payload in a batch must contain graph")
    return dgl.batch(graphs)


def _pad_ligand_unimol_tokens(ligand_features):
    # Optional ligand pretrain features stay atom-level here. Padding only makes
    # a dense [B, Amax, D] tensor; the actual fusion branch is selected in model.py.
    arrays = [feature.get("ligand_tokens") for feature in ligand_features]
    if all(array is None for array in arrays):
        return None, None
    if any(array is None for array in arrays):
        raise ValueError("Every ligand Uni-Mol payload in a batch must contain ligand_tokens when one sample has it")
    arrays = [np.asarray(array, dtype=np.float32) for array in arrays]
    dims = {array.shape[1] for array in arrays if array.ndim == 2}
    if len(dims) != 1 or any(array.ndim != 2 for array in arrays):
        raise ValueError("Ligand Uni-Mol tokens must have consistent shape [A, D] inside a batch")
    masks = []
    for feature, array in zip(ligand_features, arrays):
        mask = feature.get("ligand_mask")
        if mask is None:
            mask = np.ones((array.shape[0],), dtype=bool)
        mask = np.asarray(mask).astype(bool).reshape(-1)
        if mask.shape[0] != array.shape[0]:
            raise ValueError("Ligand Uni-Mol mask length must match ligand_tokens length")
        masks.append(mask)
    return torch.tensor(np.array(pack2D(arrays))).float(), torch.tensor(np.array(pack1D(masks))).bool()

def _pad_boltz_interface_features(features):
    pocket = [np.asarray(feature.get("pocket_emb"), dtype=np.float32) for feature in features]
    ligand = [np.asarray(feature.get("ligand_emb"), dtype=np.float32) for feature in features]
    global_feat = [np.asarray(feature.get("global_feat"), dtype=np.float32).reshape(-1) for feature in features]
    if any(value.ndim != 2 for value in pocket + ligand):
        raise ValueError("Boltz interface pocket_emb and ligand_emb must have shape [N, D]")
    dims = {value.shape[1] for value in pocket + ligand}
    if len(dims) != 1:
        raise ValueError("Boltz interface token embedding dimensions must match inside a batch")
    global_dims = {value.shape[0] for value in global_feat}
    if len(global_dims) != 1 or next(iter(global_dims)) != next(iter(dims)) * 2:
        raise ValueError("Boltz interface global features must have dimension 2 * token embedding dim")
    return {
        "pocket_emb": torch.tensor(np.array(pack2D(pocket))).float(),
        "pocket_mask": torch.tensor(np.array(get_mask(pocket))).bool(),
        "ligand_emb": torch.tensor(np.array(pack2D(ligand))).float(),
        "ligand_mask": torch.tensor(np.array(get_mask(ligand))).bool(),
        "global_feat": torch.tensor(np.stack(global_feat, axis=0)).float(),
    }



def custom_collate_fn(batch):
    batch_RNA_repre = []
    batch_seq_mask = []
    batch_Mol_Graph = []
    batch_RNA_Graph = []
    batch_RNA_feats = []
    batch_RNA_C4_coors = []
    batch_RNA_coors = []
    batch_RNA_mask = []
    batch_Mol_feats = []
    batch_Mol_coors = []
    batch_Mol_mask = []
    batch_Mol_LAS = []
    batch_label = []
    batch_dti_features = []
    batch_ligand_unimol_features = []
    batch_ligand_unimol2_features = []
    batch_pbcnet_features = []
    batch_boltz_interface_features = []
    has_dti_features = False
    has_ligand_unimol_features = False
    has_ligand_unimol2_features = False
    has_pbcnet_features = False
    has_boltz_interface_features = False

    parsed_batch = []
    # Dataset items have 10 base fields plus optional dict payloads. Base shapes after padding:
    #   RNA_repre [B, Lmax, 640], Mol_feats [B, Amax], Mol_coors [B, Amax, 3], label [B]
    # Optional payload shapes before padding are DTI [L, 256]/global [512], old Uni-Mol [A, 512], Uni-Mol2 [A, 768].
    # Classify dicts by keys so optional features can be combined without relying on fragile positional order.
    for item in batch:
        if len(item) < 10:
            raise ValueError(f"Unexpected dataset item length: {len(item)}")
        base_item = item[:10]
        dti_features = None
        ligand_unimol_features = None
        ligand_unimol2_features = None
        pbcnet_features = None
        boltz_interface_features = None
        for extra in item[10:]:
            if extra is None:
                continue
            if not isinstance(extra, dict):
                raise ValueError(f"Optional dataset payload must be a dict, got {type(extra)}")
            payload_type = extra.get("payload_type")
            if payload_type == "boltz_interface":
                boltz_interface_features = extra
                has_boltz_interface_features = True
            elif payload_type in {"pbcnet", "pbcnet_graph"}:
                pbcnet_features = extra
                has_pbcnet_features = True
            elif payload_type == "ligand_unimol2" or "ligand_unimol2_tokens" in extra:
                ligand_unimol2_features = extra
                has_ligand_unimol2_features = True
            elif "ligand_tokens" in extra or "ligand_mask" in extra:
                ligand_unimol_features = extra
                has_ligand_unimol_features = True
            else:
                dti_features = extra
                has_dti_features = True
        parsed_batch.append((base_item, dti_features, ligand_unimol_features, ligand_unimol2_features, pbcnet_features, boltz_interface_features))

    for i,(base_item,dti_features,ligand_unimol_features,ligand_unimol2_features,pbcnet_features,boltz_interface_features) in enumerate(parsed_batch):
        if has_dti_features and dti_features is None:
            raise ValueError("Cannot mix samples with and without DTI fusion features in one batch")
        if has_ligand_unimol_features and ligand_unimol_features is None:
            raise ValueError("Cannot mix samples with and without ligand Uni-Mol features in one batch")
        if has_ligand_unimol2_features and ligand_unimol2_features is None:
            raise ValueError("Cannot mix samples with and without ligand Uni-Mol2 features in one batch")
        if has_pbcnet_features and pbcnet_features is None:
            raise ValueError("Cannot mix samples with and without PBCNet fusion features in one batch")
        if has_boltz_interface_features and boltz_interface_features is None:
            raise ValueError("Cannot mix samples with and without Boltz interface features in one batch")
        rna_repre,mol_graph,rna_graph,rna_feats,rna_coors_C4,rna_coors,mol_feats,mol_coors,mol_las,label = base_item
        
        batch_RNA_repre.append(rna_repre)
        batch_Mol_Graph.append(mol_graph)
        batch_RNA_Graph.append(rna_graph)
        
        batch_RNA_feats.append(rna_feats)
        batch_RNA_C4_coors.append(rna_coors_C4)
        batch_RNA_coors.append(rna_coors)

        batch_Mol_feats.append(mol_feats)
        batch_Mol_coors.append(mol_coors)

        batch_Mol_LAS.append(mol_las)
        batch_label.append(label)
        if has_dti_features:
            batch_dti_features.append(dti_features)
        if has_ligand_unimol_features:
            batch_ligand_unimol_features.append(ligand_unimol_features)
        if has_ligand_unimol2_features:
            batch_ligand_unimol2_features.append(ligand_unimol2_features)
        if has_pbcnet_features:
            batch_pbcnet_features.append(pbcnet_features)
        if has_boltz_interface_features:
            batch_boltz_interface_features.append(boltz_interface_features)
    
    # RNA/分子的长度不同，pack* 负责补零，mask 负责告诉模型哪些位置是真 token。
    # shape after collate: RNA_repre [B, Lmax, 640], seq_mask [B, Lmax]。
    batch_seq_mask = torch.tensor( np.array(get_mask(batch_RNA_repre))).float()#.to(device)
    batch_RNA_repre = torch.tensor( np.array(pack2D(batch_RNA_repre))).float()#.to(device)
    
    # shape after PyG batch: graph.x 是所有节点拼接 [sum_nodes, feat_dim]，batch 向量记录节点属于哪个样本。
    batch_Mol_Graph = Batch.from_data_list(batch_Mol_Graph)#.to(device)
    batch_RNA_Graph = Batch.from_data_list(batch_RNA_Graph)#.to(device)
    
    # shape after collate: RNA_feats [B, R3max], RNA_coors [B, R3max, 3], RNA_mask [B, R3max]。
    batch_RNA_mask = torch.tensor( np.array(get_mask(batch_RNA_feats))).bool()#.to(device)
    batch_Mol_mask = torch.tensor( np.array(get_mask(batch_Mol_feats))).bool()#.to(device)
    
    first_rna_feats = batch_RNA_feats[0]
    if not isinstance(first_rna_feats, list) and len(np.asarray(first_rna_feats).shape) == 2:
        batch_RNA_feats = torch.tensor(np.array(pack2D(batch_RNA_feats))).float()
    else:
        batch_RNA_feats = torch.tensor(np.array(pack1D(batch_RNA_feats))).long()#.to(device)
    batch_RNA_C4_coors = torch.tensor( np.array(pack2D(batch_RNA_C4_coors)) ).float()
    batch_RNA_coors = torch.tensor( np.array(pack2D(batch_RNA_coors))).float()#.to(device)
    
    # shape after collate: Mol_feats [B, Amax], Mol_coors [B, Amax, 3], Mol_mask [B, Amax], Mol_LAS [B, Amax, Amax]。
    batch_Mol_feats = torch.tensor( np.array(pack1D(batch_Mol_feats))).long()#.to(device)
    batch_Mol_coors = torch.tensor( np.array(pack2D(batch_Mol_coors))).float()#.to(device)
    
    batch_Mol_LAS = torch.tensor( np.array(pack2D(batch_Mol_LAS))).float()#.to(device)
    batch_label = torch.FloatTensor( np.array(batch_label) )#.to(device)

    output = (
        batch_RNA_repre,
        batch_seq_mask,
        batch_Mol_Graph,
        batch_RNA_Graph,
        batch_RNA_feats,
        batch_RNA_C4_coors,
        batch_RNA_coors,
        batch_RNA_mask,
        batch_Mol_feats,
        batch_Mol_coors,
        batch_Mol_mask,
        batch_Mol_LAS,
        batch_label,
    )
    if has_dti_features and not has_ligand_unimol2_features and not has_pbcnet_features:
        # Return layout with DTI only: base 13 fields + protein tokens/mask + global.
        # Shapes: dti_protein_tokens [B, Lmax, 256], dti_protein_mask [B, Lmax], dti_global [B, 512].
        dti_protein_tokens, dti_protein_mask = _pad_dti_protein_tokens(batch_dti_features)
        dti_global = _stack_dti_global(batch_dti_features)
        output = output + (dti_protein_tokens, dti_protein_mask, dti_global)
    if has_ligand_unimol2_features or has_pbcnet_features:
        # Full optional layout: base fields, DTI fields, old ligand Uni-Mol fields,
        # Uni-Mol2 fields, then PBCNet global. train_dta.unpack_batch_for_device mirrors this order.
        # Shapes: old Uni-Mol [B, Amax, 512], Uni-Mol2 [B, Amax, 768], PBCNet global [B, D] or trainable DGL graph.
        dti_protein_tokens = dti_protein_mask = dti_global = None
        ligand_unimol_tokens = ligand_unimol_mask = None
        if has_dti_features:
            dti_protein_tokens, dti_protein_mask = _pad_dti_protein_tokens(batch_dti_features)
            dti_global = _stack_dti_global(batch_dti_features)
        if has_ligand_unimol_features:
            ligand_unimol_tokens, ligand_unimol_mask = _pad_ligand_unimol_tokens(batch_ligand_unimol_features)
        ligand_unimol2_tokens = ligand_unimol2_mask = None
        if has_ligand_unimol2_features:
            ligand_unimol2_tokens, ligand_unimol2_mask = _pad_ligand_unimol_tokens(batch_ligand_unimol2_features)
        pbcnet_global = None
        pbcnet_graph = None
        if has_pbcnet_features:
            pbcnet_payload_types = {feature.get("payload_type") for feature in batch_pbcnet_features}
            if pbcnet_payload_types == {"pbcnet_graph"}:
                pbcnet_graph = _batch_pbcnet_graph(batch_pbcnet_features)
            elif pbcnet_payload_types == {"pbcnet"}:
                pbcnet_global = _stack_pbcnet_global(batch_pbcnet_features)
            else:
                raise ValueError("Cannot mix PBCNet global and trainable graph payloads in one batch")
        output = output + (
            dti_protein_tokens, dti_protein_mask, dti_global,
            ligand_unimol_tokens, ligand_unimol_mask,
            ligand_unimol2_tokens, ligand_unimol2_mask,
            pbcnet_global, pbcnet_graph,
        )
    elif has_ligand_unimol_features:
        ligand_unimol_tokens, ligand_unimol_mask = _pad_ligand_unimol_tokens(batch_ligand_unimol_features)
        output = output + (ligand_unimol_tokens, ligand_unimol_mask)

    if has_boltz_interface_features:
        output = output + (_pad_boltz_interface_features(batch_boltz_interface_features),)

    # 返回顺序要和 train_model.py 里的 batch 解包、net/model.py.forward 参数顺序对应。
    return output
