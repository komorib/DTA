import torch
import fm
import subprocess
import argparse
import os
import sys
#sys.path.append('../')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import pandas as pd
import numpy as np
import pickle
import os

import torch
import utils.features as fea
from torch_geometric.data import Data
from Bio.PDB import PDBParser
import json
from rdkit import Chem
from rdkit.Chem import AllChem
import scipy

from utils.tankbind_feature_utils import extract_torchdrug_feature_from_mol,get_LAS_distance_constraint_mask
atom_id = {'C': 1, 'N': 2, 'Br': 3, 'O': 4, 'S': 5, 'Cl': 6, 'F': 7, 'P': 8, 'I': 9, 'As': 9, 'K': 9, 'Na': 9, 'Ge': 9, 'Se': 9, 'Au': 9, 'Pt': 9, 'Sr': 9, 'Mg': 9, 'B': 9, 'Co': 9,'Ca':9,'Rh':9, 'unknown': 9}


# 论文中的 1D RNA sequence representation：用 RNA-FM 生成每个碱基的 embedding。
def get_rna_embeddings(rna_sequence):
    """
    Generate embedding features for a given RNA sequence using the RNA-FM model.

    Args:
        rna_sequence (str): RNA sequence to process.

    Returns:
        torch.Tensor: Embedding features of shape (sequence_length, embedding_dim).
    """
    # Load RNA-FM model and related utilities
    model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # Disable dropout for deterministic results

    # Prepare data for the model
    data = [("RNA", rna_sequence)]
    _, _, batch_tokens = batch_converter(data)

    # Extract embeddings (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12])

    # Get token embeddings for the 12th layer
    token_embeddings = results["representations"][12]
    
    return token_embeddings[0][1:-1]

# 论文中的 2D RNA secondary structure：调用 RNAfold 得到 dot-bracket 结构。
def get_secondary_structure(fasta_file):
    result = subprocess.run(['RNAfold', '-i', fasta_file], capture_output=True, text=True)
    output = result.stdout.split('\n')
    if len(output) > 1:
        secondary_structure = output[2].split(' ')[0]
        return secondary_structure
    return None

# 分子 3D conformer 由 RDKit 生成；失败时退化到 2D 坐标，所以推理数据不一定全是真 3D。
def inner_smi2coords(smi, seed=42, mode='fast', remove_hs=True):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    assert len(atoms)>0, 'No atoms in molecule: {}'.format(smi)
    try:
        # will random generate conformer with seed equal to -1. else fixed random seed.
        res = AllChem.EmbedMolecule(mol, maxAttempts=5000, randomSeed=-1)
        if res == 0:
            try:
                # some conformer can not use MMFF optimize
                AllChem.MMFFOptimizeMolecule(mol)
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            except:
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
        ## for fast test... ignore this ###
        elif res == -1 and mode == 'heavy':
            AllChem.EmbedMolecule(mol, maxAttempts=5000, randomSeed=seed)
            try:
                # some conformer can not use MMFF optimize
                AllChem.MMFFOptimizeMolecule(mol)
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            except:
                AllChem.Compute2DCoords(mol)
                coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
                coordinates = coordinates_2d
        else:
            AllChem.Compute2DCoords(mol)
            coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
            coordinates = coordinates_2d
    except:
        print("Failed to generate conformer, replace with zeros.")
        coordinates = np.zeros((len(atoms),3))
    assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smi)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(coordinates_no_h), "coordinates shape is not align with {}".format(smi)
        return atoms_no_h, coordinates_no_h
    else:
        return atoms, coordinates
    
def extract_atom_coordinates(pdb_file, atoms_to_extract):
    #atoms_to_extract = ['C4\'', 'N1', 'P']
    #atoms_to_extract = ['C4\'']
    
    RNA_atom_id = { atoms_to_extract[i]:i+1 for i in range(len(atoms_to_extract))}
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)

    coordinates_list = []
    atom_list = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ':
                    atoms = {atom.name: atom for atom in residue.get_atoms()}
                    coordinates = [atoms[atom_name].get_coord() for atom_name in atoms if atom_name in atoms_to_extract]
                    atom_name = [atoms[atom_name].name for atom_name in atoms if atom_name in atoms_to_extract]
                    atom_list.extend( [RNA_atom_id[i] for i in atom_name ] )
                    coordinates_list.extend(coordinates)
    return np.array(atom_list),np.array(coordinates_list)

# dot-bracket 结构会被转成 PyG 边：括号配对边 + backbone 相邻边。
def parse_rna_structure(structure):
    edges = []
    stack = []
    for i, symbol in enumerate(structure):
        if symbol == '(':
            stack.append(i)
        elif symbol == ')':
            if stack:
                opening_bracket_index = stack.pop()
                edges.append((opening_bracket_index, i))
                edges.append((i, opening_bracket_index))
    
    for i in range(len(structure)-1):
        edges.append( (i, i+1) )
        edges.append( (i+1, i) )
    return edges

def process_fasta(fasta_file):
    with open(fasta_file, 'r') as f:
        sequences = f.read().split('>')[1:]
        sequences = [seq.split('\n', 1)[1].replace('\n', '') for seq in sequences]
    return sequences

def process_smiles(smiles_file):
    with open(smiles_file, 'r') as f:
        smiles = f.read().splitlines()
    return smiles

# def main():
#     parser = argparse.ArgumentParser(description='Process RNA and mol files')
#     parser.add_argument('--fasta', type=str, required=True, help='Path to the RNA fasta file')
#     parser.add_argument('--smiles', type=str, required=True, help='Path to the mol txt file')
#     parser.add_argument('--RhoFold_path', type=str, default="/home/xiayp/Workspace/RhoFold", required=False, help='Path to the RhoFold project')
#     parser.add_argument('--RhoFold_weight', type=str, default="/home/xiayp/Workspace/RhoFold/pretrained/model_20221010_params.pt", required=False, help='Path to the RhoFold weights')
#     args = parser.parse_args()

#     rna_seq = process_fasta(args.fasta)[0]
#     RNA_embedding = get_rna_embeddings(rna_seq)
#     secondary_structure = get_secondary_structure(args.fasta)
#     edge_index = torch.tensor(np.array(parse_rna_structure(secondary_structure) ).T, dtype=torch.long)
#     RNA_data = Data(x=RNA_embedding, edge_index=edge_index)

#     command = [
#         "python",
#         args.RhoFold_path + "/inference.py",
#         "--input_fas", args.fasta,
#         "--single_seq_pred", "True",
#         "--output_dir", "./data/tmp",
#         "--ckpt", args.RhoFold_weight
#     ]
#     subprocess.run(command, check=True, capture_output=True, text=True)

#     pdb_file_path = "data/tmp/relaxed_1000_model.pdb"  # 替换为你的 PDB 文件路径
#     atoms_to_extract = ['C4\'', 'N1', 'P']
#     RNA_3_coords = extract_atom_coordinates(pdb_file_path,atoms_to_extract)
#     atoms_to_extract = ['C4\'']
#     RNA_C4_coords = extract_atom_coordinates(pdb_file_path,atoms_to_extract)
    

#     smile_list = process_smiles(args.smiles)
#     Smile_coor_input = {}
#     for i in range(len(smile_list)):
#         atoms, coords = inner_smi2coords(smile_list[i])
#         atoms_id = [ atom_id[atom] for atom in atoms]
#         Smile_coor_input[smile_list[i]] = (atoms_id, coords)

#     RNA_repre = []
#     RNA_Graph = []
#     RNA_C4_coors = []
#     RNA_coors = []
#     RNA_feats = []

#     Mol_graph = []
#     LAS_input = []
#     Mol_coors = []
#     Mol_feats = []

#     for j in range(len(smile_list)):
#         RNA_repre.append( RNA_embedding )
#         RNA_Graph.append( RNA_data )
#         RNA_C4_coors.append( RNA_C4_coords[1] )
#         RNA_coors.append( RNA_3_coords[1] )
#         RNA_feats.append( RNA_3_coords[0] )
        
#         Mol_graph.append( fea.simplify_atom_to_graph_data_obj( Chem.MolFromSmiles(smile_list[j] ) ) )
#         LAS_input.append( get_LAS_distance_constraint_mask(Chem.MolFromSmiles(smile_list[j])) )
#         Mol_coors.append( np.array(Smile_coor_input[smile_list[j]][1]) )
#         Mol_feats.append( Smile_coor_input[smile_list[j]][0] )

#     data = [RNA_repre,Mol_graph,RNA_Graph,RNA_feats,RNA_C4_coors,RNA_coors,Mol_feats,Mol_coors,LAS_input]
#     with open('data/new_data.pkl', 'wb') as file:
#         pickle.dump(data, file)

# if __name__ == '__main__':
#     main()

# 主流程：FASTA/SMILES -> RNA-FM/RNAfold/RhoFold/RDKit 特征 -> data/new_data.pkl。
def main():
    parser = argparse.ArgumentParser(description='Process RNA and mol files')
    parser.add_argument('--fasta', type=str, required=True, help='Path to the RNA fasta file')
    parser.add_argument('--smiles', type=str, required=True, help='Path to the mol txt file')
    parser.add_argument('--RhoFold_path', type=str, default="/xcfhome/ypxia/github/RhoFold", required=False)
    parser.add_argument('--RhoFold_weight', type=str, default="/xcfhome/ypxia/github/RhoFold/pretrained/RhoFold_pretrained.pt", required=False)
    args = parser.parse_args()

    rna_records = process_fasta(args.fasta)
    smile_list = process_smiles(args.smiles)

    RNA_repre = []
    RNA_Graph = []
    RNA_C4_coors = []
    RNA_coors = []
    RNA_feats = []
    Mol_graph = []
    LAS_input = []
    Mol_coors = []
    Mol_feats = []

    for seq_idx, rna_seq in enumerate(rna_records):
        seq_output_dir = os.path.join("./data/tmp", f"seq_{seq_idx}")
        os.makedirs(seq_output_dir, exist_ok=True)

        temp_fasta_path = os.path.join(seq_output_dir, "temp.fasta")
        try:
            with open(temp_fasta_path, 'w') as temp_fasta:
                temp_fasta.write(f">seq_{seq_idx}\n{rna_seq}\n")

            # shape: RNA_embedding = [L, 640]，L 是 RNA 碱基数。
            RNA_embedding = get_rna_embeddings(rna_seq)
            secondary_structure = get_secondary_structure(temp_fasta_path)
            edge_index = torch.tensor(np.array(parse_rna_structure(secondary_structure)).T, dtype=torch.long)
            # shape: RNA_data.x = [L, 640]，edge_index = [2, E_rna]。
            RNA_data = Data(x=RNA_embedding, edge_index=edge_index)

            command = [
                "python",
                os.path.join(args.RhoFold_path, "inference.py"),
                "--input_fas", temp_fasta_path,
                "--single_seq_pred", "True",
                "--output_dir", seq_output_dir,
                "--ckpt", args.RhoFold_weight
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)

            pdb_file_path = os.path.join(seq_output_dir, "relaxed_1000_model.pdb")
            atoms_to_extract = ['C4\'', 'N1', 'P']
            # shape: RNA_3_coords = (token_ids [R3], coords [R3, 3])；R3 约为 3*L。
            RNA_3_coords = extract_atom_coordinates(pdb_file_path, atoms_to_extract)
            atoms_to_extract = ['C4\'']
            # shape: RNA_C4_coords = (token_ids [L], coords [L, 3])，用于 RNA 内部距离约束。
            RNA_C4_coords = extract_atom_coordinates(pdb_file_path, atoms_to_extract)

            Smile_coor_input = {}
            for smi in smile_list:
                atoms, coords = inner_smi2coords(smi)
                atoms_id = [atom_id[atom] for atom in atoms]
                Smile_coor_input[smi] = (atoms_id, coords)

            for j in range(len(smile_list)):
                RNA_repre.append(RNA_embedding)
                RNA_Graph.append(RNA_data)
                RNA_C4_coors.append(RNA_C4_coords[1])
                RNA_coors.append(RNA_3_coords[1])
                RNA_feats.append(RNA_3_coords[0])

                # shape: Mol_graph.x = [A, 55]，edge_index = [2, E_mol]，A 是重原子数。
                Mol_graph.append(fea.simplify_atom_to_graph_data_obj(Chem.MolFromSmiles(smile_list[j])))
                # shape: LAS_input = [A, A]，约束哪些原子对保留真实距离。
                LAS_input.append(get_LAS_distance_constraint_mask(Chem.MolFromSmiles(smile_list[j])))
                # shape: Mol_coors = [A, 3]，Mol_feats = [A]。
                Mol_coors.append(np.array(Smile_coor_input[smile_list[j]][1]))
                Mol_feats.append(Smile_coor_input[smile_list[j]][0])

        except Exception as e:
            print(f"Meet error when processing sequence {seq_idx}: {str(e)}")
            continue
        finally:
            if os.path.exists(temp_fasta_path):
                os.remove(temp_fasta_path)

    # 注意字段顺序必须和 inference 脚本的 Test_dataset 解包顺序一致。
    data = [RNA_repre, Mol_graph, RNA_Graph, RNA_feats, RNA_C4_coors, RNA_coors, Mol_feats, Mol_coors, LAS_input]
    with open('data/new_data.pkl', 'wb') as file:
        pickle.dump(data, file)

if __name__ == '__main__':
    main()