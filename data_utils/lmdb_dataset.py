import importlib
import json
import os
import pickle
import struct
import sys

import lmdb
import numpy as np
import torch
from torch.utils import data


def install_boltz_numpy_pickle_compat():
    """Allow NumPy-2 Boltz pickles (numpy._core.*) to load under NumPy 1.x."""
    try:
        import numpy as np
        import numpy.core as np_core
    except Exception:
        return
    sys.modules.setdefault("numpy._core", np_core)
    if not hasattr(np, "_core"):
        try:
            setattr(np, "_core", np_core)
        except Exception:
            pass
    for name in (
        "multiarray",
        "numeric",
        "numerictypes",
        "fromnumeric",
        "_multiarray_umath",
        "umath",
        "shape_base",
        "arrayprint",
        "records",
    ):
        try:
            module = importlib.import_module(f"numpy.core.{name}")
        except Exception:
            continue
        sys.modules.setdefault(f"numpy._core.{name}", module)


install_boltz_numpy_pickle_compat()


AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_ID = {aa: index + 1 for index, aa in enumerate(AA_LIST)}
PROTEIN_RESIDUE_NUM_TOKENS = 21
PROTEIN_ALL_ATOM_NUM_TOKENS = 8  # 0 is padding; LMDB element ids 0..6 are shifted to 1..7.
PHARMACOPHORE_POINT_TYPES = (
    "CA",
    "SIDECHAIN",
    "DONOR",
    "ACCEPTOR",
    "AROMATIC",
    "CHARGED",
    "HYDROPHOBIC",
)
PROTEIN_PHARMACOPHORE_NUM_TOKENS = 1 + len(PHARMACOPHORE_POINT_TYPES)



def sequence_to_string(sequence_value):
    if isinstance(sequence_value, np.ndarray):
        return "".join(str(item) for item in sequence_value.reshape(-1).tolist())
    if isinstance(sequence_value, (list, tuple)):
        return "".join(str(item) for item in sequence_value)
    if isinstance(sequence_value, bytes):
        return sequence_value.decode("utf-8")
    return str(sequence_value)


def ordered_chains(protein_features):
    return [chain for _, chain in ordered_chain_items(protein_features)]


def ordered_chain_items(protein_features):
    if isinstance(protein_features, dict):
        return [(key, protein_features[key]) for key in sorted(protein_features)]
    return [(str(index), chain) for index, chain in enumerate(protein_features)]


def resolve_manifest_path(manifest_dir, path_value):
    if not path_value:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(manifest_dir, path_value)


def resolve_optional_data_path(manifest_dir, path_value):
    if not path_value:
        return None
    if os.path.isabs(path_value):
        return path_value
    if os.path.exists(path_value):
        return os.path.abspath(path_value)
    return os.path.join(manifest_dir, path_value)


def _first_present(payload, keys):
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def copy_lmdb_value(value):
    if value is None:
        return None
    return bytes(value)


def normalize_pbcnet_fusion_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("PBCNet fusion LMDB value must be a dict")
    global_feature = _first_present(
        payload,
        (
            "pbcnet_global",
            "global",
            "pooled",
            "embedding",
            "complex_embedding",
        ),
    )
    if global_feature is None:
        raise ValueError("PBCNet fusion payload must contain a global feature")
    global_feature = np.asarray(global_feature, dtype=np.float32).reshape(-1)
    if global_feature.ndim != 1 or global_feature.shape[0] == 0:
        raise ValueError("PBCNet global feature must flatten to a non-empty vector")
    return {"payload_type": "pbcnet", "global": global_feature}


def select_boltz_pocket_residues(atom_coords, atom_mask, ligand_xyz, ligand_mask, radius=10.0, min_residues=16, max_residues=64):
    atom_coords = np.asarray(atom_coords, dtype=np.float32)
    atom_mask = np.asarray(atom_mask, dtype=bool)
    ligand_xyz = np.asarray(ligand_xyz, dtype=np.float32)
    ligand_mask = np.asarray(ligand_mask, dtype=bool).reshape(-1)
    if atom_coords.ndim != 3 or atom_coords.shape[-1] != 3:
        raise ValueError(f"Boltz atom_coords must have shape [P, A, 3], got {atom_coords.shape}")
    if atom_mask.shape != atom_coords.shape[:2]:
        raise ValueError("Boltz atom_mask must have shape [P, A]")
    if ligand_xyz.ndim != 2 or ligand_xyz.shape[-1] != 3:
        raise ValueError(f"Boltz ligand_xyz must have shape [L, 3], got {ligand_xyz.shape}")
    if ligand_mask.shape[0] != ligand_xyz.shape[0]:
        raise ValueError("Boltz ligand_mask length must match ligand_xyz")

    ligand_heavy = ligand_xyz[ligand_mask]
    n_residue = atom_coords.shape[0]
    if n_residue == 0 or len(ligand_heavy) == 0:
        return np.arange(min(n_residue, int(max_residues)), dtype=np.int64)

    distances = np.full(n_residue, np.inf, dtype=np.float32)
    for residue_index in range(n_residue):
        heavy_atoms = atom_coords[residue_index][atom_mask[residue_index]]
        if len(heavy_atoms) == 0:
            continue
        delta = heavy_atoms[:, None, :] - ligand_heavy[None, :, :]
        distances[residue_index] = float(np.sqrt(np.sum(delta * delta, axis=-1)).min())

    order = np.argsort(distances)
    within = order[distances[order] <= float(radius)]
    selected = within if len(within) >= int(min_residues) else order[:int(min_residues)]
    selected = selected[np.isfinite(distances[selected])][:int(max_residues)]
    return selected.astype(np.int64)


def build_boltz_interface_payload(payload, pocket_radius=10.0, min_pocket_residues=16, max_pocket_residues=64):
    if not isinstance(payload, dict):
        raise ValueError("Boltz interface LMDB value must be a dict")
    required = (
        "protein_boltz_emb",
        "ligand_boltz_emb",
        "atom_coords",
        "atom_mask",
        "ligand_xyz",
        "ligand_mask",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Boltz interface payload is missing fields: {missing}")

    protein_emb = np.asarray(payload["protein_boltz_emb"], dtype=np.float32)
    ligand_emb = np.asarray(payload["ligand_boltz_emb"], dtype=np.float32)
    ligand_mask = np.asarray(payload["ligand_mask"], dtype=bool).reshape(-1)
    if protein_emb.ndim != 2 or ligand_emb.ndim != 2:
        raise ValueError("Boltz protein_boltz_emb and ligand_boltz_emb must have shape [N, D]")
    if protein_emb.shape[1] != ligand_emb.shape[1]:
        raise ValueError(
            f"Boltz protein/ligand embedding dimensions must match: {protein_emb.shape[1]} != {ligand_emb.shape[1]}"
        )
    if ligand_mask.shape[0] != ligand_emb.shape[0]:
        raise ValueError("Boltz ligand_mask length must match ligand_boltz_emb")
    if protein_emb.shape[0] != np.asarray(payload["atom_coords"]).shape[0]:
        raise ValueError("Boltz protein_boltz_emb length must match atom_coords residue axis")

    selected = select_boltz_pocket_residues(
        payload["atom_coords"],
        payload["atom_mask"],
        payload["ligand_xyz"],
        ligand_mask,
        radius=pocket_radius,
        min_residues=min_pocket_residues,
        max_residues=max_pocket_residues,
    )
    if len(selected) == 0:
        raise ValueError("Boltz interface pocket selection produced no valid residues")
    valid_ligand = ligand_emb[ligand_mask]
    if len(valid_ligand) == 0:
        raise ValueError("Boltz interface payload has no valid ligand atoms")

    return {
        "payload_type": "boltz_interface",
        "pocket_emb": protein_emb[selected].astype(np.float32),
        "ligand_emb": valid_ligand.astype(np.float32),
        "global_feat": np.concatenate((protein_emb.sum(axis=0), ligand_emb.sum(axis=0))).astype(np.float32),
    }



def normalize_dti_fusion_payload(payload, expected_protein_length=None):
    if not isinstance(payload, dict):
        raise ValueError("DTI fusion LMDB value must be a dict")

    protein_tokens = _first_present(
        payload,
        (
            "protein_tokens",
            "prot_tokens",
            "protein_token_embeddings",
            "dti_protein_tokens",
        ),
    )
    protein_mask = _first_present(payload, ("protein_mask", "prot_mask", "dti_protein_mask"))
    global_feature = _first_present(
        payload,
        (
            "global",
            "pooled",
            "pooled_embedding",
            "global_embedding",
            "dti_global",
            "pair_embedding",
        ),
    )

    normalized = {"protein_tokens": None, "protein_mask": None, "global": None}
    if protein_tokens is not None:
        protein_tokens = np.asarray(protein_tokens, dtype=np.float32)
        if protein_tokens.ndim != 2:
            raise ValueError(f"DTI protein_tokens must have shape [L, D], got {protein_tokens.shape}")
        if expected_protein_length is not None and protein_tokens.shape[0] != int(expected_protein_length):
            raise ValueError(
                "DTI protein_tokens length must match source residue length: "
                f"got {protein_tokens.shape[0]}, expected {int(expected_protein_length)}"
            )
        if protein_mask is None:
            protein_mask = np.ones((protein_tokens.shape[0],), dtype=bool)
        else:
            protein_mask = np.asarray(protein_mask).astype(bool).reshape(-1)
            if protein_mask.shape[0] != protein_tokens.shape[0]:
                raise ValueError(
                    "DTI protein_mask length must match protein_tokens length: "
                    f"got {protein_mask.shape[0]}, expected {protein_tokens.shape[0]}"
                )
        normalized["protein_tokens"] = protein_tokens
        normalized["protein_mask"] = protein_mask

    if global_feature is not None:
        global_feature = np.asarray(global_feature, dtype=np.float32).reshape(-1)
        if global_feature.ndim != 1 or global_feature.shape[0] == 0:
            raise ValueError("DTI global feature must flatten to a non-empty vector")
        normalized["global"] = global_feature

    if normalized["protein_tokens"] is None and normalized["global"] is None:
        raise ValueError("DTI fusion payload must contain protein_tokens and/or global features")
    return normalized



def deserialize_embedding_cache_payload(blob):
    # DrugCLIP/Uni-Mol split-LMDB uses a compact binary layout:
    # [uint32 header_len][JSON header][embedding bytes][mask bytes].
    # The arrays are copied before returning so DataLoader workers do not keep
    # views into an LMDB-owned buffer after the read transaction is closed.
    if len(blob) < 4:
        raise ValueError("invalid embedding LMDB payload: too small")
    header_len = struct.unpack("<I", blob[:4])[0]
    header_start = 4
    header_end = header_start + header_len
    header = json.loads(blob[header_start:header_end].decode("utf-8"))
    emb_shape = tuple(int(v) for v in header["embedding_shape"])
    emb_dtype = np.dtype(header["embedding_dtype"])
    mask_shape = tuple(int(v) for v in header["mask_shape"])
    mask_dtype = np.dtype(header["mask_dtype"])
    emb_nbytes = int(np.prod(emb_shape)) * emb_dtype.itemsize
    mask_nbytes = int(np.prod(mask_shape)) * mask_dtype.itemsize
    emb_start = header_end
    emb_end = emb_start + emb_nbytes
    mask_end = emb_end + mask_nbytes
    embedding = np.frombuffer(blob[emb_start:emb_end], dtype=emb_dtype).reshape(emb_shape).copy()
    mask = np.frombuffer(blob[emb_end:mask_end], dtype=mask_dtype).reshape(mask_shape).copy()
    return embedding, mask, header


def is_embedding_cache_payload(payload):
    # Old ligand Uni-Mol caches are not pickle objects. Detect that binary
    # cache format before falling back to pickle.loads; otherwise Python can
    # emit noisy buffer-lifetime SystemErrors in multi-worker loading.
    if not isinstance(payload, (bytes, bytearray)) or len(payload) < 8:
        return False
    try:
        blob = bytes(payload)
        header_len = struct.unpack("<I", blob[:4])[0]
        if header_len <= 0 or header_len > len(blob) - 4:
            return False
        header = json.loads(blob[4:4 + header_len].decode("utf-8"))
        return "embedding_shape" in header and "embedding_dtype" in header and "mask_shape" in header
    except Exception:
        return False


def normalize_ligand_unimol_payload(payload, expected_ligand_length=None):
    # Normalize every supported ligand-pretrain cache into the same contract:
    #   old Uni-Mol token cache: ligand_tokens [A, 512], ligand_mask [A]
    #   Uni-Mol2 token cache: ligand_tokens [A, 768], ligand_mask [A]
    # A is the source ligand atom count after optional alignment below.
    if isinstance(payload, (bytes, bytearray)) and is_embedding_cache_payload(payload):
        ligand_tokens, ligand_mask, _ = deserialize_embedding_cache_payload(bytes(payload))
        ligand_tokens = np.asarray(ligand_tokens, dtype=np.float32)
        ligand_mask = np.asarray(ligand_mask).astype(bool).reshape(-1)
        if expected_ligand_length is not None and ligand_tokens.shape[0] != int(expected_ligand_length):
            raise ValueError(
                "Ligand Uni-Mol token length must match source ligand atom length: "
                f"got {ligand_tokens.shape[0]}, expected {int(expected_ligand_length)}"
            )
        if ligand_mask.shape[0] != ligand_tokens.shape[0]:
            raise ValueError("Ligand Uni-Mol mask length must match ligand token length")
        return {"ligand_tokens": ligand_tokens, "ligand_mask": ligand_mask}
    payload_atoms = None
    if isinstance(payload, (bytes, bytearray)):
        payload = pickle.loads(bytes(payload))
    if isinstance(payload, dict):
        if "atoms" in payload and payload["atoms"] is not None:
            payload_atoms = [str(atom) for atom in payload["atoms"]]
        ligand_tokens = None
        for key in ("ligand_tokens", "token_embedding", "tokens"):
            if key in payload:
                ligand_tokens = payload[key]
                break
        ligand_mask = None
        for key in ("ligand_mask", "token_attention_mask", "mask"):
            if key in payload:
                ligand_mask = payload[key]
                break
    else:
        ligand_tokens, ligand_mask, _ = deserialize_embedding_cache_payload(payload)
    ligand_tokens = np.asarray(ligand_tokens, dtype=np.float32)
    if ligand_tokens.ndim != 2:
        raise ValueError(f"Ligand Uni-Mol tokens must have shape [A, D], got {ligand_tokens.shape}")
    if expected_ligand_length is not None and ligand_tokens.shape[0] != int(expected_ligand_length):
        raise ValueError(
            "Ligand Uni-Mol token length must match source ligand atom length: "
            f"got {ligand_tokens.shape[0]}, expected {int(expected_ligand_length)}"
        )
    if ligand_mask is None:
        ligand_mask = np.ones((ligand_tokens.shape[0],), dtype=bool)
    ligand_mask = np.asarray(ligand_mask).astype(bool).reshape(-1)
    if ligand_mask.shape[0] != ligand_tokens.shape[0]:
        raise ValueError("Ligand Uni-Mol mask length must match ligand token length")
    normalized = {"ligand_tokens": ligand_tokens, "ligand_mask": ligand_mask}
    if payload_atoms is not None:
        normalized["atoms"] = payload_atoms
    return normalized


def align_ligand_unimol_to_source_atoms(ligand_unimol, mol, expected_ligand_length):
    # Source ligand atoms define the training-time atom order. Shape flow:
    #   already aligned: tokens [A, D] -> [A, D]
    #   heavy-atom cache: tokens [A_heavy, D] -> expanded_tokens [A_all, D]
    # If an external cache was extracted after removing H, features are placed on
    # the corresponding heavy-atom positions and H positions stay zero-masked.
    tokens = np.asarray(ligand_unimol["ligand_tokens"], dtype=np.float32)
    mask = np.asarray(ligand_unimol.get("ligand_mask", np.ones((tokens.shape[0],), dtype=bool))).astype(bool).reshape(-1)
    cache_atoms = ligand_unimol.get("atoms")
    cache_atoms = [str(atom) for atom in cache_atoms] if cache_atoms is not None else None
    expected_ligand_length = int(expected_ligand_length)
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    if tokens.shape[0] == expected_ligand_length:
        if cache_atoms is not None and cache_atoms != atom_symbols:
            raise ValueError(f"Ligand Uni-Mol atom order mismatch: cache={cache_atoms[:20]} source={atom_symbols[:20]}")
        ligand_unimol["ligand_tokens"] = tokens
        ligand_unimol["ligand_mask"] = mask
        return ligand_unimol

    if len(atom_symbols) != expected_ligand_length:
        raise ValueError(
            "Source ligand atom count from RDKit mol does not match ligand feature length: "
            f"mol_atoms={len(atom_symbols)}, expected={expected_ligand_length}"
        )
    heavy_indices = [idx for idx, symbol in enumerate(atom_symbols) if symbol != "H"]
    heavy_symbols = [atom_symbols[idx] for idx in heavy_indices]
    if len(heavy_indices) != tokens.shape[0]:
        raise ValueError(
            "Ligand Uni-Mol token length must match either source all-atom length or source heavy-atom length: "
            f"got {tokens.shape[0]}, all_atoms {expected_ligand_length}, heavy_atoms {len(heavy_indices)}"
        )
    if cache_atoms is not None and cache_atoms != heavy_symbols:
        raise ValueError(f"Ligand Uni-Mol heavy atom order mismatch: cache={cache_atoms[:20]} source={heavy_symbols[:20]}")

    expanded_tokens = np.zeros((expected_ligand_length, tokens.shape[1]), dtype=np.float32)
    expanded_mask = np.zeros((expected_ligand_length,), dtype=bool)
    expanded_tokens[np.asarray(heavy_indices, dtype=np.int64)] = tokens
    expanded_mask[np.asarray(heavy_indices, dtype=np.int64)] = mask
    return {"ligand_tokens": expanded_tokens, "ligand_mask": expanded_mask}

def chain_sequences(chains):
    return [sequence_to_string(chain.get("sequence", "")) for chain in chains]


def concat_arrays(chains, key, dtype):
    arrays = [np.asarray(chain[key], dtype=dtype) for chain in chains if key in chain and len(chain[key]) > 0]
    if not arrays:
        return np.zeros((0,), dtype=dtype)
    return np.concatenate(arrays, axis=0)


def residue_embedding(aa, position, length, dim=640):
    vector = np.zeros(dim, dtype=np.float32)
    aa_index = AA_LIST.index(aa) if aa in AA_LIST else len(AA_LIST)
    if aa_index < dim:
        vector[aa_index] = 1.0
    vector[32] = float(position) / max(length - 1, 1)
    vector[33] = float(length) / 512.0
    vector[34] = 1.0 if aa in {"D", "E"} else 0.0
    vector[35] = 1.0 if aa in {"K", "R", "H"} else 0.0
    vector[36] = 1.0 if aa in {"F", "W", "Y"} else 0.0
    vector[37] = 1.0 if aa in {"A", "V", "I", "L", "M"} else 0.0
    return vector


def build_manual_residue_representations(chains):
    sequence = "".join(chain_sequences(chains))
    return np.stack(
        [residue_embedding(aa, index, len(sequence)) for index, aa in enumerate(sequence)],
        axis=0,
    ).astype(np.float32)


def build_residue_representations(chains, embedding_mode="esm", expected_dim=640):
    if embedding_mode == "residue":
        return build_manual_residue_representations(chains)
    if embedding_mode not in {"esm", "auto"}:
        raise ValueError("protein embedding_mode must be one of: esm, residue, auto")

    embeddings = []
    for chain_index, chain in enumerate(chains):
        sequence = sequence_to_string(chain.get("sequence", ""))
        esm = chain.get("esm2_embedding")
        if esm is None:
            if embedding_mode == "auto":
                return build_manual_residue_representations(chains)
            raise ValueError(
                f"LMDB sample is missing esm2_embedding for chain index {chain_index}; "
                "run data_utils/extract_lmdb_esm.py first, or pass --protein_embedding_mode residue "
                "only for a non-ESM baseline."
            )
        esm = np.asarray(esm, dtype=np.float32)
        if esm.ndim != 2 or esm.shape[0] != len(sequence):
            if embedding_mode == "auto":
                return build_manual_residue_representations(chains)
            raise ValueError(
                f"Invalid esm2_embedding shape for chain index {chain_index}: "
                f"got {esm.shape}, expected ({len(sequence)}, {expected_dim})"
            )
        if expected_dim and esm.shape[1] != int(expected_dim):
            if embedding_mode == "auto":
                return build_manual_residue_representations(chains)
            raise ValueError(
                f"Invalid ESM hidden size for chain index {chain_index}: "
                f"got {esm.shape[1]}, expected {expected_dim}"
            )
        embeddings.append(esm)

    return np.concatenate(embeddings, axis=0).astype(np.float32)


def build_residue_ids(chains):
    sequence = "".join(chain_sequences(chains))
    return np.asarray([AA_TO_ID.get(aa, 0) for aa in sequence], dtype=np.int64)


def build_chain_ca_coords(chains):
    return concat_arrays(chains, "ca_coords", np.float32).reshape(-1, 3)


def build_full_residue_esm_inputs(chains, protein_repre=None, embedding_mode="esm", expected_dim=640):
    # Full-residue ESM 3D mode uses one complete-chain residue node per CA. Shape flow:
    #   ESM protein_repre [L_full, D] + CA coords [L_full, 3]
    #   output features [L_full, D], coords [L_full, 3]
    if protein_repre is None:
        protein_repre = build_residue_representations(
            chains,
            embedding_mode=embedding_mode,
            expected_dim=expected_dim,
        )
    protein_repre = np.asarray(protein_repre, dtype=np.float32)
    ca_coords = build_chain_ca_coords(chains)
    if protein_repre.ndim != 2:
        raise ValueError(f"Full-residue ESM features must have shape [L, D], got {protein_repre.shape}")
    if len(protein_repre) != len(ca_coords):
        raise ValueError(
            "Full-residue ESM mode requires one CA coordinate per ESM residue: "
            f"features={len(protein_repre)}, coords={len(ca_coords)}"
        )
    if len(protein_repre) == 0:
        raise ValueError("LMDB sample has no usable full-residue ESM coordinates")
    return protein_repre.astype(np.float32), ca_coords.astype(np.float32)


def build_ligand_pocket_residue_esm_inputs(chains, protein_repre, ligand_coords, cutoff=8.0):
    # Ligand-centered pocket mode selects full-chain ESM residue states whose
    # CA atom is within cutoff Angstrom of any ligand atom in the complex.
    protein_repre = np.asarray(protein_repre, dtype=np.float32)
    ca_coords = build_chain_ca_coords(chains)
    ligand_coords = np.asarray(ligand_coords, dtype=np.float32).reshape(-1, 3)
    cutoff = float(cutoff)

    if protein_repre.ndim != 2:
        raise ValueError(f"Ligand pocket ESM features must have shape [L, D], got {protein_repre.shape}")
    if len(protein_repre) != len(ca_coords):
        raise ValueError(
            "Ligand pocket ESM mode requires one CA coordinate per ESM residue: "
            f"features={len(protein_repre)}, coords={len(ca_coords)}"
        )
    if len(ca_coords) == 0:
        raise ValueError("LMDB sample has no usable protein CA coordinates")
    if len(ligand_coords) == 0:
        raise ValueError("LMDB sample has no usable ligand coordinates")
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError(f"ligand pocket cutoff must be positive, got {cutoff}")

    squared_distances = np.sum((ca_coords[:, None, :] - ligand_coords[None, :, :]) ** 2, axis=-1)
    selected = np.any(squared_distances <= cutoff ** 2, axis=1)
    if not np.any(selected):
        raise ValueError(
            f"No protein CA residue is within {cutoff:g} Angstrom of the ligand; "
            "check that protein and ligand coordinates belong to the same complex"
        )
    return protein_repre[selected].astype(np.float32), ca_coords[selected].astype(np.float32)


def build_protein_edges(ca_coords, chain_lengths, edge_mode="radius", cutoff=6.0, knn_k=16):
    ca_coords = np.asarray(ca_coords, dtype=np.float32).reshape(-1, 3)
    n_residue = len(ca_coords)
    edges = set()

    offset = 0
    for chain_length in chain_lengths:
        chain_length = int(chain_length)
        for index in range(max(chain_length - 1, 0)):
            left = offset + index
            right = left + 1
            edges.add((left, right))
            edges.add((right, left))
        offset += chain_length

    if n_residue > 1:
        distances = np.linalg.norm(ca_coords[:, None, :] - ca_coords[None, :, :], axis=-1)
        if edge_mode == "radius":
            contact_i, contact_j = np.where((distances <= float(cutoff)) & (distances > 0))
            for left, right in zip(contact_i.tolist(), contact_j.tolist()):
                edges.add((left, right))
        elif edge_mode == "knn":
            k = min(int(knn_k), n_residue - 1)
            if k > 0:
                order = np.argsort(np.where(np.eye(n_residue, dtype=bool), np.inf, distances), axis=1)
                for left in range(n_residue):
                    for right in order[left, :k].tolist():
                        edges.add((left, right))
                        edges.add((right, left))
        else:
            raise ValueError("protein edge_mode must be one of: radius, knn")

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(np.asarray(sorted(edges)).T, dtype=torch.long)


def _array_or_empty(chain, key, dtype, length=0):
    if key not in chain:
        return np.zeros((length,), dtype=dtype)
    return np.asarray(chain[key], dtype=dtype).reshape(-1)


def _string_array_or_empty(chain, key, length=0):
    if key not in chain:
        return np.asarray([""] * length, dtype=str)
    return np.asarray(chain[key], dtype=str).reshape(-1)


def pharmacophore_token(point_type_index):
    return 1 + int(point_type_index)


def build_chain_pharmacophore_inputs(chain, return_residue_indices=False):
    atom_coords = np.asarray(chain.get("atom_coords", []), dtype=np.float32).reshape(-1, 3)
    n_atoms = len(atom_coords)
    empty_tokens = np.zeros((0,), dtype=np.int64)
    empty_coords = np.zeros((0, 3), dtype=np.float32)
    empty_residue_indices = np.zeros((0,), dtype=np.int64)
    if n_atoms == 0:
        if return_residue_indices:
            return empty_tokens, empty_coords, empty_residue_indices
        return empty_tokens, empty_coords

    residue_indices = _array_or_empty(chain, "atom_residue_indices", np.int64, n_atoms)
    atom_names = _string_array_or_empty(chain, "atom_names", n_atoms)
    atom_is_backbone = _array_or_empty(chain, "atom_is_backbone", np.int64, n_atoms).astype(bool)
    atom_is_ca = _array_or_empty(chain, "atom_is_ca", np.int64, n_atoms).astype(bool)
    atom_is_donor = _array_or_empty(chain, "atom_is_donor", np.int64, n_atoms).astype(bool)
    atom_is_acceptor = _array_or_empty(chain, "atom_is_acceptor", np.int64, n_atoms).astype(bool)
    atom_is_aromatic = _array_or_empty(chain, "atom_is_aromatic", np.int64, n_atoms).astype(bool)
    atom_is_charged = _array_or_empty(chain, "atom_is_charged", np.int64, n_atoms).astype(bool)
    atom_is_hydrophobic = _array_or_empty(chain, "atom_is_hydrophobic", np.int64, n_atoms).astype(bool)

    if not atom_is_backbone.any():
        atom_is_backbone = np.isin(atom_names, ["N", "CA", "C", "O"])
    if not atom_is_ca.any():
        atom_is_ca = atom_names == "CA"

    tokens = []
    coords = []
    point_residue_indices = []

    def add_point(point_type_index, mask, residue_index):
        if np.any(mask):
            tokens.append(pharmacophore_token(point_type_index))
            coords.append(atom_coords[mask].mean(axis=0).astype(np.float32))
            point_residue_indices.append(int(residue_index))

    for residue_index in sorted(set(residue_indices.tolist())):
        residue_mask = residue_indices == residue_index
        if not np.any(residue_mask):
            continue
        add_point(0, residue_mask & atom_is_ca, residue_index)
        add_point(1, residue_mask & (~atom_is_backbone), residue_index)
        add_point(2, residue_mask & atom_is_donor, residue_index)
        add_point(3, residue_mask & atom_is_acceptor, residue_index)
        add_point(4, residue_mask & atom_is_aromatic, residue_index)
        add_point(5, residue_mask & atom_is_charged, residue_index)
        add_point(6, residue_mask & atom_is_hydrophobic, residue_index)

    if not tokens:
        if return_residue_indices:
            return empty_tokens, empty_coords, empty_residue_indices
        return empty_tokens, empty_coords
    tokens = np.asarray(tokens, dtype=np.int64)
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
    point_residue_indices = np.asarray(point_residue_indices, dtype=np.int64)
    if return_residue_indices:
        return tokens, coords, point_residue_indices
    return tokens, coords


def build_pharmacophore_inputs(chains):
    token_arrays = []
    coord_arrays = []
    for chain in chains:
        tokens, coords = build_chain_pharmacophore_inputs(chain)
        if len(tokens) > 0:
            token_arrays.append(tokens)
            coord_arrays.append(coords)
    if not token_arrays:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(token_arrays, axis=0), np.concatenate(coord_arrays, axis=0)


def build_pharmacophore_inputs_with_residue_indices(chains):
    token_arrays = []
    coord_arrays = []
    residue_index_arrays = []
    chain_index_arrays = []
    for chain_index, chain in enumerate(chains):
        tokens, coords, residue_indices = build_chain_pharmacophore_inputs(chain, return_residue_indices=True)
        if len(tokens) > 0:
            token_arrays.append(tokens)
            coord_arrays.append(coords)
            residue_index_arrays.append(residue_indices)
            chain_index_arrays.append(np.full(len(tokens), chain_index, dtype=np.int64))
    if not token_arrays:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.concatenate(token_arrays, axis=0),
        np.concatenate(coord_arrays, axis=0),
        np.concatenate(residue_index_arrays, axis=0),
        np.concatenate(chain_index_arrays, axis=0),
    )


def greedy_subsequence_indices(query_sequence, target_sequence):
    if len(query_sequence) == 0:
        return []
    indices = []
    start = 0
    for residue in query_sequence:
        position = target_sequence.find(residue, start)
        if position < 0:
            return None
        indices.append(position)
        start = position + 1
    return indices


def pocket_pharmacophore_can_align_to_feature_chains(feature_chains, structure_chains):
    if not feature_chains or not structure_chains:
        return False
    try:
        resolve_pocket_to_feature_residue_alignment(feature_chains, structure_chains)
        return True
    except ValueError:
        return False



def chain_identifier(chain, fallback_index):
    value = chain.get("chain_id", fallback_index)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0] if value.size else fallback_index
    return str(value).upper()


def residue_identity_arrays(chain):
    sequence = sequence_to_string(chain.get("sequence", ""))
    ca_coords = np.asarray(chain.get("ca_coords", []), dtype=np.float32).reshape(-1, 3)
    return sequence, ca_coords


def coordinate_residue_alignment(feature_chains, structure_chains, ca_tolerance=1e-3):
    # Prefer exact structure identity available in current LMDBs: chain id + residue type + CA coordinate.
    # This avoids greedy sequence mistakes when pocket residues are discontinuous or repeated.
    feature_lookup = {}
    for feature_chain_index, feature_chain in enumerate(feature_chains):
        feature_sequence, feature_ca = residue_identity_arrays(feature_chain)
        chain_id = chain_identifier(feature_chain, feature_chain_index)
        for residue_index, residue_name in enumerate(feature_sequence):
            if residue_index >= len(feature_ca):
                continue
            coord_key = tuple(np.round(feature_ca[residue_index] / float(ca_tolerance)).astype(np.int64).tolist())
            feature_lookup.setdefault((chain_id, residue_name, coord_key), []).append((feature_chain_index, residue_index))

    alignment = []
    for structure_chain_index, structure_chain in enumerate(structure_chains):
        structure_sequence, structure_ca = residue_identity_arrays(structure_chain)
        chain_id = chain_identifier(structure_chain, structure_chain_index)
        chain_alignment = []
        for residue_index, residue_name in enumerate(structure_sequence):
            if residue_index >= len(structure_ca):
                return None
            coord_key = tuple(np.round(structure_ca[residue_index] / float(ca_tolerance)).astype(np.int64).tolist())
            matches = feature_lookup.get((chain_id, residue_name, coord_key), [])
            if len(matches) != 1:
                return None
            chain_alignment.append(matches[0])
        alignment.append(chain_alignment)
    return alignment


def sequence_residue_alignment(feature_chains, structure_chains):
    alignment = []
    for structure_chain_index, structure_chain in enumerate(structure_chains):
        structure_sequence = sequence_to_string(structure_chain.get("sequence", ""))
        feature_chain_index = min(structure_chain_index, len(feature_chains) - 1)
        feature_chain = feature_chains[feature_chain_index]
        feature_sequence = sequence_to_string(feature_chain.get("sequence", ""))
        if len(structure_sequence) == len(feature_sequence):
            residue_alignment = list(range(len(structure_sequence)))
        else:
            residue_alignment = greedy_subsequence_indices(structure_sequence, feature_sequence)
        if residue_alignment is None or len(residue_alignment) < len(structure_sequence):
            return None
        alignment.append([(feature_chain_index, int(feature_residue_index)) for feature_residue_index in residue_alignment])
    return alignment


def resolve_pocket_to_feature_residue_alignment(feature_chains, structure_chains):
    coordinate_alignment = coordinate_residue_alignment(feature_chains, structure_chains)
    if coordinate_alignment is not None:
        return coordinate_alignment, "chain_id+residue+CA"
    sequence_alignment = sequence_residue_alignment(feature_chains, structure_chains)
    if sequence_alignment is not None:
        return sequence_alignment, "sequence"
    raise ValueError(
        "Could not align pocket residues to feature-chain ESM sequence by CA coordinate or sequence; "
        "regenerate LMDBs with original residue ids for exact mapping."
    )


def build_pocket_pharmacophore_esm_inputs(feature_chains, structure_chains, protein_repre):
    tokens, coords, residue_indices, chain_indices = build_pharmacophore_inputs_with_residue_indices(structure_chains)
    if len(tokens) == 0:
        raise ValueError("LMDB sample has no usable pocket pharmacophore coordinates")

    feature_offsets = []
    offset = 0
    for chain in feature_chains:
        feature_offsets.append(offset)
        offset += len(sequence_to_string(chain.get("sequence", "")))

    alignment_maps = []
    feature_chain_indices = []
    for structure_chain_index, structure_chain in enumerate(structure_chains):
        structure_sequence = sequence_to_string(structure_chain.get("sequence", ""))
        feature_chain_index = min(structure_chain_index, len(feature_chains) - 1)
        feature_chain = feature_chains[feature_chain_index]
        feature_sequence = sequence_to_string(feature_chain.get("sequence", ""))
        if len(structure_sequence) == len(feature_sequence):
            alignment = list(range(len(structure_sequence)))
        else:
            alignment = greedy_subsequence_indices(structure_sequence, feature_sequence)
        if alignment is None or len(alignment) < len(structure_sequence):
            raise ValueError(
                "Could not align pocket pharmacophore residues to feature-chain ESM sequence; "
                "regenerate LMDBs with original residue ids for exact mapping."
            )
        alignment_maps.append(alignment)
        feature_chain_indices.append(feature_chain_index)

    protein_repre = np.asarray(protein_repre, dtype=np.float32)
    features = []
    for token, structure_residue_index, structure_chain_index in zip(tokens, residue_indices, chain_indices):
        structure_chain_index = int(structure_chain_index)
        structure_residue_index = int(structure_residue_index)
        feature_chain_index = feature_chain_indices[structure_chain_index]
        feature_residue_index = alignment_maps[structure_chain_index][structure_residue_index]
        global_residue_index = feature_offsets[feature_chain_index] + feature_residue_index
        point_type = np.zeros((PROTEIN_PHARMACOPHORE_NUM_TOKENS,), dtype=np.float32)
        point_type[int(token)] = 1.0
        features.append(np.concatenate([point_type, protein_repre[global_residue_index]], axis=0))
    return np.asarray(features, dtype=np.float32), coords



def build_pocket_residue_esm_inputs(feature_chains, structure_chains, protein_repre):
    # Pocket residue ESM mode keeps one 3D node per pocket residue CA. Shape flow:
    #   full-chain ESM protein_repre [L_full, 640]
    #   pocket structure CA coords [L_pocket, 3]
    #   output features [L_pocket, 640], coords [L_pocket, 3]
    # Alignment priority:
    #   1) chain_id + residue letter + CA coordinate, robust to repeated residues
    #   2) sequence fallback, for old LMDBs without usable CA overlap
    feature_offsets = []
    offset = 0
    for chain in feature_chains:
        feature_offsets.append(offset)
        offset += len(sequence_to_string(chain.get("sequence", "")))

    protein_repre = np.asarray(protein_repre, dtype=np.float32)
    alignment, _alignment_source = resolve_pocket_to_feature_residue_alignment(feature_chains, structure_chains)
    features = []
    coords = []
    for structure_chain_index, structure_chain in enumerate(structure_chains):
        structure_sequence, ca_coords = residue_identity_arrays(structure_chain)
        if not structure_sequence:
            continue
        if len(ca_coords) < len(structure_sequence):
            raise ValueError(
                "Pocket residue ESM mode requires one CA coordinate per pocket residue: "
                f"coords={len(ca_coords)}, sequence={len(structure_sequence)}"
            )
        for structure_residue_index, (feature_chain_index, feature_residue_index) in enumerate(alignment[structure_chain_index]):
            global_residue_index = feature_offsets[int(feature_chain_index)] + int(feature_residue_index)
            features.append(protein_repre[global_residue_index])
            coords.append(ca_coords[structure_residue_index])

    if not features:
        raise ValueError("LMDB sample has no usable pocket residue CA coordinates")
    return np.asarray(features, dtype=np.float32), np.asarray(coords, dtype=np.float32).reshape(-1, 3)


def count_lmdb_protein_3d_tokens(chains, protein_3d_mode="residue"):
    if protein_3d_mode == "residue":
        return sum(len(sequence_to_string(chain.get("sequence", ""))) for chain in chains)
    if protein_3d_mode == "all_atom":
        return sum(int(chain.get("num_atoms", len(chain.get("atom_coords", [])))) for chain in chains)
    if protein_3d_mode in {"pocket_pharmacophore", "pocket_pharmacophore_esm"}:
        tokens, _ = build_pharmacophore_inputs(chains)
        return int(len(tokens))
    if protein_3d_mode in {"pocket_residue_esm", "ligand_pocket_residue_esm", "full_residue_esm"}:
        return sum(len(sequence_to_string(chain.get("sequence", ""))) for chain in chains)
    raise ValueError("protein_3d_mode must be one of: residue, all_atom, pocket_pharmacophore, pocket_pharmacophore_esm, pocket_residue_esm, full_residue_esm")


def select_lmdb_protein_3d_inputs(chains, protein_3d_mode="residue"):
    if protein_3d_mode == "residue":
        residue_ids = build_residue_ids(chains)
        ca_coords = build_chain_ca_coords(chains)
        return residue_ids, ca_coords, {
            "protein_3d_mode": "residue",
            "protein_3d_atoms": ["CA"],
            "protein_3d_num_tokens": PROTEIN_RESIDUE_NUM_TOKENS,
        }

    if protein_3d_mode == "all_atom":
        atom_elements = concat_arrays(chains, "atom_element", np.int64).reshape(-1)
        atom_coords = concat_arrays(chains, "atom_coords", np.float32).reshape(-1, 3)
        if len(atom_elements) != len(atom_coords) or len(atom_elements) == 0:
            raise ValueError("LMDB sample has no usable protein atom-level coordinates")
        atom_tokens = atom_elements.astype(np.int64) + 1
        return atom_tokens, atom_coords, {
            "protein_3d_mode": "all_atom",
            "protein_3d_atoms": ["all_heavy_atoms"],
            "protein_3d_num_tokens": PROTEIN_ALL_ATOM_NUM_TOKENS,
        }

    if protein_3d_mode == "pocket_pharmacophore":
        tokens, coords = build_pharmacophore_inputs(chains)
        if len(tokens) == 0:
            raise ValueError("LMDB sample has no usable pocket pharmacophore coordinates")
        return tokens, coords, {
            "protein_3d_mode": "pocket_pharmacophore",
            "protein_3d_atoms": list(PHARMACOPHORE_POINT_TYPES),
            "protein_3d_num_tokens": PROTEIN_PHARMACOPHORE_NUM_TOKENS,
        }

    raise ValueError("protein_3d_mode must be one of: residue, all_atom, pocket_pharmacophore, pocket_pharmacophore_esm, pocket_residue_esm, full_residue_esm")


def build_disabled_protein_graph(feature_dim=640):
    from torch_geometric.data import Data

    return Data(
        x=torch.zeros((1, int(feature_dim)), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )


def build_protein_graph(chains, edge_mode="radius", cutoff=6.0, knn_k=16, residue_repre=None, embedding_mode="esm", expected_dim=640):
    from torch_geometric.data import Data

    if residue_repre is None:
        residue_repre = build_residue_representations(chains, embedding_mode=embedding_mode, expected_dim=expected_dim)
    ca_coords = build_chain_ca_coords(chains)
    chain_lengths = [len(sequence_to_string(chain.get("sequence", ""))) for chain in chains]
    edge_index = build_protein_edges(ca_coords, chain_lengths, edge_mode=edge_mode, cutoff=cutoff, knn_k=knn_k)
    return Data(x=torch.tensor(residue_repre, dtype=torch.float32), edge_index=edge_index)


def ligand_tuple_from_lmdb(ligand_features_dict):
    if "mol" not in ligand_features_dict:
        raise ValueError("LMDB ligand_features does not contain RDKit mol; regenerate LMDB in the training environment")
    from data_utils.process_pdbbind import ligand_features

    return ligand_features(ligand_features_dict["mol"])


class PDBbindLmdbDataset(data.Dataset):
    def __init__(
        self,
        manifest_path,
        protein_3d_mode=None,
        protein_edge_mode="radius",
        protein_contact_cutoff=6.0,
        protein_knn_k=16,
        protein_embedding_mode="esm",
        expected_protein_embedding_dim=640,
        ligand_pocket_cutoff=8.0,
        disable_protein_2d=False,
        dti_fusion_lmdb_path=None,
        ligand_unimol_lmdb_path=None,
        ligand_unimol2_lmdb_path=None,
        pbcnet_fusion_lmdb_path=None,
        pbcnet_graph_lmdb_path=None,
        boltz_interface_lmdb_path=None,
        boltz_pocket_radius=10.0,
        boltz_min_pocket_residues=16,
        boltz_max_pocket_residues=64,
    ):
        self.manifest_path = manifest_path
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("dataset_type") != "pdbbind_lmdb":
            raise ValueError(f"not a PDBbind LMDB manifest: {manifest_path}")

        manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
        lmdb_path = resolve_manifest_path(manifest_dir, manifest["lmdb_path"])
        structure_lmdb_path = resolve_manifest_path(manifest_dir, manifest.get("structure_lmdb_path"))
        if dti_fusion_lmdb_path is None:
            dti_fusion_lmdb_path = resolve_manifest_path(manifest_dir, manifest.get("dti_fusion_lmdb_path"))
        elif not os.path.isabs(dti_fusion_lmdb_path):
            dti_fusion_lmdb_path = os.path.abspath(dti_fusion_lmdb_path)
        if ligand_unimol_lmdb_path is None:
            ligand_unimol_lmdb_path = resolve_manifest_path(manifest_dir, manifest.get("ligand_unimol_lmdb_path"))
        elif not os.path.isabs(ligand_unimol_lmdb_path):
            ligand_unimol_lmdb_path = os.path.abspath(ligand_unimol_lmdb_path)
        if ligand_unimol2_lmdb_path is None:
            ligand_unimol2_lmdb_path = resolve_manifest_path(manifest_dir, manifest.get("ligand_unimol2_lmdb_path"))
        elif not os.path.isabs(ligand_unimol2_lmdb_path):
            ligand_unimol2_lmdb_path = os.path.abspath(ligand_unimol2_lmdb_path)
        if pbcnet_fusion_lmdb_path is None:
            if pbcnet_graph_lmdb_path is None:
                pbcnet_fusion_lmdb_path = resolve_manifest_path(manifest_dir, manifest.get("pbcnet_fusion_lmdb_path"))
            else:
                pbcnet_fusion_lmdb_path = None
        elif not os.path.isabs(pbcnet_fusion_lmdb_path):
            pbcnet_fusion_lmdb_path = os.path.abspath(pbcnet_fusion_lmdb_path)
        if pbcnet_graph_lmdb_path is None:
            pbcnet_graph_lmdb_path = resolve_manifest_path(manifest_dir, manifest.get("pbcnet_graph_lmdb_path"))
        elif not os.path.isabs(pbcnet_graph_lmdb_path):
            pbcnet_graph_lmdb_path = os.path.abspath(pbcnet_graph_lmdb_path)
        if boltz_interface_lmdb_path is None:
            boltz_interface_lmdb_path = resolve_optional_data_path(manifest_dir, manifest.get("boltz_interface_lmdb_path"))
        elif not os.path.isabs(boltz_interface_lmdb_path):
            boltz_interface_lmdb_path = os.path.abspath(boltz_interface_lmdb_path)
        if structure_lmdb_path is None:
            structure_lmdb_path = lmdb_path
        manifest_protein_3d_mode = manifest.get("protein_3d_mode") or "residue"
        if protein_3d_mode is None:
            protein_3d_mode = manifest_protein_3d_mode
        elif manifest_protein_3d_mode and protein_3d_mode != manifest_protein_3d_mode:
            compatible_override = manifest_protein_3d_mode == "residue" and protein_3d_mode in {"full_residue_esm", "ligand_pocket_residue_esm"}
            if not compatible_override:
                raise ValueError(
                    f"CLI protein_3d_mode={protein_3d_mode} does not match manifest "
                    f"protein_3d_mode={manifest_protein_3d_mode}; regenerate the split or pass the matching mode."
                )
        if protein_3d_mode not in {"residue", "all_atom", "pocket_pharmacophore", "pocket_pharmacophore_esm", "pocket_residue_esm", "ligand_pocket_residue_esm", "full_residue_esm"}:
            raise ValueError("LMDB datasets support protein_3d_mode residue, all_atom, pocket_pharmacophore, pocket_pharmacophore_esm, pocket_residue_esm, or full_residue_esm")

        self.lmdb_path = lmdb_path
        self.structure_lmdb_path = structure_lmdb_path
        self.samples = manifest["samples"]
        self.protein_3d_mode = protein_3d_mode
        self.protein_edge_mode = protein_edge_mode
        self.protein_contact_cutoff = protein_contact_cutoff
        self.protein_knn_k = protein_knn_k
        self.protein_embedding_mode = protein_embedding_mode
        self.expected_protein_embedding_dim = expected_protein_embedding_dim
        self.ligand_pocket_cutoff = float(ligand_pocket_cutoff)
        if not np.isfinite(self.ligand_pocket_cutoff) or self.ligand_pocket_cutoff <= 0:
            raise ValueError(f"ligand_pocket_cutoff must be positive, got {ligand_pocket_cutoff}")
        self.disable_protein_2d = bool(disable_protein_2d)
        self.dti_fusion_lmdb_path = dti_fusion_lmdb_path
        self.ligand_unimol_lmdb_path = ligand_unimol_lmdb_path
        self.ligand_unimol2_lmdb_path = ligand_unimol2_lmdb_path
        self.pbcnet_fusion_lmdb_path = pbcnet_fusion_lmdb_path
        self.pbcnet_graph_lmdb_path = pbcnet_graph_lmdb_path
        self.boltz_interface_lmdb_path = boltz_interface_lmdb_path
        self.boltz_pocket_radius = float(boltz_pocket_radius)
        self.boltz_min_pocket_residues = int(boltz_min_pocket_residues)
        self.boltz_max_pocket_residues = int(boltz_max_pocket_residues)
        if self.boltz_pocket_radius <= 0 or self.boltz_min_pocket_residues <= 0 or self.boltz_max_pocket_residues < self.boltz_min_pocket_residues:
            raise ValueError("Invalid Boltz interface pocket configuration")
        self._env = None
        self._structure_env = None
        self._dti_fusion_env = None
        self._ligand_unimol_env = None
        self._ligand_unimol2_env = None
        self._pbcnet_fusion_env = None
        self._pbcnet_graph_env = None
        self._boltz_interface_env = None

    def __len__(self):
        return len(self.samples)

    def _open_lmdb(self, path):
        return lmdb.open(
            path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=True,
        )

    def _get_env(self):
        if self._env is None:
            self._env = self._open_lmdb(self.lmdb_path)
        return self._env

    def _get_structure_env(self):
        if self.structure_lmdb_path == self.lmdb_path:
            return self._get_env()
        if self._structure_env is None:
            self._structure_env = self._open_lmdb(self.structure_lmdb_path)
        return self._structure_env

    def _get_dti_fusion_env(self):
        if self.dti_fusion_lmdb_path is None:
            return None
        if self._dti_fusion_env is None:
            self._dti_fusion_env = self._open_lmdb(self.dti_fusion_lmdb_path)
        return self._dti_fusion_env

    def _get_pbcnet_fusion_env(self):
        if self.pbcnet_fusion_lmdb_path is None:
            return None
        if self._pbcnet_fusion_env is None:
            self._pbcnet_fusion_env = self._open_lmdb(self.pbcnet_fusion_lmdb_path)
        return self._pbcnet_fusion_env

    def _get_pbcnet_graph_env(self):
        if self.pbcnet_graph_lmdb_path is None:
            return None
        if self._pbcnet_graph_env is None:
            self._pbcnet_graph_env = self._open_lmdb(self.pbcnet_graph_lmdb_path)
        return self._pbcnet_graph_env

    def _get_boltz_interface_env(self):
        if self.boltz_interface_lmdb_path is None:
            return None
        if self._boltz_interface_env is None:
            self._boltz_interface_env = self._open_lmdb(self.boltz_interface_lmdb_path)
        return self._boltz_interface_env

    def _get_ligand_unimol_env(self):
        if self.ligand_unimol_lmdb_path is None:
            return None
        if self._ligand_unimol_env is None:
            self._ligand_unimol_env = self._open_lmdb(self.ligand_unimol_lmdb_path)
        return self._ligand_unimol_env

    def _get_ligand_unimol2_env(self):
        if self.ligand_unimol2_lmdb_path is None:
            return None
        if self._ligand_unimol2_env is None:
            self._ligand_unimol2_env = self._open_lmdb(self.ligand_unimol2_lmdb_path)
        return self._ligand_unimol2_env

    def _load_item(self, env, key, source_name):
        with env.begin(write=False) as txn:
            raw = copy_lmdb_value(txn.get(key.encode("utf-8")))
        if raw is None:
            raise KeyError(f"{source_name} LMDB key not found: {key}")
        return pickle.loads(raw)

    def __getitem__(self, index):
        sample = self.samples[index]
        key = sample["lmdb_key"]
        feature_item = self._load_item(self._get_env(), key, "feature")
        structure_item = self._load_item(self._get_structure_env(), key, "structure")
        dti_fusion = None
        ligand_unimol = None
        ligand_unimol2 = None
        pbcnet_fusion = None
        pbcnet_graph = None
        boltz_interface = None
        dti_fusion_env = self._get_dti_fusion_env()
        if dti_fusion_env is not None:
            dti_key = sample.get("dti_fusion_key", key)
            dti_payload = self._load_item(dti_fusion_env, dti_key, "DTI fusion")
            dti_fusion = dti_payload
        pbcnet_fusion_env = self._get_pbcnet_fusion_env()
        if pbcnet_fusion_env is not None:
            pbcnet_key = sample.get("pbcnet_fusion_key", key)
            pbcnet_payload = self._load_item(pbcnet_fusion_env, pbcnet_key, "PBCNet fusion")
            pbcnet_fusion = pbcnet_payload

        pbcnet_graph_env = self._get_pbcnet_graph_env()
        if pbcnet_graph_env is not None:
            pbcnet_graph_key = sample.get("pbcnet_graph_key", key)
            pbcnet_graph_payload = self._load_item(pbcnet_graph_env, pbcnet_graph_key, "PBCNet graph")
            if isinstance(pbcnet_graph_payload, dict):
                pbcnet_graph = pbcnet_graph_payload
            else:
                pbcnet_graph = {"payload_type": "pbcnet_graph", "graph": pbcnet_graph_payload}

        boltz_interface_env = self._get_boltz_interface_env()
        if boltz_interface_env is not None:
            boltz_interface_key = sample.get("boltz_interface_key", key)
            boltz_interface_payload = self._load_item(boltz_interface_env, boltz_interface_key, "Boltz interface")
            boltz_interface = build_boltz_interface_payload(
                boltz_interface_payload,
                pocket_radius=self.boltz_pocket_radius,
                min_pocket_residues=self.boltz_min_pocket_residues,
                max_pocket_residues=self.boltz_max_pocket_residues,
            )

        feature_chains = ordered_chains(feature_item["protein_features"])
        structure_chains = ordered_chains(structure_item["protein_features"])
        protein_repre = build_residue_representations(
            feature_chains,
            embedding_mode=self.protein_embedding_mode,
            expected_dim=self.expected_protein_embedding_dim,
        )
        if self.disable_protein_2d:
            protein_graph = build_disabled_protein_graph(feature_dim=self.expected_protein_embedding_dim)
        else:
            protein_graph = build_protein_graph(
                feature_chains,
                edge_mode=self.protein_edge_mode,
                cutoff=self.protein_contact_cutoff,
                knn_k=self.protein_knn_k,
                residue_repre=protein_repre,
                embedding_mode=self.protein_embedding_mode,
                expected_dim=self.expected_protein_embedding_dim,
            )
        mol_graph, mol_feats, mol_coords, mol_las = ligand_tuple_from_lmdb(feature_item["ligand_features"])
        if self.protein_3d_mode == "pocket_pharmacophore_esm":
            protein_3d_tokens, protein_3d_coords = build_pocket_pharmacophore_esm_inputs(
                feature_chains,
                structure_chains,
                protein_repre,
            )
        elif self.protein_3d_mode == "pocket_residue_esm":
            protein_3d_tokens, protein_3d_coords = build_pocket_residue_esm_inputs(
                feature_chains,
                structure_chains,
                protein_repre,
            )
        elif self.protein_3d_mode == "ligand_pocket_residue_esm":
            protein_3d_tokens, protein_3d_coords = build_ligand_pocket_residue_esm_inputs(
                feature_chains,
                protein_repre,
                mol_coords,
                cutoff=self.ligand_pocket_cutoff,
            )
        elif self.protein_3d_mode == "full_residue_esm":
            protein_3d_tokens, protein_3d_coords = build_full_residue_esm_inputs(
                feature_chains,
                protein_repre=protein_repre,
                expected_dim=self.expected_protein_embedding_dim,
            )
        else:
            protein_3d_tokens, protein_3d_coords, _ = select_lmdb_protein_3d_inputs(
                structure_chains,
                protein_3d_mode=self.protein_3d_mode,
            )
        ca_coords = build_chain_ca_coords(feature_chains)
        ligand_unimol_env = self._get_ligand_unimol_env()
        if ligand_unimol_env is not None:
            ligand_key = sample.get("ligand_unimol_key", key)
            with ligand_unimol_env.begin(write=False) as txn:
                ligand_payload = copy_lmdb_value(txn.get(ligand_key.encode("utf-8")))
                if ligand_payload is None and not ligand_key.startswith("raw:"):
                    ligand_payload = copy_lmdb_value(txn.get(("raw:" + ligand_key).encode("utf-8")))
            if ligand_payload is None:
                raise KeyError(f"Ligand Uni-Mol LMDB key not found: {ligand_key}")
            ligand_unimol = normalize_ligand_unimol_payload(ligand_payload, expected_ligand_length=None)
            ligand_unimol = align_ligand_unimol_to_source_atoms(
                ligand_unimol,
                feature_item["ligand_features"]["mol"],
                expected_ligand_length=len(mol_feats),
            )
        ligand_unimol2_env = self._get_ligand_unimol2_env()
        if ligand_unimol2_env is not None:
            ligand_key = sample.get("ligand_unimol2_key", key)
            with ligand_unimol2_env.begin(write=False) as txn:
                ligand_payload = copy_lmdb_value(txn.get(ligand_key.encode("utf-8")))
                if ligand_payload is None and not ligand_key.startswith("raw:"):
                    ligand_payload = copy_lmdb_value(txn.get(("raw:" + ligand_key).encode("utf-8")))
            if ligand_payload is None:
                raise KeyError(f"Ligand Uni-Mol2 LMDB key not found: {ligand_key}")
            ligand_unimol2 = normalize_ligand_unimol_payload(ligand_payload, expected_ligand_length=None)
            ligand_unimol2 = align_ligand_unimol_to_source_atoms(
                ligand_unimol2,
                feature_item["ligand_features"]["mol"],
                expected_ligand_length=len(mol_feats),
            )
            ligand_unimol2["payload_type"] = "ligand_unimol2"
        label = float(sample.get("label", feature_item.get("affinity")))
        if dti_fusion is not None:
            dti_fusion = normalize_dti_fusion_payload(dti_fusion, expected_protein_length=len(protein_repre))
        if pbcnet_fusion is not None:
            pbcnet_fusion = normalize_pbcnet_fusion_payload(pbcnet_fusion)
        if pbcnet_graph is not None:
            pbcnet_graph["payload_type"] = "pbcnet_graph"

        item = (
            protein_repre,
            mol_graph,
            protein_graph,
            protein_3d_tokens,
            ca_coords,
            protein_3d_coords,
            mol_feats,
            mol_coords,
            mol_las,
            label,
        )
        extras = ()
        if dti_fusion is not None:
            extras = extras + (dti_fusion,)
        if ligand_unimol is not None:
            extras = extras + (ligand_unimol,)
        if ligand_unimol2 is not None:
            extras = extras + (ligand_unimol2,)
        if pbcnet_fusion is not None:
            extras = extras + (pbcnet_fusion,)
        if pbcnet_graph is not None:
            extras = extras + (pbcnet_graph,)
        if boltz_interface is not None:
            extras = extras + (boltz_interface,)
        if extras:
            return item + extras
        return item


def lmdb_manifest_labels(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return [float(sample["label"]) for sample in manifest["samples"]]
