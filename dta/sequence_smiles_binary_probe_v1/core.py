from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


DEFAULT_THRESHOLDS = {
    "DAVIS": 7.522878745280337,  # pKd corresponding to Kd <= 30 nM
    "KIBA": 12.1,
}

_INDEX_MAGIC = b"MMIDIDX\x00\x00"
_DTYPES = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}


@dataclass
class PairSplit:
    smiles: List[str]
    proteins: List[str]
    affinity: np.ndarray

    def __len__(self) -> int:
        return len(self.smiles)


def _read_dictionary(path: Path) -> List[str]:
    tokens: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            tokens.append(line.rsplit(" ", 1)[0])
    if not tokens:
        raise ValueError(f"empty dictionary: {path}")
    return tokens


def _read_mmap_sequences(prefix: Path, dictionary: Sequence[str]) -> List[str]:
    idx_path = prefix.with_suffix(".idx")
    bin_path = prefix.with_suffix(".bin")
    raw_index = idx_path.read_bytes()
    if not raw_index.startswith(_INDEX_MAGIC):
        raise ValueError(f"unsupported mmap index magic: {idx_path}")
    if len(raw_index) < 26:
        raise ValueError(f"truncated mmap index: {idx_path}")
    dtype_code = raw_index[17]
    if dtype_code not in _DTYPES:
        raise ValueError(f"unsupported mmap dtype code {dtype_code}: {idx_path}")
    dtype = np.dtype(_DTYPES[dtype_code]).newbyteorder("<")
    count = struct.unpack_from("<Q", raw_index, 18)[0]
    sizes_offset = 26
    pointers_offset = sizes_offset + 4 * count
    expected = pointers_offset + 8 * count
    if len(raw_index) < expected:
        raise ValueError(f"truncated mmap sizes/pointers: {idx_path}")
    sizes = np.frombuffer(raw_index, dtype="<i4", count=count, offset=sizes_offset)
    pointers = np.frombuffer(raw_index, dtype="<i8", count=count, offset=pointers_offset)
    binary = np.memmap(bin_path, mode="r", dtype=np.uint8)

    decoded: List[str] = []
    for size, pointer in zip(sizes.tolist(), pointers.tolist()):
        byte_count = int(size) * dtype.itemsize
        stop = int(pointer) + byte_count
        if pointer < 0 or stop > binary.size:
            raise ValueError(f"invalid mmap pointer in {idx_path}")
        token_ids = np.frombuffer(binary, dtype=dtype, count=int(size), offset=int(pointer))
        pieces: List[str] = []
        for token_id in token_ids.tolist():
            # Fairseq reserves 0..3 for bos/pad/eos/unk; data uses 0..2 as framing.
            if token_id in (0, 1, 2):
                continue
            dictionary_index = int(token_id) - 4
            if dictionary_index < 0 or dictionary_index >= len(dictionary):
                pieces.append("?")
            else:
                pieces.append(dictionary[dictionary_index])
        decoded.append(sys.intern("".join(pieces)))
    return decoded


def load_split(dataset_dir: os.PathLike[str] | str, split: str) -> PairSplit:
    root = Path(dataset_dir)
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"invalid split: {split}")
    mol_dictionary = _read_dictionary(root / "input0" / "dict.txt")
    protein_dictionary = _read_dictionary(root / "input1" / "dict.txt")
    smiles = _read_mmap_sequences(root / "input0" / split, mol_dictionary)
    proteins = _read_mmap_sequences(root / "input1" / split, protein_dictionary)
    affinity = np.loadtxt(root / "label" / f"{split}.label", dtype=np.float64, ndmin=1)
    if len(smiles) != len(proteins) or len(smiles) != affinity.shape[0]:
        raise ValueError(
            f"split length mismatch for {root}/{split}: "
            f"smiles={len(smiles)} proteins={len(proteins)} labels={affinity.shape[0]}"
        )
    if not np.isfinite(affinity).all():
        raise ValueError(f"non-finite labels in {root}/{split}")
    return PairSplit(smiles=smiles, proteins=proteins, affinity=affinity)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def split_fingerprints(dataset_dir: Path, splits: Iterable[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    common = [Path("input0/dict.txt"), Path("input1/dict.txt")]
    paths = list(common)
    for split in splits:
        paths.extend(
            [
                Path("input0") / f"{split}.idx",
                Path("input0") / f"{split}.bin",
                Path("input1") / f"{split}.idx",
                Path("input1") / f"{split}.bin",
                Path("label") / f"{split}.label",
            ]
        )
    for relative in paths:
        result[str(relative)] = sha256_file(dataset_dir / relative)
    return result


def _sequence_descriptors(sequence: str) -> np.ndarray:
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    length = max(len(sequence), 1)
    fractions = [sequence.count(aa) / length for aa in amino_acids]
    groups = [
        sum(sequence.count(aa) for aa in "AILMFWVY") / length,
        sum(sequence.count(aa) for aa in "DEKR") / length,
        sum(sequence.count(aa) for aa in "FWY") / length,
        sum(sequence.count(aa) for aa in "GP") / length,
    ]
    return np.asarray([math.log1p(len(sequence)), *fractions, *groups], dtype=np.float32)


def _molecule_matrix(smiles_values: Sequence[str], n_bits: int) -> Tuple[sparse.csr_matrix, np.ndarray]:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors

    RDLogger.DisableLog("rdApp.warning")
    rows: List[np.ndarray] = []
    descriptors: List[List[float]] = []
    for smiles in smiles_values:
        mol = Chem.MolFromSmiles(smiles)
        bits = np.zeros(n_bits, dtype=np.uint8)
        if mol is None:
            rows.append(bits)
            descriptors.append([0.0] * 12)
            continue
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fingerprint, bits)
        rows.append(bits)
        descriptors.append(
            [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                rdMolDescriptors.CalcTPSA(mol),
                float(Lipinski.NumHDonors(mol)),
                float(Lipinski.NumHAcceptors(mol)),
                float(Lipinski.NumRotatableBonds(mol)),
                float(Lipinski.RingCount(mol)),
                float(Lipinski.HeavyAtomCount(mol)),
                float(rdMolDescriptors.CalcFractionCSP3(mol)),
                float(rdMolDescriptors.CalcNumAromaticRings(mol)),
                float(rdMolDescriptors.CalcNumAliphaticRings(mol)),
                float(rdMolDescriptors.CalcNumHeteroatoms(mol)),
            ]
        )
    return sparse.csr_matrix(np.stack(rows)), np.asarray(descriptors, dtype=np.float32)


class RepresentationEncoder:
    def __init__(
        self,
        protein_hash_dim: int = 4096,
        protein_svd_dim: int = 96,
        molecule_bits: int = 2048,
        molecule_svd_dim: int = 96,
        interaction_dim: int = 8,
        seed: int = 20260902,
    ) -> None:
        self.protein_hash_dim = int(protein_hash_dim)
        self.protein_svd_dim = int(protein_svd_dim)
        self.molecule_bits = int(molecule_bits)
        self.molecule_svd_dim = int(molecule_svd_dim)
        self.interaction_dim = int(interaction_dim)
        self.seed = int(seed)
        self.protein_vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(1, 3),
            lowercase=False,
            n_features=self.protein_hash_dim,
            alternate_sign=False,
            norm="l2",
        )
        self.protein_svd: TruncatedSVD | None = None
        self.molecule_svd: TruncatedSVD | None = None
        self.protein_scaler = StandardScaler()
        self.molecule_scaler = StandardScaler()

    @staticmethod
    def _unique(values: Sequence[str]) -> List[str]:
        return list(dict.fromkeys(values))

    def fit(self, smiles: Sequence[str], proteins: Sequence[str]) -> "RepresentationEncoder":
        unique_proteins = self._unique(proteins)
        unique_smiles = self._unique(smiles)
        protein_sparse = self.protein_vectorizer.transform(unique_proteins)
        protein_components = max(
            1, min(self.protein_svd_dim, protein_sparse.shape[0] - 1, protein_sparse.shape[1] - 1)
        )
        self.protein_svd = TruncatedSVD(
            n_components=protein_components, n_iter=7, random_state=self.seed
        ).fit(protein_sparse)
        protein_descriptors = np.stack([_sequence_descriptors(value) for value in unique_proteins])
        self.protein_scaler.fit(protein_descriptors)

        molecule_sparse, molecule_descriptors = _molecule_matrix(unique_smiles, self.molecule_bits)
        molecule_components = max(
            1, min(self.molecule_svd_dim, molecule_sparse.shape[0] - 1, molecule_sparse.shape[1] - 1)
        )
        self.molecule_svd = TruncatedSVD(
            n_components=molecule_components, n_iter=7, random_state=self.seed + 1
        ).fit(molecule_sparse)
        self.molecule_scaler.fit(molecule_descriptors)
        return self

    def _protein_lookup(self, values: Sequence[str]) -> Mapping[str, np.ndarray]:
        if self.protein_svd is None:
            raise RuntimeError("representation encoder is not fitted")
        unique = self._unique(values)
        latent = self.protein_svd.transform(self.protein_vectorizer.transform(unique)).astype(np.float32)
        descriptors = self.protein_scaler.transform(
            np.stack([_sequence_descriptors(value) for value in unique])
        ).astype(np.float32)
        combined = np.concatenate([latent, descriptors], axis=1)
        return dict(zip(unique, combined))

    def _molecule_lookup(self, values: Sequence[str]) -> Mapping[str, np.ndarray]:
        if self.molecule_svd is None:
            raise RuntimeError("representation encoder is not fitted")
        unique = self._unique(values)
        fingerprints, descriptors = _molecule_matrix(unique, self.molecule_bits)
        latent = self.molecule_svd.transform(fingerprints).astype(np.float32)
        scaled = self.molecule_scaler.transform(descriptors).astype(np.float32)
        combined = np.concatenate([latent, scaled], axis=1)
        return dict(zip(unique, combined))

    def transform(self, smiles: Sequence[str], proteins: Sequence[str]) -> np.ndarray:
        protein_lookup = self._protein_lookup(proteins)
        molecule_lookup = self._molecule_lookup(smiles)
        protein = np.stack([protein_lookup[value] for value in proteins]).astype(np.float32)
        molecule = np.stack([molecule_lookup[value] for value in smiles]).astype(np.float32)
        common = min(
            self.interaction_dim,
            self.protein_svd.n_components if self.protein_svd is not None else 0,
            self.molecule_svd.n_components if self.molecule_svd is not None else 0,
        )
        if common:
            outer = np.einsum(
                "ni,nj->nij", protein[:, :common], molecule[:, :common], optimize=True
            ).reshape(len(smiles), common * common)
        else:
            outer = np.empty((len(smiles), 0), dtype=np.float32)
        return np.concatenate([protein, molecule, outer], axis=1).astype(np.float32)


def _aggregate(values: Sequence[str], targets: np.ndarray) -> Tuple[Dict[str, float], Dict[str, int]]:
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for key, target in zip(values, targets.tolist()):
        sums[key] += float(target)
        counts[key] += 1
    return dict(sums), dict(counts)


def _aggregate_pairs(
    smiles: Sequence[str], proteins: Sequence[str], targets: np.ndarray
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], int]]:
    sums: Dict[Tuple[str, str], float] = defaultdict(float)
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for mol, protein, target in zip(smiles, proteins, targets.tolist()):
        key = (mol, protein)
        sums[key] += float(target)
        counts[key] += 1
    return dict(sums), dict(counts)


class TargetStatisticsEncoder:
    feature_names = [
        "drug_affinity_mean",
        "protein_affinity_mean",
        "pair_affinity_mean",
        "additive_affinity",
        "drug_positive_rate",
        "protein_positive_rate",
        "pair_positive_rate",
        "additive_positive_rate",
        "log_drug_count",
        "log_protein_count",
        "log_pair_count",
        "pair_seen",
        "drug_protein_affinity_gap",
        "drug_protein_positive_gap",
    ]

    def __init__(self, affinity_alpha: float = 5.0, binary_alpha: float = 10.0) -> None:
        self.affinity_alpha = float(affinity_alpha)
        self.binary_alpha = float(binary_alpha)

    def fit(
        self,
        smiles: Sequence[str],
        proteins: Sequence[str],
        affinity: np.ndarray,
        binary: np.ndarray,
    ) -> "TargetStatisticsEncoder":
        self.global_affinity = float(np.mean(affinity))
        self.global_binary = float(np.mean(binary))
        self.drug_aff_sum, self.drug_count = _aggregate(smiles, affinity)
        self.protein_aff_sum, self.protein_count = _aggregate(proteins, affinity)
        self.pair_aff_sum, self.pair_count = _aggregate_pairs(smiles, proteins, affinity)
        self.drug_bin_sum, _ = _aggregate(smiles, binary)
        self.protein_bin_sum, _ = _aggregate(proteins, binary)
        self.pair_bin_sum, _ = _aggregate_pairs(smiles, proteins, binary)
        self.train_pairs = set(self.pair_count)
        return self

    @staticmethod
    def _smoothed(total: float, count: int, prior: float, alpha: float) -> float:
        return (float(total) + alpha * prior) / (int(count) + alpha)

    def transform(
        self,
        smiles: Sequence[str],
        proteins: Sequence[str],
        affinity: np.ndarray | None = None,
        binary: np.ndarray | None = None,
        leave_one_out: bool = False,
    ) -> np.ndarray:
        if leave_one_out and (affinity is None or binary is None):
            raise ValueError("leave-one-out encoding requires targets")
        rows: List[List[float]] = []
        for index, (mol, protein) in enumerate(zip(smiles, proteins)):
            pair = (mol, protein)
            subtract_aff = float(affinity[index]) if leave_one_out else 0.0
            subtract_bin = float(binary[index]) if leave_one_out else 0.0
            decrement = 1 if leave_one_out else 0
            drug_count = max(self.drug_count.get(mol, 0) - decrement, 0)
            protein_count = max(self.protein_count.get(protein, 0) - decrement, 0)
            pair_count = max(self.pair_count.get(pair, 0) - decrement, 0)
            drug_aff = self._smoothed(
                self.drug_aff_sum.get(mol, 0.0) - subtract_aff,
                drug_count,
                self.global_affinity,
                self.affinity_alpha,
            )
            protein_aff = self._smoothed(
                self.protein_aff_sum.get(protein, 0.0) - subtract_aff,
                protein_count,
                self.global_affinity,
                self.affinity_alpha,
            )
            additive_aff = drug_aff + protein_aff - self.global_affinity
            pair_aff = self._smoothed(
                self.pair_aff_sum.get(pair, 0.0) - subtract_aff,
                pair_count,
                additive_aff,
                self.affinity_alpha,
            )
            drug_bin = self._smoothed(
                self.drug_bin_sum.get(mol, 0.0) - subtract_bin,
                drug_count,
                self.global_binary,
                self.binary_alpha,
            )
            protein_bin = self._smoothed(
                self.protein_bin_sum.get(protein, 0.0) - subtract_bin,
                protein_count,
                self.global_binary,
                self.binary_alpha,
            )
            additive_bin = float(np.clip(drug_bin + protein_bin - self.global_binary, 0.0, 1.0))
            pair_bin = self._smoothed(
                self.pair_bin_sum.get(pair, 0.0) - subtract_bin,
                pair_count,
                additive_bin,
                self.binary_alpha,
            )
            rows.append(
                [
                    drug_aff,
                    protein_aff,
                    pair_aff,
                    additive_aff,
                    drug_bin,
                    protein_bin,
                    pair_bin,
                    additive_bin,
                    math.log1p(drug_count),
                    math.log1p(protein_count),
                    math.log1p(pair_count),
                    float(pair_count > 0),
                    abs(drug_aff - protein_aff),
                    abs(drug_bin - protein_bin),
                ]
            )
        return np.asarray(rows, dtype=np.float32)

    def novel_pair_mask(self, smiles: Sequence[str], proteins: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [(mol, protein) not in self.train_pairs for mol, protein in zip(smiles, proteins)],
            dtype=bool,
        )


class BiasSVDModel:
    def __init__(self, rank: int = 32, bias_regularization: float = 10.0, seed: int = 20260902):
        self.rank = int(rank)
        self.bias_regularization = float(bias_regularization)
        self.seed = int(seed)

    def fit(
        self, smiles: Sequence[str], proteins: Sequence[str], affinity: np.ndarray
    ) -> "BiasSVDModel":
        unique_smiles = list(dict.fromkeys(smiles))
        unique_proteins = list(dict.fromkeys(proteins))
        self.drug_index = {value: index for index, value in enumerate(unique_smiles)}
        self.protein_index = {value: index for index, value in enumerate(unique_proteins)}
        drug_ids = np.asarray([self.drug_index[value] for value in smiles], dtype=np.int32)
        protein_ids = np.asarray([self.protein_index[value] for value in proteins], dtype=np.int32)
        self.global_mean = float(np.mean(affinity))
        self.drug_bias = np.zeros(len(unique_smiles), dtype=np.float64)
        self.protein_bias = np.zeros(len(unique_proteins), dtype=np.float64)
        drug_count = np.bincount(drug_ids, minlength=len(unique_smiles)).astype(np.float64)
        protein_count = np.bincount(protein_ids, minlength=len(unique_proteins)).astype(np.float64)
        for _ in range(12):
            residual = affinity - self.global_mean - self.protein_bias[protein_ids]
            self.drug_bias = np.bincount(
                drug_ids, weights=residual, minlength=len(unique_smiles)
            ) / (drug_count + self.bias_regularization)
            residual = affinity - self.global_mean - self.drug_bias[drug_ids]
            self.protein_bias = np.bincount(
                protein_ids, weights=residual, minlength=len(unique_proteins)
            ) / (protein_count + self.bias_regularization)

        pair_sum: Dict[Tuple[int, int], float] = defaultdict(float)
        pair_count: Dict[Tuple[int, int], int] = defaultdict(int)
        residual = affinity - self.global_mean - self.drug_bias[drug_ids] - self.protein_bias[protein_ids]
        for drug, protein, value in zip(drug_ids.tolist(), protein_ids.tolist(), residual.tolist()):
            pair_sum[(drug, protein)] += value
            pair_count[(drug, protein)] += 1
        keys = list(pair_sum)
        rows = np.asarray([key[0] for key in keys], dtype=np.int32)
        columns = np.asarray([key[1] for key in keys], dtype=np.int32)
        values = np.asarray([pair_sum[key] / pair_count[key] for key in keys], dtype=np.float64)
        residual_matrix = sparse.csr_matrix(
            (values, (rows, columns)), shape=(len(unique_smiles), len(unique_proteins))
        )
        components = max(1, min(self.rank, min(residual_matrix.shape) - 1))
        self.svd = TruncatedSVD(n_components=components, n_iter=10, random_state=self.seed)
        self.drug_latent = self.svd.fit_transform(residual_matrix).astype(np.float32)
        self.protein_components = self.svd.components_.astype(np.float32)
        return self

    def predict(
        self, smiles: Sequence[str], proteins: Sequence[str], rank: int | None = None
    ) -> np.ndarray:
        effective_rank = min(rank or self.rank, self.drug_latent.shape[1])
        result = np.full(len(smiles), self.global_mean, dtype=np.float64)
        for index, (mol, protein) in enumerate(zip(smiles, proteins)):
            drug_id = self.drug_index.get(mol)
            protein_id = self.protein_index.get(protein)
            if drug_id is not None:
                result[index] += self.drug_bias[drug_id]
            if protein_id is not None:
                result[index] += self.protein_bias[protein_id]
            if drug_id is not None and protein_id is not None:
                result[index] += float(
                    np.dot(
                        self.drug_latent[drug_id, :effective_rank],
                        self.protein_components[:effective_rank, protein_id],
                    )
                )
        return result


def affinity_to_score(prediction: np.ndarray, threshold: float, scale: float) -> np.ndarray:
    logits = np.clip((np.asarray(prediction) - threshold) / scale, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def best_decision_threshold(
    labels: np.ndarray, scores: np.ndarray, objective: str = "mcc"
) -> Tuple[float, Dict[str, float]]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels)
    cumulative_fp = np.cumsum(1 - sorted_labels)
    boundaries = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    tp = cumulative_tp[boundaries].astype(np.float64)
    fp = cumulative_fp[boundaries].astype(np.float64)
    positives = float(labels.sum())
    negatives = float(labels.size - labels.sum())
    fn = positives - tp
    tn = negatives - fp
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denominator, out=np.zeros_like(tp), where=denominator > 0)
    f1_denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, f1_denominator, out=np.zeros_like(tp), where=f1_denominator > 0)
    if objective == "mcc":
        ranking = np.lexsort((boundaries, f1, mcc))
    elif objective == "f1":
        ranking = np.lexsort((boundaries, mcc, f1))
    else:
        raise ValueError(f"decision objective must be mcc or f1, got {objective}")
    best = int(ranking[-1])
    threshold = float(sorted_scores[boundaries[best]])
    return threshold, {"MCC": float(mcc[best]), "F1": float(f1[best])}


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    auroc = roc_auc_score(labels, scores) if np.unique(labels).size == 2 else float("nan")
    auprc = average_precision_score(labels, scores) if np.unique(labels).size == 2 else float("nan")
    return {
        "ACC": float(accuracy_score(labels, predictions)),
        "Precision": float(precision),
        "Recall": float(sensitivity),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "MCC": float(matthews_corrcoef(labels, predictions)),
        "ROC-AUC": float(auroc),
        "PR-AUC": float(auprc),
    }


def expression_score(expression: Mapping[str, object], base_scores: Mapping[str, np.ndarray]) -> np.ndarray:
    if expression["kind"] == "base":
        return np.asarray(base_scores[str(expression["name"])], dtype=np.float64)
    if expression["kind"] == "blend":
        weight = float(expression["left_weight"])
        return weight * np.asarray(base_scores[str(expression["left"])]) + (1.0 - weight) * np.asarray(
            base_scores[str(expression["right"])]
        )
    raise ValueError(f"invalid score expression: {expression}")


def select_score_expression(
    labels: np.ndarray, base_scores: Mapping[str, np.ndarray]
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    candidates: List[Dict[str, object]] = []
    names = sorted(base_scores)
    expressions: List[Dict[str, object]] = [
        {"kind": "base", "name": name} for name in names
    ]
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            for weight in (0.25, 0.5, 0.75):
                expressions.append(
                    {"kind": "blend", "left": left, "right": right, "left_weight": weight}
                )
    for expression in expressions:
        score = expression_score(expression, base_scores)
        candidates.append(
            {
                "expression": expression,
                "AUPRC": float(average_precision_score(labels, score)),
                "AUROC": float(roc_auc_score(labels, score)),
            }
        )
    candidates.sort(key=lambda item: (item["AUPRC"], item["AUROC"]), reverse=True)
    return dict(candidates[0]["expression"]), candidates


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def fit_models(
    features: np.ndarray,
    statistics_width: int,
    affinity: np.ndarray,
    binary: np.ndarray,
    max_iter: int,
    seed: int,
) -> Dict[str, object]:
    stats = features[:, -statistics_width:]
    common = dict(
        learning_rate=0.05,
        max_iter=int(max_iter),
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=int(seed),
    )
    models: Dict[str, object] = {
        "reg_all": HistGradientBoostingRegressor(loss="squared_error", **common),
        "cls_all": HistGradientBoostingClassifier(loss="log_loss", **common),
        "reg_stats": HistGradientBoostingRegressor(
            loss="squared_error", **{**common, "max_leaf_nodes": 15, "min_samples_leaf": 30}
        ),
        "cls_stats": HistGradientBoostingClassifier(
            loss="log_loss", **{**common, "max_leaf_nodes": 15, "min_samples_leaf": 30}
        ),
    }
    models["reg_all"].fit(features, affinity)
    models["cls_all"].fit(features, binary)
    models["reg_stats"].fit(stats, affinity)
    models["cls_stats"].fit(stats, binary)
    return models


def predict_base_scores(
    models: Mapping[str, object],
    bias_svd: BiasSVDModel,
    features: np.ndarray,
    statistics_width: int,
    smiles: Sequence[str],
    proteins: Sequence[str],
    affinity_threshold: float,
    affinity_scale: float,
) -> Dict[str, np.ndarray]:
    stats = features[:, -statistics_width:]
    scores = {
        "reg_all": affinity_to_score(models["reg_all"].predict(features), affinity_threshold, affinity_scale),
        "cls_all": models["cls_all"].predict_proba(features)[:, 1],
        "reg_stats": affinity_to_score(models["reg_stats"].predict(stats), affinity_threshold, affinity_scale),
        "cls_stats": models["cls_stats"].predict_proba(stats)[:, 1],
    }
    for rank in (8, 16, 32):
        scores[f"bias_svd_rank{rank}"] = affinity_to_score(
            bias_svd.predict(smiles, proteins, rank=rank), affinity_threshold, affinity_scale
        )
    return {key: np.asarray(value, dtype=np.float64) for key, value in scores.items()}


def save_bundle(path: Path, bundle: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(dict(bundle), temporary, compress=3)
    temporary.replace(path)


def load_bundle(path: Path) -> Dict[str, object]:
    return joblib.load(path)
