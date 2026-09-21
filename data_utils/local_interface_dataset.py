import json
import os
import pickle

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from data_utils.local_interface import (
    resolve_source_lmdb,
    validate_local_interface_record,
)
from data_utils.local_interface_decoys import (
    apply_rigid_transform,
    select_epoch_decoys,
    validate_decoy_set,
)
from data_utils.local_interface_decoys_v2 import (
    BANK_NAMES as DECOY_V2_BANK_NAMES,
    select_epoch_decoys_v2,
    validate_decoy_set_v2,
)
from utils.experiment_integrity import hash_ordered_keys


ESM_DIMENSION = 640


def _open_readonly_lmdb(path):
    return lmdb.open(
        os.fspath(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=True,
    )


def _lmdb_keys(path):
    environment = _open_readonly_lmdb(path)
    try:
        with environment.begin(write=False) as transaction:
            return {
                key.decode("utf-8")
                for key in transaction.cursor().iternext(
                    keys=True, values=False
                )
                if not key.startswith(b"__")
            }
    finally:
        environment.close()


def _load_pickled(environment, key, source_name):
    with environment.begin(write=False) as transaction:
        raw = transaction.get(str(key).encode("utf-8"))
    if raw is None:
        raise KeyError(f"{source_name} LMDB key not found: {key}")
    return pickle.loads(raw)


def _sequence_to_string(sequence):
    if isinstance(sequence, np.ndarray):
        return "".join(
            str(value) for value in sequence.reshape(-1).tolist()
        )
    if isinstance(sequence, (list, tuple)):
        return "".join(str(value) for value in sequence)
    if isinstance(sequence, bytes):
        return sequence.decode("utf-8")
    return str(sequence)


def _ordered_chain_items(protein_features):
    if isinstance(protein_features, dict):
        return [
            (str(key), protein_features[key])
            for key in sorted(protein_features)
        ]
    return [
        (str(index), chain)
        for index, chain in enumerate(protein_features)
    ]


def _load_residue_context(payload, native_record, key):
    protein_features = payload.get("protein_features")
    if protein_features is None:
        raise ValueError(f"source record {key} has no protein_features")

    embeddings = []
    coordinates = []
    sequence = []
    for chain_id, chain in _ordered_chain_items(protein_features):
        chain_sequence = _sequence_to_string(
            chain.get("sequence", "")
        )
        if not chain_sequence:
            raise ValueError(
                f"source record {key} chain {chain_id} has no residues"
            )
        embedding = np.asarray(
            chain.get("esm2_embedding", []), dtype=np.float32
        )
        expected_embedding_shape = (
            len(chain_sequence),
            ESM_DIMENSION,
        )
        if embedding.shape != expected_embedding_shape:
            raise ValueError(
                f"source record {key} chain {chain_id} ESM shape "
                f"{embedding.shape} does not match "
                f"{len(chain_sequence)} residues"
            )
        ca_coords = np.asarray(
            chain.get("ca_coords", []), dtype=np.float32
        )
        if ca_coords.shape != (len(chain_sequence), 3):
            raise ValueError(
                f"source record {key} chain {chain_id} CA coordinates "
                "do not match its residues"
            )
        embeddings.append(embedding)
        coordinates.append(ca_coords)
        sequence.extend(chain_sequence)

    residue_esm = np.concatenate(embeddings, axis=0)
    residue_ca = np.concatenate(coordinates, axis=0)
    native_sequence = [
        str(value)
        for value in np.asarray(
            native_record["residue_sequence"]
        ).reshape(-1)
    ]
    if sequence != native_sequence:
        raise ValueError(
            f"source/native residue ordering does not match for {key}"
        )
    if residue_esm.shape[0] != int(native_record["residue_count"]):
        raise ValueError(
            f"source ESM residue count does not match native record {key}"
        )
    native_ca = np.asarray(
        native_record["residue_ca_coords"], dtype=np.float32
    )
    if (
        native_ca.shape != residue_ca.shape
        or not np.allclose(native_ca, residue_ca, atol=1e-5, rtol=0.0)
    ):
        raise ValueError(
            f"source/native CA residue coordinates do not match for {key}"
        )
    if not np.all(np.isfinite(residue_esm)):
        raise ValueError(f"source ESM contains nonfinite values for {key}")
    if not np.all(np.isfinite(residue_ca)):
        raise ValueError(
            f"source CA coordinates contain nonfinite values for {key}"
        )

    parents = np.asarray(
        native_record["protein_parent_residue"], dtype=np.int64
    )
    if (
        np.any(parents < 0)
        or np.any(parents >= residue_esm.shape[0])
    ):
        raise ValueError(
            f"protein parent-residue indices are invalid for ESM in {key}"
        )
    return residue_esm, residue_ca


def _to_torch(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"b", "i", "u", "f", "c"}:
            return torch.from_numpy(np.ascontiguousarray(value))
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            key: _to_torch(nested)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_to_torch(nested) for nested in value]
    if isinstance(value, tuple):
        return tuple(_to_torch(nested) for nested in value)
    return value


def recursive_to_device(value, device, non_blocking=False):
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {
            key: recursive_to_device(
                nested, device, non_blocking=non_blocking
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [
            recursive_to_device(
                nested, device, non_blocking=non_blocking
            )
            for nested in value
        ]
    if isinstance(value, tuple):
        return tuple(
            recursive_to_device(
                nested, device, non_blocking=non_blocking
            )
            for nested in value
        )
    return value


def collate_single_interface(batch):
    if len(batch) != 1:
        raise ValueError(
            "Interface-first loaders require batch_size=1"
        )
    return batch[0]


class NativeInterfaceDataset(Dataset):
    def __init__(self, manifest_path, native_lmdb_path):
        self.manifest_path = os.path.abspath(
            os.fspath(manifest_path)
        )
        with open(
            self.manifest_path, "r", encoding="utf-8"
        ) as handle:
            self.manifest = json.load(handle)
        samples = self.manifest.get("samples")
        if not isinstance(samples, list):
            raise ValueError("manifest samples must be a list")
        keys = [str(sample["lmdb_key"]) for sample in samples]
        if len(keys) != len(set(keys)):
            raise ValueError("manifest contains duplicate LMDB keys")
        self.samples = list(samples)
        self.source_lmdb_path = os.path.abspath(
            resolve_source_lmdb(
                self.manifest,
                os.path.dirname(self.manifest_path),
            )
        )
        self.native_lmdb_path = os.path.abspath(
            os.fspath(native_lmdb_path)
        )
        self._source_env = None
        self._native_env = None

    def __len__(self):
        return len(self.samples)

    def _ensure_environments(self):
        if self._source_env is None:
            self._source_env = _open_readonly_lmdb(
                self.source_lmdb_path
            )
        if self._native_env is None:
            self._native_env = _open_readonly_lmdb(
                self.native_lmdb_path
            )

    def __getitem__(self, index):
        sample = self.samples[index]
        key = str(sample["lmdb_key"])
        self._ensure_environments()
        payload = _load_pickled(
            self._source_env, key, "source structure"
        )
        native_record = _load_pickled(
            self._native_env, key, "native interface"
        )
        validate_local_interface_record(native_record)
        if str(native_record.get("key")) != key:
            raise ValueError(
                f"native interface record key does not match {key}"
            )

        residue_esm, residue_ca = _load_residue_context(
            payload, native_record, key
        )
        item = _to_torch(native_record)
        item["key"] = key
        item["label"] = torch.tensor(
            float(sample["label"]), dtype=torch.float32
        )
        item["residue_esm"] = torch.from_numpy(
            np.ascontiguousarray(residue_esm)
        )
        item["residue_ca_coords"] = torch.from_numpy(
            np.ascontiguousarray(residue_ca)
        )
        return item

    def close(self):
        for name in ("_source_env", "_native_env"):
            environment = getattr(self, name, None)
            if environment is not None:
                environment.close()
                setattr(self, name, None)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_source_env"] = None
        state["_native_env"] = None
        return state

    def __del__(self):
        self.close()


def validate_native_interface_inventory(manifest_path, native_lmdb_path):
    dataset = NativeInterfaceDataset(manifest_path, native_lmdb_path)
    try:
        expected = [str(sample["lmdb_key"]) for sample in dataset.samples]
        present = _lmdb_keys(dataset.native_lmdb_path)
        missing = [key for key in expected if key not in present]
        if missing:
            raise ValueError(
                "missing native interface records: "
                + ", ".join(missing[:5])
            )
        for index in range(len(dataset)):
            dataset[index]
        return {
            "sample_count": len(expected),
            "ordered_key_sha256": hash_ordered_keys(expected),
        }
    finally:
        dataset.close()


class PosePretrainDataset(NativeInterfaceDataset):
    def __init__(
        self,
        manifest_path,
        native_lmdb_path,
        decoy_lmdb_path,
        diagnostic_keys_path,
        seed,
        decoy_schema_version=1,
        decoy_bank=None,
        require_complete_banks=(),
    ):
        super().__init__(manifest_path, native_lmdb_path)
        if self.manifest.get("split") != "train":
            raise ValueError(
                "PosePretrainDataset requires manifest split=train"
            )
        self.diagnostic_keys_path = os.path.abspath(
            os.fspath(diagnostic_keys_path)
        )
        with open(
            self.diagnostic_keys_path, "r", encoding="utf-8"
        ) as handle:
            diagnostic = json.load(handle)
        diagnostic_keys = diagnostic.get("keys")
        if not isinstance(diagnostic_keys, list):
            raise ValueError(
                "diagnostic key file must contain a keys list"
            )
        excluded = {str(key) for key in diagnostic_keys}
        if len(excluded) != len(diagnostic_keys):
            raise ValueError(
                "diagnostic key file contains duplicate keys"
            )
        manifest_keys = {
            str(sample["lmdb_key"]) for sample in self.samples
        }
        unknown = sorted(excluded - manifest_keys)
        if unknown:
            raise ValueError(
                "diagnostic keys are absent from train manifest: "
                + ", ".join(unknown[:5])
            )

        self.decoy_lmdb_path = os.path.abspath(
            os.fspath(decoy_lmdb_path)
        )
        self.decoy_schema_version = int(decoy_schema_version)
        self.decoy_bank = decoy_bank
        self.seed = int(seed)
        self.require_complete_banks = tuple(require_complete_banks)
        if self.decoy_schema_version == 1:
            if self.decoy_bank is not None:
                raise ValueError(
                    "decoy schema version 1 does not accept decoy_bank"
                )
            if self.require_complete_banks:
                raise ValueError(
                    "decoy schema version 1 does not accept "
                    "require_complete_banks"
                )
        elif self.decoy_schema_version == 2:
            if self.decoy_bank not in DECOY_V2_BANK_NAMES:
                raise ValueError(
                    "decoy schema version 2 requires decoy_bank to be "
                    + " or ".join(DECOY_V2_BANK_NAMES)
                )
            unknown_banks = sorted(
                set(self.require_complete_banks)
                - set(DECOY_V2_BANK_NAMES)
            )
            if unknown_banks:
                raise ValueError(
                    "unknown required decoy banks: "
                    + ", ".join(unknown_banks)
                )
            if len(set(self.require_complete_banks)) != len(
                self.require_complete_banks
            ):
                raise ValueError(
                    "require_complete_banks contains duplicates"
                )
            if self.decoy_bank not in self.require_complete_banks:
                raise ValueError(
                    "selected decoy_bank must be required complete"
                )
        else:
            raise ValueError("decoy_schema_version must be 1 or 2")

        decoy_keys = _lmdb_keys(self.decoy_lmdb_path)
        unknown_decoys = sorted(decoy_keys - manifest_keys)
        if unknown_decoys:
            raise ValueError(
                "decoy keys are absent from train manifest: "
                + ", ".join(unknown_decoys[:5])
            )
        eligible_samples = [
            sample
            for sample in self.samples
            if str(sample["lmdb_key"]) not in excluded
        ]
        self.stage1_missing_decoy_keys = [
            str(sample["lmdb_key"])
            for sample in eligible_samples
            if str(sample["lmdb_key"]) not in decoy_keys
        ]
        present_samples = [
            sample
            for sample in eligible_samples
            if str(sample["lmdb_key"]) in decoy_keys
        ]
        self.stage1_incomplete_bank_keys = []
        if self.decoy_schema_version == 2:
            complete_samples = []
            environment = _open_readonly_lmdb(self.decoy_lmdb_path)
            try:
                for sample in present_samples:
                    key = str(sample["lmdb_key"])
                    decoy_set = _load_pickled(
                        environment, key, "decoy"
                    )
                    validate_decoy_set_v2(decoy_set)
                    if decoy_set.get("seed") != self.seed:
                        raise ValueError(
                            "decoy record seed does not match "
                            + str(self.seed)
                            + ": "
                            + key
                        )
                    if str(decoy_set.get("key")) != key:
                        raise ValueError(
                            "decoy record key does not match " + key
                        )
                    if all(
                        decoy_set["banks"][bank]["complete"] is True
                        for bank in self.require_complete_banks
                    ):
                        complete_samples.append(sample)
                    else:
                        self.stage1_incomplete_bank_keys.append(key)
            finally:
                environment.close()
            self.samples = complete_samples
        else:
            self.samples = present_samples

        self.stage1_included_keys = [
            str(sample["lmdb_key"]) for sample in self.samples
        ]
        self.stage1_included_ordered_key_sha256 = hash_ordered_keys(
            self.stage1_included_keys
        )
        self.stage1_missing_decoy_ordered_key_sha256 = hash_ordered_keys(
            self.stage1_missing_decoy_keys
        )
        self.stage1_incomplete_bank_ordered_key_sha256 = hash_ordered_keys(
            self.stage1_incomplete_bank_keys
        )
        self.epoch = 0
        self._decoy_env = None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _ensure_decoy_environment(self):
        if self._decoy_env is None:
            self._decoy_env = _open_readonly_lmdb(
                self.decoy_lmdb_path
            )

    def __getitem__(self, index):
        item = super().__getitem__(index)
        key = item["key"]
        self._ensure_decoy_environment()
        decoy_set = _load_pickled(
            self._decoy_env, key, "decoy"
        )
        if self.decoy_schema_version == 2:
            validate_decoy_set_v2(decoy_set)
        else:
            validate_decoy_set(decoy_set)
        if str(decoy_set.get("key")) != key:
            raise ValueError(
                f"decoy record key does not match {key}"
            )
        if self.decoy_schema_version == 2:
            selected = select_epoch_decoys_v2(
                decoy_set,
                key,
                epoch=self.epoch,
                seed=self.seed,
                bank=self.decoy_bank,
            )
        else:
            selected = select_epoch_decoys(
                decoy_set,
                key,
                epoch=self.epoch,
                seed=self.seed,
            )
        native_coords = item["ligand_coords"].numpy()
        transformed = np.stack(
            [
                apply_rigid_transform(
                    native_coords, entry["transform"]
                )
                for entry in selected
            ],
            axis=0,
        ).astype(np.float32)
        item.pop("label", None)
        item["decoy_ligand_coords"] = torch.from_numpy(
            np.ascontiguousarray(transformed)
        )
        item["decoy_strata"] = tuple(
            entry["stratum"] for entry in selected
        )
        item["decoy_transform_center"] = torch.tensor(
            [
                entry["transform"]["center"]
                for entry in selected
            ],
            dtype=torch.float32,
        )
        item["decoy_transform_rotation"] = torch.tensor(
            [
                entry["transform"]["rotation_matrix"]
                for entry in selected
            ],
            dtype=torch.float32,
        )
        item["decoy_transform_translation"] = torch.tensor(
            [
                entry["transform"]["translation"]
                for entry in selected
            ],
            dtype=torch.float32,
        )
        item["selected_decoys"] = _to_torch(selected)
        if self.decoy_schema_version == 2:
            item["decoy_bank"] = self.decoy_bank
            item["decoy_schema_version"] = self.decoy_schema_version
        return item

    def close(self):
        environment = getattr(self, "_decoy_env", None)
        if environment is not None:
            environment.close()
            self._decoy_env = None
        super().close()

    def __getstate__(self):
        state = super().__getstate__()
        state["_decoy_env"] = None
        return state
