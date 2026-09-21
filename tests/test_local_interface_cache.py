import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lmdb
import numpy as np
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AA_SYMBOLS = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_TO_CLASS = {symbol: index for index, symbol in enumerate(AA_SYMBOLS)}


def make_ligand(smiles="CO", add_hydrogens=True, heavy_coords=None):
    mol = Chem.MolFromSmiles(smiles)
    if add_hydrogens:
        mol = Chem.AddHs(mol)
    heavy_indices = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1
    ]
    if heavy_coords is None:
        heavy_coords = np.stack(
            [np.array([float(index) * 1.4, 0.0, 0.0]) for index in range(len(heavy_indices))]
        )
    heavy_coords = np.asarray(heavy_coords, dtype=np.float32).reshape(-1, 3)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    heavy_position = {atom_index: heavy_coords[index] for index, atom_index in enumerate(heavy_indices)}
    hydrogen_number = 0
    for atom in mol.GetAtoms():
        atom_index = atom.GetIdx()
        if atom_index in heavy_position:
            xyz = heavy_position[atom_index]
        else:
            parent = atom.GetNeighbors()[0].GetIdx()
            xyz = heavy_position[parent] + np.array(
                [0.0, 0.2 + 0.1 * hydrogen_number, 0.0], dtype=np.float32
            )
            hydrogen_number += 1
        conformer.SetAtomPosition(atom_index, xyz.tolist())
    mol.RemoveAllConformers()
    mol.AddConformer(conformer)
    return {
        "mol": mol,
        "atom_coords": np.asarray(conformer.GetPositions(), dtype=np.float32),
        "atom_atomic_numbers": np.asarray(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int32
        ),
        "atom_chiral_tag": np.asarray(
            [int(atom.GetChiralTag()) for atom in mol.GetAtoms()], dtype=np.int32
        ),
    }


def make_chain(
    sequence,
    atom_coords,
    residue_indices=None,
    elements=None,
    atom_names=None,
):
    atom_coords = np.asarray(atom_coords, dtype=np.float32).reshape(-1, 3)
    count = len(atom_coords)
    if residue_indices is None:
        residue_indices = np.zeros(count, dtype=np.int32)
    residue_indices = np.asarray(residue_indices, dtype=np.int32)
    if elements is None:
        elements = np.zeros(count, dtype=np.int32)
    if atom_names is None:
        atom_names = np.asarray(["C"] * count)
    ca_coords = np.stack(
        [np.array([float(index), 3.0, 0.0]) for index in range(len(sequence))]
    ).astype(np.float32)
    return {
        "sequence": np.asarray(list(sequence)),
        "atom_coords": atom_coords,
        "atom_element": np.asarray(elements, dtype=np.int32),
        "atom_names": np.asarray(atom_names),
        "atom_residue_indices": residue_indices,
        "atom_residue_type": np.asarray(
            [
                AA_TO_CLASS.get(sequence[int(residue_index)], -1)
                for residue_index in residue_indices
            ],
            dtype=np.int32,
        ),
        "atom_is_backbone": np.asarray(
            [name in {"N", "CA", "C", "O"} for name in atom_names], dtype=np.int32
        ),
        "ca_coords": ca_coords,
        "esm2_embedding": np.zeros((len(sequence), 640), dtype=np.float32),
    }


def make_payload(core_atoms=4):
    protein_coords = np.tile(np.array([[2.0, 0.0, 0.0]], dtype=np.float32), (core_atoms, 1))
    return {
        "protein_features": {
            "A": make_chain("A", protein_coords),
        },
        "ligand_features": make_ligand(),
        "affinity": 7.0,
    }


def make_two_chain_payload():
    return {
        "protein_features": {
            "B": make_chain(
                "GS",
                [[1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
                residue_indices=[0, 1],
                atom_names=["CA", "CA"],
            ),
            "A": make_chain(
                "AC",
                [[2.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
                residue_indices=[0, 1],
                atom_names=["CA", "CA"],
            ),
        },
        "ligand_features": make_ligand(),
    }


def make_payload_for_sequence(sequence):
    payload = make_payload()
    payload["protein_features"] = {
        "A": make_chain(sequence, [[2.0, 0.0, 0.0]])
    }
    return payload


def write_test_lmdb(path, records):
    environment = lmdb.open(
        str(path),
        map_size=16 * 1024 * 1024,
        subdir=True,
    )
    try:
        with environment.begin(write=True) as transaction:
            for key, value in records.items():
                transaction.put(
                    str(key).encode("utf-8"),
                    pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL),
                )
    finally:
        environment.close()


def write_test_manifest(path, source_lmdb, samples, split="train"):
    manifest = {
        "dataset_type": "pdbbind_lmdb",
        "task": "dta_regression",
        "split": split,
        "split_method": "unit_test",
        "lmdb_path": str(source_lmdb),
        "structure_lmdb_path": None,
        "samples": [
            {"lmdb_key": key, "label": float(index + 1)}
            for index, key in enumerate(samples)
        ],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


class TestLocalInterfaceGeometry(unittest.TestCase):
    def test_continuous_contact_weights_use_five_to_six_angstrom_shell(self):
        from data_utils.local_interface import continuous_contact_weights

        distances = np.array([4.9, 5.0, 5.5, 6.0, 6.1], dtype=np.float32)
        weights = continuous_contact_weights(distances)
        np.testing.assert_allclose(weights, [1.0, 1.0, 0.5, 0.0, 0.0], atol=1e-6)

    def test_six_angstrom_core_is_never_truncated(self):
        from data_utils.local_interface import LocalInterfaceConfig, extract_local_interface_record

        record = extract_local_interface_record(
            make_payload(core_atoms=300),
            "x",
            "train",
            LocalInterfaceConfig(shell_budget=2),
        )

        self.assertEqual(int(record["protein_is_core"].sum()), 300)
        self.assertEqual(record["protein_core_atom_count"], 300)
        self.assertEqual(record["protein_atom_count"], 300)

    def test_shell_budget_applies_only_outside_core(self):
        from data_utils.local_interface import LocalInterfaceConfig, extract_local_interface_record

        payload = {
            "protein_features": {
                "A": make_chain(
                    "A",
                    [[2.0, 0.0, 0.0], [7.0, 0.0, 0.0], [8.0, 0.0, 0.0], [11.0, 0.0, 0.0]],
                )
            },
            "ligand_features": make_ligand("C", heavy_coords=[[0.0, 0.0, 0.0]]),
        }
        record = extract_local_interface_record(
            payload, "shell", "train", LocalInterfaceConfig(shell_budget=1)
        )

        self.assertEqual(record["protein_source_atom_index"].tolist(), [0, 1])
        self.assertEqual(record["protein_is_core"].tolist(), [True, False])
        self.assertEqual(record["protein_shell_atom_count"], 1)

    def test_parent_indices_include_chain_offsets(self):
        from data_utils.local_interface import LocalInterfaceConfig, extract_local_interface_record

        record = extract_local_interface_record(
            make_two_chain_payload(),
            "x",
            "train",
            LocalInterfaceConfig(shell_budget=0),
        )

        self.assertTrue(np.any(record["protein_parent_residue"] >= 2))
        self.assertEqual(record["residue_chain_index"].tolist(), [0, 0, 1, 1])

    def test_core_selected_for_one_ligand_atom_does_not_cover_another_at_seven_angstrom(self):
        from data_utils.local_interface import LocalInterfaceConfig, extract_local_interface_record

        payload = {
            "protein_features": {
                "A": make_chain(
                    "A",
                    [[1.0, 0.0, 0.0], [8.0, 7.0, 0.0]],
                )
            },
            "ligand_features": make_ligand(
                "CC",
                heavy_coords=[[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
            ),
        }
        record = extract_local_interface_record(
            payload, "per_ligand_coverage", "train", LocalInterfaceConfig(shell_budget=1)
        )

        self.assertEqual(record["protein_source_atom_index"].tolist(), [0, 1])
        self.assertEqual(record["protein_is_core"].tolist(), [True, False])


class TestLocalInterfaceLigandAndSchema(unittest.TestCase):
    def test_ligand_hydrogens_are_removed_from_every_aligned_field(self):
        from data_utils.local_interface import extract_local_interface_record

        record = extract_local_interface_record(make_payload(), "x", "train")

        self.assertEqual(record["ligand_heavy_atom_count"], 2)
        self.assertEqual(record["ligand_coords"].shape, (2, 3))
        self.assertEqual(record["ligand_atomic_numbers"].tolist(), [6, 8])
        self.assertEqual(record["ligand_atom_features"].shape, (2, 55))
        self.assertEqual(record["ligand_chiral_tag"].shape, (2,))
        self.assertEqual(record["ligand_canonical_rank"].shape, (2,))
        self.assertEqual(record["ligand_automorphisms"].shape[1], 2)
        self.assertLess(int(record["ligand_edge_index"].max()), 2)

    def test_directed_explicit_hydrogens_are_removed(self):
        from data_utils.local_interface import ligand_heavy_atom_view

        parser = Chem.SmilesParserParams()
        parser.removeHs = False
        mol = Chem.MolFromSmiles("[H]N=S=N[H]", parser)
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtom().GetAtomicNum()
            end = bond.GetEndAtom().GetAtomicNum()
            if begin == 1 or end == 1:
                bond.SetBondDir(Chem.BondDir.ENDDOWNRIGHT)
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for index in range(mol.GetNumAtoms()):
            conformer.SetAtomPosition(
                index, [float(index), 0.0, 0.0]
            )
        mol.AddConformer(conformer)
        source = {
            "mol": mol,
            "atom_coords": np.asarray(
                conformer.GetPositions(), dtype=np.float32
            ),
            "atom_atomic_numbers": np.asarray(
                [atom.GetAtomicNum() for atom in mol.GetAtoms()],
                dtype=np.int32,
            ),
        }
        self.assertEqual(Chem.RemoveHs(Chem.Mol(mol)).GetNumAtoms(), 5)

        ligand = ligand_heavy_atom_view(source)

        self.assertEqual(ligand["mol"].GetNumAtoms(), 3)
        self.assertEqual(ligand["atomic_numbers"].tolist(), [7, 16, 7])
        self.assertEqual(ligand["original_heavy_indices"].tolist(), [1, 2, 3])

    def test_symmetric_ligand_stores_nonidentity_automorphism(self):
        from data_utils.local_interface import ligand_heavy_atom_view

        ligand = ligand_heavy_atom_view(make_ligand("CC"))
        automorphisms = ligand["automorphisms"]

        self.assertTrue(any(mapping.tolist() == [1, 0] for mapping in automorphisms))

    def test_chiral_tags_come_from_hydrogen_stripped_molecule(self):
        from data_utils.local_interface import ligand_heavy_atom_view

        source = make_ligand("F[C@H](Cl)Br")
        source["atom_chiral_tag"][:] = 0
        expected_mol = Chem.RemoveHs(Chem.Mol(source["mol"]))
        expected = [int(atom.GetChiralTag()) for atom in expected_mol.GetAtoms()]

        ligand = ligand_heavy_atom_view(source)

        self.assertEqual(ligand["chiral_tags"].tolist(), expected)
        self.assertTrue(any(tag != 0 for tag in expected))

    def test_automorphism_limit_allows_exactly_the_configured_count(self):
        from data_utils.local_interface import ligand_automorphisms

        mol = Chem.RemoveHs(make_ligand("CO")["mol"])

        mappings = ligand_automorphisms(mol, max_matches=1)

        self.assertEqual(mappings.shape, (1, 2))

    def test_element_tokens_map_to_fixed_symbols(self):
        from data_utils.local_interface import extract_local_interface_record

        payload = {
            "protein_features": {
                "A": make_chain(
                    "A",
                    np.tile([[2.0, 0.0, 0.0]], (4, 1)),
                    elements=[0, 1, 2, 3],
                    atom_names=["C", "N", "O", "SG"],
                )
            },
            "ligand_features": make_ligand("C"),
        }
        record = extract_local_interface_record(payload, "elements", "train")

        self.assertEqual(record["protein_element_symbol"].tolist(), ["C", "N", "O", "S"])
        self.assertEqual(record["protein_atom_name_class"].shape, (4,))
        self.assertTrue(np.all(record["protein_atom_name_class"] > 0))

    def test_unknown_protein_element_policy_is_a_hard_error(self):
        from data_utils.local_interface import extract_local_interface_record

        payload = {
            "protein_features": {
                "A": make_chain(
                    "A",
                    [[2.0, 0.0, 0.0]],
                    elements=[4],
                )
            },
            "ligand_features": make_ligand("C"),
        }

        with self.assertRaisesRegex(ValueError, "unknown element tokens"):
            extract_local_interface_record(payload, "unknown_element", "train")

    def test_chain_signature_is_order_independent_and_multiplicity_sensitive(self):
        from data_utils.local_interface import canonical_chain_signature

        left = {"B": {"sequence": list("GG")}, "A": {"sequence": list("AC")}}
        right = {"X": {"sequence": list("AC")}, "Y": {"sequence": list("GG")}}
        duplicate = {"X": {"sequence": list("AC")}, "Y": {"sequence": list("AC")}}

        self.assertEqual(canonical_chain_signature(left), canonical_chain_signature(right))
        self.assertNotEqual(canonical_chain_signature(left), canonical_chain_signature(duplicate))

    def test_structure_lmdb_falls_back_to_feature_lmdb(self):
        from data_utils.local_interface import resolve_source_lmdb

        with tempfile.TemporaryDirectory() as tmp:
            manifest = {"lmdb_path": "feature_lmdb", "structure_lmdb_path": None}
            self.assertEqual(
                resolve_source_lmdb(manifest, tmp),
                str((Path(tmp) / "feature_lmdb").resolve()),
            )

    def test_record_validation_rejects_no_schema_errors(self):
        from data_utils.local_interface import extract_local_interface_record, validate_local_interface_record

        record = extract_local_interface_record(make_payload(), "x", "train")

        self.assertIs(validate_local_interface_record(record), record)
        self.assertNotIn("esm2_embedding", record)
        self.assertNotIn("affinity", record)

    def test_validator_rejects_parent_residue_from_another_chain(self):
        from data_utils.local_interface import extract_local_interface_record, validate_local_interface_record

        record = extract_local_interface_record(make_two_chain_payload(), "x", "train")
        record["protein_parent_residue"] = record["protein_parent_residue"].copy()
        first_chain_atom = int(np.flatnonzero(record["protein_chain_index"] == 0)[0])
        first_second_chain_residue = int(
            np.flatnonzero(record["residue_chain_index"] == 1)[0]
        )
        record["protein_parent_residue"][first_chain_atom] = first_second_chain_residue

        with self.assertRaisesRegex(ValueError, "parent-residue chain"):
            validate_local_interface_record(record)

    def test_validator_rejects_inconsistent_derived_classes(self):
        from data_utils.local_interface import extract_local_interface_record, validate_local_interface_record

        record = extract_local_interface_record(make_payload(), "x", "train")
        record["protein_atom_name_class"] = record["protein_atom_name_class"].copy()
        record["protein_atom_name_class"][0] = 0
        with self.assertRaisesRegex(ValueError, "atom-name classes"):
            validate_local_interface_record(record)

        record = extract_local_interface_record(make_payload(), "x", "train")
        record["protein_residue_type"] = record["protein_residue_type"].copy()
        record["protein_residue_type"][0] = 1
        with self.assertRaisesRegex(ValueError, "residue classes"):
            validate_local_interface_record(record)


class TestLocalInterfaceCacheBuilders(unittest.TestCase):
    def test_diagnostic_split_keeps_exact_target_groups_intact(self):
        from scripts.build_local_interface_diagnostic_split import (
            build_arg_parser,
            build_diagnostic_split,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.lmdb"
            manifest_path = root / "train_manifest.json"
            output_path = root / "diagnostic_keys.json"
            records = {
                "same_a": make_payload_for_sequence("AC"),
                "same_b": make_payload_for_sequence("AC"),
                "other_c": make_payload_for_sequence("GG"),
                "other_d": make_payload_for_sequence("W"),
            }
            write_test_lmdb(source, records)
            write_test_manifest(manifest_path, source, list(records))

            args = build_arg_parser().parse_args(
                [
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_path),
                    "--fraction",
                    "0.5",
                    "--seed",
                    "42",
                ]
            )
            result = build_diagnostic_split(args)

            selected = set(result["keys"])
            self.assertTrue(selected)
            self.assertLess(len(selected), len(records))
            for group in ({"same_a", "same_b"}, {"other_c"}, {"other_d"}):
                self.assertIn(selected.intersection(group), (set(), group))
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                result,
            )

    def test_native_cache_writes_records_metadata_and_coverage(self):
        from scripts.build_local_interface_cache import (
            build_arg_parser,
            build_native_cache,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.lmdb"
            output = root / "native.lmdb"
            manifest_path = root / "train_manifest.json"
            records = {
                "a": make_payload_for_sequence("AC"),
                "b": make_payload_for_sequence("GG"),
            }
            write_test_lmdb(source, records)
            write_test_manifest(manifest_path, source, list(records))
            args = build_arg_parser().parse_args(
                [
                    "--manifest",
                    str(manifest_path),
                    "--output-lmdb",
                    str(output),
                    "--map-size-gb",
                    "0.02",
                    "--max-samples",
                    "2",
                ]
            )

            metadata = build_native_cache(args)

            self.assertEqual(metadata["schema_version"], 1)
            self.assertEqual(metadata["counts"]["written"], 2)
            self.assertIn("manifest", metadata)
            self.assertIn("source_key_sha256", metadata)
            self.assertIn("coverage_report", metadata)
            self.assertEqual(
                json.loads(
                    (output / "metadata.json").read_text(encoding="utf-8")
                ),
                metadata,
            )
            environment = lmdb.open(
                str(output), readonly=True, lock=False, subdir=True
            )
            try:
                with environment.begin(write=False) as transaction:
                    cached = pickle.loads(transaction.get(b"a"))
            finally:
                environment.close()
            self.assertEqual(cached["schema_version"], 1)
            self.assertNotIn("affinity", cached)

    def test_native_cache_resume_rejects_changed_source_hash(self):
        from scripts.build_local_interface_cache import (
            build_arg_parser,
            build_native_cache,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.lmdb"
            output = root / "native.lmdb"
            manifest_path = root / "train_manifest.json"
            records = {"a": make_payload(), "b": make_payload()}
            write_test_lmdb(source, records)
            write_test_manifest(manifest_path, source, ["a", "b"])
            parser = build_arg_parser()
            build_native_cache(
                parser.parse_args(
                    ["--manifest", str(manifest_path), "--output-lmdb", str(output)]
                )
            )
            write_test_manifest(manifest_path, source, ["b", "a"])

            with self.assertRaisesRegex(ValueError, "resume provenance"):
                build_native_cache(
                    parser.parse_args(
                        [
                            "--manifest",
                            str(manifest_path),
                            "--output-lmdb",
                            str(output),
                            "--resume",
                        ]
                    )
                )


    def test_native_cache_resume_rejects_replaced_source_record(self):
        from scripts.build_local_interface_cache import (
            build_arg_parser,
            build_native_cache,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.lmdb"
            output = root / "native.lmdb"
            manifest_path = root / "train_manifest.json"
            write_test_lmdb(source, {"a": make_payload_for_sequence("AC")})
            write_test_manifest(manifest_path, source, ["a"])
            parser = build_arg_parser()
            build_native_cache(
                parser.parse_args(
                    ["--manifest", str(manifest_path), "--output-lmdb", str(output)]
                )
            )
            write_test_lmdb(
                source, {"a": make_payload_for_sequence("GG")}
            )

            with self.assertRaisesRegex(
                ValueError, "resume provenance"
            ):
                build_native_cache(
                    parser.parse_args(
                        [
                            "--manifest",
                            str(manifest_path),
                            "--output-lmdb",
                            str(output),
                            "--resume",
                        ]
                    )
                )

    def test_native_cache_resume_rejects_changed_max_samples(self):
        from scripts.build_local_interface_cache import (
            build_arg_parser,
            build_native_cache,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.lmdb"
            output = root / "native.lmdb"
            manifest_path = root / "train_manifest.json"
            write_test_lmdb(
                source, {"a": make_payload(), "b": make_payload()}
            )
            write_test_manifest(manifest_path, source, ["a", "b"])
            parser = build_arg_parser()
            build_native_cache(
                parser.parse_args(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--output-lmdb",
                        str(output),
                        "--max-samples",
                        "1",
                    ]
                )
            )

            with self.assertRaisesRegex(
                ValueError, "resume provenance"
            ):
                build_native_cache(
                    parser.parse_args(
                        [
                            "--manifest",
                            str(manifest_path),
                            "--output-lmdb",
                            str(output),
                            "--max-samples",
                            "2",
                            "--resume",
                        ]
                    )
                )

    def test_native_cache_map_full_is_fatal(self):
        from scripts import build_local_interface_cache as builder

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.lmdb"
            output = root / "native.lmdb"
            manifest_path = root / "train_manifest.json"
            write_test_lmdb(source, {"a": make_payload()})
            write_test_manifest(manifest_path, source, ["a"])
            args = builder.build_arg_parser().parse_args(
                [
                    "--manifest",
                    str(manifest_path),
                    "--output-lmdb",
                    str(output),
                ]
            )

            with patch.object(
                builder,
                "_write_record",
                create=True,
                side_effect=lmdb.MapFullError("map full"),
            ):
                with self.assertRaises(lmdb.MapFullError):
                    builder.build_native_cache(args)

    def test_native_cache_resume_clears_transient_error(self):
        from scripts import build_local_interface_cache as builder

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.lmdb"
            output = root / "native.lmdb"
            manifest_path = root / "train_manifest.json"
            write_test_lmdb(source, {"a": make_payload()})
            write_test_manifest(manifest_path, source, ["a"])
            parser = builder.build_arg_parser()
            initial_args = parser.parse_args(
                [
                    "--manifest",
                    str(manifest_path),
                    "--output-lmdb",
                    str(output),
                ]
            )
            with patch.object(
                builder,
                "extract_local_interface_record",
                side_effect=ValueError("transient"),
            ):
                initial = builder.build_native_cache(initial_args)
            self.assertEqual(initial["counts"]["errors"], 1)

            resumed = builder.build_native_cache(
                parser.parse_args(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--output-lmdb",
                        str(output),
                        "--resume",
                    ]
                )
            )

            self.assertEqual(resumed["counts"]["errors"], 0)
            self.assertEqual(resumed["errors_preview"], [])
            self.assertEqual(resumed["counts"]["cached_total"], 1)


class TestRealLocalInterfaceRecord(unittest.TestCase):
    real_lmdb = ROOT / "data" / "PDBbind_chain_lmdb_esm_gernabind_v2"

    @unittest.skipUnless(real_lmdb.exists(), "real PDBbind LMDB is unavailable")
    def test_real_10gs_record_is_finite_and_aligned(self):
        from data_utils.local_interface import extract_local_interface_record

        environment = lmdb.open(
            str(self.real_lmdb), readonly=True, lock=False, readahead=False, subdir=True
        )
        try:
            with environment.begin(write=False) as transaction:
                payload = pickle.loads(transaction.get(b"10gs_protein"))
        finally:
            environment.close()

        record = extract_local_interface_record(payload, "10gs_protein", "train")

        self.assertTrue(np.all(np.isfinite(record["protein_coords"])))
        self.assertTrue(np.all(np.isfinite(record["ligand_coords"])))
        self.assertGreater(record["protein_core_atom_count"], 0)
        self.assertLess(int(record["protein_parent_residue"].max()), record["residue_count"])
        self.assertEqual(
            record["ligand_atom_features"].shape[0],
            record["ligand_coords"].shape[0],
        )


if __name__ == "__main__":
    unittest.main()
