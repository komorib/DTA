"""One-shot official test evaluator for anchored expert fusion v5."""
import argparse
from dataclasses import dataclass
from io import BytesIO
import hashlib, json, math, os, stat, tempfile
from numbers import Integral
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dta.anchored_expert_fusion_v5 import data as data_module
from dta.anchored_expert_fusion_v5 import selection as selection_module
from dta.anchored_expert_fusion_v5 import training as training_module
from dta.anchored_expert_fusion_v5.model import AnchoredResidualModel
from dta.anchored_expert_fusion_v5 import static_fusion as static_module
from dta.anchored_expert_fusion_v5.static_fusion import rank_percentiles
from dta.rank_fusion_v4 import cache as cache_module
from dta.rank_fusion_v4.metrics import affinity_metrics

TARGET_CI = 0.706858
OFFICIAL_TEST_COUNT = 4751
OFFICIAL_TEST_MANIFEST_SHA256 = "f8741d40837568c552ca0bd3725e0b079ba855d56d2a663c0bb95c4bc09c95b8"
OFFICIAL_TEST_ORDERED_KEY_SHA256 = "432db77b96cc1c5ff8eff4de655b64b4a487c4b448913ee7db33cdaf886b303f"
OFFICIAL_TEST_FILE_IDENTITIES = {
    "features.npz": (47273183, "b8ded10b2499f1d4d81b9321a914d5d8146f1296528c89d095369843c928abd3"),
    "metadata.json": (990, "9a539a21e71d2b3f093a43d43762a78bd5df1c92937d8559c66155a68e173fbe"),
    "coverage.json": (1070, "26672db86031ad4d6dfedf3be873867ece9537d45909c556737b2dd776c6e280"),
    "normalization.json": (153351, "65c5b00a0cc5952c8da3ce3be0e8a96b299ad07fb1acebda15c16b1e7d3ea340"),
}

@dataclass(frozen=True)
class ExpectedTestIdentity:
    count: int
    manifest_sha256: str
    ordered_key_sha256: str

    def __post_init__(self):
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count <= 0:
            raise ValueError("expected test count must be a positive integer")
        for name in ("manifest_sha256", "ordered_key_sha256"):
            value = getattr(self, name)
            if (not isinstance(value, str) or len(value) != 64 or
                    value.lower() != value):
                raise ValueError("expected test identity hash is invalid")
            try:
                int(value, 16)
            except ValueError as error:
                raise ValueError("expected test identity hash is invalid") from error

OFFICIAL_TEST_IDENTITY = ExpectedTestIdentity(
    OFFICIAL_TEST_COUNT, OFFICIAL_TEST_MANIFEST_SHA256,
    OFFICIAL_TEST_ORDERED_KEY_SHA256,
)
_START_MARKER="evaluation.start"
_COMPLETE_MARKER="evaluation.complete"
_KEYS=("CI","MAE","RMSE","Pearson","Spearman")

def _finite(value,name):
    if isinstance(value,(bool,np.bool_)):
        raise ValueError(name+" must be finite")
    try: value=float(value)
    except (TypeError,ValueError,OverflowError) as error: raise ValueError(name+" must be finite") from error
    if not math.isfinite(value): raise ValueError(name+" must be finite")
    return value

def test_status(ci): return "passed" if _finite(ci,"CI")>TARGET_CI else "complete_below_target"

def _hash(path):
    return hashlib.sha256(
        selection_module.read_verified_bytes(path, name="hashed file")
    ).hexdigest()

def _order_hash(keys): return hashlib.sha256(json.dumps(list(keys),ensure_ascii=False,separators=(",",":")).encode()).hexdigest()

def _json(path,name):
    try:
        raw = selection_module.read_verified_bytes(path, name=name)
        value = json.loads(raw.decode("utf-8"))
    except (OSError,UnicodeDecodeError,ValueError) as error:
        raise ValueError(name+" is malformed") from error
    if not isinstance(value,dict): raise ValueError(name+" is malformed")
    return value

def _package(package_dir):
    package=Path(package_dir).expanduser()
    selection_module.verify_validation_seal(package,artifact_root=package.parent)
    package=package.resolve()
    with (package/"candidate_manifest.json").open(encoding="utf-8") as handle: manifest=json.load(handle)
    reports = {
        "raw": _json(package / "raw_cross_fit_report.json", "raw report"),
        "rank": _json(package / "rank_cross_fit_report.json", "rank report"),
    }
    return (
        package, _json(package / "selection_seal.json", "selection seal"),
        _json(package / "selection.json", "selection"), manifest,
        _json(package / "selected_references.json", "selected references"),
        _json(package / "static_specs.json", "static specs"),
        _json(package / "cache_identities.json", "cache identities"),
        reports,
    )

def _ref(root,path,digest,name,allow_null=False):
    if path is None or digest is None:
        if allow_null and path is None and digest is None:return None
        raise ValueError(name+" path and hash are required")
    selection_module.validate_relative_path(path,name+" path")
    root=Path(root).resolve(); target=root/path
    try: mode=os.lstat(str(target)).st_mode
    except OSError as error: raise ValueError(name+" artifact is missing") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode): raise ValueError(name+" artifact must be a regular non-symlink file")
    try: target.resolve().relative_to(root)
    except ValueError as error: raise ValueError(name+" artifact leaves the output root") from error
    if _hash(target)!=digest: raise ValueError(name+" artifact hash does not match")
    return target

@dataclass(frozen=True)
class SealedFoldMap:
    keys: tuple
    fold_ids: np.ndarray
    sha256: str

def _load_fold_map(package):
    try:
        arrays = selection_module.load_verified_npz(
            Path(package) / "fold_map.npz", name="fold map"
        )
        if set(arrays) != {"keys", "fold_ids"}:
            raise ValueError("fold map schema does not match")
        raw_keys = arrays["keys"]
        raw_fold_ids = arrays["fold_ids"]
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and "schema" in str(error):
            raise
        raise ValueError("fold map is invalid") from error
    if (raw_keys.ndim != 1 or raw_keys.dtype.kind not in ("U", "S") or
            raw_fold_ids.ndim != 1 or
            raw_fold_ids.dtype.kind not in ("i", "u")):
        raise ValueError("fold map arrays have invalid dtypes or ranks")
    keys = tuple(str(value) for value in raw_keys.tolist())
    fold_ids = np.asarray(raw_fold_ids, dtype=np.int64).copy()
    if (not keys or len(keys) != len(set(keys)) or
            fold_ids.shape != (len(keys),) or
            set(fold_ids.tolist()) != {0, 1, 2, 3, 4}):
        raise ValueError("fold map population or five-fold coverage is invalid")
    digest = static_module.fold_map_sha256(keys, fold_ids)
    fold_ids.setflags(write=False)
    return SealedFoldMap(keys=keys, fold_ids=fold_ids, sha256=digest)

def _fold_keys(package):
    return _load_fold_map(package)

def _check_prediction(path, expected, name, expected_sha256=None):
    try:
        arrays = selection_module.load_verified_npz(
            path, expected_sha256=expected_sha256, name=name + " artifact"
        )
        if not {"keys", "labels", "predictions"}.issubset(arrays):
            raise ValueError(name+" artifact is incomplete")
        keys_array = np.asarray(arrays["keys"])
        labels_array = np.asarray(arrays["labels"])
        predictions_array = np.asarray(arrays["predictions"])
        keys = tuple(str(x) for x in keys_array.tolist())
        labels = labels_array.reshape(-1)
        predictions = predictions_array.reshape(-1)
    except (OSError, ValueError, TypeError) as error:
        if isinstance(error, ValueError) and "artifact" in str(error):
            raise
        raise ValueError(name+" artifact is invalid") from error
    if (keys != tuple(expected) or labels.shape != (len(expected),) or
            predictions.shape != labels.shape or
            labels_array.dtype != np.float32 or
            predictions_array.dtype != np.float32 or
            not np.isfinite(labels).all() or
            not np.isfinite(predictions).all()):
        raise ValueError(name+" key order or arrays do not match")
    return labels, predictions

def _sha256_value(value, name):
    if (not isinstance(value, str) or len(value) != 64 or
            value.lower() != value):
        raise ValueError(name + " must be a lowercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(name + " must be a lowercase SHA256") from error
    return value

def _identity(value, split):
    required = {
        "split", "count", "manifest_sha256",
        "ordered_key_sha256", "files",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(split + " cache identity schema is malformed")
    if value["split"] != split:
        raise ValueError(split + " cache identity split does not match")
    count = value["count"]
    if (isinstance(count, (bool, np.bool_)) or
            not isinstance(count, Integral) or int(count) <= 0):
        raise ValueError(split + " cache identity count is invalid")
    _sha256_value(value["manifest_sha256"], split + " manifest hash")
    _sha256_value(value["ordered_key_sha256"], split + " ordered key hash")
    files = value["files"]
    required_files = {
        "features.npz", "metadata.json",
        "coverage.json", "normalization.json",
    }
    if not isinstance(files, dict) or set(files) != required_files:
        raise ValueError(split + " cache identity files are malformed")
    for file_name in sorted(required_files):
        _sha256_value(files[file_name], split + " " + file_name + " hash")
    return value

def _metric_payload(value, name):
    if not isinstance(value, dict) or set(value) != set(_KEYS):
        raise ValueError(name + " metrics are malformed")
    return {key: _finite(value[key], name + " " + key) for key in _KEYS}

def _load_crossfit_report(payload, expected_family):
    required = {
        "family", "validation_keys", "fold_ids", "fold_map_hash",
        "held_out_count", "predictions", "pooled_metrics",
        "upper_bound_pooled_metrics", "folds", "fold_weights",
        "deploy_weights", "gate", "validation_available", "report_hash",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(expected_family + " cross-fit report fields are incomplete")
    if payload["family"] != expected_family:
        raise ValueError("cross-fit report family does not match")
    try:
        keys = tuple(str(value) for value in payload["validation_keys"])
        fold_ids = np.asarray(payload["fold_ids"], dtype=np.int64)
        held_out = np.asarray(payload["held_out_count"], dtype=np.int64)
        predictions = np.asarray(payload["predictions"], dtype=np.float64)
        fold_weights = np.asarray(payload["fold_weights"], dtype=np.float64)
        deploy = np.asarray(payload["deploy_weights"], dtype=np.float64)
        raw_available = np.asarray(payload["validation_available"])
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(expected_family + " cross-fit arrays are malformed") from error
    count = len(keys)
    if (not keys or len(keys) != len(set(keys)) or
            fold_ids.shape != (count,) or held_out.shape != (count,) or
            predictions.shape != (count,) or fold_weights.shape != (5, 3) or
            deploy.shape != (3,) or raw_available.dtype != np.bool_ or
            raw_available.shape != (count, 3)):
        raise ValueError(expected_family + " cross-fit shapes do not match")
    if (not np.isfinite(predictions).all() or
            not np.isfinite(fold_weights).all() or
            not np.isfinite(deploy).all()):
        raise ValueError(expected_family + " cross-fit arrays are nonfinite")
    if not np.all(held_out == 1):
        raise ValueError(expected_family + " held-out coverage is incomplete")
    if payload["fold_map_hash"] != static_module.fold_map_sha256(keys, fold_ids):
        raise ValueError(expected_family + " fold map hash does not match")
    pooled = _metric_payload(payload["pooled_metrics"], expected_family + " pooled")
    upper_pooled = _metric_payload(
        payload["upper_bound_pooled_metrics"],
        expected_family + " upper pooled",
    )
    if not isinstance(payload["folds"], list) or len(payload["folds"]) != 5:
        raise ValueError(expected_family + " fold inventory is incomplete")
    folds = []
    fold_fields = {
        "fold_id", "held_out_indices", "selected_weights",
        "candidate_metrics", "upper_bound_metrics", "search_metrics",
        "slope", "intercept", "stage_counts", "searched_count",
    }
    for expected_fold, value in enumerate(payload["folds"]):
        if not isinstance(value, dict) or set(value) != fold_fields:
            raise ValueError(expected_family + " fold fields are incomplete")
        if value["fold_id"] != expected_fold:
            raise ValueError(expected_family + " fold id does not match")
        try:
            held_indices = np.asarray(value["held_out_indices"], dtype=np.int64)
            selected_weights = np.asarray(
                value["selected_weights"], dtype=np.float64
            )
            stage_counts = tuple(int(item) for item in value["stage_counts"])
            searched_count = int(value["searched_count"])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(expected_family + " fold values are malformed") from error
        expected_indices = np.flatnonzero(fold_ids == expected_fold)
        if (not np.array_equal(held_indices, expected_indices) or
                selected_weights.shape != (3,) or
                not np.isfinite(selected_weights).all() or
                not np.array_equal(selected_weights, fold_weights[expected_fold]) or
                not stage_counts or any(item <= 0 for item in stage_counts) or
                searched_count != sum(stage_counts)):
            raise ValueError(expected_family + " fold provenance does not match")
        folds.append(static_module.FoldResult(
            fold_id=expected_fold, held_out_indices=held_indices,
            selected_weights=selected_weights,
            candidate_metrics=_metric_payload(
                value["candidate_metrics"], expected_family + " fold candidate"
            ),
            upper_bound_metrics=_metric_payload(
                value["upper_bound_metrics"], expected_family + " fold upper"
            ),
            search_metrics=_metric_payload(
                value["search_metrics"], expected_family + " fold search"
            ),
            slope=_finite(value["slope"], expected_family + " fold slope"),
            intercept=_finite(
                value["intercept"], expected_family + " fold intercept"
            ),
            stage_counts=stage_counts, searched_count=searched_count,
        ))
    expected_deploy = static_module.deploy_weights(fold_weights)
    if not np.array_equal(deploy, expected_deploy):
        raise ValueError(expected_family + " deploy weights do not match folds")
    gate_value = payload["gate"]
    if (not isinstance(gate_value, dict) or
            set(gate_value) != {"passed", "reasons"} or
            not isinstance(gate_value["passed"], bool) or
            not isinstance(gate_value["reasons"], list) or
            any(not isinstance(item, str) for item in gate_value["reasons"])):
        raise ValueError(expected_family + " gate is malformed")
    recomputed_gate = static_module.evaluate_gate(
        pooled, upper_pooled,
        [fold.candidate_metrics for fold in folds],
        [fold.upper_bound_metrics for fold in folds],
        fold_weights, held_out,
    )
    if (gate_value["passed"] != recomputed_gate.passed or
            tuple(gate_value["reasons"]) != recomputed_gate.reasons):
        raise ValueError(expected_family + " gate reasons do not match")
    report_hash = payload["report_hash"]
    if not isinstance(report_hash, str) or len(report_hash) != 64:
        raise ValueError(expected_family + " report hash is malformed")
    try:
        int(report_hash, 16)
    except ValueError as error:
        raise ValueError(expected_family + " report hash is malformed") from error
    report = static_module.CrossFitReport(
        family=expected_family, validation_keys=keys, fold_ids=fold_ids,
        fold_map_hash=payload["fold_map_hash"], held_out_count=held_out,
        predictions=predictions, pooled_metrics=pooled,
        upper_bound_pooled_metrics=upper_pooled, folds=tuple(folds),
        fold_weights=fold_weights, deploy_weights=deploy, gate=recomputed_gate,
        validation_available=raw_available.astype(np.bool_, copy=False),
        report_hash=report_hash,
    )
    if static_module.report_sha256(report) != report_hash:
        raise ValueError(expected_family + " canonical report hash does not match")
    return report

def _validate_formal_package(
    selection, manifest, refs, identities, reports, static_specs, fold_map,
    artifact_root=None,
):
    if not isinstance(selection, dict) or selection.get("status") != "complete":
        raise ValueError("sealed selection is incomplete")
    controls = selection.get("inventory_controls")
    if (selection.get("inventory_mode") != "formal" or
            not isinstance(controls, dict) or
            controls.get("mode") != "formal" or
            tuple(controls.get("budgets", ())) != selection_module.FORMAL_BUDGETS or
            tuple(controls.get("seeds", ())) != selection_module.FORMAL_SEEDS or
            controls.get("limit") is not None):
        raise ValueError("official evaluation requires a formal validation package")
    if not isinstance(reports, dict) or set(reports) != {"raw", "rank"}:
        raise ValueError("cross-fit reports are malformed")
    report_objects = {
        family: _load_crossfit_report(reports[family], family)
        for family in ("raw", "rank")
    }
    passing = tuple(
        family for family in ("raw", "rank")
        if report_objects[family].gate.passed
    )
    report_hashes = {
        family: report_objects[family].report_hash
        for family in ("raw", "rank")
    }
    if passing not in (("raw",), ("rank",), ("raw", "rank")):
        raise ValueError("formal package must contain one or two passing families")
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("candidate manifest is invalid")
    records = tuple(selection_module.CandidateRecord.from_dict(item)
                    for item in manifest)
    experts = tuple(item for item in records if item.type == "expert")
    statics = tuple(item for item in records if item.type == "static")
    residuals = tuple(item for item in records if item.type == "residual")
    ensembles = tuple(item for item in records
                      if item.type in ("raw_ensemble", "rank_ensemble"))
    inventory = selection_module.build_candidate_inventory(
        experts, statics, residuals, ensembles,
        expected_families=passing,
        expected_budgets=selection_module.FORMAL_BUDGETS,
        expected_seeds=selection_module.FORMAL_SEEDS,
    )
    expected_count = 25 if len(passing) == 1 else 47
    if len(records) != expected_count or len(inventory) != expected_count:
        raise ValueError("formal candidate inventory count does not match")
    names = tuple(item.name for item in records)
    if (selection.get("candidate_count") != expected_count or
            tuple(selection.get("candidate_names", ())) != names):
        raise ValueError("candidate manifest count or order does not match selection")
    train = _identity(identities.get("train"), "train")
    valid = _identity(identities.get("valid"), "valid")
    if not isinstance(fold_map, SealedFoldMap):
        raise ValueError("sealed fold map binding is missing")
    if len(fold_map.keys) != valid["count"]:
        raise ValueError("fold map population does not match valid identity count")
    if _order_hash(fold_map.keys) != valid["ordered_key_sha256"]:
        raise ValueError("fold map ordered keys do not match valid identity")
    for family in ("raw", "rank"):
        report = report_objects[family]
        if (report.validation_keys != fold_map.keys or
                not np.array_equal(report.fold_ids, fold_map.fold_ids) or
                report.fold_map_hash != fold_map.sha256 or
                len(report.validation_keys) != valid["count"] or
                set(report.fold_ids.tolist()) != {0, 1, 2, 3, 4}):
            raise ValueError(family + " report does not match sealed fold map")
    if not isinstance(static_specs, dict) or set(static_specs) != set(passing):
        raise ValueError("static specs do not match passing families")
    bound_specs = {}
    spec_hashes = {}
    for family in passing:
        try:
            spec = training_module.StaticSpec.from_dict(static_specs[family])
        except (TypeError, ValueError) as error:
            raise ValueError(family + " static spec is malformed") from error
        report = report_objects[family]
        if (spec.family != family or
                spec.cross_fit_report_hash != report.report_hash or
                spec.train_cache_identity != train or
                spec.valid_cache_identity != valid or
                not np.array_equal(
                    np.asarray(spec.deploy_weights, dtype=np.float64),
                    report.deploy_weights,
                )):
            raise ValueError(family + " static spec does not match sealed report")
        if family == "raw":
            expected_slope, expected_intercept = 1.0, 0.0
        else:
            expected_slope = float(np.median(
                [fold.slope for fold in report.folds]
            ))
            expected_intercept = float(np.median(
                [fold.intercept for fold in report.folds]
            ))
        if (spec.rank_slope != expected_slope or
                spec.rank_intercept != expected_intercept):
            raise ValueError(family + " deploy calibration does not match report")
        bound_specs[family] = spec
        spec_hashes[family] = training_module.static_spec_sha256(spec)
    for record in records:
        if record.type == "expert":
            expected_lineage = {
                "valid_manifest": valid["manifest_sha256"],
                "valid_ordered_keys": valid["ordered_key_sha256"],
                "prediction_source": valid["files"]["features.npz"],
            }
        else:
            if record.family not in passing:
                raise ValueError(record.name + " family did not pass validation")
            expected_lineage = {
                "train_manifest": train["manifest_sha256"],
                "train_ordered_keys": train["ordered_key_sha256"],
                "valid_manifest": valid["manifest_sha256"],
                "valid_ordered_keys": valid["ordered_key_sha256"],
                "cross_fit_report": report_hashes[record.family],
                "static_spec_sha256": spec_hashes[record.family],
            }
            if record.type in ("raw_ensemble", "rank_ensemble"):
                member_payload = json.dumps(
                    dict(record.member_hashes), sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                expected_lineage["member_set"] = hashlib.sha256(
                    member_payload
                ).hexdigest()
        if dict(record.lineage_hashes) != expected_lineage:
            raise ValueError(record.name + " static_spec/report lineage does not match package")
    by_name = {record.name: record for record in records}
    for child in residuals + ensembles:
        parent = by_name.get(child.parent_static_name)
        if (parent is None or parent.type != "static" or
                parent.family != child.family or
                parent.lineage_hashes.get("static_spec_sha256") !=
                child.lineage_hashes.get("static_spec_sha256")):
            raise ValueError(child.name + " parent static lineage does not resolve")
        if child.type in ("raw_ensemble", "rank_ensemble"):
            for member_name in child.members:
                member = by_name.get(member_name)
                if (member is None or member.type != "residual" or
                        member.parent_static_name != child.parent_static_name or
                        child.member_hashes.get(member_name) !=
                        member.prediction_sha256 or
                        member.lineage_hashes.get("static_spec_sha256") !=
                        child.lineage_hashes.get("static_spec_sha256")):
                    raise ValueError(child.name + " member lineage does not resolve")
    fields = {"name", "prediction_path", "prediction_sha256",
              "checkpoint_path", "checkpoint_sha256"}
    if not isinstance(refs, dict) or set(refs) != fields:
        raise ValueError("selected references are malformed")
    matches = tuple(item for item in records if item.name == refs["name"])
    if len(matches) != 1:
        raise ValueError("selected candidate is not one exact manifest record")
    selected = matches[0]
    if (selection.get("selected_name") != selected.name or
            selection.get("selected_type") != selected.type or
            selection.get("selected_family") != selected.family or
            selection.get("selected_validation_metrics") !=
            dict(selected.validation_metrics)):
        raise ValueError("selected candidate fields do not match manifest")
    expected_refs = {
        "name": selected.name,
        "prediction_path": selected.prediction_path,
        "prediction_sha256": selected.prediction_sha256,
        "checkpoint_path": selected.checkpoint_path,
        "checkpoint_sha256": selected.checkpoint_sha256,
    }
    if refs != expected_refs:
        raise ValueError("selected references do not match manifest")
    if artifact_root is not None:
        _validate_candidate_artifacts(
            records, artifact_root, fold_map.keys, bound_specs
        )
        selected_again = selection_module.select_validation_candidate(
            experts, statics, residuals, ensembles
        )
        if selected_again.name != selected.name:
            raise ValueError("sealed selected candidate no longer wins validation policy")
    return records, selected, train, valid, passing

def _load_verified_test_cache(test_cache_dir, expected_files):
    root = Path(test_cache_dir).expanduser()
    try:
        mode = os.lstat(str(root)).st_mode
    except OSError as error:
        raise ValueError("official test cache is missing") from error
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ValueError("official test cache must be a regular directory")
    root = root.resolve()
    with tempfile.TemporaryDirectory(prefix=".v5-test-cache-") as snapshot:
        snapshot_path = Path(snapshot)
        for name, (size, digest) in expected_files.items():
            value = selection_module.read_verified_bytes(
                root / name, expected_sha256=digest, expected_size=size,
                name="official " + name,
            )
            (snapshot_path / name).write_bytes(value)
        return cache_module.load_fusion_cache(
            snapshot_path, expected_split="test"
        )


def _validate_candidate_artifacts(records, artifact_root, fold_keys, specs):
    root = Path(artifact_root).resolve()
    arrays = {}
    labels = None
    for record in records:
        path = _ref(
            root, record.prediction_path, record.prediction_sha256,
            record.name + " prediction",
        )
        current_labels, predictions = _check_prediction(
            path, fold_keys, record.name + " prediction",
            expected_sha256=record.prediction_sha256,
        )
        if labels is None:
            labels = current_labels
        elif not np.array_equal(labels, current_labels):
            raise ValueError("candidate artifacts do not share exact validation labels")
        actual_metrics = _metrics(
            current_labels, predictions, record.name + " validation"
        )
        if any(abs(actual_metrics[key] - record.validation_metrics[key]) > 1e-6
               for key in _KEYS):
            raise ValueError(record.name + " validation metrics do not match artifact")
        arrays[record.name] = predictions

    by_name = {record.name: record for record in records}
    for record in records:
        if record.type not in ("raw_ensemble", "rank_ensemble"):
            continue
        members = [arrays[name] for name in record.members]
        if record.type == "raw_ensemble":
            expected = np.stack(members).mean(axis=0)
        else:
            expected = np.stack(
                [rank_percentiles(value) for value in members], axis=0
            ).mean(axis=0)
            spec = specs[record.family]
            expected = expected * spec.rank_slope + spec.rank_intercept
        expected = np.asarray(expected, dtype=np.float32)
        if not np.array_equal(expected, arrays[record.name]):
            raise ValueError(record.name + " ensemble artifact does not match sealed members")
    if labels is None:
        raise ValueError("candidate artifacts are empty")
    return labels


def _validate_test_cache(cache, expected):
    if not isinstance(expected, ExpectedTestIdentity):
        raise ValueError("expected test identity is invalid")
    metadata = cache.metadata
    if metadata.get("split") != "test":
        raise ValueError("test evaluator requires expected_split=test")
    keys = tuple(str(value) for value in cache.keys)
    if (len(keys) != expected.count or
            metadata.get("count") != expected.count):
        raise ValueError("test cache count does not match expected identity")
    if len(keys) != len(set(keys)):
        raise ValueError("test cache contains duplicate keys")
    computed_order = _order_hash(keys)
    if (metadata.get("ordered_key_sha256") != computed_order or
            computed_order != expected.ordered_key_sha256):
        raise ValueError("test cache ordered key hash does not match")
    if metadata.get("manifest_sha256") != expected.manifest_sha256:
        raise ValueError("test cache manifest hash does not match")
    count = len(keys)
    expected_shapes = {
        "labels": (count,),
        "predictions": (count, 3),
        "standardized_predictions": (count, 3),
        "available": (count, 3),
    }
    for name, shape in expected_shapes.items():
        if np.asarray(getattr(cache, name)).shape != shape:
            raise ValueError("test cache " + name + " shape does not match")
    available = np.asarray(cache.available)
    if available.dtype != np.bool_ or not available[:, 0].all():
        raise ValueError("test cache Upper-Bound coverage is incomplete")
    latent_names = ("upper_bound_latents", "boltz_latents",
                    "three_dprot_latents")
    for name in latent_names:
        value = np.asarray(getattr(cache, name))
        if value.ndim != 2 or value.shape[0] != count or value.shape[1] <= 0:
            raise ValueError("test cache " + name + " shape does not match")
    numeric = (cache.labels, cache.predictions,
               cache.standardized_predictions,
               *(getattr(cache, name) for name in latent_names))
    if any(not np.isfinite(np.asarray(value)).all() for value in numeric):
        raise ValueError("test cache contains nonfinite arrays")
    dimensions = {
        "upper_bound": int(cache.upper_bound_latents.shape[1]),
        "boltz_pairwise": int(cache.boltz_latents.shape[1]),
        "three_dprot": int(cache.three_dprot_latents.shape[1]),
    }
    if metadata.get("latent_dimensions") != dimensions:
        raise ValueError("test cache latent dimensions do not match metadata")
    return metadata, keys

def _spec(specs,family,train,valid):
    try: result=training_module.StaticSpec.from_dict(specs[family])
    except (KeyError,TypeError,ValueError) as error: raise ValueError("sealed static reconstruction metadata is missing") from error
    if result.family!=family or result.train_cache_identity!=train or result.valid_cache_identity!=valid: raise ValueError("sealed static cache lineage does not match package")
    return result

def _torch_load(path, expected_sha256=None):
    return selection_module.load_verified_torch(
        path, expected_sha256=expected_sha256, map_location="cpu",
        name="checkpoint",
    )

def _model(path, record, spec, widths, device, expected_sha256=None, strict_metrics=False):
    payload = _torch_load(
        path, expected_sha256=expected_sha256
    )
    config = payload.get("model_config") if isinstance(payload, dict) else None
    state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else None
    run_config = payload.get("run_config") if isinstance(payload, dict) else None
    cache_lineage = payload.get("cache_lineage") if isinstance(payload, dict) else None
    if (not isinstance(config, dict) or not isinstance(state_dict, dict) or
            not isinstance(run_config, dict) or
            not isinstance(cache_lineage, dict)):
        raise ValueError(record.name + " checkpoint reconstruction metadata is missing")
    expected_config = {
        "class": "AnchoredResidualModel",
        "latent_widths": list(widths),
        "budget": float(record.budget),
        "dropout": config.get("dropout"),
        "adapter_width": 64,
        "hidden_widths": [256, 128, 64],
        "dtype": "float32",
    }
    if config != expected_config:
        raise ValueError(record.name + " checkpoint configuration does not match manifest/cache")
    spec_payload = spec.to_dict()
    spec_hash = training_module.static_spec_sha256(spec)
    expected_lineage = {
        "train": spec_payload["train_cache_identity"],
        "valid": spec_payload["valid_cache_identity"],
    }
    required_run = {
        "schema_version": 1,
        "static_spec_sha256": spec_hash,
        "static_spec": spec_payload,
        "family": spec.family,
        "deploy_weights": list(spec.deploy_weights),
        "rank_calibration": {
            "slope": spec.rank_slope,
            "intercept": spec.rank_intercept,
        },
        "cache_lineage": expected_lineage,
        "budget": float(record.budget),
        "seed": int(record.seed),
        "model": config,
    }
    if any(run_config.get(key) != value
           for key, value in required_run.items()):
        raise ValueError(record.name + " checkpoint run_config lineage does not match")
    if (cache_lineage != expected_lineage or
            payload.get("budget") != record.budget or
            payload.get("seed") != record.seed or
            payload.get("static_spec_sha256") != spec_hash):
        raise ValueError(record.name + " checkpoint lineage does not match")
    if strict_metrics:
        checkpoint_metrics = payload.get("metrics")
        if (not isinstance(checkpoint_metrics, dict) or
                set(checkpoint_metrics) != set(_KEYS) or
                any(abs(float(checkpoint_metrics[key]) -
                        float(record.validation_metrics[key])) > 1e-6
                    for key in _KEYS)):
            raise ValueError(record.name + " checkpoint metrics do not match candidate")
    model = AnchoredResidualModel(
        latent_widths=widths, budget=record.budget,
        dropout=float(config["dropout"]),
    ).float()
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(record.name + " checkpoint state_dict does not match") from error
    return model.to(device).eval()

def _move(batch,device):
    return {"base_predictions":batch["base_predictions"].to(device=device,dtype=torch.float32),"raw_predictions":batch["raw_predictions"].to(device=device,dtype=torch.float32),"standardized_predictions":batch["standardized_predictions"].to(device=device,dtype=torch.float32),"available":batch["available"].to(device),"latents":[x.to(device=device,dtype=torch.float32) for x in batch["latents"]]}

@torch.inference_mode()
def _residual(cache,base,record,spec,checkpoint,device,strict_metrics=False):
    widths=(cache.upper_bound_latents.shape[1],cache.boltz_latents.shape[1],cache.three_dprot_latents.shape[1]); model = _model(
        checkpoint, record, spec, widths, device,
        expected_sha256=record.checkpoint_sha256,
        strict_metrics=strict_metrics,
    )
    loader=DataLoader(data_module.AnchoredFusionDataset(cache,base),batch_size=256,shuffle=False,num_workers=0,collate_fn=data_module.collate_anchored_fusion); values=[]
    for batch in loader:
        output=model(_move(batch,device)).float()
        if not bool(torch.isfinite(output).all()): raise FloatingPointError(record.name+" produced nonfinite predictions")
        values.append(output.cpu().numpy())
    result=np.concatenate(values).astype(float)
    if result.shape!=cache.labels.shape or not np.isfinite(result).all(): raise ValueError(record.name+" predictions are invalid")
    if np.any(np.abs(result-np.asarray(base,dtype=float))>float(record.budget)+4*np.finfo(np.float32).eps): raise ValueError(record.name+" correction exceeds sealed budget")
    return result

def _predict(package,records,selected,specs,train,valid,cache,device,fold_keys,strict_checkpoint=False):
    root=package.parent
    prediction=_ref(
        root, selected.prediction_path, selected.prediction_sha256,
        selected.name+" prediction",
    )
    _check_prediction(
        prediction, fold_keys, selected.name+" prediction",
        expected_sha256=selected.prediction_sha256,
    )
    checkpoint=_ref(root,selected.checkpoint_path,selected.checkpoint_sha256,selected.name+" checkpoint",allow_null=selected.type!="residual")
    if selected.type=="expert":
        if selected.name not in data_module.EXPERT_ORDER: raise ValueError("selected expert is invalid")
        return data_module.fallback_predictions(cache.predictions,cache.available)[:,data_module.EXPERT_ORDER.index(selected.name)].astype(float)
    spec=_spec(specs,selected.family,train,valid); base=training_module.static_predictions(cache,spec).astype(float)
    if selected.type=="static": return base
    if selected.type=="residual":
        if selected.parent_static_name!="static_"+selected.family or checkpoint is None: raise ValueError("sealed residual parent or checkpoint is invalid")
        return _residual(
            cache, base, selected, spec, checkpoint, device,
            strict_metrics=strict_checkpoint,
        )
    if selected.type not in ("raw_ensemble","rank_ensemble") or selected.ensemble_rule not in ("arithmetic_mean","stable_average_percentile_mean"): raise ValueError("sealed ensemble rule is invalid")
    by_name={x.name:x for x in records}; members=[]
    for name in selected.members:
        member=by_name.get(name)
        if member is None or member.type!="residual" or member.family!=selected.family or selected.member_hashes.get(name)!=member.prediction_sha256: raise ValueError("sealed ensemble member lineage is invalid")
        member_prediction=_ref(
            root, member.prediction_path, member.prediction_sha256,
            member.name+" prediction",
        )
        _, member_artifact = _check_prediction(
            member_prediction, fold_keys, member.name+" prediction",
            expected_sha256=member.prediction_sha256,
        )
        member_checkpoint=_ref(root,member.checkpoint_path,member.checkpoint_sha256,member.name+" checkpoint")
        member_spec=_spec(specs,member.family,train,valid); member_base=training_module.static_predictions(cache,member_spec).astype(float)
        members.append(_residual(cache,member_base,member,member_spec,member_checkpoint,device))
    if selected.type=="raw_ensemble":
        if selected.ensemble_rule!="arithmetic_mean": raise ValueError("raw ensemble rule does not match")
        reconstructed = np.stack(members).mean(axis=0)
        return reconstructed
    if selected.ensemble_rule!="stable_average_percentile_mean": raise ValueError("rank ensemble rule does not match")
    percentile=np.stack([rank_percentiles(x) for x in members]).mean(axis=0)
    reconstructed = (
        percentile * spec.rank_slope + spec.rank_intercept
        if selected.family == "rank" else percentile
    )
    return reconstructed

def _metrics(labels,predictions,name):
    raw=affinity_metrics(np.asarray(labels,dtype=float),np.asarray(predictions,dtype=float))
    return {key:_finite(raw[key],name+" "+key) for key in _KEYS}

def _atomic(path,value):
    path=Path(path); descriptor,temp=tempfile.mkstemp(dir=str(path.parent),prefix="."+path.name+".",suffix=".tmp"); os.close(descriptor)
    try:
        with Path(temp).open("wb") as handle: handle.write(value); handle.flush(); os.fsync(handle.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)

def _atomic_json(path,value): _atomic(path,json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()+b"\n")
def _atomic_npz(path,**arrays):
    buffer=BytesIO(); np.savez_compressed(buffer,**arrays); _atomic(path,buffer.getvalue())

def _reserve(output):
    output=Path(output).expanduser().resolve(); output.mkdir(parents=True,exist_ok=True)
    if any((output/name).exists() for name in ("result.json","predictions.npz",_COMPLETE_MARKER)): raise RuntimeError("already evaluated")
    marker=output/_START_MARKER
    try: descriptor=os.open(str(marker),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
    except FileExistsError as error: raise RuntimeError("evaluation already in progress") from error
    try: os.write(descriptor,b"started\n"); os.fsync(descriptor)
    finally: os.close(descriptor)
    return output,marker

def _complete(output):
    marker=output/_COMPLETE_MARKER
    try: descriptor=os.open(str(marker),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
    except FileExistsError as error: raise RuntimeError("already evaluated") from error
    try: os.write(descriptor,b"complete\n"); os.fsync(descriptor)
    finally: os.close(descriptor)

def _evaluate_impl(
    package_dir, test_cache_dir, output_dir, *, device="cpu",
    expected_identity=OFFICIAL_TEST_IDENTITY, expected_file_identities=None,
):
    package, seal, selection, manifest, refs, specs, identities, reports = _package(package_dir)
    fold_binding = _fold_keys(package)
    records, selected, train, valid, passing = _validate_formal_package(
        selection, manifest, refs, identities, reports, specs, fold_binding,
        artifact_root=package.parent,
    )
    fold_keys = (
        fold_binding.keys
        if isinstance(fold_binding, SealedFoldMap)
        else tuple(fold_binding)
    )
    output, marker = _reserve(output_dir)
    try:
        try: device_value=torch.device(device)
        except (TypeError,RuntimeError) as error: raise ValueError("device is invalid") from error
        if device_value.type not in ("cpu","cuda"): raise ValueError("device must be cpu or cuda")
        if device_value.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA test evaluation requested but CUDA is unavailable")
        file_identities = (
            OFFICIAL_TEST_FILE_IDENTITIES
            if expected_file_identities is None else expected_file_identities
        )
        if Path(test_cache_dir).is_dir():
            cache = _load_verified_test_cache(test_cache_dir, file_identities)
        else:
            # Unit tests may inject an already-loaded cache; the public path is
            # always a directory and therefore takes the verified snapshot path.
            cache = cache_module.load_fusion_cache(
                test_cache_dir, expected_split="test"
            )
        metadata, keys = _validate_test_cache(cache, expected_identity)
        predictions = np.asarray(
            _predict(
                package, records, selected, specs, train, valid, cache,
                device_value, fold_keys, strict_checkpoint=True,
            ),
            dtype=np.float32,
        ).reshape(-1)
        if (predictions.shape != (expected_identity.count,) or
                not np.isfinite(predictions).all()):
            raise ValueError("selected test predictions are invalid")
        selected_metrics = selection.get("selected_validation_metrics")
        if not isinstance(selected_metrics, dict):
            raise ValueError("sealed selected validation metrics are missing")
        selected_metrics = {
            key: _finite(selected_metrics.get(key), "selected validation " + key)
            for key in _KEYS
        }
        if selected_metrics != dict(selected.validation_metrics):
            raise ValueError("selected validation metrics do not match manifest")
        upper = np.asarray(
            data_module.fallback_predictions(
                cache.predictions, cache.available
            )[:, 0],
            dtype=np.float32,
        )
        if not np.isfinite(upper).all():
            raise ValueError("upper-bound test predictions are invalid")
        _atomic_npz(
            output / "predictions.npz",
            keys=np.asarray(keys, dtype=np.str_),
            labels=np.asarray(cache.labels, dtype=np.float32),
            predictions=predictions,
            upper_bound_predictions=upper,
        )
        persisted = selection_module.load_verified_npz(
            output / "predictions.npz", name="persisted predictions"
        )
        persisted_predictions = np.asarray(persisted["predictions"])
        persisted_upper = np.asarray(persisted["upper_bound_predictions"])
        if (persisted_predictions.dtype != np.float32 or
                persisted_upper.dtype != np.float32 or
                not np.array_equal(persisted_predictions, predictions) or
                not np.array_equal(persisted_upper, upper)):
            raise ValueError("persisted prediction arrays do not match inference")
        metrics = _metrics(cache.labels, persisted_predictions, "test")
        upper_metrics = _metrics(
            cache.labels, persisted_upper, "upper bound test"
        )
        prediction_sha256 = _hash(output / "predictions.npz")
        result = {
            "status": test_status(metrics["CI"]),
            "count": expected_identity.count,
            "metrics": metrics,
            "coverage": 1.0,
            "selected_validation_metrics": selected_metrics,
            "upper_bound_test_metrics": upper_metrics,
            "delta_ci_vs_upper_bound": metrics["CI"] - upper_metrics["CI"],
            "target_ci": TARGET_CI,
            "delta_ci_vs_target": metrics["CI"] - TARGET_CI,
            "selection_seal_sha256": _hash(package / "selection_seal.json"),
            "prediction_sha256": prediction_sha256,
            "selected_name": selected.name,
            "selected_type": selected.type,
            "test_manifest_sha256": metadata["manifest_sha256"],
            "test_ordered_key_sha256": metadata["ordered_key_sha256"],
        }
        _atomic_json(output / "result.json", result)
        _complete(output)
        marker.unlink()
        return result
    except BaseException:
        for name in ("evaluation.complete", "result.json", "predictions.npz"):
            try:
                (output / name).unlink()
            except FileNotFoundError:
                pass
        try:
            marker.unlink()
        except FileNotFoundError:
            pass
        raise

def evaluate_official_test(package, test_cache, output, *, device="cpu"):
    return _evaluate_impl(package, test_cache, output, device=device, expected_identity=OFFICIAL_TEST_IDENTITY)
def evaluate_sealed_selection(package,test_cache,output,*,device="cpu"): return evaluate_official_test(package,test_cache,output,device=device)
def evaluate_test(package,test_cache,output,*,device="cpu"): return evaluate_official_test(package,test_cache,output,device=device)

def build_arg_parser():
    parser=argparse.ArgumentParser(description="Evaluate one sealed anchored expert fusion v5 selection on test"); parser.add_argument("--package",required=True); parser.add_argument("--test-cache",required=True); parser.add_argument("--output",required=True); parser.add_argument("--device",default="cpu"); return parser

def main(argv=None):
    args=build_arg_parser().parse_args(argv); print(json.dumps(evaluate_official_test(args.package,args.test_cache,args.output,device=args.device),sort_keys=True),flush=True); return 0

if __name__=="__main__": raise SystemExit(main())
__all__ = [
    "ExpectedTestIdentity", "OFFICIAL_TEST_COUNT",
    "OFFICIAL_TEST_MANIFEST_SHA256",
    "OFFICIAL_TEST_ORDERED_KEY_SHA256", "OFFICIAL_TEST_FILE_IDENTITIES", "TARGET_CI",
    "build_arg_parser", "evaluate_official_test",
    "evaluate_sealed_selection", "evaluate_test", "main", "test_status",
]
