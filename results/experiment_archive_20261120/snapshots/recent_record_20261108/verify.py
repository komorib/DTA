"""Verify archived aggregate evidence and displayed tables; no model/data reads."""
import hashlib
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
METRICS = ("CI", "RMSE", "MAE", "Pearson", "Spearman")


def read(name):
    return json.loads((BASE / "sources" / name).read_text(encoding="utf-8"))


def main():
    manifest = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["artifacts"]:
        raw = (BASE / entry["archive"]).read_bytes()
        assert len(raw) == entry["bytes"], entry["archive"]
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"], entry["archive"]
        json.loads(raw)
    assert manifest["artifact_count"] == len(manifest["artifacts"]) == 39
    report = (BASE.parents[1] / "reports" / "LP_PDBBind_recent_experiments_20261108.md").read_text(encoding="utf-8")
    tables = []
    for line in report.splitlines():
        if line.startswith("| "):
            tables.append([cell.strip() for cell in line.split("|")[1:-1]])

    def row(label, metrics, count=None, width=None):
        matching = [r for r in tables if r[0] == label and (width is None or len(r) == width)]
        assert len(matching) == 1, (label, matching)
        actual = matching[0]
        assert actual[-5:] == ["%.6f" % metrics[k] for k in METRICS], (label, actual)
        if count is not None:
            assert int(actual[1].replace(",", "")) == count, label

    depth4 = read("depth4_summary.json")["aligned_msa"]
    sam = read("sam_summary.json")["aligned_msa"]
    row("四层 affinity-only AdamW", depth4["validation_metrics"])
    row("四层 SAM-AdamW", sam["validation_metrics"])
    for arm in (depth4, sam):
        assert arm["optimizer_steps"] == 20700
        assert arm["checkpoint_reload_maximum_prediction_delta"] == 0
    assert len(read("depth4_history.json")) == len(read("sam_history.json")) == 30
    comparison = read("msa_context_comparison.json")
    row("完整 ESM＋MSA，strict v2", comparison["metrics"]["old_aligned"])
    row("MSA-only context", comparison["metrics"]["new_aligned"], width=7)
    esm = read("esm_only_comparison.json")["rows"]
    row("完整 ESM＋MSA，v3", esm["esm_msa_v3"]["metrics"])
    row("真正 ESM-only v1", esm["esm_only"]["metrics"])
    joint8 = read("joint8_selection.json")
    row("8行 MSA＋2层交互，单 A", joint8["validation_metrics"])
    assert joint8["optimizer_steps"] == 20700
    joint32 = read("joint32_history_snapshot.json")
    assert len(joint32) == 16 and joint32[-1]["epoch"] == 15
    best = max(joint32, key=lambda x: x["validation_metrics"]["CI"])
    assert best["epoch"] == 6
    row("32行 MSA＋4层交互，修补版单 A", best["validation_metrics"])
    assert read("joint32_failure.json")["error_type"] == "KeyboardInterrupt"
    random = read("random_validation_ablation.json")
    row("完整 FullChainESMMSAAll v2", random["metrics"]["full_esm_msa"])
    row("完整模型参照", random["metrics"]["full_esm_msa"])
    # Two distinct row labels describe the random MSA-only arm vs original LP.
    random_rows = [r for r in tables if r[0] == "MSA-only context" and len(r) == 6]
    assert len(random_rows) == 1
    assert random_rows[0][-5:] == ["%.6f" % random["metrics"]["msa_context_only"][k] for k in METRICS]
    common = read("common_validation_three.json")
    assert common["common_count"] == 1783
    for label, key in (("完整自研 v2", "fullchain_esm_msa_all_v2"), ("3DProtDTA", "3dprot"), ("GraphDTA-GIN", "graphdta")):
        row(label, common["common_metrics"][key])
    groups = read("coldstart_group_metrics.json")["groups"]
    for label, key in (("全部", "all"), ("cold_ligand", "cold_ligand"), ("cold_protein", "cold_protein"), ("cold_both", "cold_both"), ("双已见未标类别", "both_seen_unmarked")):
        row(label, groups[key]["metrics"], groups[key]["count"])
    counts = read("annotation_summary.json")["counts"]
    assert counts["identity_unknown"] == 0
    assert sum(groups[k]["count"] for k in groups if k != "all") == counts["test_total"] == 1813
    no3d = read("no3d_summary.json")
    row("no_3d", no3d["reported_validation_metrics"])
    assert no3d["selection"]["optimizer_steps"] == 27210
    assert no3d["selection"]["checkpoint_reload_maximum_prediction_delta"] == 0
    smoke = read("kermt_gpu5_smoke_passed.json")
    assert smoke["status"] == "real_gpu_smoke_passed"
    assert smoke["initial_zero_forward_equal"] and smoke["projection_updated"]
    assert smoke["projection_gradient_absolute_sum"] > 0
    assert smoke["reload_maximum_prediction_delta"] == 0
    assert not smoke["validation_metrics_computed"]
    print(json.dumps(dict(status="archived_evidence_and_report_tables_verified", artifact_count=39,
                          report_tables_checked=True, source_files_modified=False,
                          snapshot_no1d_completed_epochs=len(read("no1d_history_snapshot.json")),
                          model_forward_calls=0, training_steps=0,
                          test_row_labels_or_predictions_read=False), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
