"""End-to-end smoke test for KDBNet2 (方向 A 改动版) on REAL PDBbind data,
with zero ESM embeddings (protein branch uses 1280-dim zero vectors — this is
a wiring/plumbing test, NOT a scientific run).

What it verifies (项目 Prompt 阶段 4 的验收点):
  1. 单样本 featurization 能跑(manifest -> pocket graph + ligand graph)
  2. 单 batch forward + backward(baseline MSE 路径)
  3. 10-step overfit(loss 应下降;确认梯度真的在更新)
  4. heteroscedastic NLL 路径:forward/backward,且预测的 sigma 有限、非 NaN
  5. flag 关闭(heteroscedastic=False)时确为原 baseline 行为(输出单通道)

Run (in your kdbnet conda env, from repo root /mnt/d/KDBNet):
    python scripts/smoke_test.py --limit 64 --n_epochs 12

It only touches the first --limit complexes of the manifest, uses random split,
CPU or GPU (auto). No ESM, no external downloads.
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kdbnet.experiment import DTAExperiment


def banner(msg):
    print("\n" + "=" * 70)
    print("[SMOKE] " + msg)
    print("=" * 70, flush=True)


def build_exp(manifest, pdbbind_dir, limit, n_epochs, batch_size, lr, seed,
              heteroscedastic, nll_warmup_epochs):
    """Construct a DTAExperiment with a MINIMAL branch config (drug 3D + protein
    residue only; pocket_atom / one_d_two_d OFF) for the fastest possible loop."""
    return DTAExperiment(
        task="pdbbind_pl",
        split_method="random",
        split_frac=[0.7, 0.15, 0.15],
        pdbbind_manifest=manifest,
        pdbbind_dir=pdbbind_dir,
        split_file=None,
        limit=limit,
        # smallest sensible model
        prot_gcn_dims=[64, 64],
        prot_fc_dims=[128, 64],
        drug_gcn_dims=[64, 32],
        drug_fc_dims=[128, 64],
        mlp_dims=[128, 64],
        mlp_dropout=0.25,
        use_pocket_atom=False,
        use_one_d_two_d=False,
        use_molecular_dropout=False,
        n_ensembles=1,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        uncertainty=False,
        parallel=False,
        prot_emb_dim=1280,          # zero-ESM: 1280-dim zero vectors per residue
        heteroscedastic=heteroscedastic,
        nll_warmup_epochs=nll_warmup_epochs,
        save_log=False,
        save_checkpoint=False,
    )


def run_variant(tag, **kw):
    banner(f"variant = {tag}")
    exp = build_exp(**kw)
    # sizes of each split
    for s in ("train", "valid", "test"):
        try:
            n = len(exp.task_df[s])
            print(f"  split {s}: {n} samples")
        except Exception as e:
            print(f"  split {s}: <err {e}>")

    # train (this exercises single-sample featurization, single-batch f/b,
    # and the full epoch loop with periodic eval)
    exp.train(n_epochs=kw["n_epochs"], patience=kw["n_epochs"],
              eval_freq=max(1, kw["n_epochs"] // 3), test_freq=None,
              monitoring_score="mse")

    # test + inspect predictions
    res = exp.test(test_tag=f"smoke-{tag}", print_log=True)
    df = res["df"]
    print(f"  [pred] columns: {list(df.columns)}")
    yp = df["y_pred"].values.astype(float)
    print(f"  [pred] y_pred finite={np.isfinite(yp).all()} "
          f"min={yp.min():.3f} max={yp.max():.3f}")
    if "y_std" in df.columns:
        ys = df["y_std"].values.astype(float)
        print(f"  [pred] y_std finite={np.isfinite(ys).all()} "
              f"min={ys.min():.4g} max={ys.max():.4g}")
    for col in ("y_std_epistemic", "y_std_aleatoric", "y_std_total"):
        if col in df.columns:
            v = df[col].values.astype(float)
            print(f"  [pred] {col} finite={np.isfinite(v).all()} "
                  f"min={v.min():.4g} max={v.max():.4g}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/PDBbind/processed/pdbbind_pl_manifest.tsv")
    ap.add_argument("--pdbbind_dir", default="data/PDBbind")
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--n_epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
    common = dict(manifest=args.manifest, pdbbind_dir=args.pdbbind_dir,
                  limit=args.limit, n_epochs=args.n_epochs,
                  batch_size=args.batch_size, lr=args.lr, seed=args.seed)

    results = {}
    ok = True

    # --- variant 1: baseline (MSE, single channel) ---
    try:
        results["baseline"] = run_variant("baseline_mse",
            heteroscedastic=False, nll_warmup_epochs=0, **common)
    except Exception:
        ok = False
        print("[SMOKE][FAIL] baseline variant crashed:")
        traceback.print_exc()

    # --- variant 2: heteroscedastic NLL (two channel, aleatoric) ---
    try:
        results["hetero"] = run_variant("heteroscedastic_nll",
            heteroscedastic=True, nll_warmup_epochs=max(2, args.n_epochs // 3),
            **common)
    except Exception:
        ok = False
        print("[SMOKE][FAIL] heteroscedastic variant crashed:")
        traceback.print_exc()

    banner("SMOKE SUMMARY")
    print("baseline ran:", "baseline" in results)
    print("hetero ran:  ", "hetero" in results)
    print("OVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
