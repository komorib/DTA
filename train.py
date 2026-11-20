"""Train the standalone early-only model on the original FullChain data layout."""

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from data import EarlyKVCollator, EarlyKVDataset, MoleculeCache
from losses import affinity_only_loss
from model import FullChainMoLFormerEarlyKVV1
from dta.lp_pdbbind_fullchain_esm_msa_cross_v1.run_training import _fit_arm
from dta.msa_interface_pair_pooling_v18.training import _seed_everything
from dta.msa_interface_pair_pooling_v18.training import TrainingConfig


def loss_for_runner(output, normalized_labels, _batch, config):
    return affinity_only_loss(output, normalized_labels, config.minimum_rank_gap)


def make_dataset(args, molecule_cache, split):
    return EarlyKVDataset(
        Path(args.manifest_root) / (split + "_manifest.json"),
        Path(args.complex_root) / split,
        Path(args.feature_cache) / args.mode,
        args.mode,
        Path(args.source_manifest),
        Path(args.targets),
        args.weight_sha256,
        molecule_cache=molecule_cache,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--complex-root", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--weight-sha256", required=True)
    parser.add_argument("--molecule-cache", required=True)
    parser.add_argument("--mode", default="aligned_msa")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--label-mean", type=float, default=None)
    parser.add_argument("--label-std", type=float, default=None)
    args = parser.parse_args()

    config = TrainingConfig(
        pretrained_checkpoint="none", device=args.device, seed=args.seed,
        batch_size=args.batch_size, evaluation_batch_size=args.eval_batch_size,
        num_workers=args.workers, amp_dtype="bfloat16", learning_rate=1e-4,
        weight_decay=1e-4, maximum_epochs=args.epochs, model_width=512,
        layers=4, heads=8, dropout=0.2, latent_width=1024,
        msa_width=192, msa_layers=4, msa_heads=6, pair_factor=8,
        pair_width=96, interface_blocks=2, pair_token_width=256,
        pair_token_heads=8, pair_token_layers=2, pair_token_count=96,
        inference_subsets=1, rank_weight=0.40, mse_weight=0.15,
        ema_decay=0.995,
    )
    molecule_cache = MoleculeCache(args.molecule_cache)
    train_data = make_dataset(args, molecule_cache, "train")
    valid_data = make_dataset(args, molecule_cache, "valid")
    label_mean = (float(train_data.labels.mean()) if args.label_mean is None
                  else float(args.label_mean))
    label_std = (float(train_data.labels.std()) if args.label_std is None
                 else float(args.label_std))
    if not math.isfinite(label_mean) or not math.isfinite(label_std) or label_std <= 0:
        raise ValueError("training labels cannot be normalized")
    try:
        _seed_everything(args.seed)
        model = FullChainMoLFormerEarlyKVV1(**config.model_kwargs())
        initial_state = {name: value.detach().cpu().clone()
                         for name, value in model.state_dict().items()}
        runner_args = SimpleNamespace(
            seed=args.seed, device=args.device, batch_size=args.batch_size,
            evaluation_batch_size=args.eval_batch_size, num_workers=args.workers,
            learning_rate=1e-4, weight_decay=1e-4, maximum_epochs=args.epochs,
        )
        output = Path(args.output).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        summary = _fit_arm(
            mode="early_kv", initial_state=initial_state, config=config,
            train_data=train_data, valid_data=valid_data, args=runner_args,
            label_mean=label_mean, label_std=label_std, baseline={},
            output=output, lineage={}, model_type=FullChainMoLFormerEarlyKVV1,
            collator_type=EarlyKVCollator,
            version="lp-pdbbind-fullchain-molformer-early-kv-v1-standalone",
            protocol_fields={
                "architecture_flags": {
                    "molecule_kv_enabled": True,
                    "molecule_branch_enabled": False,
                },
                "auxiliary_effective_scale": 0.0,
            }, loss_function=loss_for_runner,
            selection_policy="ci_lexicographic_v1",
        )
        print(summary)
    finally:
        train_data.close()
        valid_data.close()


if __name__ == "__main__":
    main()
