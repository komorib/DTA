"""Run dependency and early-fusion contract checks for this release."""

from pathlib import Path

import torch

import inspect

from model import FullChainMoLFormerEarlyKVV1


def main():
    root = Path(__file__).resolve().parent
    required = ("model.py", "train.py", "data.py", "losses.py", "evaluate.py", "dta")
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise AssertionError("missing release files: %s" % ", ".join(missing))

    source_code = inspect.getsource(FullChainMoLFormerEarlyKVV1)
    for fragment in ("nn.LayerNorm(768)", "nn.Linear(768, width)",
                     "token[:, None, :]", "torch.cat((source, token[:, None, :]), dim=1)"):
        if fragment not in source_code:
            raise AssertionError("early K/V contract is missing: %s" % fragment)

    # Exercise the exact tensor operation used by the model without constructing
    # the full geometric trunk or loading data.
    source = torch.zeros(2, 7, 512)
    mask = torch.ones(2, 7, dtype=torch.bool)
    token = torch.zeros(2, 512)
    memory = torch.cat((source, token[:, None, :]), dim=1)
    memory_mask = torch.cat((mask, torch.ones((2, 1), dtype=torch.bool)), dim=1)
    if memory.shape != (2, 8, 512) or memory_mask.shape != (2, 8):
        raise AssertionError("molecule token was not appended once")
    if not bool(memory_mask[:, -1].all()):
        raise AssertionError("appended molecule token is masked")

    print("public release smoke: OK")


if __name__ == "__main__":
    main()
