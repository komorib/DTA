"""
Molecular dropout for KDBNet's ``DTAModel`` -- faithful port of DockTDeep.

Mechanism (training only):
  * per sample, with probability ``p`` a drop fires
  * mode="complex": the dropped partner is protein vs ligand, chosen 50/50
  * a dropped partner contributes a ZERO embedding to the fusion concat
        - "ligand"  -> zero the drug branch (xd)
        - "protein" -> zero BOTH protein branches (xp AND the pocket-atom xpa),
          because in KDBNet the protein is represented by the residue/ESM branch
          *and* the pocket-atom branch
  * whenever a drop fires, that sample's regression target is set to the
    "no-binding" anchor.  For raw pKd labels that is 0.0
    (pKd = 0  <=>  Kd = 1 M  <=>  DockTDeep's delta_G = 0).

This operates at the EMBEDDING level (after each per-branch FC, before the
concat); it never touches coordinates.  It is active only when the training
loop passes keep masks into ``DTAModel.forward``; validation / test / inference
call the model without masks and are completely unaffected.
"""

import torch


class MolecularDropout:
    def __init__(self, p=0.06, no_binding_value=0.0, mode="complex"):
        assert mode in ("complex", "protein", "ligand")
        self.p = float(p)
        self.anchor = float(no_binding_value)
        self.mode = mode

    @torch.no_grad()
    def sample(self, batch_size, labels):
        """Make the per-sample drop decision ONCE (mirrors DockTDeep drawing
        alpha/beta a single time and reusing them for both the embedding mask
        and the label edit).

        Parameters
        ----------
        batch_size : int
        labels     : (B,) or (B, 1) tensor of true affinities (raw pKd)

        Returns
        -------
        keep_drug : (B, 1) float in {0., 1.} -- 0 => drug branch dropped
        keep_prot : (B, 1) float in {0., 1.} -- 0 => BOTH protein branches dropped
        labels    : labels with every fired sample overwritten by the anchor
        """
        device = labels.device
        alpha = torch.rand(batch_size, device=device)   # trigger draw
        beta = torch.rand(batch_size, device=device)    # which-partner coin
        fire = alpha <= self.p

        if self.mode == "complex":
            drop_drug = fire & (beta > 0.5)
            drop_prot = fire & (beta <= 0.5)
        elif self.mode == "protein":
            drop_prot, drop_drug = fire, torch.zeros_like(fire)
        else:  # "ligand"
            drop_drug, drop_prot = fire, torch.zeros_like(fire)

        keep_drug = (~drop_drug).float().unsqueeze(-1)
        keep_prot = (~drop_prot).float().unsqueeze(-1)

        fire_b = fire.view([-1] + [1] * (labels.dim() - 1))
        labels = torch.where(fire_b, torch.full_like(labels, self.anchor), labels)
        return keep_drug, keep_prot, labels
