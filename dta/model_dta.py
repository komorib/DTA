import torch
import torch.nn.functional as F

from net.model import GerNA


class GerNADTA(GerNA):
    """GerNA backbone with an unconstrained scalar regression head.

    The base GerNA class already builds a Linear(..., 1) head when
    uncertainty=False, but its Affinity_pred_module applies sigmoid to that
    scalar. DTA needs raw pAffinity values, so only the final activation is
    changed here.
    """

    def Affinity_pred_module(
        self,
        batch_size,
        RNA_1d_fea=None,
        RNA_2d_fea=None,
        Mol_2d_fea=None,
        seq_mask=None,
        vertex_mask=None,
        RNA_feats=None,
        Mol_feats=None,
        RNA_mask=None,
        Mol_mask=None,
        pairwise_pred_2d=None,
        pairwise_pred_3d=None,
        dti_global=None,
        pbcnet_global=None,
        boltz_interface=None,
    ):
        if getattr(self, "_captured_affinity_features", None) is not None:
            raise RuntimeError("captured GerNA affinity features are unread")
        legacy_capture_requested = bool(
            getattr(self, "_affinity_feature_capture_requested", False)
        )
        if getattr(self, "_captured_multiscale_affinity_features", None) is not None:
            raise RuntimeError("captured GerNA multiscale features are unread")
        multiscale_capture_requested = bool(
            getattr(self, "_multiscale_affinity_feature_capture_requested", False)
        )
        self._affinity_feature_capture_requested = False
        self._multiscale_affinity_feature_capture_requested = False
        self._captured_affinity_features = None
        self._captured_multiscale_affinity_features = None
        capture_requested = legacy_capture_requested or multiscale_capture_requested
        if self.mol_graph:
            Mol_2d_fea = F.leaky_relu(self.mol_2d_final(Mol_2d_fea), 0.1)
            if self.rna_repre:
                RNA_1d_fea = F.leaky_relu(self.rna_1d_final(RNA_1d_fea), 0.1)
                if self.rna_graph:
                    RNA_2d_fea = F.leaky_relu(self.rna_2d_final(RNA_2d_fea), 0.1)
                    self.fusion_grad_frac = 0.5
                    RNA_2d_fea = self.fusion_grad_frac * RNA_2d_fea + (1 - self.fusion_grad_frac) * RNA_1d_fea
                else:
                    RNA_2d_fea = RNA_1d_fea
            else:
                RNA_2d_fea = F.leaky_relu(self.rna_2d_final(RNA_2d_fea), 0.1)
        if self.coors:
            RNA_feats = F.leaky_relu(self.rna_3d_final(RNA_feats), 0.1)
            Mol_feats = F.leaky_relu(self.mol_3d_final(Mol_feats), 0.1)

        if self.coors:
            mf_3d, rf_3d = self.dma_gru_3d(batch_size, RNA_feats, Mol_feats, RNA_mask, Mol_mask, pairwise_pred_3d)
            interaction_3d = F.leaky_relu(
                torch.einsum("bi,bj->bij", mf_3d, rf_3d),
                0.1,
            )
            kroneck_3d = interaction_3d.flatten(1)
        if self.mol_graph:
            mf_2d, rf_2d = self.dma_gru_2d(batch_size, RNA_2d_fea, Mol_2d_fea, seq_mask, vertex_mask, pairwise_pred_2d)
            interaction_2d = F.leaky_relu(
                torch.einsum("bi,bj->bij", mf_2d, rf_2d),
                0.1,
            )
            kroneck_2d = interaction_2d.flatten(1)

        if self.mol_graph and self.coors:
            kroneck = torch.cat((kroneck_2d, kroneck_3d), dim=1)
        elif self.mol_graph:
            kroneck = kroneck_2d
        elif self.coors:
            kroneck = kroneck_3d
        else:
            raise ValueError("GerNADTA requires at least one molecular/protein representation branch")

        kroneck = self.apply_dti_late_fusion(kroneck, dti_global)
        kroneck = self.apply_pbcnet_late_fusion(kroneck, pbcnet_global)
        kroneck = self.apply_boltz_interface_fusion(kroneck, boltz_interface)
        captured_head = []
        handle = None
        if capture_requested:
            handle = self.W_out[-1].register_forward_pre_hook(
                lambda module, inputs: captured_head.append(inputs[0])
            )
        try:
            prediction = self.W_out(kroneck)
        finally:
            if handle is not None:
                handle.remove()
        if capture_requested and len(captured_head) != 1:
            raise RuntimeError("failed to capture GerNA affinity head")
        if legacy_capture_requested:
            self._captured_affinity_features = {
                "gerna_2d": kroneck_2d if self.mol_graph else None,
                "gerna_3d": kroneck_3d if self.coors else None,
                "gerna_head": captured_head[0],
            }
        if multiscale_capture_requested:
            self._captured_multiscale_affinity_features = {
                "gerna_2d_molecule": mf_2d,
                "gerna_2d_protein": rf_2d,
                "gerna_3d_molecule": mf_3d,
                "gerna_3d_protein": rf_3d,
                "gerna_2d_interaction": interaction_2d,
                "gerna_3d_interaction": interaction_3d,
                "gerna_head": captured_head[0],
            }
        return prediction

    def request_affinity_features(self):
        if getattr(self, "_affinity_feature_capture_requested", False):
            raise RuntimeError("GerNA affinity features already requested")
        if getattr(self, "_captured_affinity_features", None) is not None:
            raise RuntimeError("captured GerNA affinity features are unread")
        self._affinity_feature_capture_requested = True

    def discard_affinity_features(self):
        self._affinity_feature_capture_requested = False
        self._captured_affinity_features = None

    def request_multiscale_affinity_features(self):
        if getattr(self, "_multiscale_affinity_feature_capture_requested", False):
            raise RuntimeError("GerNA multiscale features already requested")
        if getattr(self, "_captured_multiscale_affinity_features", None) is not None:
            raise RuntimeError("captured GerNA multiscale features are unread")
        self._multiscale_affinity_feature_capture_requested = True

    def discard_multiscale_affinity_features(self):
        self._multiscale_affinity_feature_capture_requested = False
        self._captured_multiscale_affinity_features = None

    def take_multiscale_affinity_features(self):
        features = getattr(self, "_captured_multiscale_affinity_features", None)
        if features is None:
            raise RuntimeError(
                "no captured GerNA multiscale features; run forward first"
            )
        self._captured_multiscale_affinity_features = None
        return features

    def take_affinity_features(self):
        features = getattr(self, "_captured_affinity_features", None)
        if features is None:
            raise RuntimeError(
                "no captured GerNA affinity features; run forward first"
            )
        self._captured_affinity_features = None
        return features

    def high_level_affinity_parameters(self):
        modules = [self.W_out]
        if self.mol_graph:
            modules.extend(
                [
                    self.mol_2d_final,
                    self.mc1,
                    self.mp1,
                    self.hc0,
                    self.hp0,
                    self.hc1,
                    self.hp1,
                    self.m_to_r_transform,
                    self.r_to_m_transform,
                    self.GRU_dma,
                ]
            )
            if hasattr(self, "rna_1d_final"):
                modules.append(self.rna_1d_final)
            if hasattr(self, "rna_2d_final"):
                modules.append(self.rna_2d_final)
        if self.coors:
            modules.extend(
                [
                    self.mol_3d_final,
                    self.rna_3d_final,
                    self.mc1_3d,
                    self.mp1_3d,
                    self.hc0_3d,
                    self.hp0_3d,
                    self.hc1_3d,
                    self.hp1_3d,
                    self.m_to_r_transform_3d,
                    self.r_to_m_transform_3d,
                    self.GRU_dma_3d,
                ]
            )
        if self.dti_late_fusion == "concat":
            modules.extend([self.dti_late_norm, self.dti_late_proj])
        if self.pbcnet_late_fusion == "concat":
            modules.extend([self.pbcnet_late_norm, self.pbcnet_late_proj])
        if self.boltz_interface_fusion:
            modules.append(self.boltz_interface_head)

        parameters = []
        seen = set()
        for module in modules:
            for parameter in module.parameters():
                identifier = id(parameter)
                if identifier not in seen:
                    seen.add(identifier)
                    parameters.append(parameter)
        return parameters
