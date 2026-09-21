import torch
import torch.nn as nn
import torch_geometric
from kdbnet.gvp import GVP, GVPConvLayer, LayerNorm
from kdbnet.one_d_two_d import OneDTwoDBranch
from kdbnet.interface_pairformer import PocketLigandInterfacePairformer

class Prot3DGraphModel(nn.Module):
    def __init__(self,
        d_vocab=21, d_embed=20,
        d_dihedrals=6, d_pretrained_emb=1280, d_edge=39,
        d_gcn=[128, 256, 256],
    ):
        super(Prot3DGraphModel, self).__init__()
        d_gcn_in = d_gcn[0]
        self.embed = nn.Embedding(d_vocab, d_embed)
        self.proj_node = nn.Linear(d_embed + d_dihedrals + d_pretrained_emb, d_gcn_in)
        self.proj_edge = nn.Linear(d_edge, d_gcn_in)
        gcn_layer_sizes = [d_gcn_in] + d_gcn
        layers = []
        for i in range(len(gcn_layer_sizes) - 1):            
            layers.append((
                torch_geometric.nn.TransformerConv(
                    gcn_layer_sizes[i], gcn_layer_sizes[i + 1], edge_dim=d_gcn_in),
                'x, edge_index, edge_attr -> x'
            ))            
            layers.append(nn.LeakyReLU())            
        
        self.gcn = torch_geometric.nn.Sequential(
            'x, edge_index, edge_attr', layers)        
        self.pool = torch_geometric.nn.global_mean_pool
        

    def forward(self, data):
        x, edge_index = data.seq, data.edge_index
        batch = data.batch

        x = self.embed(x)
        s = data.node_s
        emb = data.seq_emb
        x = torch.cat([x, s, emb], dim=-1)

        edge_attr = data.edge_s

        x = self.proj_node(x)
        edge_attr = self.proj_edge(edge_attr)

        x = self.gcn(x, edge_index, edge_attr)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        return x

class DrugGVPModel(nn.Module):
    def __init__(self, 
        node_in_dim=[66, 1], node_h_dim=[128, 64],
        edge_in_dim=[16, 1], edge_h_dim=[32, 1],
        num_layers=3, drop_rate=0.1,
    ):
        """
        Parameters
        ----------
        node_in_dim : list of int
            Input dimension of drug node features (si, vi).
            Scalar node feartures have shape (N, si).
            Vector node features have shape (N, vi, 3).
        node_h_dims : list of int
            Hidden dimension of drug node features (so, vo).
            Scalar node feartures have shape (N, so).
            Vector node features have shape (N, vo, 3).
        """
        super(DrugGVPModel, self).__init__()
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers))

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

    def forward(self, xd):
        # Unpack input data
        h_V = (xd.node_s, xd.node_v)
        h_E = (xd.edge_s, xd.edge_v)
        edge_index = xd.edge_index
        batch = xd.batch

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)

        # per-graph mean
        out = torch_geometric.nn.global_add_pool(out, batch)

        return out


class PocketAtomGVPModel(nn.Module):
    def __init__(self,
        node_in_dim=[63, 1], node_h_dim=[128, 32],
        edge_in_dim=[16, 1], edge_h_dim=[32, 1],
        num_layers=3, drop_rate=0.1,
    ):
        super(PocketAtomGVPModel, self).__init__()
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )
        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers)
        )
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0))
        )

    def forward(self, data):
        h_V = (data.node_s, data.node_v)
        h_E = (data.edge_s, data.edge_v)
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, data.edge_index, h_E)
        out = self.W_out(h_V)
        return torch_geometric.nn.global_add_pool(out, data.batch)


class DTAModel(nn.Module):
    def __init__(self,
            prot_emb_dim=1280,
            prot_gcn_dims=[128, 256, 256],
            prot_fc_dims=[1024, 128],
            drug_node_in_dim=[66, 1], drug_node_h_dims=[128, 64],
            drug_edge_in_dim=[16, 1], drug_edge_h_dims=[32, 1],            
            drug_fc_dims=[1024, 128],
            use_pocket_atom=False,
            pocket_atom_node_in_dim=[63, 1],
            pocket_atom_node_h_dims=[128, 32],
            pocket_atom_edge_in_dim=[16, 1],
            pocket_atom_edge_h_dims=[32, 1],
            pocket_atom_fc_dims=[1024, 128],
            use_one_d_two_d=False,
            one_d_two_d_hidden_dim=64,
            one_d_two_d_output_dim=128,
            one_d_two_d_gnn_layers=3,
            one_d_two_d_triangle_layers=2,
            one_d_two_d_dropout=0.25,
            one_d_two_d_use_protein_2d=True,
            one_d_two_d_fc_dims=[1024, 128],
            use_interface_pairformer=False,
            interface_pairformer_single_dim=128,
            interface_pairformer_pair_dim=64,
            interface_pairformer_output_dim=128,
            interface_pairformer_blocks=4,
            interface_pairformer_heads=4,
            interface_pairformer_num_rbf=32,
            interface_pairformer_fine_num_rbf=32,
            interface_pairformer_max_pocket_residues=64,
            interface_pairformer_max_ligand_atoms=64,
            interface_pairformer_max_residue_atoms=14,
            interface_pairformer_pocket_radius=10.0,
            interface_pairformer_min_pocket_residues=16,
            interface_pairformer_distance_max=20.0,
            interface_pairformer_fine_distance_max=8.0,
            interface_pairformer_dropout=0.15,
            interface_pairformer_gate_hidden_dim=128,
            mlp_dims=[1024, 512], mlp_dropout=0.25,
            predict_logvar=False):
        super(DTAModel, self).__init__()
        self.use_pocket_atom = use_pocket_atom
        self.use_one_d_two_d = use_one_d_two_d
        self.use_interface_pairformer = use_interface_pairformer
        self.predict_logvar = predict_logvar
        raw_prot_emb_dim = prot_emb_dim

        self.drug_model = DrugGVPModel(
            node_in_dim=drug_node_in_dim, node_h_dim=drug_node_h_dims,
            edge_in_dim=drug_edge_in_dim, edge_h_dim=drug_edge_h_dims,
        )
        drug_emb_dim = drug_node_h_dims[0]

        self.prot_model = Prot3DGraphModel(
            d_pretrained_emb=prot_emb_dim, d_gcn=prot_gcn_dims
        )
        prot_emb_dim = prot_gcn_dims[-1]
        if self.use_pocket_atom:
            self.pocket_atom_model = PocketAtomGVPModel(
                node_in_dim=pocket_atom_node_in_dim,
                node_h_dim=pocket_atom_node_h_dims,
                edge_in_dim=pocket_atom_edge_in_dim,
                edge_h_dim=pocket_atom_edge_h_dims,
            )
            pocket_atom_emb_dim = pocket_atom_node_h_dims[0]
            self.pocket_atom_fc = self.get_fc_layers(
                [pocket_atom_emb_dim] + pocket_atom_fc_dims,
                dropout=mlp_dropout, batchnorm=False,
                no_last_dropout=True, no_last_activation=True)
        else:
            pocket_atom_fc_dims = []

        if self.use_one_d_two_d:
            self.one_d_two_d_model = OneDTwoDBranch(
                prot_emb_dim=raw_prot_emb_dim,
                hidden_dim=one_d_two_d_hidden_dim,
                output_dim=one_d_two_d_output_dim,
                gnn_layers=one_d_two_d_gnn_layers,
                triangle_layers=one_d_two_d_triangle_layers,
                dropout=one_d_two_d_dropout,
                use_protein_2d=one_d_two_d_use_protein_2d,
            )
            self.one_d_two_d_fc = self.get_fc_layers(
                [one_d_two_d_output_dim] + one_d_two_d_fc_dims,
                dropout=mlp_dropout, batchnorm=False,
                no_last_dropout=True, no_last_activation=True)
        else:
            one_d_two_d_fc_dims = []

        if self.use_interface_pairformer:
            self.interface_pairformer = PocketLigandInterfacePairformer(
                protein_esm_dim=raw_prot_emb_dim,
                protein_scalar_dim=6,
                ligand_scalar_dim=drug_node_in_dim[0],
                single_dim=interface_pairformer_single_dim,
                pair_dim=interface_pairformer_pair_dim,
                output_dim=interface_pairformer_output_dim,
                num_blocks=interface_pairformer_blocks,
                num_heads=interface_pairformer_heads,
                num_rbf=interface_pairformer_num_rbf,
                fine_num_rbf=interface_pairformer_fine_num_rbf,
                max_pocket_residues=interface_pairformer_max_pocket_residues,
                max_ligand_atoms=interface_pairformer_max_ligand_atoms,
                max_residue_atoms=interface_pairformer_max_residue_atoms,
                pocket_radius=interface_pairformer_pocket_radius,
                min_pocket_residues=interface_pairformer_min_pocket_residues,
                distance_max=interface_pairformer_distance_max,
                fine_distance_max=interface_pairformer_fine_distance_max,
                dropout=interface_pairformer_dropout,
            )
            self.interface_pairformer_output_dim = interface_pairformer_output_dim
        else:
            self.interface_pairformer_output_dim = 0

        self.drug_fc = self.get_fc_layers(
            [drug_emb_dim] + drug_fc_dims,
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)
       
        self.prot_fc = self.get_fc_layers(
            [prot_emb_dim] + prot_fc_dims,
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)

        top_in_dim = drug_fc_dims[-1] + prot_fc_dims[-1]
        if self.use_pocket_atom:
            top_in_dim += pocket_atom_fc_dims[-1]
        if self.use_one_d_two_d:
            top_in_dim += one_d_two_d_fc_dims[-1]
        top_out_dim = 2 if self.predict_logvar else 1
        if self.use_interface_pairformer:
            if len(mlp_dims) == 0:
                global_hidden_dim = top_in_dim
                self.global_trunk = nn.Identity()
            else:
                global_hidden_dim = mlp_dims[-1]
                self.global_trunk = self.get_fc_layers(
                    [top_in_dim] + mlp_dims,
                    dropout=mlp_dropout,
                    batchnorm=False,
                    no_last_dropout=False,
                    no_last_activation=False,
                )
            gate_hidden = int(interface_pairformer_gate_hidden_dim)
            local_hidden = max(gate_hidden, interface_pairformer_output_dim)
            self.global_mu_head = nn.Linear(global_hidden_dim, 1)
            self.local_delta_mu_head = nn.Sequential(
                nn.LayerNorm(interface_pairformer_output_dim),
                nn.Linear(interface_pairformer_output_dim, local_hidden),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(local_hidden, 1),
            )
            self.local_gate = nn.Sequential(
                nn.LayerNorm(global_hidden_dim + interface_pairformer_output_dim),
                nn.Linear(global_hidden_dim + interface_pairformer_output_dim, gate_hidden),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(gate_hidden, 1),
            )
            if self.predict_logvar:
                self.logvar_head = nn.Sequential(
                    nn.LayerNorm(global_hidden_dim + interface_pairformer_output_dim + 1),
                    nn.Linear(
                        global_hidden_dim + interface_pairformer_output_dim + 1,
                        gate_hidden,
                    ),
                    nn.GELU(),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(gate_hidden, 1),
                )
            self.last_interface_gate = None
        else:
            self.top_fc = self.get_fc_layers(
                [top_in_dim] + mlp_dims + [top_out_dim],
                dropout=mlp_dropout, batchnorm=False,
                no_last_dropout=True, no_last_activation=True)

    def get_fc_layers(self, hidden_sizes,
            dropout=0, batchnorm=False,
            no_last_dropout=True, no_last_activation=True):
        act_fn = torch.nn.LeakyReLU()
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if not no_last_activation or i != len(hidden_sizes) - 2:
                layers.append(act_fn)
            if dropout > 0:
                if not no_last_dropout or i != len(hidden_sizes) - 2:
                    layers.append(nn.Dropout(dropout))
            if batchnorm and i != len(hidden_sizes) - 2:
                layers.append(nn.BatchNorm1d(out_dim))
        return nn.Sequential(*layers)

    def forward(self, xd, xp, xpa=None, x12d=None, keep_drug=None, keep_prot=None):
        local_interface = None
        if self.use_interface_pairformer:
            # The local branch consumes the original residue/atom coordinates
            # before the independent KDBNet encoders pool them away.
            local_interface = self.interface_pairformer(xp, xd)

        xd = self.drug_model(xd)
        xp = self.prot_model(xp)

        xd = self.drug_fc(xd)
        xp = self.prot_fc(xp)

        # keep_drug / keep_prot are (B, 1) in {0., 1.}. Gating happens AFTER the
        # branch FCs so the branch contributes an exact zero embedding.
        # The protein is represented by BOTH xp and the pocket-atom xpa, so a
        # single keep_prot mask gates them together.
        if keep_drug is not None:
            xd = xd * keep_drug
        if keep_prot is not None:
            xp = xp * keep_prot

        parts = [xd, xp]
        if self.use_pocket_atom:
            if xpa is None:
                raise ValueError("Pocket atom branch is enabled but xpa is None")
            xpa = self.pocket_atom_model(xpa)
            xpa = self.pocket_atom_fc(xpa)
            if keep_prot is not None:
                xpa = xpa * keep_prot
            parts.append(xpa)

        if self.use_one_d_two_d:
            if x12d is None:
                raise ValueError("1D2D branch is enabled but x12d is None")
            x12d = self.one_d_two_d_model(x12d)
            x12d = self.one_d_two_d_fc(x12d)
            if keep_drug is not None and keep_prot is not None:
                x12d = x12d * keep_drug * keep_prot
            elif keep_drug is not None:
                x12d = x12d * keep_drug
            elif keep_prot is not None:
                x12d = x12d * keep_prot
            parts.append(x12d)

        x = torch.cat(parts, dim=1)
        if not self.use_interface_pairformer:
            return self.top_fc(x)

        global_hidden = self.global_trunk(x)
        if keep_drug is not None or keep_prot is not None:
            local_keep = None
            if keep_drug is not None and keep_prot is not None:
                local_keep = keep_drug * keep_prot
            elif keep_drug is not None:
                local_keep = keep_drug
            else:
                local_keep = keep_prot
            local_interface = local_interface * local_keep
        else:
            local_keep = None

        gate_input = torch.cat([global_hidden, local_interface], dim=-1)
        gate = torch.sigmoid(self.local_gate(gate_input))
        if local_keep is not None:
            gate = gate * local_keep
        mu_global = self.global_mu_head(global_hidden)
        delta_mu_local = self.local_delta_mu_head(local_interface)
        mu = mu_global + gate * delta_mu_local
        self.last_interface_gate = gate.detach()

        if not self.predict_logvar:
            return mu

        logvar = self.logvar_head(
            torch.cat([global_hidden, local_interface, gate], dim=-1)
        )
        return torch.cat([mu, logvar], dim=-1)
