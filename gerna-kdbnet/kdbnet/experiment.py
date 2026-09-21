import copy
import random
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
try:
    import uncertainty_toolbox as uct
except ImportError:
    uct = None

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric
torch.set_num_threads(1)

from kdbnet.dta import KIBA, DAVIS, PDBbindPL
from kdbnet.pdbbind_lmdb import PDBbindLmdb
from kdbnet.pocket_atom_graph import pocket_atom_node_dim
from kdbnet.model import DTAModel
from kdbnet.molecular_dropout import MolecularDropout
from kdbnet.losses import HeteroscedasticCriterion
from kdbnet.utils import(
    Logger,
    Saver,
    EarlyStopping
)
from kdbnet.metrics import evaluation_metrics

REGRESSION_METRICS = ['mse', 'spearman', 'pearson', 'rmse', 'mae', 'ci']
LOWER_BETTER_METRICS = {'mse', 'rmse', 'mae'}


def _seed_everything(seed):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _parallel_train_per_epoch(
    kwargs=None, test_loader=None,
    n_epochs=None, eval_freq=None, test_freq=None,
    monitoring_score='pearson',
    loss_fn=None, logger=None,
    test_after_train=True,
    molecular_dropout=None,
):
    midx = kwargs['midx']
    model = kwargs['model']
    optimizer = kwargs['optimizer']
    train_loader = kwargs['train_loader']
    valid_loader = kwargs['valid_loader']
    device = kwargs['device']
    stopper = kwargs['stopper']
    best_model_state_dict = kwargs['best_model_state_dict']
    if stopper.early_stop:
        return kwargs

    for epoch in range(1, n_epochs + 1):
        model.train()
        if hasattr(loss_fn, 'current_epoch'):
            loss_fn.current_epoch = epoch
        total_loss = 0
        for step, batch in enumerate(train_loader, start=1):
            xd = batch['drug'].to(device)
            xp = batch['protein'].to(device)
            xpa = batch['pocket_atom'].to(device) if 'pocket_atom' in batch else None
            x12d = batch['one_d_two_d'].to(device) if 'one_d_two_d' in batch else None
            y = batch['y'].to(device).view(-1, 1)
            optimizer.zero_grad()
            if molecular_dropout is not None:
                keep_drug, keep_prot, y = molecular_dropout.sample(y.size(0), y)
                yh = model(xd, xp, xpa, x12d, keep_drug=keep_drug, keep_prot=keep_prot)
            else:
                yh = model(xd, xp, xpa, x12d)
            loss = loss_fn(yh, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite training loss ({}) at M-{} epoch {} step {}. "
                    "If running heteroscedastic NLL with clamp disabled, the "
                    "log-variance likely diverged; re-enable a clamp bound "
                    "(nll_logvar_max) or lengthen nll_warmup_epochs.".format(
                        loss.item(), midx, epoch, step))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / step
        if epoch % eval_freq == 0:
            val_results = _parallel_test(
                {'model': model, 'midx': midx, 'test_loader': valid_loader, 'device': device},
                loss_fn=loss_fn, logger=logger
            )
            is_best = stopper.update(val_results['metrics'][monitoring_score])
            if is_best:
                best_model_state_dict = copy.deepcopy(model.state_dict())
            logger.info(f"M-{midx} E-{epoch} | Train Loss: {train_loss:.4f} | Valid Loss: {val_results['loss']:.4f} | "\
                + ' | '.join([f'{k}: {v:.4f}' for k, v in val_results['metrics'].items()])
                + f" | best {monitoring_score}: {stopper.best_score:.4f}"
                )
        if test_freq is not None and epoch % test_freq == 0:
            test_results = _parallel_test(
                {'midx': midx, 'model': model, 'test_loader': test_loader, 'device': device},
                loss_fn=loss_fn, logger=logger
            )
            logger.info(f"M-{midx} E-{epoch} | Test Loss: {test_results['loss']:.4f} | "\
                + ' | '.join([f'{k}: {v:.4f}' for k, v in test_results['metrics'].items()])
                )

        if stopper.early_stop:
            logger.info('Early stop at epoch {}'.format(epoch))
            break

    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
    if test_after_train:
        test_results = _parallel_test(
            {'midx': midx, 'model': model, 'test_loader': test_loader, 'device': device},
            loss_fn=loss_fn,
            test_tag=f"Model {midx}", print_log=True, logger=logger
        )
    rets = dict(midx = midx, model = model)
    return rets


def _parallel_test(
    kwargs=None, loss_fn=None, 
    test_tag=None, print_log=False, logger=None,
):
    midx = kwargs['midx']
    model = kwargs['model']
    test_loader = kwargs['test_loader']
    device = kwargs['device']
    model.eval()
    # Use the same log-variance clamp bounds the criterion was configured with,
    # so eval-time aleatoric variance matches the trained model's configuration.
    logvar_min = getattr(loss_fn, 'logvar_min', -7.0)
    logvar_max = getattr(loss_fn, 'logvar_max', 7.0)
    yt, yp, yv, yg, total_loss = (
        torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), 0
    )
    has_aleatoric = False
    has_interface_gate = False
    with torch.no_grad():
        for step, batch in enumerate(test_loader, start=1):
            xd = batch['drug'].to(device)
            xp = batch['protein'].to(device)
            xpa = batch['pocket_atom'].to(device) if 'pocket_atom' in batch else None
            x12d = batch['one_d_two_d'].to(device) if 'one_d_two_d' in batch else None
            y = batch['y'].to(device)
            yh = model(xd, xp, xpa, x12d)
            interface_gate = getattr(model, 'last_interface_gate', None)
            if interface_gate is not None:
                has_interface_gate = True
                yg = torch.cat([yg, interface_gate.detach().cpu()], dim=0)
            loss = loss_fn(yh, y.view(-1, 1))
            total_loss += loss.item()
            if yh.dim() == 2 and yh.size(1) == 2:
                # heteroscedastic head: column 0 = mu, column 1 = logvar
                has_aleatoric = True
                mu = yh[:, 0:1]
                logvar = yh[:, 1:2]
                if logvar_min is not None or logvar_max is not None:
                    logvar = torch.clamp(logvar, min=logvar_min, max=logvar_max)
                aleatoric_var = torch.exp(logvar)
                yp = torch.cat([yp, mu.detach().cpu()], dim=0)
                yv = torch.cat([yv, aleatoric_var.detach().cpu()], dim=0)
            else:
                yp = torch.cat([yp, yh.detach().cpu()], dim=0)
            yt = torch.cat([yt, y.detach().cpu()], dim=0)
    yt = yt.numpy()
    yp = yp.view(-1).numpy()
    results = {
        'midx': midx,
        'y_true': yt,
        'y_pred': yp,
        'loss': total_loss / step,
    }
    if has_aleatoric:
        results['y_var_aleatoric'] = yv.view(-1).numpy()
    if has_interface_gate:
        results['interface_gate'] = yg.view(-1).numpy()
    eval_metrics = evaluation_metrics(
        yt, yp,
        eval_metrics=REGRESSION_METRICS
    )
    results['metrics'] = eval_metrics
    if print_log:
        logger.info(f"{test_tag} | Test Loss: {results['loss']:.4f} | "\
            + ' | '.join([f'{k}: {v:.4f}' for k, v in results['metrics'].items()]))
    return results


def _unpack_evidential_output(output):
    mu, v, alpha, beta = torch.split(output, output.shape[1]//4, dim=1)
    inverse_evidence = 1. / ((alpha - 1) * v)
    var = beta * inverse_evidence
    return mu, var, inverse_evidence


class DTAExperiment(object):
    def __init__(self,
        task=None,
        split_method='protein',
        split_frac=[0.7, 0.1, 0.2],
        prot_gcn_dims=[128, 128, 128], prot_gcn_bn=False,
        prot_fc_dims=[1024, 128],
        drug_in_dim=66, drug_fc_dims=[1024, 128], drug_gcn_dims=[128, 64],
        use_pocket_atom=False, pocket_atom_cutoff=5.0,
        pocket_atom_edge_cutoff=4.5, pocket_atom_include_hydrogen=False,
        pocket_atom_include_hetatm=False, pocket_atom_node_in_dim=None,
        pocket_atom_gcn_dims=[128, 32], pocket_atom_fc_dims=[1024, 128],
        use_one_d_two_d=False,
        one_d_two_d_protein_edge_cutoff=8.0,
        one_d_two_d_max_protein_residues=128,
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
        num_pos_emb=16, num_rbf=16,
        contact_cutoff=8.,
        n_ensembles=1, n_epochs=500, batch_size=256,
        lr=0.001,        
        seed=42, onthefly=False,
        uncertainty=False, parallel=False,
        output_dir='../output', save_log=False, save_checkpoint=False,
        pdbbind_manifest=None, pdbbind_dir='../data/PDBbind',
        split_file=None, limit=None,
        pdbbind_lmdb_dir=None, pdbbind_lmdb_index=None,
        pdbbind_lmdb_structure_root='data/PDBbind_full/extracted',
        pdbbind_lmdb_esm_key='esm2_embedding',
        pdbbind_lmdb_key_suffix='_protein',
        pdbbind_lmdb_target_field='affinity',
        pdbbind_lmdb_ligand_source='sdf',
        pdbbind_lmdb_allow_zero_embedding=False,
        pdbbind_lmdb_cache_graphs=False,
        pdbbind_lmdb_split_missing_policy='error',
        prot_emb_dim=1280,
        use_molecular_dropout=False,
        molecular_dropout_p=0.06,
        molecular_dropout_anchor=0.0,
        molecular_dropout_mode='complex',
        heteroscedastic=False,
        nll_warmup_epochs=5,
        nll_logvar_min=None,
        nll_logvar_max=None,
        **unused_kwargs,
    ):
        self.saver = Saver(output_dir)
        self.logger = Logger(logfile=self.saver.save_dir/'exp.log' if save_log else None)
        if unused_kwargs:
            raise TypeError(
                "DTAExperiment received unknown keyword argument(s): {}".format(
                    sorted(unused_kwargs)
                )
            )
        self.save_checkpoint = save_checkpoint

        self.uncertainty = uncertainty
        self.parallel = parallel
        self.n_ensembles = n_ensembles
        if self.uncertainty and self.n_ensembles < 2:
            raise ValueError('n_ensembles must be greater than 1 when uncertainty is True')            
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = int(seed) if seed is not None else None
        _seed_everything(self.seed)
        self.logger.info("Random seed set to {}".format(self.seed))
        if pocket_atom_node_in_dim is None:
            pocket_atom_node_in_dim = pocket_atom_node_dim(num_rbf=num_rbf)
        task = task.lower() if isinstance(task, str) else task
        dataset_klass = {
            'kiba': KIBA,
            'davis': DAVIS,
            'pdbbind_pl': PDBbindPL,
            'pdbbind_lmdb': PDBbindLmdb,
        }[task]
        dataset_kwargs = dict(
            split_method=split_method,
            split_frac=split_frac,
            seed=self.seed,
            onthefly=onthefly,
            num_pos_emb=num_pos_emb,
            num_rbf=num_rbf,
            contact_cutoff=contact_cutoff,
        )
        if task == 'pdbbind_pl':
            dataset_kwargs.update(dict(
                manifest_path=pdbbind_manifest,
                pdbbind_dir=pdbbind_dir,
                split_file=split_file,
                limit=limit,
            ))
        elif task == 'pdbbind_lmdb':
            dataset_kwargs.update(dict(
                lmdb_dir=pdbbind_lmdb_dir,
                index_path=pdbbind_lmdb_index,
                structure_root=pdbbind_lmdb_structure_root,
                split_file=split_file,
                key_suffix=pdbbind_lmdb_key_suffix,
                target_field=pdbbind_lmdb_target_field,
                esm_key=pdbbind_lmdb_esm_key,
                prot_emb_dim=prot_emb_dim,
                allow_zero_embedding=pdbbind_lmdb_allow_zero_embedding,
                ligand_source=pdbbind_lmdb_ligand_source,
                use_pocket_atom=use_pocket_atom,
                pocket_atom_cutoff=pocket_atom_cutoff,
                pocket_atom_edge_cutoff=pocket_atom_edge_cutoff,
                pocket_atom_include_hydrogen=pocket_atom_include_hydrogen,
                pocket_atom_include_hetatm=pocket_atom_include_hetatm,
                use_one_d_two_d=use_one_d_two_d,
                one_d_two_d_protein_edge_cutoff=one_d_two_d_protein_edge_cutoff,
                one_d_two_d_max_protein_residues=one_d_two_d_max_protein_residues,
                use_interface_pairformer=use_interface_pairformer,
                interface_pairformer_pocket_radius=interface_pairformer_pocket_radius,
                interface_pairformer_min_pocket_residues=interface_pairformer_min_pocket_residues,
                interface_pairformer_max_pocket_residues=interface_pairformer_max_pocket_residues,
                limit=limit,
                cache_graphs=pdbbind_lmdb_cache_graphs,
                split_missing_policy=pdbbind_lmdb_split_missing_policy,
            ))

        self.dataset = dataset_klass(**dataset_kwargs)
        if hasattr(self.dataset, 'split_report') and self.dataset.split_report:
            self.logger.info("Split merge report: {}".format(self.dataset.split_report))
        self._task_data_df_split = None
        self._task_loader = None

        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            if self.parallel and self.n_ensembles > 1:
                self.logger.warning("No visible CUDA devices; ensemble models will run on CPU")
            self.devices = [torch.device('cpu') for _ in range(self.n_ensembles)]
        else:
            if self.parallel and n_gpus < self.n_ensembles:
                self.logger.warning(f"Visible GPUs ({n_gpus}) is fewer than "
                f"number of models ({self.n_ensembles}). Some models will be run on the same GPU"
                )
            self.devices = [torch.device(f'cuda:{i % n_gpus}')
                for i in range(self.n_ensembles)]
        self.model_config = dict(
            prot_emb_dim=prot_emb_dim,
            prot_gcn_dims=prot_gcn_dims,            
            prot_fc_dims=prot_fc_dims,
            drug_node_in_dim=[drug_in_dim, 1],
            drug_node_h_dims=drug_gcn_dims,
            drug_fc_dims=drug_fc_dims,            
            use_pocket_atom=use_pocket_atom,
            pocket_atom_node_in_dim=[pocket_atom_node_in_dim, 1],
            pocket_atom_node_h_dims=pocket_atom_gcn_dims,
            pocket_atom_edge_in_dim=[num_rbf, 1],
            pocket_atom_fc_dims=pocket_atom_fc_dims,
            use_one_d_two_d=use_one_d_two_d,
            one_d_two_d_hidden_dim=one_d_two_d_hidden_dim,
            one_d_two_d_output_dim=one_d_two_d_output_dim,
            one_d_two_d_gnn_layers=one_d_two_d_gnn_layers,
            one_d_two_d_triangle_layers=one_d_two_d_triangle_layers,
            one_d_two_d_dropout=one_d_two_d_dropout,
            one_d_two_d_use_protein_2d=one_d_two_d_use_protein_2d,
            one_d_two_d_fc_dims=one_d_two_d_fc_dims,
            use_interface_pairformer=use_interface_pairformer,
            interface_pairformer_single_dim=interface_pairformer_single_dim,
            interface_pairformer_pair_dim=interface_pairformer_pair_dim,
            interface_pairformer_output_dim=interface_pairformer_output_dim,
            interface_pairformer_blocks=interface_pairformer_blocks,
            interface_pairformer_heads=interface_pairformer_heads,
            interface_pairformer_num_rbf=interface_pairformer_num_rbf,
            interface_pairformer_fine_num_rbf=interface_pairformer_fine_num_rbf,
            interface_pairformer_max_pocket_residues=interface_pairformer_max_pocket_residues,
            interface_pairformer_max_ligand_atoms=interface_pairformer_max_ligand_atoms,
            interface_pairformer_max_residue_atoms=interface_pairformer_max_residue_atoms,
            interface_pairformer_pocket_radius=interface_pairformer_pocket_radius,
            interface_pairformer_min_pocket_residues=interface_pairformer_min_pocket_residues,
            interface_pairformer_distance_max=interface_pairformer_distance_max,
            interface_pairformer_fine_distance_max=interface_pairformer_fine_distance_max,
            interface_pairformer_dropout=interface_pairformer_dropout,
            interface_pairformer_gate_hidden_dim=interface_pairformer_gate_hidden_dim,
            mlp_dims=mlp_dims, mlp_dropout=mlp_dropout,
            predict_logvar=heteroscedastic)
        self.build_model()
        self.heteroscedastic = heteroscedastic
        if heteroscedastic:
            self.criterion = HeteroscedasticCriterion(
                warmup_epochs=nll_warmup_epochs,
                logvar_min=nll_logvar_min,
                logvar_max=nll_logvar_max,
            )
            self.logger.info(
                "Heteroscedastic NLL head enabled: warmup_epochs={}, "
                "logvar_min={}, logvar_max={}".format(
                    nll_warmup_epochs, nll_logvar_min, nll_logvar_max))
        else:
            self.criterion = F.mse_loss
        self.molecular_dropout = (
            MolecularDropout(
                p=molecular_dropout_p,
                no_binding_value=molecular_dropout_anchor,
                mode=molecular_dropout_mode,
            ) if use_molecular_dropout else None
        )
        if self.molecular_dropout is not None:
            self.logger.info(
                "Molecular dropout enabled: p={}, anchor={}, mode={}".format(
                    molecular_dropout_p, molecular_dropout_anchor,
                    molecular_dropout_mode))
        if use_one_d_two_d:
            self.logger.info(
                "1D2D branch enabled: protein_edge_cutoff={}, max_protein_residues={}, "
                "hidden_dim={}, output_dim={}, gnn_layers={}, triangle_layers={}, dropout={}, "
                "use_protein_2d={}".format(
                    one_d_two_d_protein_edge_cutoff,
                    one_d_two_d_max_protein_residues,
                    one_d_two_d_hidden_dim,
                    one_d_two_d_output_dim,
                    one_d_two_d_gnn_layers,
                    one_d_two_d_triangle_layers,
                    one_d_two_d_dropout,
                    one_d_two_d_use_protein_2d,
                )
            )
        if use_interface_pairformer:
            self.logger.info(
                "Pocket-ligand interface Pairformer enabled: single_dim={}, "
                "pair_dim={}, output_dim={}, blocks={}, heads={}, max_pocket={}, "
                "max_ligand={}, max_residue_atoms={}, pocket_radius={}, min_pocket={}, "
                "num_rbf={}, fine_num_rbf={}, distance_max={}, fine_distance_max={}, "
                "dropout={}, gate_hidden={}".format(
                    interface_pairformer_single_dim,
                    interface_pairformer_pair_dim,
                    interface_pairformer_output_dim,
                    interface_pairformer_blocks,
                    interface_pairformer_heads,
                    interface_pairformer_max_pocket_residues,
                    interface_pairformer_max_ligand_atoms,
                    interface_pairformer_max_residue_atoms,
                    interface_pairformer_pocket_radius,
                    interface_pairformer_min_pocket_residues,
                    interface_pairformer_num_rbf,
                    interface_pairformer_fine_num_rbf,
                    interface_pairformer_distance_max,
                    interface_pairformer_fine_distance_max,
                    interface_pairformer_dropout,
                    interface_pairformer_gate_hidden_dim,
                )
            )

        self.split_method = split_method
        self.split_frac = split_frac

        self.logger.info(self.models[0])
        self.logger.info(self.optimizers[0])

    def build_model(self):
        _seed_everything(self.seed)
        self.models = [DTAModel(**self.model_config).to(self.devices[i])
                        for i in range(self.n_ensembles)]
        self.optimizers = [optim.Adam(model.parameters(), lr=self.lr) for model in self.models]

    def _get_data_loader(self, dataset, shuffle=False):
        generator = None
        if shuffle and self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
        return torch_geometric.loader.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    pin_memory=False,
                    num_workers=0,
                    generator=generator,
                )

    @property
    def task_data_df_split(self):
        if self._task_data_df_split is None:
            (data, df) = self.dataset.get_split(return_df=True)
            self._task_data_df_split = (data, df)
        return self._task_data_df_split

    @property
    def task_data(self):
        return self.task_data_df_split[0]

    @property
    def task_df(self):
        return self.task_data_df_split[1]

    @property
    def task_loader(self):
        if self._task_loader is None:
            _loader = {
                s: self._get_data_loader(
                    self.task_data[s], shuffle=(s == 'train'))
                for s in self.task_data
            }
            self._task_loader = _loader
        return self._task_loader

    def recalibrate_std(self, df, recalib_df):
        if uct is None:
            raise ImportError('uncertainty_toolbox is required for uncertainty recalibration')
        y_mean = recalib_df['y_pred'].values
        y_std = recalib_df['y_std'].values
        y_true = recalib_df['y_true'].values
        std_ratio = uct.recalibration.optimize_recalibration_ratio(
            y_mean, y_std, y_true, criterion="miscal")
        df['y_std_recalib'] = df['y_std'] * std_ratio
        return df

    def _format_predict_df(self, results,
            test_df=None, esb_yp=None, recalib_df=None,
            var_epistemic=None, var_aleatoric=None, var_total=None,
            interface_gate=None):
        """
        results: dict with keys y_pred, y_true, y_var
        """
        df = self.task_df['test'].copy() if test_df is None else test_df.copy()
        assert np.allclose(results['y_true'], df['y'].values)
        df = df.rename(columns={'y': 'y_true'})
        df['y_pred'] = results['y_pred']
        if interface_gate is not None:
            df['interface_gate'] = interface_gate
        if esb_yp is not None:
            has_aleatoric = var_aleatoric is not None
            if has_aleatoric:
                df['y_std_epistemic'] = np.sqrt(var_epistemic)
                df['y_std_aleatoric'] = np.sqrt(var_aleatoric)
                df['y_std_total'] = np.sqrt(var_total)
                # keep the legacy column name usable; recalibration still
                # operates on y_std.
                df['y_std'] = df['y_std_total']
                if recalib_df is not None:
                    df = self.recalibrate_std(df, recalib_df)
            elif self.uncertainty:
                df['y_std'] = np.std(esb_yp, axis=0)
                if recalib_df is not None:
                    df = self.recalibrate_std(df, recalib_df)
            for i in range(self.n_ensembles):
                df[f'y_pred_{i + 1}'] = esb_yp[i]
        return df

    def train(self, n_epochs=None, patience=None,
                eval_freq=1, test_freq=None,
                monitoring_score='pearson',
                train_data=None, valid_data=None,                
                rebuild_model=False,
                test_after_train=False):
        n_epochs = n_epochs or self.n_epochs
        if rebuild_model:
            self.build_model()
        tl, vl = self.task_loader['train'], self.task_loader['valid']
        rets_list = []
        for i in range(self.n_ensembles):
            stp = EarlyStopping(eval_freq=eval_freq, patience=patience,
                                    higher_better=(monitoring_score not in LOWER_BETTER_METRICS))
            rets = dict(
                midx = i + 1,
                model = self.models[i],
                optimizer = self.optimizers[i],
                device = self.devices[i],
                train_loader = tl,
                valid_loader = vl,
                stopper = stp,
                best_model_state_dict = None,
            )
            rets_list.append(rets)

        # Each ensemble member gets its own criterion instance so that the
        # per-epoch `current_epoch` state (used by HeteroscedasticCriterion to
        # switch between MSE warm-up and NLL) cannot race across threads when
        # parallel=True. The MSE baseline keeps the exact same F.mse_loss
        # object reference (deepcopy is only applied to stateful criteria).
        def _member_criterion():
            if isinstance(self.criterion, HeteroscedasticCriterion):
                return copy.deepcopy(self.criterion)
            return self.criterion

        rets_list = Parallel(n_jobs=(self.n_ensembles if self.parallel else 1), prefer="threads")(
            delayed(_parallel_train_per_epoch)(
                kwargs=rets_list[i],
                test_loader=self.task_loader['test'],
                n_epochs=n_epochs, eval_freq=eval_freq, test_freq=test_freq,
                monitoring_score=monitoring_score,
                loss_fn=_member_criterion(), logger=self.logger,
                test_after_train=test_after_train,
                molecular_dropout=self.molecular_dropout,
            ) for i in range(self.n_ensembles))

        for i, rets in enumerate(rets_list):
            self.models[rets['midx'] - 1] = rets['model']
        if self.save_checkpoint:
            for i, model in enumerate(self.models):
                filename = 'best_model_{}.pt'.format(i + 1)
                self.saver.save_ckp(model.state_dict(), filename)
                self.logger.info("Saved checkpoint: {}".format(
                    self.saver.save_dir / filename
                ))


    def test(self, test_model=None, test_loader=None,
                test_data=None, test_df=None,
                recalib_df=None,
                save_prediction=False, save_df_name='prediction.tsv',
                test_tag=None, print_log=False):
        test_models = self.models if test_model is None else [test_model]
        if test_data is not None:
            assert test_df is not None, 'test_df must be provided if test_data used'
            test_loader = self._get_data_loader(test_data)
        elif test_loader is not None:
            assert test_df is not None, 'test_df must be provided if test_loader used'
        else:
            test_loader = self.task_loader['test']
        rets_list = []
        # DTAExperiment keeps the pristine criterion object.  During ensemble
        # training, stateful criteria are copied per member, so this object may
        # still report current_epoch=1.  Evaluation must always use NLL rather
        # than accidentally falling back to the warm-up MSE path.
        eval_criterion = self.criterion
        if isinstance(self.criterion, HeteroscedasticCriterion):
            eval_criterion = copy.deepcopy(self.criterion)
            eval_criterion.current_epoch = eval_criterion.warmup_epochs + 1
        for i, model in enumerate(test_models):
            rets = _parallel_test(
                kwargs={
                    'midx': i + 1,
                    'model': model,
                    'test_loader': test_loader,
                    'device': self.devices[i],
                },
                loss_fn=eval_criterion,
                test_tag=f"Model {i+1}", print_log=True, logger=self.logger
            )
            rets_list.append(rets)


        esb_yp, esb_var, esb_gate, esb_loss = None, None, None, 0
        has_aleatoric = all('y_var_aleatoric' in rets for rets in rets_list)
        has_interface_gate = all('interface_gate' in rets for rets in rets_list)
        for rets in rets_list:
            esb_yp = rets['y_pred'].reshape(1, -1) if esb_yp is None else\
                np.vstack((esb_yp, rets['y_pred'].reshape(1, -1)))
            if has_aleatoric:
                v = rets['y_var_aleatoric'].reshape(1, -1)
                esb_var = v if esb_var is None else np.vstack((esb_var, v))
            if has_interface_gate:
                gate = rets['interface_gate'].reshape(1, -1)
                esb_gate = gate if esb_gate is None else np.vstack((esb_gate, gate))
            esb_loss += rets['loss']

        y_true = rets['y_true']
        y_pred = np.mean(esb_yp, axis=0)
        esb_loss /= len(test_models)
        results = {
            'y_true': y_true,
            'y_pred': y_pred,
            'loss': esb_loss,
        }

        var_epistemic = var_aleatoric = var_total = None
        if has_aleatoric:
            var_epistemic = np.var(esb_yp, axis=0)
            var_aleatoric = np.mean(esb_var, axis=0)
            var_total = var_epistemic + var_aleatoric
            results['y_var_epistemic'] = var_epistemic
            results['y_var_aleatoric'] = var_aleatoric
            results['y_var_total'] = var_total

        eval_metrics = evaluation_metrics(
            y_true, y_pred,
            eval_metrics=REGRESSION_METRICS
        )
        results['metrics'] = eval_metrics
        interface_gate = np.mean(esb_gate, axis=0) if has_interface_gate else None
        if interface_gate is not None:
            results['interface_gate'] = interface_gate
        results['df'] = self._format_predict_df(results,
            test_df=test_df, esb_yp=esb_yp, recalib_df=recalib_df,
            var_epistemic=var_epistemic, var_aleatoric=var_aleatoric,
            var_total=var_total, interface_gate=interface_gate)
        if save_prediction:
            self.saver.save_df(results['df'], save_df_name, float_format='%g')
        if print_log:
            self.logger.info(f"{test_tag} | Test Loss: {results['loss']:.4f} | "\
                + ' | '.join([f'{k}: {v:.4f}' for k, v in results['metrics'].items()]))
        return results
