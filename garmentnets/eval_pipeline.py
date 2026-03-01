# %%
# import
import argparse
import os
import pathlib
import pdb
import pickle

import numpy as np
import torch
import wandb
import yaml
import hydra
from joblib import Parallel
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import pytorch_lightning.utilities.seed as seed_utils
from pytorch_lightning.loggers import WandbLogger

from garmentnets.datasets.conv_implicit_wnf_dataset import ConvImplicitWNFDataModule
from garmentnets.networks.conv_implicit_wnf import ConvImplicitWNFPipeline
from garmentnets.networks.hrnet_nocs import HRNet2NOCS
from garmentnets.networks.pointnet2_nocs import PointNet2NOCS
from utils.data_utils import update_config, find_best_checkpoint

from torch_geometric.data import Dataset, Data, DataLoader

from visualization.plot_utils import save_numpy_as_gif


def get_default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='release_new_check',
                        help='Name of the experiment')
    parser.add_argument('--log_dir', type=str, default='data/test/',
                        help='Logging directory')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument('--max_test_num', type=int, default=1, help='max number of test examples')
    parser.add_argument('--cloth_type', type=str, default='Trousers', help='cloth type')
    parser.add_argument("--mc_thres", type=float, default=0.05, help='threshold for marching cubes')
    parser.add_argument('--model_path', type=str, default='gn_test_trousers_pipe/trousers_pipe')
    parser.add_argument('--tt_finetune', default=False, action="store_true")
    parser.add_argument('--make_opt_gif', default=False, action="store_true")
    parser.add_argument("--opt_mesh_density", type=str, default='dense',
                        help="Mesh density for optimization")
    parser.add_argument("--opt_mesh_init", type=str, default='task',
                        help="Mesh initialization for optimization")
    parser.add_argument("--opt_lr", type=float, default=1e-3)
    parser.add_argument("--opt_iter_total", type=int, default=100,
                        help="Number of iterations for optimization")
    parser.add_argument("--chamfer3d_w", type=float, default=1.,
                        help="Weight for 3D chamfer loss")
    parser.add_argument("--laplacian_w", type=float, default=0.01,
                        help="Weight for laplacian loss")
    parser.add_argument("--normal_w", type=float, default=0.,
                        help="Weight for normal loss")
    parser.add_argument("--edge_w", type=float, default=0.02,
                        help="Weight for edge length loss")
    parser.add_argument("--depth_w", type=float, default=0.,
                        help="Weight for depth loss through differential rendering")
    parser.add_argument("--silhouette_w", type=float, default=0.,
                        help="Weight for silhouette loss through differential rendering")
    parser.add_argument("--obs_consist_w", type=float, default=10.,
                        help="Weight for observation consistency loss")
    parser.add_argument("--consist_iter", type=int, default=50,
                        help="Number of iterations for observation consistency loss")
    parser.add_argument("--table_w", type=float, default=10.,
                        help="Weight for table loss (prevent penetration)")
    parser.add_argument('--use_wandb', type=bool, default=False, help='Use weight and bias for logging')
    parser.add_argument('--collapse_shell', default=False, action="store_true",
                        help='Collapse MC shell mesh to thin surface (for thin cloth)')
    parser.add_argument('--traj_first', default=False, action="store_true",
                        help='Evaluate only the first frame of each trajectory (stride=85)')
    parser.add_argument('--frames_per_traj', type=int, default=85,
                        help='Number of frames per trajectory (used with --traj_first)')

    args = parser.parse_args()
    return args.__dict__


def run_task(args):
    cfg = OmegaConf.load(args['model_path'] + '/config.yaml')
    cfg = update_config(cfg, args)
    cfg.log_dir = os.path.join(cfg.log_dir, cfg.exp_name)
    cfg.finetune = True
    cfg.conv_implicit_model.vis_per_items = 1
    cfg.conv_implicit_model.max_vis_per_epoch_val = 40
    pred_cfg = OmegaConf.load('garmentnets/config/predict_default.yaml')
    cfg['prediction'] = pred_cfg.prediction
    seed_utils.seed_everything(1234)
    finetune_cfg = {'opt_mesh_density': cfg['opt_mesh_density'],
                    'opt_mesh_init': cfg['opt_mesh_init'],
                    'opt_iter_total': cfg['opt_iter_total'],
                    'chamfer_mode': 'scipy', 'chamfer3d_w': cfg['chamfer3d_w'],
                    'laplacian_w': cfg['laplacian_w'], 'normal_w': cfg['normal_w'],
                    'edge_w': cfg['edge_w'], 'rest_edge_len': 0.,
                    'depth_w': cfg['depth_w'], 'silhouette_w': cfg['silhouette_w'],
                    'obs_consist_w': cfg['obs_consist_w'], 'consist_iter': cfg['consist_iter'],
                    'table_w': cfg['table_w'],
                    'lr': cfg['opt_lr'],
                    }
    finetune_cfg = OmegaConf.create(finetune_cfg)
    cfg['finetune_cfg'] = finetune_cfg

    cfg.datamodule.batch_size = 1
    datamodule = ConvImplicitWNFDataModule(cfg, **cfg.datamodule)

    if cfg.use_wandb:
        wandb.init(project='occluded cloth',
                   name=cfg.exp_name,
                   resume='allow',
                   )

    datamodule.prepare_data(test_only=True)
    batch_size = datamodule.kwargs['batch_size']
    joblib_parallel = Parallel(n_jobs=10, verbose=1)
    if cfg.input_type == 'pc':
        pointnet2_model = PointNet2NOCS.load_from_checkpoint(
            find_best_checkpoint(cfg.canon_checkpoint))
    else:
        pointnet2_model = HRNet2NOCS.load_from_checkpoint(
            find_best_checkpoint(cfg.canon_checkpoint))

    pointnet2_params = dict(pointnet2_model.hparams)
    pipeline_model = ConvImplicitWNFPipeline(
        cfg,
        pointnet2_params=pointnet2_params,
        batch_size=batch_size, **cfg.conv_implicit_model,
        cloth_nocs_aabb=datamodule.cloth_nocs_aabb,
    )
    pipeline_model.pointnet2_nocs = pointnet2_model
    pipeline_model.batch_size = batch_size

    model_path = find_best_checkpoint(cfg['model_path'])
    state_dict = torch.load(model_path)['state_dict']
    pipeline_model.load_state_dict(state_dict, strict=False)
    pipeline_model = pipeline_model.cuda()
    pipeline_model.eval()

    data_loader = datamodule.test_dataloader()
    os.makedirs(cfg.log_dir, exist_ok=True)

    print('Current working directory is ', os.getcwd())
    torch.manual_seed(1234)
    all_metrics = []
    # Build list of sample indices to evaluate
    if cfg.get('traj_first', False):
        frames_per_traj = cfg.get('frames_per_traj', 85)
        total_samples = len(data_loader.dataset)
        eval_indices = list(range(0, total_samples, frames_per_traj))
        if cfg.max_test_num < len(eval_indices):
            eval_indices = eval_indices[:cfg.max_test_num + 1]
        print(f'[traj_first] Evaluating {len(eval_indices)} trajectories '
              f'(first frame of each, stride={frames_per_traj})')
    else:
        eval_indices = None

    eval_count = 0
    for batch_idx, batch in enumerate(data_loader):
        # Skip samples not in eval_indices when traj_first is enabled
        if eval_indices is not None and batch_idx not in eval_indices:
            continue
        batch = batch.to('cuda')
        try:
            results = pipeline_model.predict_mesh(batch,
                                                  finetune_cfg=finetune_cfg,
                                                  make_gif=cfg.make_opt_gif,
                                                  parallel=joblib_parallel,
                                                  )[0]
            metrics = pipeline_model.eval_metrics(batch, results, self_log=False)
            if '3d_plot' in metrics and metrics['3d_plot'] is not None:
                metrics['3d_plot'].write_html(f'{cfg.log_dir}/{eval_count}_vis.html')
            if cfg.make_opt_gif:
                save_numpy_as_gif(np.array(results['optimization_gif']),
                                  f'{cfg.log_dir}/opt_{eval_count}.gif',
                                  fps=10, add_index_rate=-1)
            print(f'[{eval_count}] (sample {batch_idx}) chamfer_mesh={metrics.get("test/chamfer_mesh", "N/A")}, '
                  f'chamfer_pc={metrics.get("test/chamfer_pc", "N/A")}')
            all_metrics.append(metrics)
        except Exception as e:
            print(f'[{eval_count}] (sample {batch_idx}) ERROR: {e}')
        eval_count += 1
        if eval_indices is None and batch_idx > cfg.max_test_num:
            break
    # Print summary
    if all_metrics:
        for key in ['test/chamfer_mesh', 'test/chamfer_pc', 'test/chamfer_mesh_opt', 'test/chamfer_pc_opt']:
            vals = [float(m[key]) for m in all_metrics if key in m and m[key] is not None]
            if vals:
                print(f'{key}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}')


def main():
    args = get_default_args()
    run_task(args)


if __name__ == '__main__':
    main()
