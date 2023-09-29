import torch
import argparse
import os
import pytorch_lightning as pl
from torch_geometric.datasets import QM9, ModelNet
from md17 import MD17
from torch_geometric.loader import DataLoader
import torch_geometric as tg
import numpy as np
import models.transforms as T
from callbacks import EMA, EpochTimer


from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from itertools import repeat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Run parameters
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--warmup', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-16,
                        help='weight decay')
    parser.add_argument('--print', type=int, default=100,
                        help='print interval')
    parser.add_argument('--log', type=bool, default=False,
                        help='logging flag')
    parser.add_argument('--enable_progress_bar', type=bool, default=False,
                        help='enable progress bar')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Num workers in dataloader')
    parser.add_argument('--save_dir', type=str, default="saved models",
                        help='Directory in which to save models')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Data parameters
    parser.add_argument('--dataset', type=str, default="md17",
                        help='Data set')
    parser.add_argument('--root', type=str, default="datasets",
                        help='Data set location')
    parser.add_argument('--download', type=bool, default=False,
                        help='Download flag')

    # QM9 parameters
    parser.add_argument('--target', type=str, default="revised aspirin",
                        help='MD17 target')
    parser.add_argument('--lifting_radius', type=float, default=None,
                        help='Radius (Angstrom) between which atoms to add links.')
    parser.add_argument('--n', type=int, default=12,
                        help='grid size (on SO(d) or SO(d)/SO(d-1)).')
    parser.add_argument('--radius', type=float, default=100.,
                        help='Radius (Angstrom) between which atoms to add links.')

    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='max degree of hidden rep')
    parser.add_argument('--basis_dim', type=int, default=None,
                        help='number of basis functions')
    parser.add_argument('--layers', type=int, default=4,
                        help='Number of message passing layers')
    parser.add_argument('--droprate', type=float, default=0.,
                        help='Dropout in the conv blocks.')
    parser.add_argument('--sigma_x', type=float, default=1.0,
                    help='RFF sigma for angular part')
    parser.add_argument('--sigma_R', type=float, default=5.0,
                    help='RFF sigma for angular part')
    parser.add_argument('--lambda_F', type=float, default=500.0,
                    help='coefficient in front of the force loss')
    parser.add_argument('--min_dist', type=float, default=0.9,  # Warning MD17 specific parameter
                    help='When weighting edges we assume this the min dist in the range of expected distances')
    parser.add_argument('--norm', type=str, default="layer", # or "layer"
                        help='Type of normalization ("none", or "layer"), default is "layer"')
    parser.add_argument('--M', type=str, default="R3S2", 
                        help='Type of normalization ("R3", "R3S2", "SE3"), default is "R3S2"')
    parser.add_argument('--separable', type=int, default=1,
                        help='Use int=0 for false, all other ints evaluate to true. Flag for whether or not channel mixing is done in ConvNextBlock')
    parser.add_argument('--widening_factor', type=int, default=4,
                        help='Number of message passing layers')
    parser.add_argument('--repeats', type=int, default=10,
                        help='number of repeated forward passes at test-time')
    parser.add_argument('--memory_friendly', type=int, default=0,
                        help='Use int=0 for False (default), all other ints evaluate to True. Flag for memory friendly repeated forward passes')
    parser.add_argument('--degree', type=int, default=2,
                        help='degree of the polynomial embedding')

    # Parallel computing stuff
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus to use (assumes all are on one node)')

    args = parser.parse_args()

    # Devices
    if args.gpus > 0:
        accelerator = "gpu"
        devices = args.gpus
    else:
        accelerator = "cpu"
        devices = "auto"
    if args.num_workers == -1:
        args.num_workers = os.cpu_count()

    # Load the dataset and set the dataset specific settings
    transform = []
    transform = [
        T.Kcal2meV(),
        T.OneHotTransform(9),
    ] + transform
    transform = tg.transforms.Compose(transform)
    dataset = MD17(
        args.root, name=args.target, transform=transform
    )
    # Random split
    random_state = np.random.RandomState(seed=args.seed)
    perm = torch.from_numpy(random_state.permutation(np.arange(len(dataset))))
    train_idx, val_idx, test_idx = perm[:950], perm[950:1000], perm[1000:]
    # Custom split
    test_idx = list(range(min(len(dataset),100000)))
    train_idx = test_idx[::100]
    del test_idx[::100]
    val_idx = train_idx[::20]
    del train_idx[::20]

    datasets = {'train': dataset[train_idx], 'valid': dataset[val_idx], 'test': dataset[test_idx]}
    
    # The model
    from models.md17 import SIVA_MD17
    model = SIVA_MD17(args)
    model.set_dataset_statistics(datasets['train'])

    # Make the dataloaders
    dataloaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == 'train'), num_workers=args.num_workers)
        for split, dataset in datasets.items()}

    # logging
    if args.log:
        if args.dataset == "qm9":
            logger = pl.loggers.WandbLogger(project="SIVA-" + args.dataset + "-" + model.qm9_target_dict[args.target],
                                        name='siva', config=args)
        else:
            logger = pl.loggers.WandbLogger(project="SIVA-" + args.dataset + "-" + args.target.replace(" ", "_"),
                                        name='siva', config=args)
    else:
        logger = None

    pl.seed_everything(args.seed, workers=True)
    torch.autograd.set_detect_anomaly(True)
    # Do the training and testing
    callbacks = []
    callbacks = [EMA(0.99)]
    callbacks += [pl.callbacks.ModelCheckpoint(monitor='valid MAE (energy)', mode = 'min')]
    callbacks += [EpochTimer()]
    # callbacks = []
    if args.log:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, callbacks=callbacks, inference_mode=not(args.dataset=='md17'),
                         gradient_clip_val=1., accelerator=accelerator, devices=devices, enable_progress_bar=args.enable_progress_bar)
    trainer.fit(model, dataloaders['train'], dataloaders['valid'])
    trainer.test(model, dataloaders['test'], ckpt_path = "best")
