import torch
import argparse
import os
from n_body_system.dataset_nbody import NBodyDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import pytorch_lightning as pl
import torch_geometric.nn as nn
from callbacks import EMA, EpochTimer

def make_pyg_loader(dataset, batch_size, shuffle, num_workers, radius=1000., loop=False):
    data_list = []
    for data in dataset:
        loc, vel, edge_attr, charges, loc_end = data
        graph = Data(pos=loc, vel=vel, charges=charges, y=loc_end)
        graph.edge_index = nn.radius_graph(x=graph.pos, r=radius, loop=loop, max_num_neighbors=1000)
        data_list.append(graph)
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Run parameters
    parser.add_argument('--epochs', type=int, default=10000,
                        help='number of epochs')
    parser.add_argument('--warmup', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-16,
                        help='weight decay')
    parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                        help='how many epochs to wait before logging test')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Num workers in dataloader')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--enable_progress_bar', type=bool, default=False,
                    help='enable progress bar')

    # Model parameters
    parser.add_argument('--n', type=int, default=20,
                        help='grid size (on SO(d) or SO(d)/SO(d-1)).')
    parser.add_argument('--radius', type=float, default=100.,
                        help='Radius (Angstrom) between which atoms to add links.')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='max degree of hidden rep')
    parser.add_argument('--basis_dim', type=int, default=256,
                        help='number of basis functions')
    parser.add_argument('--layers', type=int, default=5,
                        help='Number of message passing layers')
    parser.add_argument('--M', type=str, default="R3S2", 
                        help='Type of normalization ("R3", "R3S2", "SE3"), default is "R3S2"')
    parser.add_argument('--separable', type=int, default=1,
                        help='Use int=0 for false, all other ints evaluate to true. Flag for whether or not channel mixing is done in ConvNextBlock')
    parser.add_argument('--degree', type=int, default=3,
                        help='degree of the polynomial embedding')
    parser.add_argument('--widening_factor', type=int, default=4,
                    help='Number of message passing layers')
    
    # Dataset
    parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
    parser.add_argument('--dataset', type=str, default="nbody_small", metavar='N',
                    help='nbody_small, nbody')

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
    dataset_train = NBodyDataset(partition='train', dataset_name=args.dataset,
                                 max_samples=args.max_training_samples)
    dataset_val = NBodyDataset(partition='val', dataset_name="nbody_small")
    dataset_test = NBodyDataset(partition='test', dataset_name="nbody_small")

    datasets = {'train': dataset_train, 'valid': dataset_val, 'test': dataset_test}

    # The model
    from models.nbody import SIVA_NBODY
    model = SIVA_NBODY(args)

    # Make the dataloaders
    dataloaders = {
        split: make_pyg_loader(dataset, batch_size=args.batch_size, shuffle=(split == 'train'), num_workers=args.num_workers)
        for split, dataset in datasets.items()}

    # logging
    logger = pl.loggers.WandbLogger(project="PONITA-" + args.dataset, name='siva', config=args)

    pl.seed_everything(args.seed, workers=True)
    torch.autograd.set_detect_anomaly(True)
    # Do the training and testing
    callbacks = []
    callbacks.append(pl.callbacks.ModelCheckpoint(monitor='valid MSE', mode = 'min'))
    callbacks.append(EpochTimer())
    callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, callbacks=callbacks, gradient_clip_val=1., accelerator=accelerator, devices=devices, check_val_every_n_epoch=args.test_interval,enable_progress_bar=args.enable_progress_bar)
    trainer.fit(model, dataloaders['train'], dataloaders['valid'])
    trainer.test(model, dataloaders['test'], ckpt_path = "best")
