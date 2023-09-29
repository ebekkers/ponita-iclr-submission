import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from .siva_nbody import SIVA
import numpy as np
from callbacks import EMAOptimizer
from siva.geometry.r3s2 import random_rotation
from torch_geometric.data import Batch
from torch_geometric.data import Data


class SIVA_NBODY(pl.LightningModule):
    """Graph Neural Network module"""

    def __init__(self, args):
        super().__init__()

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup = args.warmup


        self.train_metric = torchmetrics.MeanSquaredError()
        self.valid_metric = torchmetrics.MeanSquaredError()
        self.test_metric = torchmetrics.MeanSquaredError()

        # Dataset specifics:
        in_channels_scalar = 2  # Charge, Velocity norm
        in_channels_vec = 2  # Velocity, rel_pos
        out_channels_scalar = 0  # None
        out_channels_vec = 1  # Output velocity

        # Make the model
        self.model = SIVA(in_channels_scalar,
                        args.hidden_dim,
                        out_channels_scalar,
                        args.layers,
                        input_dim_vec = in_channels_vec,
                        output_dim_vec = out_channels_vec,
                        n=args.n,
                        radius = args.radius,
                        M=args.M,
                        basis_dim=args.basis_dim,
                        separable=not(args.separable==0),
                        degree=args.degree,
                        widening_factor=args.widening_factor)


    def forward(self, graph):
        pred = self.model(graph)
        return pred

    def training_step(self, graph):

        pos_pred = graph.pos + self.model(graph)
        loss = torch.mean((pos_pred - graph.y)**2)
        self.train_metric(pos_pred, graph.y)

        return loss

    def on_training_epoch_end(self):
        self.log("train MSE", self.train_metric, prog_bar=True)

    def validation_step(self, graph, batch_idx):
        pos_pred = graph.pos + self.model(graph)
        self.valid_metric(pos_pred, graph.y)  

    def on_validation_epoch_end(self):
        self.log("valid MSE", self.valid_metric, prog_bar=True)
    
    def test_step(self, graph, batch_idx):
        pos_pred = graph.pos + self.model(graph)
        self.test_metric(pos_pred, graph.y)  

    def on_test_epoch_end(self):
        self.log("test MSE", self.test_metric)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = CosineWarmupScheduler(optimizer, self.warmup, self.trainer.max_epochs)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid MAE"}
        return {"optimizer": optimizer, "monitor": "valid MAE"}
    

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor