import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from .siva_grid_classification import SIVA
import numpy as np
from callbacks import EMAOptimizer
from siva.geometry.r3s2 import random_rotation

class SIVA_ModelNet(pl.LightningModule):
    """Graph Neural Network module"""

    def __init__(self, args):
        super().__init__()

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup = args.warmup

        self.train_metric = torchmetrics.Accuracy('multiclass', num_classes=40)
        self.valid_metric = torchmetrics.Accuracy('multiclass', num_classes=40)
        self.test_metric = torchmetrics.Accuracy('multiclass', num_classes=40)
        self.test_metric_avg = torchmetrics.Accuracy('multiclass', num_classes=40)

        # Dataset specifics:
        in_channels = 1
        out_channels = 40

        # Make the model
        self.model = SIVA(in_channels,
                        args.hidden_dim,
                        out_channels,
                        args.layers,
                        droprate=args.droprate,
                        task="graph",
                        lifting_radius=args.lifting_radius,
                        n=args.n,
                        radius = args.radius,
                        sigma_x=args.sigma_x,
                        sigma_R=args.sigma_R,
                        min_dist = args.min_dist,
                        layer_norm=args.norm == "layer",
                        M=args.M,
                        basis_dim=args.basis_dim,
                        depthwise=not(args.depthwise==0))
    
    def set_dataset_statistics(self, dataset):
            # Label balance
            ys = np.array([data.y.item() for data in dataset])
            counts = np.array([(ys == i).sum() for i in range(40)])
            counts = counts / counts.max()
            weights = 1 / counts
            weights = weights / weights.max()
            self.class_weights = torch.tensor(weights)
            
    def forward(self, graph):
        pred = self.model(graph)
        return pred.squeeze(-1)

    def training_step(self, graph):
        pred = self(graph)
        loss = F.cross_entropy(pred, graph.y, self.class_weights.type_as(pred))
        self.train_metric(pred, graph.y)
        return loss

    def training_epoch_end(self, outs):
        self.log("train ACC)", self.train_metric, prog_bar=True)

    def validation_step(self, graph, batch_idx):
        pred = self(graph)
        self.valid_metric(pred, graph.y)

    def validation_epoch_end(self, outs):
        self.log("valid ACC", self.valid_metric, prog_bar=True)
    
    def test_step(self, graph, batch_idx):
            runs = 5
            pred_avg = 0
            for i in range(runs):
                pred = self(graph)
                pred_avg += pred / runs
            self.test_metric(pred, graph.y)
            self.test_metric_avg(pred_avg, graph.y)
    
    def test_epoch_end(self, outs):
        self.log("test ACC avg", self.test_metric_avg)
        self.log("test ACC", self.test_metric)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = CosineWarmupScheduler(optimizer, self.warmup, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    

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