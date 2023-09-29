import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from .siva_fiber import SIVA
import numpy as np
from callbacks import EMAOptimizer
from siva.geometry.r3s2 import random_rotation
from torch_geometric.data import Batch


class SIVA_MD17(pl.LightningModule):
    """Graph Neural Network module"""

    def __init__(self, args):
        super().__init__()

        self.repeats = args.repeats
        self.memory_friendly = args.memory_friendly == 1
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup = args.warmup
        self.lambda_F = args.lambda_F

        self.shift = 0.
        self.scale = 1.

        self.train_metric = torchmetrics.MeanAbsoluteError()
        self.train_metric_force = torchmetrics.MeanAbsoluteError()
        self.valid_metric = torchmetrics.MeanAbsoluteError()
        self.valid_metric_force = torchmetrics.MeanAbsoluteError()
        self.test_metrics_energy = nn.ModuleList([torchmetrics.MeanAbsoluteError() for r in range(self.repeats)])
        self.test_metrics_force = nn.ModuleList([torchmetrics.MeanAbsoluteError() for r in range(self.repeats)])

        # Dataset specifics:
        in_channels = 9
        out_channels = 1

        # Make the model
        self.model = SIVA(in_channels,
                        args.hidden_dim,
                        out_channels,
                        args.layers,
                        n=args.n,
                        radius = args.radius,
                        M=args.M,
                        basis_dim=args.basis_dim,
                        separable=not(args.separable==0),
                        degree=args.degree,
                        widening_factor=args.widening_factor)

    def set_dataset_statistics(self, dataset):
        ys = np.array([data.energy.item() for data in dataset])
        forces = np.concatenate([data.force.numpy() for data in dataset])
        self.shift = np.mean(ys)
        # self.scale = np.sqrt(np.mean((ys - self.shift)**2))
        self.scale = np.sqrt(np.mean(forces**2))
        self.min_dist = 1e10
        self.max_dist = 0
        for data in dataset:
            pos = data.pos
            edm = np.linalg.norm(pos[:,None,:] - pos[None,:,:],axis=-1)
            min_dist = np.min(edm + np.eye(edm.shape[0]) * 1e10)
            max_dist = np.max(edm)
            if min_dist < self.min_dist:
                self.min_dist = min_dist 
            if max_dist > self.max_dist:
                self.max_dist = max_dist 
        print('Min-max range of distances between atoms in the dataset:', self.min_dist, '-', self.max_dist)

    def forward(self, graph):
        # if self.training:
        #     graph.pos = graph.pos + (torch.rand_like(graph.pos) - 0.5) * (self.trainer.max_epochs - self.current_epoch) / self.trainer.max_epochs
        pred = self.model(graph)
        return pred.squeeze(-1)

    @torch.enable_grad()
    def pred_energy_and_force(self, graph):
        graph.pos = torch.autograd.Variable(graph.pos, requires_grad=True)
        pred_energy = self(graph)
        sign = -1.0
        pred_force = sign * torch.autograd.grad(
            pred_energy,
            graph.pos,
            grad_outputs=torch.ones_like(pred_energy),
            create_graph=True,
            retain_graph=True
        )[0]
        # Return result
        return pred_energy, pred_force

    def training_step(self, graph):
        pred_energy, pred_force = self.pred_energy_and_force(graph)
        
        energy_loss = torch.mean((pred_energy - (graph.energy - self.shift) / self.scale)**2)
        force_loss = torch.mean(torch.sum((pred_force - graph.force / self.scale)**2,-1)) / 3.
        loss = energy_loss / self.lambda_F + force_loss

        self.train_metric(pred_energy * self.scale + self.shift, graph.energy)
        self.train_metric_force(pred_force * self.scale, graph.force)

        return loss

    def on_training_epoch_end(self):
        self.log("train MAE (energy)", self.train_metric, prog_bar=True)
        self.log("train MAE (force)", self.train_metric_force, prog_bar=True)

    def validation_step(self, graph, batch_idx):
        pred_energy, pred_force = self.pred_energy_and_force(graph)
        self.valid_metric(pred_energy * self.scale + self.shift, graph.energy)
        self.valid_metric_force(pred_force * self.scale, graph.force)        

    def on_validation_epoch_end(self):
        self.log("valid MAE (energy)", self.valid_metric, prog_bar=True)
        self.log("valid MAE (force)", self.valid_metric_force, prog_bar=True)
    
    def test_step(self, graph, batch_idx):
            if self.memory_friendly:
                pred_energy_repeated = []
                pred_force_repeated = []
                for r in range(self.repeats):
                    pred_energy, pred_force = self.pred_energy_and_force(graph)
                    pred_energy_repeated.append(pred_energy)
                    pred_force_repeated.append(pred_force)
                pred_energy_repeated = torch.stack(pred_energy_repeated, 0)
                pred_force_repeated = torch.stack(pred_force_repeated, 0)

                for r in range(self.repeats):
                    pred_energy, pred_force = pred_energy_repeated[:r+1].mean(0), pred_force_repeated[:r+1].mean(0)
                    self.test_metrics_energy[r](pred_energy * self.scale + self.shift, graph.energy)
                    self.test_metrics_force[r](pred_force * self.scale, graph.force)

                energy_std = pred_energy_repeated.std(0).mean() * self.scale
                force_std = pred_force_repeated.std(0).mean() * self.scale

                return torch.tensor([energy_std, force_std])
            else:
                batch_size = graph.batch.max() + 1
                batch_length = graph.batch.shape[0]
                graph_repeated = Batch.from_data_list([graph] * self.repeats)
                pred_energy_repeated, pred_force_repeated = self.pred_energy_and_force(graph_repeated)
                pred_energy_repeated = pred_energy_repeated.unflatten(0, (self.repeats, batch_size))
                pred_force_repeated = pred_force_repeated.unflatten(0, (self.repeats, batch_length))
                
                for r in range(self.repeats):
                    pred_energy, pred_force = pred_energy_repeated[:r+1].mean(0), pred_force_repeated[:r+1].mean(0)
                    self.test_metrics_energy[r](pred_energy * self.scale + self.shift, graph.energy)
                    self.test_metrics_force[r](pred_force * self.scale, graph.force)

                energy_std = pred_energy_repeated.std(0).mean() * self.scale
                force_std = pred_force_repeated.std(0).mean() * self.scale

    def on_test_epoch_end(self):
        for r in range(self.repeats):
            self.log("test MAE (energy) x"+str(r+1), self.test_metrics_energy[r])
            self.log("test MAE (force) x"+str(r+1), self.test_metrics_force[r])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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