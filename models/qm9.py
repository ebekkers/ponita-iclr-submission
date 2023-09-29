import torch
import torchmetrics
import pytorch_lightning as pl
# from .poni import PONI
from .siva import SIVA
import numpy as np


class QM9Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.target_mean = 0.
        self.target_mad = 1.
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup = args.warmup

        self.train_metric = torchmetrics.MeanAbsoluteError()
        self.valid_metric = torchmetrics.MeanAbsoluteError()

        self.qm9_target_dict = {
                                    0: 'dipole_moment',
                                    1: 'isotropic_polarizability',
                                    2: 'homo',
                                    3: 'lumo',
                                    4: 'gap',
                                    5: 'electronic_spatial_extent',
                                    6: 'zpve',
                                    7: 'energy_U0',
                                    8: 'energy_U',
                                    9: 'enthalpy_H',
                                    10: 'free_energy',
                                    11: 'heat_capacity',
                                }

    def set_dataset_statistics(self, dataset):
        ys = np.array([data.y.item() for data in dataset])
        self.target_mean = np.mean(ys)
        self.target_mad = np.mean(np.abs(ys - self.target_mean))

    def training_step(self, batch, batch_idx):
        # Compute the loss for the optimizer
        graph = batch
        out = self.net(graph).squeeze()
        self.lr_schedulers()
        loss = torch.nn.functional.l1_loss(out, (graph.y - self.target_mean) / self.target_mad)
        # To log
        self.log("train loss", loss, sync_dist=True)
        self.train_metric(out * self.target_mad + self.target_mean, graph.y)
        # Return
        return loss

    def training_epoch_end(self, outs):
        self.log("train MAE", self.train_metric.compute(), sync_dist=True)
        self.train_metric.reset()

    def validation_step(self, batch, batch_idx):
        graph = batch
        out = self.net(graph).squeeze()
        self.valid_metric(out * self.target_mad + self.target_mean, graph.y)

    def validation_epoch_end(self, outs):
        self.log("valid MAE", self.valid_metric.compute(), sync_dist=True)
        self.valid_metric.reset()

    def test_step(self, batch, batch_idx):
        graph = batch
        out = self.net(graph).squeeze()
        self.valid_metric(out * self.target_mad + self.target_mean, graph.y)

    def test_epoch_end(self, outs):
        self.log("test MAE", self.valid_metric.compute(), sync_dist=True)
        self.valid_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = CosineWarmupScheduler(optimizer, self.warmup, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


class SIVA_QM9(QM9Model):
    def __init__(self, args):
        super().__init__(args)

        # Dataset specifics:
        in_channels = 11
        out_channels = 1

        # Make the model
        self.net = SIVA(in_channels,
                        args.hidden_features,
                        out_channels,
                        args.layers,
                        droprate=args.droprate,
                        pool=args.pool,
                        task="graph",
                        sigma_1=args.sigma_1,
                        sigma_2=args.sigma_2)


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