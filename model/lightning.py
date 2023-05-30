import torch
from torch import nn as nn

import lightning as pl
from loss.loss import LNCCLoss
from .model import AutoEncoder


class AutoEncoderTrainer(pl.pytorch.LightningModule):
    def __init__(self, mod='t1', learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters({'learning_rate': learning_rate,
                                   'mod': mod})
        self.model = AutoEncoder()
        self.l1_loss = nn.L1Loss()
        self.ncc_loss = LNCCLoss(window_size=7)
        self.learning_rate = learning_rate
        self.mod = mod.lower()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.mod == 't1':
            x = batch['target']
        elif self.mod == 't2':
            x = batch['source']
        else:
            raise ValueError(f'Unknown modality: {self.mod}')

        y, features = self.model(x)

        l1_loss = self.l1_loss(y, x)
        ncc_loss = self.ncc_loss(y, x)

        total_loss = l1_loss + ncc_loss

        self.log('l1_loss', l1_loss)
        self.log('ncc_loss', ncc_loss)
        self.log('total_loss', total_loss)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
