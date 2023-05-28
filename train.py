import torch
import torch.nn as nn
import lightning as pl
import torchio as tio
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from dataset import BrainMRInterSubj3D
from model import AutoEncoder
from loss import LNCCLoss


class AutoEncoderTrainer(pl.LightningModule):
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


def main():
    mod = 't2'
    epochs = 200
    batch_size = 1
    lr = 1e-3

    transform = tio.Compose([
        tio.CropOrPad(128),
        tio.RandomFlip(),
        tio.RandomGamma(),
        tio.Blur(std=(1, 1, 1)),
        tio.ToCanonical()
    ])

    train_dataset = BrainMRInterSubj3D(
        data_dir_path='/vol/alan/projects/camcan_malpem/train/',
        crop_size=[176, 192, 176],
        evaluate=False,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    ckpt_callback = ModelCheckpoint(save_last=False,
                                    dirpath='./checkpoints/',
                                    filename=f'ae-mod-{mod}-epochs-{epochs}.ckpt',
                                    verbose=True
                                    )

    logger = WandbLogger(project='ContraRegAutoEncoders', tags=[mod])

    trainer = pl.Trainer(logger=logger,
                         callbacks=[ckpt_callback],
                         gpus=1,
                         max_epochs=epochs)
    ae_trainer = AutoEncoderTrainer(mod=mod, learning_rate=lr)

    torch.set_float32_matmul_precision('medium')
    trainer.fit(ae_trainer, train_loader)


if __name__ == '__main__':
    main()
