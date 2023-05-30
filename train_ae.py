import torch
import lightning as pl
import torchio as tio
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from data.dataset import BrainMRInterSubj3D
from model.lightning import AutoEncoderTrainer


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

    trainer = pl.Trainer(
                         logger=logger,
                         callbacks=[ckpt_callback],
                         gpus=1,
                         max_epochs=epochs)
    ae_trainer = AutoEncoderTrainer(mod=mod, learning_rate=lr)

    torch.set_float32_matmul_precision('medium')
    trainer.fit(ae_trainer, train_loader)


if __name__ == '__main__':
    main()
