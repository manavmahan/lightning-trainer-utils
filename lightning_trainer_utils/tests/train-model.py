import os
from attr import dataclass
import yaml
import pytorch_lightning as pl
import torch

torch.backends.cudnn.benchmark = True
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from data_utils.data_module import SharedDataModule
from trainers.model_wrapper import ModelWrapper
from trainers.callbacks import (
    SaveCheckpoint,
    LogLearningRate,
    LogGradient,
    LogETL,
)

from argparse import ArgumentParser


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.length = kwargs.get("length", 100)
        super().__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        vertices = torch.randn(32, 2)
        segments = torch.randint(0, 32, (10, 2))
        return {
            "vertices": vertices,
            "segments": segments,
        }


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


if __name__ == "__main__":
    data_kwargs = yaml.safe_load(open("configs/data.yaml", "r"))
    model_kwargs = yaml.safe_load(open("configs/model.yaml", "r"))
    trainer_kwargs = yaml.safe_load(open("configs/trainer.yaml", "r"))

    train_data_kwargs = data_kwargs["data"]["general"].copy()
    train_data_kwargs.update(data_kwargs["data"]["train"])

    val_data_kwargs = data_kwargs["data"]["general"].copy()
    val_data_kwargs.update(data_kwargs["data"]["validation"])

    datamodule = SharedDataModule(
        dataloader_class=torch.utils.data.DataLoader,
        dataset_class=DummyDataset,
        training_kwargs=train_data_kwargs,
        val_kwargs=val_data_kwargs,
        dl_kwargs=data_kwargs["DATALOADER"],
    )

    model = DummyModel(**model_kwargs)
    wrapped_model = ModelWrapper(model, trainer_kwargs.get("wrapper", dict()))

    ckpt_path = trainer_kwargs.get("load_from", None)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        model.on_load_checkpoint(torch.load(ckpt_path, weights_only=True))
    else:
        print(f"Checkpoint not found at {ckpt_path}. Starting from scratch.")
        ckpt_path = None

    wandb_logger = WandbLogger(**trainer_kwargs.get("wandb"), id=model.wandb_id)
    early_stop_callback = EarlyStopping(
        monitor="validation/transformer-loss",  # Metric to monitor
        patience=10,  # How many epochs to wait after last improvement
        verbose=True,  # Print logs
        mode="min",  # "min" for loss, "max" for accuracy, etc.
    )
    callbacks = [SaveCheckpoint(), LogETL(), LogGradient(), LogLearningRate()]

    trainer = pl.Trainer(
        **trainer_kwargs.get("trainer"), logger=wandb_logger, callbacks=callbacks
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
