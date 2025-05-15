import os
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
    StepProgressBar,
)


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
    def __init__(self, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, vertices, **kwargs):
        x = self.linear(vertices)
        loss = torch.mean(x)
        report = {"loss": loss}

        return {"loss": loss, "report": report, "output": x}


if __name__ == "__main__":
    all_data_kwargs = yaml.safe_load(open("configs/data.yaml", "r"))
    model_kwargs = yaml.safe_load(open("configs/model.yaml", "r"))
    trainer_kwargs = yaml.safe_load(open("configs/trainer.yaml", "r"))

    data_kwargs = all_data_kwargs.get("data", dict())
    train_data_kwargs = data_kwargs.get("general", dict()).copy()
    train_data_kwargs.update(data_kwargs.get("train", dict()))

    data_kwargs = all_data_kwargs.get("data", dict())
    val_data_kwargs = data_kwargs.get("general", dict()).copy()
    val_data_kwargs.update(data_kwargs.get("validataion", dict()))

    datamodule = SharedDataModule(
        dataloader_class=torch.utils.data.DataLoader,
        dataset_class=DummyDataset,
        training_kwargs=train_data_kwargs,
        validation_kwargs=val_data_kwargs,
        dataloader_kwargs=all_data_kwargs.get("dataloader", dict()),
    )

    model = DummyModel(**model_kwargs)
    wrapped_model = ModelWrapper(model=model, **trainer_kwargs.get("wrapper", dict()))

    ckpt_path = trainer_kwargs.get("load_from", None)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        wrapped_model.on_load_checkpoint(torch.load(ckpt_path, weights_only=True))
    else:
        print(f"Checkpoint not found at {ckpt_path}. Starting from scratch.")
        ckpt_path = None

    wandb_logger = WandbLogger(
        **trainer_kwargs.get("wandb", dict()), id=wrapped_model.wandb_id
    )
    callbacks = [SaveCheckpoint(), LogETL(), LogGradient(), LogLearningRate()]

    print(trainer_kwargs.get("trainer", dict()))
    trainer = pl.Trainer(
        **trainer_kwargs.get("trainer", dict()),
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    trainer.fit(wrapped_model, datamodule=datamodule, ckpt_path=ckpt_path)
