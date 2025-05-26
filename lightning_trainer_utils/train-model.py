import os
import yaml
import pytorch_lightning as pl
import torch

torch.backends.cudnn.benchmark = True
from pytorch_lightning.loggers import WandbLogger

from data_utils.data_module import SharedDataModule, DictDataLoader
from trainer_utils.model_wrapper import ModelWrapper, extract_weights
from trainer_utils.callbacks import (
    SaveCheckpoint,
    LogLearningRate,
    GradientClipLogger,
    LogETL,
)


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.length = kwargs.get("length", 256)
        super().__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        vertices = torch.randn(32, 2)
        segments = (vertices[:, 0] * vertices[:, 0]) + (vertices[:, 1] * vertices[:, 1]) > 0.5
        return {
            "vertices": vertices,
            "segments": segments,
        }


class DummyModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(2, 200000)

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
        dataloader_class=DictDataLoader,
        dataset_class=DummyDataset,
        training_kwargs=train_data_kwargs,
        validation_kwargs=val_data_kwargs,
        dataloader_kwargs=all_data_kwargs.get("dataloader", dict()),
    )

    model = DummyModel(**model_kwargs)
    wrapped_model = ModelWrapper(model=model, **trainer_kwargs.get("wrapper", dict()))

    ckpt_path = trainer_kwargs.get("ckpt_path", None)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        torch.load(ckpt_path, weights_only=True)
    else:
        print(f"Checkpoint not found at {ckpt_path}. Starting from scratch.")
        ckpt_path = None

    wandb_logger = WandbLogger(
        **trainer_kwargs.get("wandb", dict()), id=wrapped_model.wandb_id
    )
    callbacks = [SaveCheckpoint(), LogETL(), GradientClipLogger(), LogLearningRate()]

    trainer = pl.Trainer(
        **trainer_kwargs.get("trainer", dict()),
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    trainer.fit(wrapped_model, datamodule=datamodule, ckpt_path=ckpt_path)

    extract_weights(
        model=model,
        checkpoint_path="checkpoints/last-v1.ckpt",
        save_to="checkpoints/weights.safetensors",
        wrapper_kwargs=trainer_kwargs.get("wrapper", dict()),
    )

    wrapped_model.on_load_checkpoint(
        torch.load("checkpoints/last-v1.ckpt", map_location="cpu")
    )

    weights = torch.load("checkpoints/weights.safetensors", map_location="cpu")
