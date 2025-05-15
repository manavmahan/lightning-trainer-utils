import time
import torch
import torch.nn.utils as utils

import pytorch_lightning as pl
from pytorch_custom_utils import get_adam_optimizer

from beartype import beartype

from .ema import EMA
from .optimizer_scheduler import (
    get_cosine_schedule_with_warmup,
)

from collections import namedtuple

ModelOuput = namedtuple(
    "ModelOuput", ["loss", "report", "output"], defaults=[None, None, None]
)


class ModelWrapper(pl.LightningModule):
    @beartype
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        use_ema: bool = False,
        scheduler_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        forward_kwargs: dict = dict(),
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.forward_kwargs = forward_kwargs

        self.optimizer = optimizer_kwargs.pop("optimizer", None)
        self.schedueler = scheduler_kwargs.pop("scheduler", None)

        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs

        self.max_norm = optimizer_kwargs.get("max_norm", 1.0)

        self.wandb_id = None
        self.start_epoch = 0

        self.use_ema = use_ema
        if self.use_ema:
            self.ema_model = EMA(self.model)

        print("Unuserd kwargs:", kwargs)

    def configure_optimizers(self):
        optimizer = (
            self.optimizer
            if self.optimizer is not None
            else get_adam_optimizer(self.model.parameters(), **self.optimizer_kwargs)
        )
        scheduler = (
            self.schedueler
            if self.schedueler is not None
            else get_cosine_schedule_with_warmup(optimizer, **self.scheduler_kwargs)
        )
        return [optimizer], [scheduler]

    def optimizer_step(self, *args, **kwargs):
        if self.max_norm is not None:
            utils.clip_grad_norm_(self.parameters(), self.max_norm)
        super().optimizer_step(*args, **kwargs)

    def forward(self, x):
        output = self.model(**x, **self.forward_kwargs)
        return ModelOuput(**output)

    def training_step(self, batch, batch_idx):
        fwd_out = self(batch)
        loss = fwd_out.loss
        report = fwd_out.report

        for k, v in report.items():
            self.log("training/" + k, v, logger=True, sync_dist=True)

        if self.use_ema and self.global_rank == 0:
            self.ema_model.step(self.model)
        return loss

    def validation_step(self, batch, batch_idx):
        fwd_out = self(batch)
        loss = fwd_out.loss
        report = fwd_out.report
        for k, v in report.items():
            self.log("validation/" + k, v, logger=True, sync_dist=True)
        return loss

    def on_save_checkpoint(self, checkpoint):
        """Manually save additional metrics."""
        checkpoint["metrics"] = self.trainer.callback_metrics
        if self.use_ema:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()
        checkpoint["wandb_id"] = self.trainer.logger.experiment.id

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        self.trainer.callback_metrics = checkpoint.get("metrics", {})

        self.wandb_id = checkpoint.get("wandb_id", None)
        self.start_epoch = checkpoint.get("epoch", 0)
        if self.use_ema:
            if "ema_state_dict" in checkpoint:
                self.ema_model.load_state_dict(checkpoint["ema_state_dict"])
                print("EMA state dict found in checkpoint.")
            else:
                print("No EMA state dict found in checkpoint.")
