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


class LogInfGradient(pl.Callback):
    def __init__(self, should_stop: bool = True):
        super().__init__()
        self.should_stop = should_stop

    def on_after_backward(self, trainer, pl_module):
        # Calculate the total gradient norm
        total_norm = 0.0
        for param in pl_module.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        # Convert total_norm to a tensor and check for NaN or Inf
        total_norm_tensor = torch.tensor(total_norm)
        
        if torch.isinf(total_norm_tensor) or torch.isnan(total_norm_tensor):
            print(f"Infinite/NaN gradient norm @ {trainer.current_epoch} epoch.")
            trainer.save_checkpoint(
                f"inf_nan_gradient_epoch_{trainer.current_epoch}.ckpt",
                weights_only=True,
            )
            trainer.should_stop = self.should_stop


class LogETL(pl.Callback):
    def on_fit_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed_time = time.time() - self.start_time
        elapsed_epoch = trainer.current_epoch - pl_module.start_epoch
        if elapsed_epoch < 1:
            trainer.start_epoch = trainer.current_epoch
            elapsed_epoch = 1
        remaining_time = elapsed_time * (trainer.max_epochs - trainer.current_epoch) / elapsed_epoch
        pl_module.log("ETL (min)", remaining_time / 60, sync_dist=True)  # Log ETA in minutes


class ModelWrapper(pl.LightningModule):
    @beartype
    def __init__(
        self,
        model: SegmentVQVAE | Transformer,
        scheduler_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
    ):
        super().__init__()
        self.model = model
        self.optimizer_kwargs = optimizer_kwargs
        self.max_norm = optimizer_kwargs.get("max_norm", 1.0)
        self.scheduler_kwargs = scheduler_kwargs

        self.curr_total_norm = 0.0
        self.start_time = None
        self.wandb_id = None
        self.start_epoch = 0

        if isinstance(model, SegmentVQVAE):
            self.ema_model = EMA(self.model)
        else:
            self.ema_model = None

    def on_train_epoch_start(self):
        if self.start_time is None:
            self.start_time = time.time()

    def on_train_epoch_end(self):
        if self.start_epoch > self.current_epoch:
            self.start_epoch = self.current_epoch
        elapsed_time = time.time() - self.start_time
        avg_epoch_time = elapsed_time / (self.current_epoch - self.start_epoch + 1)
        remaining_epochs = self.trainer.max_epochs - self.current_epoch - 1
        remaining_time = avg_epoch_time * remaining_epochs
        # self.log("ETA (min)", remaining_time / 60, sync_dist=True)  # Log ETA in minutes

    def configure_optimizers(self):
        optimizer = get_adam_optimizer(self.model.parameters(), **self.optimizer_kwargs)
        scheduler = get_cosine_schedule_with_warmup(optimizer, **self.scheduler_kwargs)
        return [optimizer], [scheduler]

    def optimizer_step(self, *args, **kwargs):
        self.curr_total_norm = utils.clip_grad_norm_(self.parameters(), self.max_norm)
        super().optimizer_step(*args, **kwargs)

    def forward(self, x):
        if isinstance(self.model, Transformer):
            return self.model(**x)
        loss, report = self.model(**x, return_loss_breakdown=True)
        return loss, report

    def training_step(self, batch, batch_idx):
        fwd_out = self(batch)
        loss, loss_dict = fwd_out
        cur_lr = get_lr(self.optimizers())

        self.log(
            "train/lr",
            cur_lr,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/total_norm",
            self.curr_total_norm,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        for k, v in loss_dict.items():
            self.log(
                "train/" + k,
                v,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
        if self.ema_model is not None and self.global_rank == 0:
            self.ema_model.step(self.model)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self(batch)
        if "commit_loss" in loss_dict:
            del loss_dict["commit_loss"]
        for k, v in loss_dict.items():
            self.log(
                "validation/" + k,
                v,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
        return loss

    def on_save_checkpoint(self, checkpoint):
        """Manually save additional metrics."""
        for k, v in self.trainer.callback_metrics.items():
            if k.startswith("train/") or k.startswith("validation/"):
                checkpoint[k] = v
        if self.ema_model is not None:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()
        checkpoint["wandb_id"] = self.trainer.logger.experiment.id

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        self.wandb_id = checkpoint.get("wandb_id", None)
        self.start_epoch = checkpoint.get("epoch", 0)
        if self.ema_model is not None:
            if "ema_state_dict" in checkpoint:
                self.ema_model.load_state_dict(checkpoint["ema_state_dict"])
                print("EMA state dict found in checkpoint.")
            else:
                print("No EMA state dict found in checkpoint.")
