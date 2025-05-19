import time
import torch
import pytorch_lightning as pl


class SaveCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, **kwargs):
        """
        Initializes the callback with the given configuration.
        Args:
            kwargs (dict): A configuration dictionary containing the following keys:
                - dirpath (str, optional): Directory path where checkpoints will be saved.
                  Defaults to "checkpoints/".
                - filename (str, optional): Filename format for the checkpoints.
                  Defaults to "step-{step}".
                - save_top_k (int, optional): Number of best models to save.
                  Defaults to -1 (save all checkpoints).
                - every_n_train_steps (int, optional): Frequency (in training steps)
                  at which checkpoints are saved. Defaults to 512.
                - save_weights_only (bool, optional): Whether to save only model weights
                  instead of the full model. Defaults to False.
                - **kwargs: Additional keyword arguments passed to the parent class initializer.
        """

        dirpath = kwargs.pop("dirpath", "checkpoints/")
        filename = kwargs.pop("filename", "{step:05d}")
        save_top_k = kwargs.pop("save_top_k", -1)
        every_n_train_steps = kwargs.pop("every_n_train_steps", 512)
        save_weights_only = kwargs.pop("save_weights_only", False)
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            save_top_k=save_top_k,
            every_n_train_steps=every_n_train_steps,
            save_weights_only=save_weights_only,
            save_last=True,
            monitor="step",
            mode="max",
            **kwargs,
        )
        print(
            "Save checkpoint strategy initialized with the following parameters:"
            f"\n- dirpath: {dirpath}"
            f"\n- filename: {filename}"
            f"\n- save_top_k: {save_top_k}"
            f"\n- every_n_train_steps: {every_n_train_steps}"
            f"\n- save_weights_only: {save_weights_only}"
        )


class LogLearningRate(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        # Get the learning rate from the optimizer
        for param_group in trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
            break
        pl_module.log("trainer/lr", lr, on_step=True, logger=True, sync_dist=True)


class LogGradient(pl.Callback):
    def __init__(self, should_stop: bool = False):
        super().__init__()
        self.should_stop = should_stop

    def on_after_backward(self, trainer, pl_module):
        total_norm = pl_module.total_norm
        if total_norm is None:
            return
        pl_module.log("trainer/norm", total_norm, on_step=True, logger=True, sync_dist=True)
        if torch.isinf(total_norm) or torch.isnan(total_norm):
            print(f"Infinite/NaN gradient norm @ {trainer.current_epoch} epoch.")
            trainer.save_checkpoint(
                f"checkpoints/inf_nan_gradient_epoch_{trainer.current_epoch}.ckpt",
                weights_only=True,
            )
            trainer.should_stop = self.should_stop


class LogETL(pl.Callback):
    def on_fit_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.max_epochs==-1:
            return
        elapsed_time = time.time() - self.start_time
        elapsed_epochs = trainer.current_epoch - pl_module.start_epoch
        if elapsed_epochs < 1:
            pl_module.start_epoch = trainer.current_epoch + 1
            elapsed_epochs = 1
        remaining_time = (elapsed_time / elapsed_epochs) * (
            trainer.max_epochs - trainer.current_epoch
        )
        pl_module.log(
            "trainer/ETL (min)", remaining_time / 60, on_epoch=True, logger=True, sync_dist=True
        )

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        if trainer.max_steps==-1:
            return
        elapsed_time = time.time() - self.start_time
        elapsed_steps = trainer.global_step - pl_module.start_step
        if elapsed_steps < 1:
            pl_module.start_step = trainer.global_step + 1
            elapsed_steps = 1
        remaining_time = (elapsed_time / elapsed_steps) * (
            trainer.max_steps - trainer.global_step
        )
        pl_module.log(
            "trainer/ETL (min)", remaining_time / 60, on_step=True, logger=True, sync_dist=True
        )