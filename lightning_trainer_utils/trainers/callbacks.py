import time
import torch
import pytorch_lightning as pl


class SaveCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, cfg=dict()):
        """
        Initializes the callback with the given configuration.
        Args:
            cfg (dict): A configuration dictionary containing the following keys:
                - dirpath (str, optional): Directory path where checkpoints will be saved.
                  Defaults to "checkpoints/".
                - filename (str, optional): Filename format for the checkpoints.
                  Defaults to "step-{step}".
                - save_top_k (int, optional): Number of best models to save.
                  Defaults to -1 (save all checkpoints).
                - every_n_train_steps (int, optional): Frequency (in training steps)
                  at which checkpoints are saved. Defaults to 500.
                - save_weights_only (bool, optional): Whether to save only model weights
                  instead of the full model. Defaults to True.
                - **cfg: Additional keyword arguments passed to the parent class initializer.
        """

        dirpath = cfg.get("dirpath", "checkpoints/")
        filename = cfg.get("filename", "step-{step}")
        save_top_k = cfg.get("save_top_k", -1)
        every_n_train_steps = cfg.get("every_n_train_steps", 500)
        save_weights_only = cfg.get("save_weights_only", True)
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            save_top_k=save_top_k,
            every_n_train_steps=every_n_train_steps,
            save_weights_only=save_weights_only,
            **cfg,
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
    def on_train_step_end(self, trainer, pl_module):
        # Get the learning rate from the optimizer
        for param_group in trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
            break

        # Log the learning rate
        pl_module.log("train/lr", lr, on_step=True, logger=True, sync_dist=True)


class LogGradient(pl.Callback):
    def __init__(self, should_stop: bool = True):
        super().__init__()
        self.should_stop = should_stop

    def on_after_backward(self, trainer, pl_module):
        # Calculate the total gradient norm
        total_norm = 0.0
        for param in pl_module.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5

        # Convert total_norm to a tensor and check for NaN or Inf
        total_norm_tensor = torch.tensor(total_norm)

        pl_module.log("train/norm", total_norm, logger=True, sync_dist=True)
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
        remaining_time = (
            elapsed_time * (trainer.max_epochs - trainer.current_epoch) / elapsed_epoch
        )
        pl_module.log("ETL (min)", remaining_time / 60, sync_dist=True)
