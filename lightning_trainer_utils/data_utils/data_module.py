import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typeguard import typechecked

from .data_loader import DictDataLoader


@typechecked
class SharedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_class: type[Dataset],
        dataloader_class: type[DataLoader] = DictDataLoader,
        training_kwargs: dict = dict(),
        validation_kwargs: dict = dict(),
        dataloader_kwargs: dict = dict(),
    ):
        super().__init__()
        self.dataset_class = dataset_class
        self.dataloader_class = dataloader_class
        self.train_kwargs = training_kwargs
        self.val_kwargs = validation_kwargs
        self.dl_kwargs = dataloader_kwargs

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if self.train_dataset is None:
            self.train_dataset = self.dataset_class(**self.train_kwargs)

        if self.val_kwargs is not None:
            if self.val_dataset is None:
                self.val_dataset = self.dataset_class(**self.val_kwargs)

    def train_dataloader(self):
        return self.dataloader_class(self.train_dataset, **self.dl_kwargs, shuffle=True)

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return self.dataloader_class(self.val_dataset, **self.dl_kwargs, shuffle=False)
