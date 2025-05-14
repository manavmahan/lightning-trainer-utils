from functools import partial
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def dict_collate_fn(batch, pad_value):
    """
    Custom collate function to handle dictionaries in a batch.
    """
    assert isinstance(batch, dict), "Batch must be a dictionary."
    keys = batch[0].keys()
    batch = [d.values() for d in batch]

    output = [
        (
            pad_sequence(d, batch_first=True, padding_value=pad_value)
            if torch.is_tensor(d[0])
            else list(d)
        )
        for d in zip(*batch)
    ]
    return dict(zip(keys, output))


class DictDataLoader(DataLoader):
    """
    A DataLoader that returns a dictionary of data.
    """

    def __init__(self, *args, **kwargs):
        pad_value = kwargs.pop("pad_value", -1)
        collate_fn = kwargs.pop(
            "collate_fn", partial(dict_collate_fn, pad_value=pad_value)
        )
        super().__init__(collate_fn=collate_fn, *args, **kwargs)
