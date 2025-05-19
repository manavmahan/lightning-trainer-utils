[![PyPI version](https://badge.fury.io/py/lightning-trainer-utils.svg)](https://pypi.org/project/lightning-trainer-utils/)
# PyTorch Lightning Trainer Utilities

## Installation
```bash
pip install lightning-trainer-utils
```

## ML Model Assumptions

### forward
- The model wrapper uses the forward function as follows:
```python
    output = self.model(**x, **self.forward_kwargs)
    return ModelOuput(**output)
```
It expects `batch` as `dict` and returns a `dict` with keys `[loss, report, output]`.

### return
- ML model should return a dict with the following keys:
    - `loss`
    - `report`
    - `output` [optional]


## Trainer
### Global Step
`batch_step = num_samples / (batch_size * num_devices)
trainer_global_step = num_samples / (batch_size * num_devices * grad_accumulation)`
`SaveCheckpoint` also use `trainer_global_step`.

