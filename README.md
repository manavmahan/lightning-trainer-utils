# PyTorch Lightning Trainer Utilities

## ML Model Assumptions

### forward

- The batch will be passed as `kwargs` to `forward` function of the ML model.
```python
self.model(**batch)
```

### return
- ML model should return a dict with the following keys:
    - `loss`
    - `loss-dict`
    - `output` [optional]