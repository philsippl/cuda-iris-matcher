## iris-hamming

PyTorch CUDA extension for masked fractional Hamming distance on IrisCode-like bit tensors.

### Install (editable)

```bash
pip install -e .
```

### Use

```python
import torch
import iris_hamming as ih

data = torch.randint(0, 2**31, (1024, 400), dtype=torch.int32, device="cuda")
mask = torch.full((1024, 400), 0x7FFFFFFF, dtype=torch.int32, device="cuda")

D, pairs, match_count = ih.masked_hamming_cuda(
    data, mask,
    write_output=False,
    collect_pairs=False,
    threshold=1.0,
    max_pairs=0,
)
```

### Tests

```bash
pytest -q
```


