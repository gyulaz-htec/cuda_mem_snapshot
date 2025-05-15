# cuda_mem_snapshot

This is a small Python library that provides a context manager for CUDA memory snaphsoting using PyTorch.
Uses torch.cuda.memory._record_memory_history and torch.cuda.memory._dump_snapshot and saves `.pickle` files as snapshot for each GPU separately.

## Installation

```bash
pip install .
```

## Usage

```python
from cuda_mem_snapshot import CudaMemorySnapshotManager

save_dir = "/path/to/save"
snapshot_name = "memory_snapshot"

with CudaMemorySnapshotManager(save_dir=save_dir, snapshot_name=snapshot_name, enable=True):
    # Your code that uses torch and CUDA
```