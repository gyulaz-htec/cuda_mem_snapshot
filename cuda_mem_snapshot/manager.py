import torch

class CudaMemorySnapshotManager:
    def __init__(self,  save_dir, snapshot_name, enable=True):
        rank = int(os.environ["LOCAL_RANK"])
        self.snapshot_path = f"{save_dir}/{snapshot_name}_{rank}.pickle"
        self.enable = enable

    def __enter__(self):
        torch.cuda.memory._record_memory_history(enabled='all' if self.enable else None)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.memory._dump_snapshot(self.snapshot_path)
        torch.cuda.memory._record_memory_history(enabled=None)
        return False
