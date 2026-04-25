"""Tests for UnifiedDataLoader."""
import pytest
import torch
from unifiedefficientloader import UnifiedDataLoader

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "image": torch.ones(3, 16, 16, dtype=torch.float32) * idx,
            "label": torch.tensor(idx, dtype=torch.long)
        }

def test_unified_data_loader_basic():
    dataset = DummyDataset(size=10)
    loader = UnifiedDataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    
    batches = list(loader)
    assert len(batches) == 5
    
    first_batch = batches[0]
    assert first_batch["image"].shape == (2, 3, 16, 16)
    assert first_batch["label"].shape == (2,)
    assert first_batch["label"][0].item() == 0
    assert first_batch["label"][1].item() == 1

def test_unified_data_loader_drop_last():
    dataset = DummyDataset(size=11)
    loader = UnifiedDataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, drop_last=True)
    
    batches = list(loader)
    assert len(batches) == 5

def test_unified_data_loader_workers():
    dataset = DummyDataset(size=10)
    loader = UnifiedDataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)
    
    batches = list(loader)
    assert len(batches) == 5

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_unified_data_loader_direct_gpu():
    dataset = DummyDataset(size=10)
    loader = UnifiedDataLoader(dataset, batch_size=2, shuffle=False, num_workers=2, direct_gpu=True)
    
    batches = list(loader)
    assert len(batches) == 5
    first_batch = batches[0]
    assert first_batch["image"].is_cuda
    assert first_batch["label"].is_cuda
    assert first_batch["label"][0].item() == 0
    assert first_batch["label"][1].item() == 1
