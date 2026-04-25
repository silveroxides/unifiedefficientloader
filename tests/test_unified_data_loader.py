"""Tests for UnifiedDataLoader."""
import pytest
import torch
from unifiedefficientloader import UnifiedDataLoader, UnifiedSafetensorsLoader
import os

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

def test_unified_data_loader_load_fn():
    def my_load_fn(idx):
        return {"a": torch.tensor([idx]), "b": torch.tensor([idx * 2])}
        
    loader = UnifiedDataLoader(load_fn=my_load_fn, length=10, batch_size=2, num_workers=2)
    batches = list(loader)
    
    assert len(batches) == 5
    assert batches[0]["a"].shape == (2, 1)
    assert batches[0]["a"][0].item() == 0
    assert batches[0]["a"][1].item() == 1
    assert batches[0]["b"][1].item() == 2

def test_unified_data_loader_safetensors_fast_path(tmp_path):
    from safetensors.torch import save_file
    st_path = os.path.join(tmp_path, "test.safetensors")
    
    tensors = {
        "item_0": torch.ones(2, 2) * 0,
        "item_1": torch.ones(2, 2) * 1,
        "item_2": torch.ones(2, 2) * 2,
        "item_3": torch.ones(2, 2) * 3,
    }
    save_file(tensors, st_path)
    
    st_loader = UnifiedSafetensorsLoader(st_path, low_memory=True)
    
    loader = UnifiedDataLoader(st_loader, batch_size=2, num_workers=2)
    batches = list(loader)
    
    assert len(batches) == 2
    assert batches[0].shape == (2, 2, 2) # Collated 2 items of shape (2,2)
    
    # We don't guarantee exact order if shuffle=False right now because async stream yields 
    # as things complete, but with batch=2 and 4 items, let's just check the values.
    # Async stream preserves order if prefetch is bounded properly but let's check broadly:
    
    all_vals = torch.cat(batches, dim=0).flatten().unique().tolist()
    assert set(all_vals) == {0.0, 1.0, 2.0, 3.0}
