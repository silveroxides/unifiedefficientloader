import argparse
import logging
import time
import sys
import torch
import warnings
from torch.utils.data import Dataset, DataLoader
from unifiedefficientloader import UnifiedDataLoader

# Silence internal PyTorch dataloader deprecation warnings
warnings.filterwarnings("ignore", message=".*The argument 'device' of Tensor.*")

def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

class SyntheticDataset(Dataset):
    """Generates synthetic image tensors in memory for pure throughput testing."""
    def __init__(self, size, shape=(3, 256, 256), dtype=torch.float32):
        self.size = size
        self.shape = shape
        self.dtype = dtype

        # Pre-allocate one tensor and share it to avoid allocation overhead in getitem
        self.tensor = torch.randn(*shape, dtype=dtype)
        self.label = torch.tensor(1, dtype=torch.long)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "image": self.tensor,
            "label": self.label
        }

def run_benchmark(loader, name, total_items, device="cuda"):
    logger = logging.getLogger(__name__)
    logger.info(f"--- Starting {name} Benchmark ---")

    start_time = time.time()

    total_batches = 0
    total_bytes = 0

    for batch in loader:
        images = batch["image"]
        # Ensure we wait for GPU transfers if any
        if getattr(images, 'is_cuda', False) or getattr(images, 'device', torch.device('cpu')).type == 'cuda':
             torch.cuda.current_stream().synchronize()

        total_batches += 1
        total_bytes += images.numel() * images.element_size()

    end_time = time.time()
    duration = end_time - start_time

    mb = total_bytes / (1024 * 1024)
    gb = mb / 1024
    throughput = gb / duration if duration > 0 else 0
    items_per_sec = total_items / duration if duration > 0 else 0

    logger.info(f"[{name}] Completed in {duration:.4f}s")
    logger.info(f"[{name}] Total data processed: {gb:.2f} GB in {total_batches} batches")
    logger.info(f"[{name}] Throughput: {throughput:.2f} GB/s ({items_per_sec:.0f} items/s)")

    return {
        "duration": duration,
        "throughput": throughput,
        "items_per_sec": items_per_sec
    }

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark comparison: UnifiedDataLoader vs torch DataLoader"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose debug logging"
    )
    parser.add_argument(
        "--size", type=int, default=10000, help="Number of items in synthetic dataset"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of background workers"
    )
    parser.add_argument(
        "--shape", type=str, default="3,256,256", help="Comma-separated image shape"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device"
    )
    args = parser.parse_args()

    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    shape = tuple(int(x) for x in args.shape.split(","))
    device = torch.device(args.device)

    logger.info(f"Configuration: Size={args.size}, Batch={args.batch_size}, Workers={args.workers}, Shape={shape}")

    dataset = SyntheticDataset(args.size, shape=shape)

    # 1. Standard DataLoader (No Pinning)
    logger.info("Preparing Standard PyTorch DataLoader (pin_memory=False)")
    torch_loader_no_pin = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=False
    )

    # 2. Standard DataLoader (Pinned)
    logger.info("Preparing Standard PyTorch DataLoader (pin_memory=True)")
    torch_loader_pin = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True
    )

    # 3. UnifiedDataLoader (CPU threaded)
    logger.info("Preparing UnifiedDataLoader (CPU Threaded)")
    unified_loader_cpu = UnifiedDataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=False,
        direct_gpu=False
    )

    # 4. UnifiedDataLoader (Direct GPU)
    unified_loader_gpu = None
    if device.type == "cuda":
        logger.info("Preparing UnifiedDataLoader (Direct GPU)")
        unified_loader_gpu = UnifiedDataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            direct_gpu=True,
            device=device.type
        )

    # Run Benchmarks
    results = {}

    results["PyTorch_NoPin"] = run_benchmark(torch_loader_no_pin, "PyTorch DataLoader (Unpinned)", args.size, device)
    results["PyTorch_Pin"] = run_benchmark(torch_loader_pin, "PyTorch DataLoader (Pinned)", args.size, device)
    results["Unified_CPU"] = run_benchmark(unified_loader_cpu, "UnifiedDataLoader (CPU Threaded)", args.size, device)

    if unified_loader_gpu:
        results["Unified_GPU"] = run_benchmark(unified_loader_gpu, "UnifiedDataLoader (Direct GPU)", args.size, device)

    # Summary Table
    logger.info("=========================================================")
    logger.info("                 BENCHMARK SUMMARY                       ")
    logger.info("=========================================================")
    logger.info(f"{'Loader Type':<35} | {'Time (s)':<10} | {'GB/s':<8} | {'Items/s':<8}")
    logger.info("-" * 65)
    for name, stats in results.items():
        logger.info(f"{name:<35} | {stats['duration']:<10.4f} | {stats['throughput']:<8.2f} | {stats['items_per_sec']:<8.0f}")
    logger.info("=========================================================")

if __name__ == "__main__":
    main()
