import argparse
import gc
import logging
import os
import shutil
import threading
import time
import warnings

import torch
from torch.utils.data import Dataset, DataLoader

from unifiedefficientloader import UnifiedDataLoader, UnifiedSafetensorsLoader

warnings.filterwarnings("ignore", message=".*The argument 'device' of Tensor.*")

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

TMP_DIR = "tmp_bench"
SAFETENSORS_PATH = os.path.join(TMP_DIR, "dataset.safetensors")


def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _rss_mb():
    if not _PSUTIL:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def generate_disk_data(n_items, shape, dtype=torch.float32):
    """Write n_items .pt files and one .safetensors file to TMP_DIR."""
    logger = logging.getLogger(__name__)
    os.makedirs(TMP_DIR, exist_ok=True)

    logger.info(f"Generating {n_items} items, shape={shape} to {TMP_DIR}/")

    for i in range(n_items):
        t = torch.randn(*shape, dtype=dtype)
        torch.save(t, os.path.join(TMP_DIR, f"item_{i}.pt"))

    try:
        from safetensors.torch import save_file
        tensors = {f"item_{i}": torch.randn(*shape, dtype=dtype) for i in range(n_items)}
        save_file(tensors, SAFETENSORS_PATH)
        logger.info(f"Wrote {SAFETENSORS_PATH}")
    except ImportError:
        logger.warning("safetensors not installed — skipping .safetensors generation")

    logger.info("Disk data generation complete.")


def cleanup_disk_data():
    """Remove TMP_DIR. Retries on Windows file-lock with brief delay."""
    logger = logging.getLogger(__name__)
    if not os.path.exists(TMP_DIR):
        return
    for attempt in range(3):
        try:
            shutil.rmtree(TMP_DIR)
            logger.info(f"Cleaned up {TMP_DIR}/")
            return
        except PermissionError as e:
            if attempt < 2:
                import time as _t
                _t.sleep(0.5)
            else:
                logger.warning(f"Could not fully clean up {TMP_DIR}: {e}")


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class DiskDatasetPT(Dataset):
    """Loads individual .pt files — one torch.load per __getitem__."""
    def __init__(self, n_items, size):
        self.n_items = n_items
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        path = os.path.join(TMP_DIR, f"item_{idx % self.n_items}.pt")
        tensor = torch.load(path, weights_only=True)
        return {"image": tensor, "label": torch.tensor(idx % self.n_items, dtype=torch.long)}


class DiskDatasetSafetensors(Dataset):
    """
    Loads tensors from a single .safetensors file via UnifiedSafetensorsLoader
    in low-memory streaming mode. One tensor read per __getitem__. No bulk preload.
    Uses threading.local() so each worker thread opens its own file handle —
    no contention on seek/read across concurrent workers.
    """
    def __init__(self, n_items, size, use_mmap=False):
        self.n_items = n_items
        self.size = size
        self.use_mmap = use_mmap
        self._thread_local = threading.local()
        self._all_loaders_lock = threading.Lock()
        self._all_loaders = []

    def _get_loader(self):
        if not hasattr(self._thread_local, "loader") or self._thread_local.loader is None:
            loader = UnifiedSafetensorsLoader(
                SAFETENSORS_PATH,
                low_memory=True,
                use_mmap=self.use_mmap,
            )
            self._thread_local.loader = loader
            with self._all_loaders_lock:
                self._all_loaders.append(loader)
        return self._thread_local.loader

    def close(self):
        """Close all per-thread loaders. Call before cleanup on Windows."""
        with self._all_loaders_lock:
            for loader in self._all_loaders:
                try:
                    loader.close()
                except Exception:
                    pass
            self._all_loaders.clear()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        loader = self._get_loader()
        key = f"item_{idx % self.n_items}"
        tensor = loader.get_tensor(key)
        return {"image": tensor.clone(), "label": torch.tensor(idx % self.n_items, dtype=torch.long)}

    def __getstate__(self):
        # threading.local is not picklable — strip it for multiprocessing safety
        state = self.__dict__.copy()
        state["_thread_local"] = None
        state["_all_loaders"] = []
        state["_all_loaders_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._thread_local = threading.local()
        self._all_loaders = []
        self._all_loaders_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(loader, name, total_items, device):
    logger = logging.getLogger(__name__)
    logger.info(f"--- Starting: {name} ---")

    rss_before = _rss_mb()
    start_time = time.perf_counter()

    total_batches = 0
    total_bytes = 0

    for i, batch in enumerate(loader):
        images = batch["image"]

        if images.device.type == "cuda":
            torch.cuda.current_stream().synchronize()

        total_batches += 1
        total_bytes += images.numel() * images.element_size()

        del batch
        if i % 20 == 0:
            gc.collect()

    duration = time.perf_counter() - start_time
    rss_after = _rss_mb()

    gb = total_bytes / (1024 ** 3)
    throughput = gb / duration if duration > 0 else 0
    items_per_sec = total_items / duration if duration > 0 else 0

    logger.info(
        f"[{name}] {duration:.4f}s | {gb:.3f} GB | "
        f"{throughput:.2f} GB/s | {items_per_sec:.0f} items/s"
    )
    if rss_before is not None:
        logger.info(
            f"[{name}] RSS: {rss_before:.0f} MB -> {rss_after:.0f} MB "
            f"(delta {rss_after - rss_before:+.0f} MB)"
        )

    return {
        "duration": duration,
        "throughput": throughput,
        "items_per_sec": items_per_sec,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: UnifiedDataLoader vs torch.utils.data.DataLoader"
    )
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    parser.add_argument("--size", type=int, default=5000, help="Total items to iterate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=4, help="Background workers")
    parser.add_argument("--shape", type=str, default="1,512,512", help="Tensor shape C,H,W")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device",
    )
    args = parser.parse_args()

    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    shape = tuple(int(x) for x in args.shape.split(","))
    device = torch.device(args.device)
    n_items = args.batch_size * 3

    logger.info(
        f"Config: size={args.size}, batch={args.batch_size}, workers={args.workers}, "
        f"shape={shape}, device={device}, n_disk_items={n_items}"
    )

    # Track dataset instances that hold file handles so we can close them
    # before cleanup — necessary on Windows to release file locks.
    datasets_to_close = []

    try:
        generate_disk_data(n_items, shape)

        results = {}

        # 1. PyTorch DataLoader — unpinned .pt
        ds_pt = DiskDatasetPT(n_items, args.size)
        loader = DataLoader(
            ds_pt,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=False,
        )
        results["PyTorch_NoPin"] = run_benchmark(
            loader, "PyTorch DataLoader (unpinned .pt)", args.size, device
        )
        del loader

        # 2. PyTorch DataLoader — pinned .pt
        ds_pt2 = DiskDatasetPT(n_items, args.size)
        loader = DataLoader(
            ds_pt2,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
        )
        results["PyTorch_Pin"] = run_benchmark(
            loader, "PyTorch DataLoader (pinned .pt)", args.size, device
        )
        del loader

        # 3. UnifiedDataLoader — CPU threaded, safetensors
        ds_st = DiskDatasetSafetensors(n_items, args.size, use_mmap=False)
        datasets_to_close.append(ds_st)
        loader = UnifiedDataLoader(
            ds_st,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=False,
            direct_gpu=False,
        )
        results["Unified_CPU"] = run_benchmark(
            loader, "UnifiedDataLoader (CPU, safetensors)", args.size, device
        )
        del loader

        # 4. UnifiedDataLoader — Direct GPU, safetensors
        if device.type == "cuda":
            ds_st2 = DiskDatasetSafetensors(n_items, args.size, use_mmap=False)
            datasets_to_close.append(ds_st2)
            loader = UnifiedDataLoader(
                ds_st2,
                batch_size=args.batch_size,
                num_workers=args.workers,
                direct_gpu=True,
                device=device.type,
            )
            results["Unified_GPU"] = run_benchmark(
                loader, "UnifiedDataLoader (direct GPU, safetensors)", args.size, device
            )
            del loader
        else:
            logger.warning("Skipping direct GPU benchmark — no CUDA device.")

        # 5. UnifiedDataLoader — MMAP, safetensors
        try:
            from unifiedefficientloader.uel import control
            control.init()
            if control.lib is None:
                raise RuntimeError("uel lib not loaded")

            ds_mmap = DiskDatasetSafetensors(n_items, args.size, use_mmap=True)
            # Probe: open loader to confirm mmap actually initialised (not fallen back)
            probe_loader = ds_mmap._get_loader()
            if not probe_loader.use_mmap:
                ds_mmap.close()
                raise RuntimeError("mmap fallback triggered — native lib unavailable")

            datasets_to_close.append(ds_mmap)
            loader = UnifiedDataLoader(
                ds_mmap,
                batch_size=args.batch_size,
                num_workers=args.workers,
                direct_gpu=device.type == "cuda",
                device=device.type if device.type == "cuda" else "cpu",
            )
            label = (
                "UnifiedDataLoader (mmap + direct GPU, safetensors)"
                if device.type == "cuda"
                else "UnifiedDataLoader (mmap CPU, safetensors)"
            )
            results["Unified_MMAP"] = run_benchmark(loader, label, args.size, device)
            del loader
        except Exception as e:
            logger.warning(f"Skipping MMAP benchmark: {e}")

        # Summary table
        logger.info("=" * 70)
        logger.info("                      BENCHMARK SUMMARY")
        logger.info("=" * 70)
        logger.info(f"{'Loader':<44} | {'Time(s)':<8} | {'GB/s':<6} | Items/s")
        logger.info("-" * 70)
        for label, stats in results.items():
            logger.info(
                f"{label:<44} | {stats['duration']:<8.4f} | "
                f"{stats['throughput']:<6.2f} | {stats['items_per_sec']:.0f}"
            )
        logger.info("=" * 70)

    finally:
        # Close all safetensors loaders before rmtree — Windows file lock release
        for ds in datasets_to_close:
            try:
                ds.close()
            except Exception:
                pass
        gc.collect()
        cleanup_disk_data()


if __name__ == "__main__":
    main()
