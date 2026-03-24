# UnifiedEfficientLoader Testing Examples

This document provides clear examples of how to use `tests/manual_test.py` for various testing and benchmarking scenarios. This script is designed to test the performance and correctness of `UnifiedSafetensorsLoader` and its pinned transfer capabilities.

## Prerequisites

Ensure you have created a dummy safetensors file for testing if you don't have a real one:

```bash
python create_dummy.py
```

## Basic Usage Scenarios

### 1. Sequential Loading (Baseline)
This is the simplest way to run the benchmark. It loads tensors one by one and transfers them to the GPU sequentially.

```bash
python tests/manual_test.py dummy.safetensors
```
*   **What it does:** Performs a standard sequential load, transfer to GPU, and return to CPU for every tensor.
*   **Use case:** Establishing a baseline performance for comparison.

### 2. Quick Verification (Limiting Tensors)
If you just want to verify that the script works without processing the entire file:

```bash
python tests/manual_test.py dummy.safetensors --limit 10
```
*   **What it does:** Limits the benchmark to the first 10 U8 (dictionary) tensors and the first 10 standard tensors.

### 3. Debugging and Verbose Logging
To see detailed logs including exactly which tensors are being loaded and converted:

```bash
python tests/manual_test.py dummy.safetensors --debug --limit 2
```
*   **What it does:** Enables `DEBUG` level logging for more granular information.

## Advanced Benchmarking Scenarios

### 4. Asynchronous Loading via DataLoader
Use the PyTorch `DataLoader` integration to overlap I/O and processing.

```bash
python tests/manual_test.py dummy.safetensors --async-batch 16
```
*   **What it does:** Uses `UnifiedSafetensorsLoader.to_iterable_dataset()` to feed a `DataLoader`. Tensors are loaded in background threads in batches of 16.
*   **Use case:** Benchmarking performance when I/O is the bottleneck.

### 5. High-Throughput Batch Pinning and Transfer
This mode maximizes GPU throughput by pinning a large batch of tensors in memory before transferring them to the GPU.

```bash
python tests/manual_test.py dummy.safetensors --async-batch 64 --batch-transfer
```
*   **What it does:**
    1.  Loads a batch of 64 tensors asynchronously.
    2.  `DataLoader` automatically pins the memory for the entire batch.
    3.  `manual_test.py` transfers the entire pinned batch to the GPU in one go.
*   **Use case:** Stress testing GPU transfer speeds and memory efficiency.

### 6. Sub-Batch GPU Transfers
To control how many tensors are sent to the GPU at once within a larger `async-batch`, use `--transfer-count`.

```bash
python tests/manual_test.py dummy.safetensors --async-batch 128 --batch-transfer --transfer-count 32
```
*   **What it does:** Loads and pins 128 tensors at a time, but transfers them to the GPU in smaller sub-batches of 32.
*   **Use case:** Optimizing for GPU memory limits while still benefiting from large-batch asynchronous loading and pinning.

## Comparison Benchmarking

To compare the memory-efficient loader against the standard preloading approach, use the following two scripts.

### 1. Unified Efficient Loader (Streaming Mode)
This uses `UnifiedSafetensorsLoader` in low-memory mode (streaming from disk).

```bash
python tests/manual_test.py dummy.safetensors
```
*   **Performance:** Lower RAM usage, faster startup (lazy loading), potentially slower total roundtrip depending on disk speed.

### 2. Standard Safetensors Loader (Preload Mode)
This uses the baseline script `tests/benchmark_standard.py` which preloads everything to RAM first.

```bash
python tests/benchmark_standard.py dummy.safetensors
```
*   **Performance:** Higher RAM usage, slower startup (full preload), faster transfer loop (data already in RAM).

## Argument Reference Summary

| Argument | Description | Default |
| :--- | :--- | :--- |
| `file` | Path to the safetensors file | (Required) |
| `--debug` | Enable verbose debug logging | `False` |
| `--device` | Target device for pinned transfers | `cuda` (if available) |
| `--limit` | Max tensors to process per category (0 for all) | `0` |
| `--chunk-size` | Number of tensors to process before logging summary | `100` |
| `--async-batch` | Use PyTorch DataLoader with this batch size | `0` (Disabled) |
| `--batch-transfer` | Pin entire batch first, then transfer to GPU | `False` |
| `--transfer-count` | Sub-batch size for GPU transfers (0 for all) | `0` |

## Understanding the Results

At the end of each run, the script provides a **GRAND TOTAL SUMMARY**:
- **Total U8 Dictionaries:** Count, size, and load/decode times for JSON-encoded metadata tensors.
- **Total Standard Tensors:** Count, total shape, total size, and detailed timing for:
    - **Shape/NDIM Time:** Overhead of fetching metadata from the header.
    - **Data Loading Time:** Time spent reading raw bytes from disk (and pinning if using `async-batch`).
    - **Pinned GPU Transfer:** Time to move pinned CPU memory to GPU.
    - **CPU Return Transfer:** Time to move data back to CPU (for verification).
    - **Memory Cleanup Time:** Time spent on manual memory management/deletion.
- **FULL ROUNDTRIP TIME:** Sum of all standard tensor processing steps.
- **Total Script Time:** Wall-clock time for the entire execution.
