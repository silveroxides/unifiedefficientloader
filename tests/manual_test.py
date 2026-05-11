import argparse
import contextlib
import logging
import os
import time
import sys
import math
import torch
import warnings
from unifiedefficientloader import (
    UnifiedSafetensorsLoader,
    tensor_to_dict,
    transfer_to_gpu_pinned,
    IncrementalSafetensorsWriter,
)

# Silence internal PyTorch dataloader deprecation warnings
warnings.filterwarnings("ignore", message=".*The argument 'device' of Tensor.*")


def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Manual test script and benchmark for unifiedefficientloader"
    )
    parser.add_argument("file", help="Path to the safetensors file to load")
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose debug logging"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device for pinned transfer tests",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of tensors to test per category (0 for no limit)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Number of tensors to process before logging a summary chunk",
    )
    parser.add_argument(
        "--async-batch",
        type=int,
        default=0,
        help="If >0, uses async_stream with this batch size instead of sequential load",
    )
    parser.add_argument(
        "--direct-gpu",
        action="store_true",
        help="Enable direct-to-GPU streaming pipeline",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        default=True,
        help="Enable memory-efficient streaming mode",
    )
    parser.add_argument(
        "--no-low-memory",
        action="store_false",
        dest="low_memory",
        help="Disable memory-efficient streaming mode (preload everything)",
    )
    parser.add_argument(
        "--batch-transfer",
        action="store_true",
        help="If set, enables sequential pinning in main thread via async_stream",
    )
    parser.add_argument(
        "--mmap",
        action="store_true",
        help="Benchmark mmap mode (use_mmap=True): zero-copy OS page-cache mapping",
    )
    # ── Writer flags ────────────────────────────────────────────────────────
    parser.add_argument(
        "--writer-test",
        action="store_true",
        help=(
            "Add IncrementalSafetensorsWriter to the standard-tensor benchmark loop. "
            "Each tensor is written to an output file AFTER its GPU round-trip, "
            "benchmarking the full load->GPU->CPU->write pipeline in a single pass. "
            "Sequential mode uses write(); async_stream mode uses write_batch(). "
            "Output file is placed next to the input file and deleted when done."
        ),
    )
    parser.add_argument(
        "--writer-workers",
        type=int,
        default=4,
        help="Background worker threads for IncrementalSafetensorsWriter (default: 4)",
    )
    parser.add_argument(
        "--writer-verify",
        action="store_true",
        help=(
            "After the writer context closes, re-open the output file and verify "
            "tensor shapes, dtypes, and values (torch.allclose) against the source"
        ),
    )
    args = parser.parse_args()

    filepath = args.file
    debug = args.debug
    low_memory = args.low_memory
    no_low_memory = not args.low_memory
    chunk_size = args.chunk_size
    async_batch = args.async_batch
    direct_gpu = args.direct_gpu
    batch_transfer = args.batch_transfer
    setup_logging(debug)
    logger = logging.getLogger(__name__)
    limit = args.limit if args.limit > 0 else "ALL"
    device = torch.device(args.device)
    logger.info(
        f"Running unifiedefficientloader benchmark on file: {filepath} | Limit: {limit} tensors per category | Device: {device} | Low Memory: {args.low_memory}"
    )

    if direct_gpu:
        if no_low_memory:
            logger.warning(
                "direct_gpu=True requires low_memory=True. Forcing low_memory=True."
            )
            low_memory = True
        if async_batch == 0:
            logger.info(
                "direct_gpu=True requires async_stream. Forcing --async-batch=1."
            )
            async_batch = 1

    logger.info(f"--- Starting Benchmark for {filepath} ---")
    if direct_gpu:
        logger.info("[Benchmark Mode] Direct-to-GPU Pipeline Active")
    if args.writer_test:
        logger.info(
            "[Benchmark Mode] Writer Test Active — write() / write_batch() "
            "called after GPU round-trip inside the standard-tensor loop"
        )
    script_start_time = time.time()

    # ── Grand Totals (loader) ─────────────────────────────────────────────────
    total_u8_tensors = 0
    total_u8_bytes = 0
    total_u8_load_time = 0.0
    total_u8_convert_time = 0.0

    total_std_tensors = 0
    total_std_elements = 0
    total_std_bytes = 0
    total_std_shape_time = 0.0
    total_std_load_time = 0.0
    total_std_transfer_gpu_time = 0.0
    total_std_transfer_cpu_time = 0.0
    total_std_mark_time = 0.0

    # ── Grand Totals (writer — only when --writer-test) ───────────────────────
    total_writer_enqueue_time = 0.0
    total_writer_finalize_time = 0.0
    writer_output_path = None
    writer_verify_ok = 0
    writer_verify_fail = 0

    # 1. Benchmark Header Loading
    start_time = time.time()
    try:
        loader = UnifiedSafetensorsLoader(
            filepath, low_memory=low_memory, direct_gpu=direct_gpu
        )
    except Exception as e:
        logger.error(f"Failed to load file {filepath}: {e}")
        sys.exit(1)

    header_time = time.time() - start_time
    logger.info(
        f"[Benchmark] Header initialization (low_memory={low_memory}) took {header_time:.5f} seconds"
    )

    with loader:
        # 2. Benchmark Finding U8 Dictionary Tensors (1D only)
        start_time = time.time()
        uint8_tensor_keys = [
            k
            for k, v in loader._header.items()
            if isinstance(v, dict)
            and v.get("dtype") == "U8"
            and len(v.get("shape", [])) == 1
        ]
        find_u8_time = time.time() - start_time
        logger.info(
            f"[Benchmark] Scanning header for 1D U8 tensors took {find_u8_time:.5f} seconds. Found {len(uint8_tensor_keys)}."
        )

        # 3. Benchmark Loading and Converting U8 Tensors
        test_u8_keys = (
            uint8_tensor_keys[: args.limit] if args.limit > 0 else uint8_tensor_keys
        )
        logger.info(f"--- Benchmarking {len(test_u8_keys)} U8 tensor(s) ---")

        chunk_count = 0
        chunk_load_time = 0.0
        chunk_convert_time = 0.0
        chunk_bytes = 0

        for idx, key in enumerate(test_u8_keys, 1):
            # Load
            start_time = time.time()
            tensor = loader.get_tensor(key)
            l_time = time.time() - start_time
            chunk_load_time += l_time
            total_u8_load_time += l_time

            b_size = tensor.numel() * tensor.element_size()
            chunk_bytes += b_size
            total_u8_bytes += b_size

            # Convert
            start_time = time.time()
            try:
                extracted_dict = tensor_to_dict(tensor)
                c_time = time.time() - start_time
                chunk_convert_time += c_time
                total_u8_convert_time += c_time
            except Exception as e:
                logger.warning(f"Failed to decode '{key}' as JSON dict: {e}")

            chunk_count += 1
            total_u8_tensors += 1

            if chunk_count >= chunk_size or idx == len(test_u8_keys):
                logger.info(
                    f"[U8 Chunk Summary] Processed {chunk_count} tensors ({chunk_bytes / 1024:.2f} KB) | "
                    f"Load: {chunk_load_time:.4f}s | Decode: {chunk_convert_time:.4f}s"
                )
                chunk_count = 0
                chunk_load_time = 0.0
                chunk_convert_time = 0.0
                chunk_bytes = 0

        # 4. Benchmark Standard Tensors
        standard_keys = [k for k in loader.keys() if k not in uint8_tensor_keys]
        if standard_keys:
            test_keys = standard_keys[: args.limit] if args.limit > 0 else standard_keys

            # Build writer context: IncrementalSafetensorsWriter when --writer-test
            # is active, nullcontext otherwise.  Both branches use the same loop code.
            if args.writer_test:
                writer_output_path = (
                    os.path.splitext(os.path.abspath(filepath))[0]
                    + "_writer_test.safetensors"
                )
                logger.info(
                    f"[Writer] Output: {writer_output_path} | Workers: {args.writer_workers}"
                )
                _writer_ctx = IncrementalSafetensorsWriter(
                    writer_output_path,
                    metadata=loader.metadata(),
                    max_workers=args.writer_workers,
                )
            else:
                _writer_ctx = contextlib.nullcontext()

            # loop_end_time is stamped immediately after the inner loop exits so
            # that time.time() after the with-block measures __exit__ duration.
            loop_end_time = None

            if async_batch > 0:
                logger.info(
                    f"--- Benchmarking {len(test_keys)} standard tensor(s) ASYNCHRONOUSLY "
                    f"via async_stream (batch={async_batch}, pin={batch_transfer}"
                    + (", +write_batch" if args.writer_test else "") + ") ---"
                )

                stream_start_time = time.time()

                chunk_count = 0
                chunk_shape_time = 0.0
                chunk_load_time = 0.0
                chunk_transfer_time = 0.0
                chunk_transfer_back_time = 0.0
                chunk_mark_time = 0.0
                chunk_enqueue_time = 0.0
                chunk_bytes = 0
                chunk_elements = 0

                if direct_gpu:
                    pin_memory = True
                else:
                    pin_memory = batch_transfer
                stream = loader.async_stream(
                    test_keys, batch_size=async_batch, pin_memory=pin_memory
                )

                with _writer_ctx as writer:
                    for batch in stream:
                        # Collect post-GPU-round-trip tensors for write_batch
                        processed_batch = []

                        for k, tensor in batch:
                            start_time = time.time()
                            shape = loader.get_shape(k)
                            s_time = time.time() - start_time
                            chunk_shape_time += s_time
                            total_std_shape_time += s_time

                            elements = math.prod(shape) if shape else 0
                            chunk_elements += elements
                            total_std_elements += elements

                            b_size = tensor.numel() * tensor.element_size()
                            chunk_bytes += b_size
                            total_std_bytes += b_size

                            start_time = time.time()
                            gpu_tensor = transfer_to_gpu_pinned(tensor, device=device)
                            t_time = time.time() - start_time
                            chunk_transfer_time += t_time
                            total_std_transfer_gpu_time += t_time

                            start_time = time.time()
                            if gpu_tensor.device.type == "cuda":
                                torch.cuda.current_stream().synchronize()
                            cpu_tensor = gpu_tensor.to("cpu")
                            tb_time = time.time() - start_time
                            chunk_transfer_back_time += tb_time
                            total_std_transfer_cpu_time += tb_time

                            start_time = time.time()
                            loader.mark_processed(k)
                            m_time = time.time() - start_time
                            chunk_mark_time += m_time
                            total_std_mark_time += m_time

                            if args.writer_test:
                                # Hold cpu_tensor until write_batch below
                                processed_batch.append((k, cpu_tensor))
                            else:
                                del cpu_tensor

                            del tensor, gpu_tensor
                            if direct_gpu:
                                import gc
                                gc.collect()

                            chunk_count += 1
                            total_std_tensors += 1

                            if chunk_count >= chunk_size or total_std_tensors == len(
                                test_keys
                            ):
                                total_chunk_time = time.time() - stream_start_time
                                approx_load = max(
                                    0,
                                    total_chunk_time
                                    - (
                                        chunk_transfer_time
                                        + chunk_transfer_back_time
                                        + chunk_mark_time
                                        + chunk_shape_time
                                        + chunk_enqueue_time
                                    ),
                                )
                                chunk_load_time = approx_load
                                total_std_load_time += approx_load

                                msg = (
                                    f"[Async Chunk Summary] Processed {chunk_count} tensors "
                                    f"(Total Shape: {chunk_elements}, "
                                    f"{chunk_bytes / (1024 * 1024):.2f} MB) | "
                                    f"Async Load/Pin: {approx_load:.4f}s | "
                                    f"Transfer to GPU: {chunk_transfer_time:.4f}s | "
                                    f"Transfer to CPU: {chunk_transfer_back_time:.4f}s"
                                )
                                if args.writer_test:
                                    msg += f" | Write Enqueue: {chunk_enqueue_time:.4f}s"
                                logger.info(msg)

                                chunk_count = 0
                                chunk_shape_time = 0.0
                                chunk_load_time = 0.0
                                chunk_transfer_time = 0.0
                                chunk_transfer_back_time = 0.0
                                chunk_mark_time = 0.0
                                chunk_enqueue_time = 0.0
                                chunk_bytes = 0
                                chunk_elements = 0
                                stream_start_time = time.time()

                        # write_batch with the post-GPU-round-trip tensors
                        if args.writer_test and processed_batch:
                            t0 = time.time()
                            writer.write_batch(processed_batch)
                            e_time = time.time() - t0
                            chunk_enqueue_time += e_time
                            total_writer_enqueue_time += e_time
                            for _, ct in processed_batch:
                                del ct

                        del processed_batch
                        batch.clear()
                        del batch

                    loop_end_time = time.time()

            else:
                logger.info(
                    f"--- Benchmarking {len(test_keys)} standard tensor(s) SEQUENTIALLY"
                    + (" +writer" if args.writer_test else "") + " ---"
                )

                chunk_count = 0
                chunk_shape_time = 0.0
                chunk_load_time = 0.0
                chunk_transfer_time = 0.0
                chunk_transfer_back_time = 0.0
                chunk_mark_time = 0.0
                chunk_enqueue_time = 0.0
                chunk_bytes = 0
                chunk_elements = 0

                with _writer_ctx as writer:
                    for idx, sample_key in enumerate(test_keys, 1):
                        # Shape & Ndim
                        start_time = time.time()
                        shape = loader.get_shape(sample_key)
                        s_time = time.time() - start_time
                        chunk_shape_time += s_time
                        total_std_shape_time += s_time

                        elements = math.prod(shape) if shape else 0
                        chunk_elements += elements
                        total_std_elements += elements

                        # Load
                        start_time = time.time()
                        tensor = loader.get_tensor(sample_key)
                        l_time = time.time() - start_time
                        chunk_load_time += l_time
                        total_std_load_time += l_time

                        b_size = tensor.numel() * tensor.element_size()
                        chunk_bytes += b_size
                        total_std_bytes += b_size

                        # Transfer to GPU
                        start_time = time.time()
                        gpu_tensor = transfer_to_gpu_pinned(tensor, device=device)
                        t_time = time.time() - start_time
                        chunk_transfer_time += t_time
                        total_std_transfer_gpu_time += t_time

                        # Transfer back to CPU
                        start_time = time.time()
                        if gpu_tensor.device.type == "cuda":
                            torch.cuda.current_stream().synchronize()
                        cpu_tensor = gpu_tensor.to("cpu")
                        tb_time = time.time() - start_time
                        chunk_transfer_back_time += tb_time
                        total_std_transfer_cpu_time += tb_time

                        # Write result of GPU round-trip to output file
                        if args.writer_test:
                            t0 = time.time()
                            writer.write(sample_key, cpu_tensor)
                            e_time = time.time() - t0
                            chunk_enqueue_time += e_time
                            total_writer_enqueue_time += e_time

                        # Mark Processed
                        start_time = time.time()
                        loader.mark_processed(sample_key)
                        m_time = time.time() - start_time
                        chunk_mark_time += m_time
                        total_std_mark_time += m_time

                        del tensor, gpu_tensor, cpu_tensor

                        chunk_count += 1
                        total_std_tensors += 1

                        if chunk_count >= chunk_size or idx == len(test_keys):
                            msg = (
                                f"[Standard Chunk Summary] Processed {chunk_count} tensors "
                                f"(Total Shape: {chunk_elements}, "
                                f"{chunk_bytes / (1024 * 1024):.2f} MB) | "
                                f"Shape: {chunk_shape_time:.4f}s | Load: {chunk_load_time:.4f}s | "
                                f"Transfer to GPU: {chunk_transfer_time:.4f}s | "
                                f"Transfer to CPU: {chunk_transfer_back_time:.4f}s | "
                                f"Cleanup: {chunk_mark_time:.4f}s"
                            )
                            if args.writer_test:
                                msg += f" | Write Enqueue: {chunk_enqueue_time:.4f}s"
                            logger.info(msg)

                            chunk_count = 0
                            chunk_shape_time = 0.0
                            chunk_load_time = 0.0
                            chunk_transfer_time = 0.0
                            chunk_transfer_back_time = 0.0
                            chunk_mark_time = 0.0
                            chunk_enqueue_time = 0.0
                            chunk_bytes = 0
                            chunk_elements = 0

                    loop_end_time = time.time()

            # Measure writer __exit__ (header flush + thread join)
            if args.writer_test and loop_end_time is not None:
                total_writer_finalize_time = time.time() - loop_end_time
                logger.info(
                    f"[Writer] Finalized '{writer_output_path}' | "
                    f"Finalize: {total_writer_finalize_time:.4f}s"
                )

                # Optional verification
                if args.writer_verify:
                    logger.info(
                        f"[Writer Verification] Verifying {len(test_keys)} tensors ..."
                    )
                    try:
                        with UnifiedSafetensorsLoader(
                            writer_output_path, low_memory=True
                        ) as vl:
                            for vkey in test_keys:
                                try:
                                    orig = loader.get_tensor(vkey)
                                    written = vl.get_tensor(vkey)
                                    shape_ok = orig.shape == written.shape
                                    dtype_ok = orig.dtype == written.dtype
                                    value_ok = torch.allclose(
                                        orig, written.to(orig.dtype)
                                    )
                                    if shape_ok and dtype_ok and value_ok:
                                        writer_verify_ok += 1
                                    else:
                                        writer_verify_fail += 1
                                        logger.warning(
                                            f"  [FAIL] '{vkey}': shape={shape_ok} "
                                            f"dtype={dtype_ok} values={value_ok}"
                                        )
                                    del orig, written
                                except Exception as e:
                                    writer_verify_fail += 1
                                    logger.warning(f"  [ERROR] '{vkey}': {e}")
                    except Exception as e:
                        logger.error(
                            f"[Writer Verification] Could not open output: {e}"
                        )
                    total_verified = writer_verify_ok + writer_verify_fail
                    logger.info(
                        f"[Writer Verification] {writer_verify_ok}/{total_verified} OK | "
                        f"{writer_verify_fail}/{total_verified} FAILED"
                    )

                # Cleanup output file
                try:
                    if os.path.exists(writer_output_path):
                        os.remove(writer_output_path)
                        logger.info(
                            f"[Writer] Removed output file: {writer_output_path}"
                        )
                except OSError as e:
                    logger.warning(f"[Writer] Could not remove output file: {e}")

        else:
            logger.info("No standard tensors found to test.")

    # 5. mmap Benchmark
    if args.mmap:
        logger.info("--- Starting mmap Benchmark ---")
        mmap_total_tensors = 0
        mmap_total_bytes = 0
        mmap_map_time = 0.0
        mmap_view_time = 0.0
        mmap_clone_time = 0.0
        mmap_transfer_time = 0.0
        mmap_bounce_time = 0.0

        mmap_start = time.time()
        try:
            mmap_loader = UnifiedSafetensorsLoader(
                filepath, low_memory=True, use_mmap=True
            )
        except Exception as e:
            logger.error(f"Failed to open file in mmap mode: {e}")
            mmap_loader = None

        if mmap_loader is not None:
            mmap_map_time = time.time() - mmap_start
            if not mmap_loader.use_mmap:
                logger.warning(
                    "[mmap] Native lib unavailable — fell back to standard IO. Results reflect IO path, not true mmap."
                )
            else:
                logger.info(
                    f"[mmap] Mapping init took {mmap_map_time:.5f}s  |  use_mmap={mmap_loader.use_mmap}"
                )

            with mmap_loader:
                mmap_keys = mmap_loader.keys()
                test_mmap_keys = (
                    mmap_keys[: args.limit] if args.limit > 0 else mmap_keys
                )
                logger.info(f"[mmap] Benchmarking {len(test_mmap_keys)} tensor(s)")

                chunk_count = 0
                chunk_view_time = 0.0
                chunk_clone_time = 0.0
                chunk_transfer_time = 0.0
                chunk_bytes = 0

                for idx, key in enumerate(test_mmap_keys, 1):
                    t0 = time.time()
                    tensor = mmap_loader.get_tensor(key)
                    v_time = time.time() - t0
                    chunk_view_time += v_time
                    mmap_view_time += v_time

                    b_size = tensor.numel() * tensor.element_size()
                    chunk_bytes += b_size
                    mmap_total_bytes += b_size

                    t0 = time.time()
                    writable = tensor.clone()
                    c_time = time.time() - t0
                    chunk_clone_time += c_time
                    mmap_clone_time += c_time

                    if device.type == "cuda":
                        t0 = time.time()
                        gpu_tensor = transfer_to_gpu_pinned(writable, device=device)
                        tr_time = time.time() - t0
                        chunk_transfer_time += tr_time
                        mmap_transfer_time += tr_time
                        del gpu_tensor

                    del tensor, writable
                    chunk_count += 1
                    mmap_total_tensors += 1

                    if chunk_count >= chunk_size or idx == len(test_mmap_keys):
                        msg = (
                            f"[mmap Chunk Summary] Processed {chunk_count} tensors "
                            f"({chunk_bytes / (1024 * 1024):.2f} MB) | "
                            f"View: {chunk_view_time:.4f}s | Clone: {chunk_clone_time:.4f}s"
                        )
                        if device.type == "cuda":
                            msg += f" | GPU Transfer: {chunk_transfer_time:.4f}s"
                        logger.info(msg)
                        chunk_count = 0
                        chunk_view_time = 0.0
                        chunk_clone_time = 0.0
                        chunk_transfer_time = 0.0
                        chunk_bytes = 0

                if mmap_loader._mmap is not None:
                    t0 = time.time()
                    bounced = mmap_loader._mmap.bounce()
                    mmap_bounce_time = time.time() - t0
                    logger.info(
                        f"[mmap] bounce() returned {bounced} in {mmap_bounce_time:.5f}s"
                    )

        logger.info(
            "----------------------------------------------------------------------"
        )
        logger.info(
            f"[mmap Grand Total] Tensors: {mmap_total_tensors} | {mmap_total_bytes / (1024 * 1024):.2f} MB"
        )
        logger.info(f"  -> Mapping init  : {mmap_map_time:.4f}s")
        logger.info(
            f"  -> View time     : {mmap_view_time:.4f}s  (zero-copy frombuffer)"
        )
        logger.info(
            f"  -> Clone time    : {mmap_clone_time:.4f}s  (forces page faults / actual reads)"
        )
        if device.type == "cuda":
            logger.info(f"  -> GPU transfer  : {mmap_transfer_time:.4f}s")
        logger.info(f"  -> bounce() time : {mmap_bounce_time:.4f}s")
        logger.info(
            "----------------------------------------------------------------------"
        )

    total_script_time = time.time() - script_start_time

    logger.info(
        "======================================================================"
    )
    logger.info(
        "                        GRAND TOTAL SUMMARY                           "
    )
    logger.info(
        "======================================================================"
    )
    logger.info(
        f"Total U8 Dictionaries   : {total_u8_tensors} tensors ({total_u8_bytes / 1024:.2f} KB)"
    )
    logger.info(f"  -> Loading Time       : {total_u8_load_time:.4f}s")
    logger.info(f"  -> Decoding Time      : {total_u8_convert_time:.4f}s")
    logger.info("")
    logger.info(
        f"Total Standard Tensors  : {total_std_tensors} tensors (Total Shape: {total_std_elements}, {total_std_bytes / (1024 * 1024):.2f} MB)"
    )
    loading_label = "Direct GPU (Disk->GPU)" if direct_gpu else "Data Loading Time  "
    logger.info(f"  -> Shape/NDIM Time    : {total_std_shape_time:.4f}s")
    logger.info(f"  -> {loading_label}  : {total_std_load_time:.4f}s")
    logger.info(f"  -> Pinned GPU Transfer: {total_std_transfer_gpu_time:.4f}s")
    logger.info(f"  -> CPU Return Transfer: {total_std_transfer_cpu_time:.4f}s")
    logger.info(f"  -> Memory Cleanup Time: {total_std_mark_time:.4f}s")
    if args.writer_test:
        logger.info(
            f"  -> Write Enqueue Time : {total_writer_enqueue_time:.4f}s  "
            f"(async dispatch, overlaps GPU work)"
        )
        logger.info(
            f"  -> Writer Finalize    : {total_writer_finalize_time:.4f}s  "
            f"(header flush + thread join)"
        )

    total_roundtrip_time = (
        total_std_shape_time
        + total_std_load_time
        + total_std_transfer_gpu_time
        + total_std_transfer_cpu_time
        + total_std_mark_time
    )
    if args.writer_test:
        total_roundtrip_time += total_writer_enqueue_time + total_writer_finalize_time
    logger.info(f"  => FULL ROUNDTRIP TIME: {total_roundtrip_time:.4f}s")

    if args.writer_test:
        throughput_mbs = (
            (total_std_bytes / (1024 * 1024)) / total_roundtrip_time
            if total_roundtrip_time > 0
            else 0.0
        )
        logger.info(f"  => Write Throughput   : {throughput_mbs:.2f} MB/s")
        if args.writer_verify:
            total_verified = writer_verify_ok + writer_verify_fail
            logger.info(
                f"  [Verification]        : {writer_verify_ok}/{total_verified} OK | "
                f"{writer_verify_fail}/{total_verified} FAILED"
            )

    logger.info(
        "----------------------------------------------------------------------"
    )
    logger.info(f"Total Script Time       : {total_script_time:.4f}s")
    logger.info(
        "======================================================================"
    )
    logger.info("--- Benchmark Complete ---")


if __name__ == "__main__":
    main()
