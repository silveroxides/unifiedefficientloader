import argparse
import logging
import time
import sys
import math
import torch
import warnings
from safetensors import safe_open
from unifiedefficientloader import (
    tensor_to_dict,
    transfer_to_gpu_pinned
)

# Silence internal PyTorch dataloader deprecation warnings
warnings.filterwarnings("ignore", message=".*The argument 'device' of Tensor.*")

def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def main():
    parser = argparse.ArgumentParser(description="Standard safetensors loading benchmark (baseline)")
    parser.add_argument("file", help="Path to the safetensors file to load")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device for pinned transfer tests")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of tensors to test per category (0 for no limit)")
    parser.add_argument("--chunk-size", type=int, default=100, help="Number of tensors to process before logging a summary chunk")
    args = parser.parse_args()

    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    logger.info(f"--- Starting Standard Benchmark (Preload) for {args.file} ---")
    script_start_time = time.time()

    # Step 1: Preload everything into RAM (Standard safetensors usage)
    logger.info("Loading entire model into system memory via safetensors.safe_open...")
    preload_start_time = time.time()
    
    loaded_tensors = {}
    try:
        with safe_open(args.file, framework="pt", device="cpu") as f:
            all_keys = f.keys()
            
            # Progress bar if possible
            try:
                from tqdm import tqdm
                iterator = tqdm(all_keys, desc="Preloading tensors")
            except ImportError:
                iterator = all_keys
                
            for k in iterator:
                loaded_tensors[k] = f.get_tensor(k)
    except Exception as e:
        logger.error(f"Failed to preload file {args.file}: {e}")
        sys.exit(1)
        
    preload_time = time.time() - preload_start_time
    logger.info(f"[Benchmark] Full model preload took {preload_time:.5f} seconds (Found {len(loaded_tensors)} tensors)")

    # Grand Totals
    total_u8_tensors = 0
    total_u8_bytes = 0
    total_u8_convert_time = 0.0

    total_std_tensors = 0
    total_std_elements = 0
    total_std_bytes = 0
    total_std_shape_time = 0.0
    total_std_load_time = 0.0
    total_std_transfer_gpu_time = 0.0
    total_std_transfer_cpu_time = 0.0
    total_std_mark_time = 0.0

    # Step 2: Categorize tensors (U8 dicts vs Standard)
    # Matching the logic in manual_test.py
    uint8_tensor_keys = [
        k for k, v in loaded_tensors.items()
        if v.dtype == torch.uint8 and v.ndim == 1
    ]
    standard_keys = [k for k in loaded_tensors.keys() if k not in uint8_tensor_keys]

    # Step 3: Benchmark U8 Tensors
    test_u8_keys = uint8_tensor_keys[:args.limit] if args.limit > 0 else uint8_tensor_keys
    if test_u8_keys:
        logger.info(f"--- Benchmarking {len(test_u8_keys)} U8 tensor(s) (decoded from RAM) ---")
        chunk_count = 0
        chunk_convert_time = 0.0
        chunk_bytes = 0
        
        for idx, key in enumerate(test_u8_keys, 1):
            tensor = loaded_tensors[key]
            b_size = tensor.numel() * tensor.element_size()
            chunk_bytes += b_size
            total_u8_bytes += b_size
            
            # Convert (decode JSON)
            start_time = time.time()
            try:
                _ = tensor_to_dict(tensor)
                c_time = time.time() - start_time
                chunk_convert_time += c_time
                total_u8_convert_time += c_time
            except Exception as e:
                logger.warning(f"Failed to decode '{key}' as JSON dict: {e}")
            
            chunk_count += 1
            total_u8_tensors += 1
            
            if chunk_count >= args.chunk_size or idx == len(test_u8_keys):
                logger.info(f"[U8 Chunk Summary] Processed {chunk_count} tensors ({chunk_bytes / 1024:.2f} KB) | "
                            f"Decode: {chunk_convert_time:.4f}s")
                chunk_count = 0
                chunk_convert_time = 0.0
                chunk_bytes = 0

    # Step 4: Benchmark Standard Tensors
    test_std_keys = standard_keys[:args.limit] if args.limit > 0 else standard_keys
    if test_std_keys:
        logger.info(f"--- Benchmarking {len(test_std_keys)} standard tensor(s) (transfer from RAM) ---")
        
        chunk_count = 0
        chunk_shape_time = 0.0
        chunk_load_time = 0.0
        chunk_transfer_time = 0.0
        chunk_transfer_back_time = 0.0
        chunk_mark_time = 0.0
        chunk_bytes = 0
        chunk_elements = 0

        for idx, sample_key in enumerate(test_std_keys, 1):
            # Already in RAM, but we still time these steps for parity with manual_test.py
            start_time = time.time()
            tensor = loaded_tensors[sample_key]
            shape = tensor.shape
            s_time = time.time() - start_time
            chunk_shape_time += s_time
            total_std_shape_time += s_time
            
            elements = math.prod(shape) if shape else 0
            chunk_elements += elements
            total_std_elements += elements

            # "Load" time is just lookup (but we time it anyway for parity with manual_test.py)
            start_time = time.time()
            _ = loaded_tensors[sample_key]
            l_time = time.time() - start_time
            chunk_load_time += l_time
            total_std_load_time += l_time
            
            b_size = tensor.numel() * tensor.element_size()
            chunk_bytes += b_size
            total_std_bytes += b_size

            # Transfer to GPU
            start_time = time.time()
            gpu_tensor = transfer_to_gpu_pinned(tensor, device=args.device)
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

            # Memory Cleanup (removing from RAM dictionary)
            start_time = time.time()
            del loaded_tensors[sample_key]
            del gpu_tensor, cpu_tensor
            # gc.collect() # Optional: could be slow
            m_time = time.time() - start_time
            chunk_mark_time += m_time
            total_std_mark_time += m_time
            
            chunk_count += 1
            total_std_tensors += 1
            
            if chunk_count >= args.chunk_size or idx == len(test_std_keys):
                logger.info(
                    f"[Standard Chunk Summary] Processed {chunk_count} tensors (Total Shape: {chunk_elements}, "
                    f"{chunk_bytes / (1024*1024):.2f} MB) | "
                    f"Shape/Lookup: {chunk_shape_time:.4f}s | "
                    f"Transfer to GPU: {chunk_transfer_time:.4f}s | Transfer to CPU: {chunk_transfer_back_time:.4f}s | Cleanup: {chunk_mark_time:.4f}s"
                )
                chunk_count = 0
                chunk_shape_time = 0.0
                chunk_load_time = 0.0
                chunk_transfer_time = 0.0
                chunk_transfer_back_time = 0.0
                chunk_mark_time = 0.0
                chunk_bytes = 0
                chunk_elements = 0
    else:
        logger.info("No standard tensors found to test.")

    total_script_time = time.time() - script_start_time

    logger.info("======================================================================")
    logger.info("                  STANDARD (PRELOAD) GRAND SUMMARY                    ")
    logger.info("======================================================================")
    logger.info(f"Model Preload Time      : {preload_time:.4f}s")
    logger.info("")
    logger.info(f"Total U8 Dictionaries   : {total_u8_tensors} tensors ({total_u8_bytes / 1024:.2f} KB)")
    logger.info(f"  -> Decoding Time      : {total_u8_convert_time:.4f}s")
    logger.info("")
    logger.info(f"Total Standard Tensors  : {total_std_tensors} tensors (Total Shape: {total_std_elements}, {total_std_bytes / (1024*1024):.2f} MB)")
    logger.info(f"  -> Shape/NDIM Time    : {total_std_shape_time:.4f}s")
    logger.info(f"  -> RAM Lookup Time    : {total_std_load_time:.4f}s")
    logger.info(f"  -> Pinned GPU Transfer: {total_std_transfer_gpu_time:.4f}s")
    logger.info(f"  -> CPU Return Transfer: {total_std_transfer_cpu_time:.4f}s")
    logger.info(f"  -> Memory Cleanup Time: {total_std_mark_time:.4f}s")
    
    total_transfer_loop_time = (
        total_std_shape_time + 
        total_std_load_time + 
        total_std_transfer_gpu_time + 
        total_std_transfer_cpu_time + 
        total_std_mark_time
    )
    logger.info(f"  => TRANSFER LOOP TIME : {total_transfer_loop_time:.4f}s")
    logger.info("----------------------------------------------------------------------")
    logger.info(f"Total Combined Time     : {(preload_time + total_transfer_loop_time + total_u8_convert_time):.4f}s")
    logger.info(f"Total Script Time       : {total_script_time:.4f}s")
    logger.info("======================================================================")
    logger.info("--- Standard Benchmark Complete ---")

if __name__ == "__main__":
    main()
