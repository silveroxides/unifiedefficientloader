import argparse
import logging
import time
import sys
import math
import torch
import warnings
from unifiedefficientloader import (
    UnifiedSafetensorsLoader,
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
    parser = argparse.ArgumentParser(description="Manual test script and benchmark for unifiedefficientloader")
    parser.add_argument("file", help="Path to the safetensors file to load")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device for pinned transfer tests")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of tensors to test per category (0 for no limit)")
    parser.add_argument("--chunk-size", type=int, default=100, help="Number of tensors to process before logging a summary chunk")
    parser.add_argument("--async-batch", type=int, default=0, help="If >0, uses async_stream with this batch size instead of sequential load")
    parser.add_argument("--low-memory", action="store_true", default=True, help="Enable memory-efficient streaming mode")
    parser.add_argument("--no-low-memory", action="store_false", dest="low_memory", help="Disable memory-efficient streaming mode (preload everything)")
    parser.add_argument("--workers", type=int, default=4, help="Number of background workers for async_stream (ignored, uses internal max)")
    parser.add_argument("--batch-transfer", action="store_true", help="If set, enables sequential pinning in main thread via async_stream")
    parser.add_argument("--transfer-count", type=int, default=0, help="Ignored in this version")
    args = parser.parse_args()

    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    logger.info(f"--- Starting Benchmark for {args.file} ---")
    script_start_time = time.time()

    # Grand Totals
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

    # 1. Benchmark Header Loading
    start_time = time.time()
    try:
        loader = UnifiedSafetensorsLoader(args.file, low_memory=args.low_memory)
    except Exception as e:
        logger.error(f"Failed to load file {args.file}: {e}")
        sys.exit(1)
    
    header_time = time.time() - start_time
    logger.info(f"[Benchmark] Header initialization (low_memory={args.low_memory}) took {header_time:.5f} seconds")

    with loader:
        # 2. Benchmark Finding U8 Dictionary Tensors (1D only)
        start_time = time.time()
        uint8_tensor_keys = [
            k for k, v in loader._header.items()
            if isinstance(v, dict) and v.get("dtype") == "U8" and len(v.get("shape", [])) == 1
        ]
        find_u8_time = time.time() - start_time
        logger.info(f"[Benchmark] Scanning header for 1D U8 tensors took {find_u8_time:.5f} seconds. Found {len(uint8_tensor_keys)}.")

        # 3. Benchmark Loading and Converting U8 Tensors
        test_u8_keys = uint8_tensor_keys[:args.limit] if args.limit > 0 else uint8_tensor_keys
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
            
            if chunk_count >= args.chunk_size or idx == len(test_u8_keys):
                logger.info(f"[U8 Chunk Summary] Processed {chunk_count} tensors ({chunk_bytes / 1024:.2f} KB) | "
                            f"Load: {chunk_load_time:.4f}s | Decode: {chunk_convert_time:.4f}s")
                chunk_count = 0
                chunk_load_time = 0.0
                chunk_convert_time = 0.0
                chunk_bytes = 0

        # 4. Benchmark Standard Tensors
        standard_keys = [k for k in loader.keys() if k not in uint8_tensor_keys]
        if standard_keys:
            if args.limit > 0:
                test_keys = standard_keys[:args.limit]
            else:
                test_keys = standard_keys

            if args.async_batch > 0:
                logger.info(f"--- Benchmarking {len(test_keys)} standard tensor(s) ASYNCHRONOUSLY via async_stream (batch={args.async_batch}, pin={args.batch_transfer}) ---")
                
                stream_start_time = time.time()
                
                chunk_count = 0
                chunk_shape_time = 0.0
                chunk_load_time = 0.0
                chunk_transfer_time = 0.0
                chunk_transfer_back_time = 0.0
                chunk_mark_time = 0.0
                chunk_bytes = 0
                chunk_elements = 0

                # Use async_stream directly.
                # If --batch-transfer is set, we enable sequential pinning in main thread.
                stream = loader.async_stream(test_keys, batch_size=args.async_batch, pin_memory=args.batch_transfer)

                for batch in stream:
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
                        gpu_tensor = transfer_to_gpu_pinned(tensor, device=args.device)
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
                        
                        del tensor, gpu_tensor, cpu_tensor
                        
                        chunk_count += 1
                        total_std_tensors += 1
                        
                        if chunk_count >= args.chunk_size or total_std_tensors == len(test_keys):
                            total_chunk_time = time.time() - stream_start_time
                            approx_load = max(0, total_chunk_time - (chunk_transfer_time + chunk_transfer_back_time + chunk_mark_time + chunk_shape_time))
                            chunk_load_time += approx_load
                            total_std_load_time += approx_load
                            
                            logger.info(
                                f"[Async Chunk Summary] Processed {chunk_count} tensors (Total Shape: {chunk_elements}, "
                                f"{chunk_bytes / (1024*1024):.2f} MB) | "
                                f"Async Load/Pin: {approx_load:.4f}s | "
                                f"Transfer to GPU: {chunk_transfer_time:.4f}s | Transfer to CPU: {chunk_transfer_back_time:.4f}s"
                            )
                            chunk_count = 0
                            chunk_shape_time = 0.0
                            chunk_load_time = 0.0
                            chunk_transfer_time = 0.0
                            chunk_transfer_back_time = 0.0
                            chunk_mark_time = 0.0
                            chunk_bytes = 0
                            chunk_elements = 0
                            stream_start_time = time.time()
                    
                    batch.clear()
                    del batch
            else:
                logger.info(f"--- Benchmarking {len(test_keys)} standard tensor(s) SEQUENTIALLY ---")
                
                chunk_count = 0
                chunk_shape_time = 0.0
                chunk_load_time = 0.0
                chunk_transfer_time = 0.0
                chunk_transfer_back_time = 0.0
                chunk_mark_time = 0.0
                chunk_bytes = 0
                chunk_elements = 0

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

                    # Mark Processed
                    start_time = time.time()
                    loader.mark_processed(sample_key)
                    m_time = time.time() - start_time
                    chunk_mark_time += m_time
                    total_std_mark_time += m_time
                    
                    # Cleanup manually
                    del tensor, gpu_tensor, cpu_tensor
                    
                    chunk_count += 1
                    total_std_tensors += 1
                    
                    if chunk_count >= args.chunk_size or idx == len(test_keys):
                        logger.info(
                            f"[Standard Chunk Summary] Processed {chunk_count} tensors (Total Shape: {chunk_elements}, "
                            f"{chunk_bytes / (1024*1024):.2f} MB) | "
                            f"Shape: {chunk_shape_time:.4f}s | Load: {chunk_load_time:.4f}s | "
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
    logger.info("                        GRAND TOTAL SUMMARY                           ")
    logger.info("======================================================================")
    logger.info(f"Total U8 Dictionaries   : {total_u8_tensors} tensors ({total_u8_bytes / 1024:.2f} KB)")
    logger.info(f"  -> Loading Time       : {total_u8_load_time:.4f}s")
    logger.info(f"  -> Decoding Time      : {total_u8_convert_time:.4f}s")
    logger.info("")
    logger.info(f"Total Standard Tensors  : {total_std_tensors} tensors (Total Shape: {total_std_elements}, {total_std_bytes / (1024*1024):.2f} MB)")
    logger.info(f"  -> Shape/NDIM Time    : {total_std_shape_time:.4f}s")
    logger.info(f"  -> Data Loading Time  : {total_std_load_time:.4f}s")
    logger.info(f"  -> Pinned GPU Transfer: {total_std_transfer_gpu_time:.4f}s")
    logger.info(f"  -> CPU Return Transfer: {total_std_transfer_cpu_time:.4f}s")
    logger.info(f"  -> Memory Cleanup Time: {total_std_mark_time:.4f}s")
    
    total_roundtrip_time = (
        total_std_shape_time + 
        total_std_load_time + 
        total_std_transfer_gpu_time + 
        total_std_transfer_cpu_time + 
        total_std_mark_time
    )
    logger.info(f"  => FULL ROUNDTRIP TIME: {total_roundtrip_time:.4f}s")
    logger.info("----------------------------------------------------------------------")
    logger.info(f"Total Script Time       : {total_script_time:.4f}s")
    logger.info("======================================================================")
    logger.info("--- Benchmark Complete ---")

if __name__ == "__main__":
    main()
