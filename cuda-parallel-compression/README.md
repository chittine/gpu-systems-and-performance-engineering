# CUDA Parallel Compression

GPU-accelerated run-length encoding (RLE) compression and decompression pipeline using parallel scan and irregular parallel algorithms.

This project explores high-performance GPU compression techniques focused on parallel prefix sum (scan), run detection, scatter operations, and hybrid decompression strategies for image and sparse-data workloads.

---

## Overview

The project implements a complete CUDA-based run-length encoding pipeline including:
- Parallel scan (prefix sum)
- Run boundary detection
- Compression
- Decompression
- Hybrid workload mapping strategies

The implementation focuses on minimizing memory traffic, avoiding race conditions, and improving throughput for irregular parallel workloads.

---

## Implemented Components

### Parallel Scan
Hierarchical GPU prefix-sum implementation using:
- block-level scans
- shared memory
- recursive scan propagation
- multi-element processing per thread

### Run-Length Compression
Implemented GPU-based RLE compression:
- run boundary detection
- scan-based output positioning
- scatter operations
- compressed output generation

### Parallel Decompression
Implemented hybrid GPU decompression strategies:
- one-thread-per-run mapping
- one-block-per-run mapping
- optimized handling for sparse and dense workloads

---

## Performance Highlights

- Achieved ~134 GB/s scan throughput
- Achieved ~0.95 ms compression runtime on 16M input
- Achieved ~0.62 ms decompression on image benchmarks
- Achieved ~0.06 ms decompression on sparse-data benchmarks

---

## Techniques Explored

- Parallel prefix sum (scan)
- Irregular parallelism
- Scatter/gather operations
- GPU synchronization
- Memory-efficient GPU algorithms
- Race-condition avoidance
- Workload mapping optimization
- CUDA performance engineering

---

## Files

| File | Description |
|---|---|
| `scan.cu` | Parallel GPU prefix sum implementation |
| `shuffle.cu` | GPU shuffle and helper operations |
| `rle_compress.cu` | CUDA RLE compression implementation |
| `rle_decompress.cu` | CUDA RLE decompression implementation |

---

## Notes

This project demonstrates GPU acceleration of irregular compression workloads using parallel scan and efficient workload decomposition strategies. The implementation emphasizes throughput optimization, reduced memory traffic, and scalable GPU-based compression/decompression pipelines.
