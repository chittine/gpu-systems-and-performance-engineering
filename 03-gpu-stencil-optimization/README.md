# GPU Stencil Optimization

CUDA-based stencil and wave simulation optimization project focused on memory-bandwidth efficiency, shared-memory optimization, and GPU performance analysis.

This project explores stencil computation workloads commonly used in scientific computing, numerical simulation, and high-performance computing applications. The implementation evaluates GPU acceleration, memory hierarchy behavior, and optimization tradeoffs for large-scale iterative simulations.

---

## Overview

The project implements a GPU-accelerated wave simulation using stencil computation techniques and compares performance across CPU and GPU implementations.

Key focus areas include:
- GPU memory bandwidth utilization
- Shared-memory optimization
- DRAM-bound performance analysis
- Cache behavior
- Parallel stencil computation
- GPU acceleration of iterative numerical workloads

---

## Implemented Optimizations

### Naive GPU Stencil Kernel
Baseline CUDA stencil implementation using global memory accesses.

### Shared Memory Optimization
Implemented shared-memory tiling to improve data reuse and reduce redundant global memory accesses.

### Memory Bandwidth Analysis
Analyzed:
- DRAM traffic
- L2 cache limitations
- memory-bound execution behavior
- throughput bottlenecks

---

## Performance Highlights

- Achieved ~98.5× speedup from CPU to GPU execution
- Optimized large-scale iterative stencil workloads
- Evaluated DRAM-bound versus cache-bound behavior
- Explored shared-memory tradeoffs and synchronization overhead

---

## Techniques Explored

- CUDA stencil computation
- Shared memory tiling
- GPU memory hierarchy analysis
- Bandwidth-bound optimization
- Parallel iterative simulation
- Numerical wave propagation workloads
- GPU performance engineering

---

## Files

| File | Description |
|---|---|
| `wave.cu` | CUDA stencil and wave simulation implementation |
---

## Notes

The project demonstrates how memory bandwidth and cache behavior influence stencil computation performance on GPUs. The optimization study highlights the tradeoffs between shared-memory reuse, synchronization overhead, and overall throughput for iterative simulation workloads.
