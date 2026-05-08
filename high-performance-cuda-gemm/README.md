# High-Performance CUDA GEMM

Optimization study of large-scale CUDA matrix multiplication (GEMM) kernels focused on shared-memory tiling, register blocking, occupancy optimization, and Tensor Core acceleration.

This project explores progressive GPU optimization techniques for dense matrix multiplication workloads, moving from baseline CUDA kernels to highly optimized implementations designed to improve throughput, memory reuse, and hardware utilization.

---

## Overview

The project implements and evaluates multiple CUDA GEMM optimization strategies for large-scale matrix multiplication workloads.

Optimization stages include:
- Naive GPU GEMM
- Shared-memory tiling
- Register tiling
- Occupancy optimization
- Tensor Core / MMA acceleration

The implementation focuses on balancing:
- memory bandwidth
- compute throughput
- register pressure
- occupancy
- cache reuse
- warp-level execution efficiency

---

## Implemented Optimizations

### Naive GEMM Kernel
Baseline CUDA matrix multiplication implementation using global memory accesses.

### Shared Memory Tiling
Implemented block tiling using shared memory:
- reduced global memory traffic
- increased data reuse
- improved arithmetic intensity

### Register Tiling
Added register-level microtiling:
- partial sums stored in registers
- improved reuse of shared-memory data
- reduced shared-memory traffic

### Occupancy Optimization
Explored:
- block size tuning
- register pressure balancing
- occupancy tradeoffs
- warp scheduling efficiency

### Tensor Core / MMA Optimization
Implemented Tensor Core acceleration using MMA instructions:
- warp-level matrix multiply operations
- hardware-accelerated fused matrix computation
- increased floating-point throughput

---

## Performance Highlights

- Optimized 3072×3072×3072 GEMM workload
- Achieved ~5.69 ms runtime on optimized kernel
- Reached ~10 TFLOP/s throughput
- Improved memory reuse through shared-memory tiling
- Applied roofline analysis for compute-bound and memory-bound evaluation

---

## Techniques Explored

- CUDA kernel optimization
- Shared-memory tiling
- Register blocking
- Warp-level parallelism
- Occupancy optimization
- Tensor Core acceleration
- MMA instructions
- Roofline performance analysis
- Memory hierarchy optimization
- Throughput benchmarking

---

## Optimization Pipeline

```text
Naive GEMM
    ↓
Shared-memory tiling
    ↓
Register tiling
    ↓
Occupancy optimization
    ↓
Tensor Core / MMA acceleration
