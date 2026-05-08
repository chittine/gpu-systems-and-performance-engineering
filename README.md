# GPU Systems and Performance Engineering

Collection of high-performance computing and GPU optimization projects focused on SIMD vectorization, multicore parallelism, CUDA kernel optimization, Tensor Core acceleration, memory hierarchy analysis, and parallel compression algorithms.

The repository explores performance engineering techniques across CPU and GPU architectures, including shared-memory tiling, register blocking, occupancy optimization, stencil computation, and parallel scan/compression pipelines.

---

## Technologies and Concepts

- CUDA
- C/C++
- AVX512 SIMD Vectorization
- Multithreading and Multicore Parallelism
- Shared Memory Tiling
- Register Tiling
- Tensor Core / MMA Optimization
- Occupancy Optimization
- Warp-Level Parallelism
- Roofline Performance Analysis
- Prefix Sum (Scan)
- Parallel Compression Algorithms
- GPU Memory Hierarchy Analysis

---

## Projects

### 1. SIMD and Multicore Optimization

Progressive CPU optimization pipeline for Mandelbrot set generation workloads.

Implemented:
- AVX512 SIMD vectorization
- Instruction-level parallelism (ILP)
- Multicore parallelism
- Multithreading
- Workload partitioning

Focus areas:
- SIMD lane utilization
- Thread scheduling
- CPU throughput scaling
- Parallel workload decomposition

---

### 2. GPU Architecture Performance Analysis

Microarchitecture-focused CUDA experiments exploring:
- Cache latency
- Memory coalescing
- Warp scheduling
- Throughput saturation
- GPU memory hierarchy behavior

Includes throughput benchmarking and performance visualization using custom analysis scripts.

---

### 3. GPU Stencil Optimization

CUDA stencil and wave simulation workload optimized for memory bandwidth efficiency.

Key observations:
- Achieved ~98.5× speedup from CPU to GPU
- Explored shared-memory optimization tradeoffs
- Analyzed DRAM-bound behavior and cache limitations
- Evaluated memory traffic and bandwidth bottlenecks

---

### 4. High-Performance CUDA GEMM

Optimization study of large-scale CUDA matrix multiplication kernels.

Optimization stages:
- Naive GPU GEMM
- Shared-memory tiling
- Register tiling
- Occupancy optimization
- Tensor Core / MMA acceleration

Performance highlights:
- Optimized 3072×3072×3072 GEMM workload
- Achieved ~5.69 ms runtime
- Reached ~10 TFLOP/s throughput
- Explored compute-bound vs memory-bound behavior
- Applied roofline performance analysis

Techniques explored:
- Shared-memory reuse
- Register blocking
- Warp-level optimization
- Occupancy balancing
- Memory hierarchy optimization

---

### 5. CUDA Parallel Compression

GPU-accelerated run-length encoding (RLE) compression and decompression pipeline.

Implemented:
- Parallel scan (prefix sum)
- Run detection
- Scatter-based compression
- Hybrid decompression strategies

Performance highlights:
- ~134 GB/s scan throughput
- ~0.95 ms compression runtime on 16M input
- ~0.62 ms decompression on image benchmarks
- ~0.06 ms decompression on sparse workloads

Focus areas:
- Irregular parallelism
- Race-condition avoidance
- Memory-efficient GPU algorithms

---

## Performance Highlights

| Project | Result |
|---|---|
| CUDA GEMM | ~5.69 ms for 3072³ GEMM |
| CUDA GEMM | ~10 TFLOP/s throughput |
| GPU Scan | ~134 GB/s throughput |
| RLE Compression | ~0.95 ms on 16M input |
| GPU Stencil | ~98.5× CPU→GPU speedup |

---
