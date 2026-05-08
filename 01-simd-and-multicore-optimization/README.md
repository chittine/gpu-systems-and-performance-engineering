# SIMD and Multicore Optimization

Progressive CPU optimization study for Mandelbrot set generation workloads, exploring SIMD vectorization, instruction-level parallelism (ILP), multicore execution, and multithreading techniques for high-performance CPU computing.

This project focuses on improving throughput and scalability across modern CPU architectures using AVX512 intrinsics, workload partitioning, and parallel execution strategies.

---

## Implemented Optimizations

### Scalar Baseline
Initial scalar Mandelbrot implementation used as a correctness and performance reference.

### AVX512 SIMD Vectorization
Implemented vectorized Mandelbrot computation using AVX512 intrinsics:
- SIMD lane parallelism
- Vector masking
- Packed floating-point operations
- Lane-wise iteration tracking

### Instruction-Level Parallelism (ILP)
Improved throughput by processing multiple independent vectors simultaneously:
- Reduced pipeline stalls
- Increased functional unit utilization
- Improved instruction overlap

### Multicore Parallelism
Implemented multicore execution using:
- pthreads
- workload partitioning across CPU cores
- row-based parallel decomposition

### Multithreading
Extended multicore execution with additional software threads:
- improved CPU utilization
- increased parallel workload scheduling
- better throughput scaling on multi-core systems

---

## Techniques Explored

- AVX512 SIMD intrinsics
- Vector masking
- Instruction-level parallelism
- Multicore scheduling
- Thread synchronization
- Parallel workload decomposition
- CPU throughput optimization
- Mandelbrot set computation
- High-performance numerical computing

---

## Performance Focus

The project investigates:
- SIMD lane utilization
- Throughput scaling
- CPU parallel efficiency
- Vectorized floating-point workloads
- Parallel scheduling overhead
- Scalability across cores and threads

---

## Files

| File | Description |
|---|---|
| `mandelbrot_cpu.cpp` | Scalar and SIMD vectorized Mandelbrot implementation |
| `mandelbrot_cpu_2.cpp` | Extended ILP, multicore, and multithreaded implementations |

---

## Example Optimizations

- SIMD vectorized Mandelbrot iteration using AVX512
- Active lane masking for escape-time computation
- Parallel row decomposition across threads
- Multi-vector ILP execution
- Thread-per-core and multi-thread-per-core scheduling strategies

---

## Notes

This project serves as a CPU-side performance engineering study and forms the foundation for later GPU-focused optimization work involving CUDA kernels, memory hierarchy analysis, and high-performance parallel algorithms.
