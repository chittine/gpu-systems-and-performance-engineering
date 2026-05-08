#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Utility Functions

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

template <typename Op>
void print_array(
    size_t n,
    typename Op::Data const *x // allowed to be either a CPU or GPU pointer
);

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Already Written)

template <typename Op>
void scan_cpu(size_t n, typename Op::Data const *x, typename Op::Data *out) {
    using Data = typename Op::Data;
    Data accumulator = Op::identity();
    for (size_t i = 0; i < n; i++) {
        accumulator = Op::combine(accumulator, x[i]);
        out[i] = accumulator;
    }
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

namespace scan_gpu {

constexpr int BLOCK_SIZE = 256;
constexpr int ITEMS_PER_THREAD = 8;
constexpr int ELEMS_PER_BLOCK = BLOCK_SIZE * ITEMS_PER_THREAD;

template <typename Op>
__global__ void scan_chunked_block_kernel(
    size_t n,
    typename Op::Data const *x,
    typename Op::Data *out,
    typename Op::Data *block_sums
) {
    using Data = typename Op::Data;

    extern __shared__ __align__(16) char shmem_raw[];
    Data *shmem = reinterpret_cast<Data *>(shmem_raw);

    int tid = threadIdx.x;
    size_t block_base = static_cast<size_t>(blockIdx.x) * ELEMS_PER_BLOCK;
    size_t thread_base = block_base + static_cast<size_t>(tid) * ITEMS_PER_THREAD;

    Data local_prefix[ITEMS_PER_THREAD];
    Data running = Op::identity();

    int valid_count = 0;
    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
        size_t idx = thread_base + k;
        if (idx < n) {
            running = Op::combine(running, x[idx]);
            local_prefix[k] = running;
            valid_count++;
        } else {
            local_prefix[k] = Op::identity();
        }
    }

    shmem[tid] = running;
    __syncthreads();

    // Inclusive scan over per-thread totals
    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        Data temp = shmem[tid];
        if (tid >= offset) {
            temp = Op::combine(shmem[tid - offset], temp);
        }
        __syncthreads();
        shmem[tid] = temp;
        __syncthreads();
    }

    Data thread_prefix = (tid == 0) ? Op::identity() : shmem[tid - 1];

    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
        size_t idx = thread_base + k;
        if (idx < n) {
            out[idx] = Op::combine(thread_prefix, local_prefix[k]);
        }
    }

    if (block_sums != nullptr) {
        size_t elems_in_block = 0;
        if (block_base < n) {
            elems_in_block = n - block_base;
            if (elems_in_block > ELEMS_PER_BLOCK) elems_in_block = ELEMS_PER_BLOCK;
        }

        if (elems_in_block > 0) {
            int valid_threads = static_cast<int>((elems_in_block + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD);
            if (tid == valid_threads - 1) {
                block_sums[blockIdx.x] = shmem[tid];
            }
        }
    }
}

template <typename Op>
__global__ void add_block_prefixes_kernel(
    size_t n,
    typename Op::Data *out,
    typename Op::Data const *scanned_block_sums
) {
    using Data = typename Op::Data;

    int tid = threadIdx.x;
    size_t block_base = static_cast<size_t>(blockIdx.x) * ELEMS_PER_BLOCK;
    size_t thread_base = block_base + static_cast<size_t>(tid) * ITEMS_PER_THREAD;

    if (blockIdx.x == 0) return;
    if (block_base >= n) return;

    Data prefix = scanned_block_sums[blockIdx.x - 1];

    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
        size_t idx = thread_base + k;
        if (idx < n) {
            out[idx] = Op::combine(prefix, out[idx]);
        }
    }
}

template <typename Op>
size_t get_workspace_size(size_t n) {
    using Data = typename Op::Data;

    if (n == 0) return sizeof(Data);

    size_t num_blocks = (n + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;

    size_t total = n * sizeof(Data);

    if (num_blocks > 1) {
        total += num_blocks * sizeof(Data);
        total += get_workspace_size<Op>(num_blocks);
    }

    return total;
}

template <typename Op>
typename Op::Data *launch_scan(
    size_t n,
    typename Op::Data *x,
    void *workspace
) {
    using Data = typename Op::Data;

    Data *workspace_data = reinterpret_cast<Data *>(workspace);

    if (n == 0) {
        return workspace_data;
    }

    size_t num_blocks = (n + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
    Data *out = workspace_data;

    if (num_blocks == 1) {
        size_t shmem_size = BLOCK_SIZE * sizeof(Data);
        scan_chunked_block_kernel<Op><<<1, BLOCK_SIZE, shmem_size>>>(n, x, out, nullptr);
        CUDA_CHECK(cudaGetLastError());
        return out;
    }

    Data *block_sums = out + n;
    void *recursive_workspace = reinterpret_cast<void *>(block_sums + num_blocks);

    size_t shmem_size = BLOCK_SIZE * sizeof(Data);

    scan_chunked_block_kernel<Op>
        <<<static_cast<int>(num_blocks), BLOCK_SIZE, shmem_size>>>(
            n, x, out, block_sums);
    CUDA_CHECK(cudaGetLastError());

    Data *scanned_block_sums = launch_scan<Op>(num_blocks, block_sums, recursive_workspace);

    add_block_prefixes_kernel<Op>
        <<<static_cast<int>(num_blocks), BLOCK_SIZE>>>(
            n, out, scanned_block_sums);
    CUDA_CHECK(cudaGetLastError());

    return out;
}

} // namespace scan_gpu 

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

struct DebugRange {
    uint32_t lo;
    uint32_t hi;

    static constexpr uint32_t INVALID = 0xffffffff;

    static __host__ __device__ __forceinline__ DebugRange invalid() {
        return {INVALID, INVALID};
    }

    __host__ __device__ __forceinline__ bool operator==(const DebugRange &other) const {
        return lo == other.lo && hi == other.hi;
    }

    __host__ __device__ __forceinline__ bool operator!=(const DebugRange &other) const {
        return !(*this == other);
    }

    __host__ __device__ bool is_empty() const { return lo == hi; }

    __host__ __device__ bool is_valid() const { return lo != INVALID; }

    std::string to_string() const {
        if (lo == INVALID) {
            return "INVALID";
        } else {
            return std::to_string(lo) + ":" + std::to_string(hi);
        }
    }
};

struct DebugRangeConcatOp {
    using Data = DebugRange;

    static __host__ __device__ __forceinline__ Data identity() { return {0, 0}; }

    static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
        if (a.is_empty()) {
            return b;
        } else if (b.is_empty()) {
            return a;
        } else if (a.is_valid() && b.is_valid() && a.hi == b.lo) {
            return {a.lo, b.hi};
        } else {
            return Data::invalid();
        }
    }

    static std::string to_string(Data d) { return d.to_string(); }
};

struct SumOp {
    using Data = uint32_t;

    static __host__ __device__ __forceinline__ Data identity() { return 0; }

    static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
        return a + b;
    }

    static std::string to_string(Data d) { return std::to_string(d); }
};

constexpr size_t max_print_array_output = 1025;
static thread_local size_t total_print_array_output = 0;

template <typename Op> void print_array(size_t n, typename Op::Data const *x) {
    using Data = typename Op::Data;

    // copy 'x' from device to host if necessary
    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, x));
    auto x_host_buf = std::vector<Data>();
    Data const *x_host_ptr = nullptr;
    if (attr.type == cudaMemoryTypeDevice) {
        x_host_buf.resize(n);
        x_host_ptr = x_host_buf.data();
        CUDA_CHECK(
            cudaMemcpy(x_host_buf.data(), x, n * sizeof(Data), cudaMemcpyDeviceToHost));
    } else {
        x_host_ptr = x;
    }

    if (total_print_array_output >= max_print_array_output) {
        return;
    }

    printf("[\n");
    for (size_t i = 0; i < n; i++) {
        auto s = Op::to_string(x_host_ptr[i]);
        printf("  [%zu] = %s,\n", i, s.c_str());
        total_print_array_output++;
        if (total_print_array_output > max_print_array_output) {
            printf("  ... (output truncated)\n");
            break;
        }
    }
    printf("]\n");

    if (total_print_array_output >= max_print_array_output) {
        printf("(Reached maximum limit on 'print_array' output; skipping further calls "
               "to 'print_array')\n");
    }

    total_print_array_output++;
}

template <typename Reset, typename F>
double benchmark_ms(double target_time_ms, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        f();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms);
    }
    return best_time_ms;
}

struct Results {
    double time_ms;
    double bandwidth_gb_per_sec;
};

enum class Mode {
    TEST,
    BENCHMARK,
};

template <typename Op>
Results run_config(Mode mode, std::vector<typename Op::Data> const &x) {
    // Allocate buffers
    using Data = typename Op::Data;
    size_t n = x.size();
    size_t workspace_size = scan_gpu::get_workspace_size<Op>(n);
    Data *x_gpu;
    Data *workspace_gpu;
    CUDA_CHECK(cudaMalloc(&x_gpu, n * sizeof(Data)));
    CUDA_CHECK(cudaMalloc(&workspace_gpu, workspace_size));
    CUDA_CHECK(cudaMemcpy(x_gpu, x.data(), n * sizeof(Data), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));

    // Test correctness
    auto expected = std::vector<Data>(n);
    scan_cpu<Op>(n, x.data(), expected.data());
    auto out_gpu = scan_gpu::launch_scan<Op>(n, x_gpu, workspace_gpu);
    if (out_gpu == nullptr) {
        printf("'launch_scan' function not yet implemented (returned nullptr)\n");
        exit(1);
    }
    auto actual = std::vector<Data>(n);
    CUDA_CHECK(
        cudaMemcpy(actual.data(), out_gpu, n * sizeof(Data), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) {
        if (actual.at(i) != expected.at(i)) {
            auto actual_str = Op::to_string(actual.at(i));
            auto expected_str = Op::to_string(expected.at(i));
            printf(
                "Mismatch at position %zu: %s != %s\n",
                i,
                actual_str.c_str(),
                expected_str.c_str());
            if (n <= 128) {
                printf("Input:\n");
                print_array<Op>(n, x.data());
                printf("\nExpected:\n");
                print_array<Op>(n, expected.data());
                printf("\nActual:\n");
                print_array<Op>(n, actual.data());
            }
            exit(1);
        }
    }
    if (mode == Mode::TEST) {
        return {0.0, 0.0};
    }

    // Benchmark
    double target_time_ms = 200.0;
    double time_ms = benchmark_ms(
        target_time_ms,
        [&]() {
            CUDA_CHECK(
                cudaMemcpy(x_gpu, x.data(), n * sizeof(Data), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
        },
        [&]() { scan_gpu::launch_scan<Op>(n, x_gpu, workspace_gpu); });
    double bytes_processed = n * sizeof(Data) * 2;
    double bandwidth_gb_per_sec = bytes_processed / time_ms / 1e6;

    // Cleanup
    CUDA_CHECK(cudaFree(x_gpu));
    CUDA_CHECK(cudaFree(workspace_gpu));

    return {time_ms, bandwidth_gb_per_sec};
}

std::vector<DebugRange> gen_debug_ranges(uint32_t n) {
    auto ranges = std::vector<DebugRange>();
    for (uint32_t i = 0; i < n; ++i) {
        ranges.push_back({i, i + 1});
    }
    return ranges;
}

template <typename Rng> std::vector<uint32_t> gen_random_data(Rng &rng, uint32_t n) {
    auto uniform = std::uniform_int_distribution<uint32_t>(0, 100);
    auto data = std::vector<uint32_t>();
    for (uint32_t i = 0; i < n; ++i) {
        data.push_back(uniform(rng));
    }
    return data;
}

template <typename Op, typename GenData>
void run_tests(std::vector<uint32_t> const &sizes, GenData &&gen_data) {
    for (auto size : sizes) {
        auto data = gen_data(size);
        printf("  Testing size %8u\n", size);
        run_config<Op>(Mode::TEST, data);
        printf("  OK\n\n");
    }
}

int main(int argc, char const *const *argv) {
    auto correctness_sizes = std::vector<uint32_t>{
        16,
        10,
        128,
        100,
        1024,
        1000,
        1 << 20,
        1'000'000,
        16 << 20,
        64 << 20,
    };

    auto rng = std::mt19937(0xCA7CAFE);

    printf("Correctness:\n\n");
    printf("Testing scan operation: debug range concatenation\n\n");
    run_tests<DebugRangeConcatOp>(correctness_sizes, gen_debug_ranges);
    printf("Testing scan operation: integer sum\n\n");
    run_tests<SumOp>(correctness_sizes, [&](uint32_t n) {
        return gen_random_data(rng, n);
    });

    printf("Performance:\n\n");

    size_t n = 64 << 20;
    auto data = gen_random_data(rng, n);

    printf("Benchmarking scan operation: integer sum, size %zu\n\n", n);

    // Warmup
    run_config<SumOp>(Mode::BENCHMARK, data);
    // Benchmark
    auto results = run_config<SumOp>(Mode::BENCHMARK, data);
    printf("  Time: %.2f ms\n", results.time_ms);
    printf("  Throughput: %.2f GB/s\n", results.bandwidth_gb_per_sec);

    return 0;
}