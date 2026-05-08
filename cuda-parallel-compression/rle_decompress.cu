#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
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

////////////////////////////////////////////////////////////////////////////////
// Simple Caching GPU Memory Allocator

class GpuAllocCache {
  public:
    GpuAllocCache() = default;

    ~GpuAllocCache();

    GpuAllocCache(GpuAllocCache const &) = delete;
    GpuAllocCache &operator=(GpuAllocCache const &) = delete;
    GpuAllocCache(GpuAllocCache &&) = delete;
    GpuAllocCache &operator=(GpuAllocCache &&) = delete;

    void *alloc(size_t size);
    void reset();

  private:
    void *buffer = nullptr;
    size_t capacity = 0;
    bool active = false;
};

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation

void rle_decompress_cpu(
    uint32_t compressed_count,
    char const *compressed_data,
    uint32_t const *compressed_lengths,
    std::vector<char> &raw) {
    raw.clear();
    for (uint32_t i = 0; i < compressed_count; i++) {
        char c = compressed_data[i];
        uint32_t run_length = compressed_lengths[i];
        for (uint32_t j = 0; j < run_length; j++) {
            raw.push_back(c);
        }
    }
}

struct Decompressed {
    uint32_t count;
    char const *data;
};

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

namespace rle_gpu {

constexpr int BLOCK_SIZE = 256;
constexpr int ITEMS_PER_THREAD = 8;
constexpr int ELEMS_PER_BLOCK = BLOCK_SIZE * ITEMS_PER_THREAD;

__global__ void scan_block_kernel(
    uint32_t n,
    uint32_t const *x,
    uint32_t *out,
    uint32_t *block_sums
) {
    extern __shared__ uint32_t shmem[];

    int tid = threadIdx.x;
    uint32_t block_base = blockIdx.x * ELEMS_PER_BLOCK;
    uint32_t thread_base = block_base + tid * ITEMS_PER_THREAD;

    uint32_t local[ITEMS_PER_THREAD];
    uint32_t running = 0;

    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
        uint32_t idx = thread_base + k;
        if (idx < n) {
            running += x[idx];
            local[k] = running;
        } else {
            local[k] = 0;
        }
    }

    shmem[tid] = running;
    __syncthreads();

    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        uint32_t val = shmem[tid];
        if (tid >= offset) {
            val += shmem[tid - offset];
        }
        __syncthreads();
        shmem[tid] = val;
        __syncthreads();
    }

    uint32_t prefix = (tid == 0) ? 0 : shmem[tid - 1];

    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
        uint32_t idx = thread_base + k;
        if (idx < n) {
            out[idx] = prefix + local[k];
        }
    }

    if (block_sums != nullptr) {
        uint32_t elems_left = (block_base < n) ? (n - block_base) : 0;
        uint32_t elems_here =
            elems_left > ELEMS_PER_BLOCK ? ELEMS_PER_BLOCK : elems_left;

        uint32_t valid_threads =
            (elems_here + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD;

        if (elems_here > 0 && tid == valid_threads - 1) {
            block_sums[blockIdx.x] = shmem[tid];
        }
    }
}

__global__ void add_block_prefix_kernel(
    uint32_t n,
    uint32_t *out,
    uint32_t const *scanned_block_sums
) {
    if (blockIdx.x == 0) return;

    uint32_t prefix = scanned_block_sums[blockIdx.x - 1];

    uint32_t tid = threadIdx.x;
    uint32_t block_base = blockIdx.x * ELEMS_PER_BLOCK;
    uint32_t thread_base = block_base + tid * ITEMS_PER_THREAD;

    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
        uint32_t idx = thread_base + k;
        if (idx < n) {
            out[idx] += prefix;
        }
    }
}

uint32_t *launch_scan_uint32(
    uint32_t n,
    uint32_t const *x,
    uint32_t *workspace
) {
    if (n == 0) return workspace;

    uint32_t num_blocks = (n + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
    uint32_t *out = workspace;

    size_t shmem_size = BLOCK_SIZE * sizeof(uint32_t);

    if (num_blocks == 1) {
        scan_block_kernel<<<1, BLOCK_SIZE, shmem_size>>>(n, x, out, nullptr);
        CUDA_CHECK(cudaGetLastError());
        return out;
    }

    uint32_t *block_sums = out + n;
    uint32_t *recursive_workspace = block_sums + num_blocks;

    scan_block_kernel<<<num_blocks, BLOCK_SIZE, shmem_size>>>(
        n,
        x,
        out,
        block_sums);
    CUDA_CHECK(cudaGetLastError());

    uint32_t *scanned_block_sums =
        launch_scan_uint32(num_blocks, block_sums, recursive_workspace);

    add_block_prefix_kernel<<<num_blocks, BLOCK_SIZE>>>(
        n,
        out,
        scanned_block_sums);
    CUDA_CHECK(cudaGetLastError());

    return out;
}

size_t scan_workspace_size(uint32_t n) {
    if (n == 0) return sizeof(uint32_t);

    uint32_t num_blocks = (n + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
    size_t total = n * sizeof(uint32_t);

    if (num_blocks > 1) {
        total += num_blocks * sizeof(uint32_t);
        total += scan_workspace_size(num_blocks);
    }

    return total;
}

__global__ void exclusive_starts_kernel(
    uint32_t n,
    uint32_t const *inclusive_scan,
    uint32_t *starts
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    starts[i] = (i == 0) ? 0 : inclusive_scan[i - 1];
}

// Good for few long runs, e.g. sparse benchmark.
__global__ void decompress_runs_block_kernel(
    uint32_t compressed_count,
    char const *compressed_data,
    uint32_t const *compressed_lengths,
    uint32_t const *run_starts,
    char *out
) {
    uint32_t run = blockIdx.x;
    if (run >= compressed_count) return;

    char c = compressed_data[run];
    uint32_t start = run_starts[run];
    uint32_t len = compressed_lengths[run];

    for (uint32_t k = threadIdx.x; k < len; k += blockDim.x) {
        out[start + k] = c;
    }
}

// Good for many short runs, e.g. image benchmark.
__global__ void decompress_runs_thread_kernel(
    uint32_t compressed_count,
    char const *compressed_data,
    uint32_t const *compressed_lengths,
    uint32_t const *run_starts,
    char *out
) {
    uint32_t run = blockIdx.x * blockDim.x + threadIdx.x;
    if (run >= compressed_count) return;

    char c = compressed_data[run];
    uint32_t start = run_starts[run];
    uint32_t len = compressed_lengths[run];

    for (uint32_t k = 0; k < len; k++) {
        out[start + k] = c;
    }
}

Decompressed launch_rle_decompress(
    uint32_t compressed_count,
    char const *compressed_data,
    uint32_t const *compressed_lengths,
    GpuAllocCache &workspace_alloc_1,
    GpuAllocCache &workspace_alloc_2
) {
    if (compressed_count == 0) {
        return {0, nullptr};
    }

    size_t scan_bytes = scan_workspace_size(compressed_count);
    size_t starts_bytes = compressed_count * sizeof(uint32_t);

    uint8_t *workspace1 =
        reinterpret_cast<uint8_t *>(workspace_alloc_1.alloc(scan_bytes + starts_bytes));

    uint32_t *scan_workspace =
        reinterpret_cast<uint32_t *>(workspace1);

    uint32_t *run_starts =
        reinterpret_cast<uint32_t *>(workspace1 + scan_bytes);

    uint32_t *inclusive_scan =
        launch_scan_uint32(compressed_count, compressed_lengths, scan_workspace);

    uint32_t decompressed_count = 0;
    CUDA_CHECK(cudaMemcpy(
        &decompressed_count,
        inclusive_scan + compressed_count - 1,
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost));

    char *out =
        reinterpret_cast<char *>(workspace_alloc_2.alloc(decompressed_count));

    int threads = 256;
    int blocks = (compressed_count + threads - 1) / threads;

    exclusive_starts_kernel<<<blocks, threads>>>(
        compressed_count,
        inclusive_scan,
        run_starts);
    CUDA_CHECK(cudaGetLastError());

    if (compressed_count > 65536) {
        decompress_runs_thread_kernel<<<blocks, threads>>>(
            compressed_count,
            compressed_data,
            compressed_lengths,
            run_starts,
            out);
        CUDA_CHECK(cudaGetLastError());
    } else {
        decompress_runs_block_kernel<<<compressed_count, threads>>>(
            compressed_count,
            compressed_data,
            compressed_lengths,
            run_starts,
            out);
        CUDA_CHECK(cudaGetLastError());
    }

    return {decompressed_count, out};
}

} // namespace rle_gpu

////////////////////////////////////////////////////////////////////////////////
// Code Below Here Unchanged

GpuAllocCache::~GpuAllocCache() {
    if (buffer) {
        CUDA_CHECK(cudaFree(buffer));
    }
}

void *GpuAllocCache::alloc(size_t size) {
    if (active) {
        printf("Error: GpuAllocCache::alloc called while active\n");
        exit(1);
    }

    if (size > capacity) {
        if (buffer) {
            CUDA_CHECK(cudaFree(buffer));
        }
        CUDA_CHECK(cudaMalloc(&buffer, size));
        CUDA_CHECK(cudaMemset(buffer, 0, size));
        capacity = size;
    }

    active = true;
    return buffer;
}

void GpuAllocCache::reset() {
    if (active) {
        CUDA_CHECK(cudaMemset(buffer, 0, capacity));
    }
    active = false;
}

void rle_compress_cpu(
    uint32_t raw_count,
    char const *raw,
    std::vector<char> &compressed_data,
    std::vector<uint32_t> &compressed_lengths) {
    compressed_data.clear();
    compressed_lengths.clear();

    uint32_t i = 0;
    while (i < raw_count) {
        char c = raw[i];
        uint32_t run_length = 1;
        i++;
        while (i < raw_count && raw[i] == c) {
            run_length++;
            i++;
        }
        compressed_data.push_back(c);
        compressed_lengths.push_back(run_length);
    }
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
        double this_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms);
    }
    return best_time_ms;
}

struct Results {
    double time_ms;
};

enum class Mode {
    TEST,
    BENCHMARK,
};

Results run_config(Mode mode, std::vector<char> const &original_raw) {
    auto compressed_data = std::vector<char>();
    auto compressed_lengths = std::vector<uint32_t>();

    rle_compress_cpu(
        original_raw.size(),
        original_raw.data(),
        compressed_data,
        compressed_lengths);

    char *compressed_data_gpu;
    uint32_t *compressed_lengths_gpu;

    CUDA_CHECK(cudaMalloc(&compressed_data_gpu, compressed_data.size()));
    CUDA_CHECK(cudaMalloc(
        &compressed_lengths_gpu,
        compressed_lengths.size() * sizeof(uint32_t)));

    auto workspace_alloc_1 = GpuAllocCache();
    auto workspace_alloc_2 = GpuAllocCache();

    CUDA_CHECK(cudaMemcpy(
        compressed_data_gpu,
        compressed_data.data(),
        compressed_data.size(),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        compressed_lengths_gpu,
        compressed_lengths.data(),
        compressed_lengths.size() * sizeof(uint32_t),
        cudaMemcpyHostToDevice));

    auto reset = [&]() {
        workspace_alloc_1.reset();
        workspace_alloc_2.reset();
    };

    auto f = [&]() {
        rle_gpu::launch_rle_decompress(
            compressed_data.size(),
            compressed_data_gpu,
            compressed_lengths_gpu,
            workspace_alloc_1,
            workspace_alloc_2);
    };

    auto decompressed = rle_gpu::launch_rle_decompress(
        compressed_data.size(),
        compressed_data_gpu,
        compressed_lengths_gpu,
        workspace_alloc_1,
        workspace_alloc_2);

    std::vector<char> raw(decompressed.count);

    CUDA_CHECK(cudaMemcpy(
        raw.data(),
        decompressed.data,
        decompressed.count,
        cudaMemcpyDeviceToHost));

    bool correct = true;

    if (raw.size() != original_raw.size()) {
        printf("Mismatch in decompressed size:\n");
        printf("  Expected: %zu\n", original_raw.size());
        printf("  Actual:   %zu\n", raw.size());
        correct = false;
    }

    if (correct) {
        for (size_t i = 0; i < raw.size(); i++) {
            if (raw[i] != original_raw[i]) {
                printf("Mismatch in decompressed data at index %zu:\n", i);
                printf(
                    "  Expected: 0x%02x\n",
                    static_cast<unsigned char>(original_raw[i]));
                printf(
                    "  Actual:   0x%02x\n",
                    static_cast<unsigned char>(raw[i]));
                correct = false;
                break;
            }
        }
    }

    if (!correct) {
        exit(1);
    }

    if (mode == Mode::TEST) {
        return {};
    }

    double target_time_ms = 1000.0;
    double time_ms = benchmark_ms(target_time_ms, reset, f);

    CUDA_CHECK(cudaFree(compressed_data_gpu));
    CUDA_CHECK(cudaFree(compressed_lengths_gpu));

    return {time_ms};
}

template <typename Rng>
std::vector<char> generate_test_data(uint32_t size, Rng &rng) {
    auto random_byte = std::uniform_int_distribution<int32_t>(
        std::numeric_limits<char>::min(),
        std::numeric_limits<char>::max());

    constexpr uint32_t alphabet_size = 4;
    auto alphabet = std::vector<char>();

    for (uint32_t i = 0; i < alphabet_size; i++) {
        alphabet.push_back(random_byte(rng));
    }

    auto random_symbol =
        std::uniform_int_distribution<uint32_t>(0, alphabet_size - 1);

    auto data = std::vector<char>();

    for (uint32_t i = 0; i < size; i++) {
        data.push_back(alphabet.at(random_symbol(rng)));
    }

    return data;
}

template <typename Rng>
std::vector<char> generate_sparse(uint32_t size, uint32_t nonzero_count, Rng &rng) {
    auto data = std::vector<char>(size, 0);

    auto random_index = std::uniform_int_distribution<uint32_t>(0, size - 1);
    auto random_byte = std::uniform_int_distribution<int32_t>(
        std::numeric_limits<char>::min(),
        std::numeric_limits<char>::max());

    for (uint32_t i = 0; i < nonzero_count; i++) {
        data.at(random_index(rng)) = random_byte(rng);
    }

    char fill = random_byte(rng);

    for (uint32_t i = 0; i < size; i++) {
        if (data.at(i) == 0) {
            data.at(i) = fill;
        } else {
            fill = random_byte(rng);
        }
    }

    return data;
}

int main(int argc, char const *const *argv) {
    auto rng = std::mt19937(0xCA7CAFE);

    auto test_sizes = std::vector<uint32_t>{
        16,
        10,
        128,
        100,
        1 << 10,
        1000,
        1 << 20,
        1'000'000,
        16 << 20,
    };

    printf("Correctness:\n\n");

    for (auto test_size : test_sizes) {
        auto raw = generate_test_data(test_size, rng);
        printf("  Testing decompression for size %u\n", test_size);
        run_config(Mode::TEST, raw);
        printf("  OK\n\n");
    }

    auto test_data_search_paths = std::vector<std::string>{".", "/"};
    std::string test_data_path;

    for (auto test_data_search_path : test_data_search_paths) {
        auto candidate_path = test_data_search_path + "/rle_raw.bmp";
        if (std::filesystem::exists(candidate_path)) {
            test_data_path = candidate_path;
            break;
        }
    }

    if (test_data_path.empty()) {
        printf("Could not find test data file.\n");
        exit(1);
    }

    auto raw = std::vector<char>();

    {
        auto file = std::ifstream(test_data_path, std::ios::binary);
        if (!file) {
            printf("Could not open test data file '%s'.\n", test_data_path.c_str());
            exit(1);
        }

        file.seekg(0, std::ios::end);
        raw.resize(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(raw.data(), raw.size());
    }

    printf("Performance:\n\n");

    printf("  Testing decompression on file 'rle_raw.bmp' (size %zu)\n", raw.size());
    auto results = run_config(Mode::BENCHMARK, raw);
    printf("  Time: %.2f ms\n", results.time_ms);

    auto raw_sparse = generate_sparse(16 << 20, 1 << 10, rng);

    printf("\n  Testing decompression on sparse data (size %u)\n", 16 << 20);
    results = run_config(Mode::BENCHMARK, raw_sparse);
    printf("  Time: %.2f ms\n", results.time_ms);

    return 0;
}