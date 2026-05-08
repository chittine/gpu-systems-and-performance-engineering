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
// CPU Reference Implementation

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

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

namespace rle_gpu {

constexpr int BLOCK_SIZE = 256;
constexpr int ITEMS_PER_THREAD = 4;
constexpr int ELEMS_PER_BLOCK = BLOCK_SIZE * ITEMS_PER_THREAD;

__global__ void scan_flags_block_kernel(
    uint32_t n,
    char const *raw,
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
            uint32_t flag = (idx == 0 || raw[idx] != raw[idx - 1]) ? 1u : 0u;
            running += flag;
            local[k] = running;
        } else {
            local[k] = 0;
        }
    }

    shmem[tid] = running;
    __syncthreads();

    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        uint32_t val = shmem[tid];
        if (tid >= offset) val += shmem[tid - offset];
        __syncthreads();
        shmem[tid] = val;
        __syncthreads();
    }

    uint32_t prefix = (tid == 0) ? 0 : shmem[tid - 1];

    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
        uint32_t idx = thread_base + k;
        if (idx < n) out[idx] = prefix + local[k];
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
        if (tid >= offset) val += shmem[tid - offset];
        __syncthreads();
        shmem[tid] = val;
        __syncthreads();
    }

    uint32_t prefix = (tid == 0) ? 0 : shmem[tid - 1];

    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
        uint32_t idx = thread_base + k;
        if (idx < n) out[idx] = prefix + local[k];
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
        if (idx < n) out[idx] += prefix;
    }
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

uint32_t *launch_scan_uint32(uint32_t n, uint32_t *x, void *workspace) {
    uint32_t *workspace_u32 = reinterpret_cast<uint32_t *>(workspace);
    if (n == 0) return workspace_u32;

    uint32_t num_blocks = (n + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
    uint32_t *out = workspace_u32;
    size_t shmem_size = BLOCK_SIZE * sizeof(uint32_t);

    if (num_blocks == 1) {
        scan_block_kernel<<<1, BLOCK_SIZE, shmem_size>>>(n, x, out, nullptr);
        CUDA_CHECK(cudaGetLastError());
        return out;
    }

    uint32_t *block_sums = out + n;
    void *recursive_workspace =
        reinterpret_cast<void *>(block_sums + num_blocks);

    scan_block_kernel<<<num_blocks, BLOCK_SIZE, shmem_size>>>(
        n, x, out, block_sums);
    CUDA_CHECK(cudaGetLastError());

    uint32_t *scanned_block_sums =
        launch_scan_uint32(num_blocks, block_sums, recursive_workspace);

    add_block_prefix_kernel<<<num_blocks, BLOCK_SIZE>>>(
        n, out, scanned_block_sums);
    CUDA_CHECK(cudaGetLastError());

    return out;
}

uint32_t *launch_scan_flags(
    uint32_t n,
    char const *raw,
    void *workspace
) {
    uint32_t *workspace_u32 = reinterpret_cast<uint32_t *>(workspace);
    uint32_t num_blocks = (n + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
    uint32_t *out = workspace_u32;
    size_t shmem_size = BLOCK_SIZE * sizeof(uint32_t);

    if (num_blocks == 1) {
        scan_flags_block_kernel<<<1, BLOCK_SIZE, shmem_size>>>(n, raw, out, nullptr);
        CUDA_CHECK(cudaGetLastError());
        return out;
    }

    uint32_t *block_sums = out + n;
    void *recursive_workspace =
        reinterpret_cast<void *>(block_sums + num_blocks);

    scan_flags_block_kernel<<<num_blocks, BLOCK_SIZE, shmem_size>>>(
        n, raw, out, block_sums);
    CUDA_CHECK(cudaGetLastError());

    uint32_t *scanned_block_sums =
        launch_scan_uint32(num_blocks, block_sums, recursive_workspace);

    add_block_prefix_kernel<<<num_blocks, BLOCK_SIZE>>>(
        n, out, scanned_block_sums);
    CUDA_CHECK(cudaGetLastError());

    return out;
}

__global__ void scatter_runs_kernel(
    uint32_t n,
    char const *raw,
    uint32_t const *scan,
    char *compressed_data,
    uint32_t *run_starts
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (i == 0 || raw[i] != raw[i - 1]) {
        uint32_t pos = scan[i] - 1;
        compressed_data[pos] = raw[i];
        run_starts[pos] = i;
    }
}

__global__ void make_lengths_kernel(
    uint32_t raw_count,
    uint32_t compressed_count,
    uint32_t const *run_starts,
    uint32_t *compressed_lengths
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= compressed_count) return;

    uint32_t start = run_starts[i];
    uint32_t end = (i + 1 < compressed_count)
        ? run_starts[i + 1]
        : raw_count;

    compressed_lengths[i] = end - start;
}

size_t get_workspace_size(uint32_t raw_count) {
    return scan_workspace_size(raw_count)
         + raw_count * sizeof(uint32_t);
}

uint32_t launch_rle_compress(
    uint32_t raw_count,
    char const *raw,
    void *workspace,
    char *compressed_data,
    uint32_t *compressed_lengths
) {
    if (raw_count == 0) return 0;

    uint32_t *scan_workspace = reinterpret_cast<uint32_t *>(workspace);
    uint32_t *run_starts =
        reinterpret_cast<uint32_t *>(
            reinterpret_cast<char *>(workspace) + scan_workspace_size(raw_count));

    int threads = 256;
    int blocks = (raw_count + threads - 1) / threads;

    uint32_t *scanned_flags =
        launch_scan_flags(raw_count, raw, scan_workspace);

    uint32_t compressed_count = 0;
    CUDA_CHECK(cudaMemcpy(
        &compressed_count,
        scanned_flags + raw_count - 1,
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost));

    scatter_runs_kernel<<<blocks, threads>>>(
        raw_count,
        raw,
        scanned_flags,
        compressed_data,
        run_starts);
    CUDA_CHECK(cudaGetLastError());

    int length_blocks = (compressed_count + threads - 1) / threads;

    make_lengths_kernel<<<length_blocks, threads>>>(
        raw_count,
        compressed_count,
        run_starts,
        compressed_lengths);
    CUDA_CHECK(cudaGetLastError());

    return compressed_count;
}

} // namespace rle_gpu

////////////////////////////////////////////////////////////////////////////////
// Code Below Here Unchanged

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

Results run_config(Mode mode, std::vector<char> const &raw) {
    size_t workspace_size = rle_gpu::get_workspace_size(raw.size());

    char *raw_gpu;
    void *workspace;
    char *compressed_data_gpu;
    uint32_t *compressed_lengths_gpu;

    CUDA_CHECK(cudaMalloc(&raw_gpu, raw.size()));
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    CUDA_CHECK(cudaMalloc(&compressed_data_gpu, raw.size()));
    CUDA_CHECK(cudaMalloc(&compressed_lengths_gpu, raw.size() * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(raw_gpu, raw.data(), raw.size(), cudaMemcpyHostToDevice));

    auto reset = [&]() {
        CUDA_CHECK(cudaMemset(compressed_data_gpu, 0, raw.size()));
        CUDA_CHECK(cudaMemset(
            compressed_lengths_gpu,
            0,
            raw.size() * sizeof(uint32_t)));
    };

    auto f = [&]() {
        rle_gpu::launch_rle_compress(
            raw.size(),
            raw_gpu,
            workspace,
            compressed_data_gpu,
            compressed_lengths_gpu);
    };

    reset();

    uint32_t compressed_count = rle_gpu::launch_rle_compress(
        raw.size(),
        raw_gpu,
        workspace,
        compressed_data_gpu,
        compressed_lengths_gpu);

    std::vector<char> compressed_data(compressed_count);
    std::vector<uint32_t> compressed_lengths(compressed_count);

    CUDA_CHECK(cudaMemcpy(
        compressed_data.data(),
        compressed_data_gpu,
        compressed_count,
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        compressed_lengths.data(),
        compressed_lengths_gpu,
        compressed_count * sizeof(uint32_t),
        cudaMemcpyDeviceToHost));

    std::vector<char> compressed_data_expected;
    std::vector<uint32_t> compressed_lengths_expected;

    rle_compress_cpu(
        raw.size(),
        raw.data(),
        compressed_data_expected,
        compressed_lengths_expected);

    bool correct = true;

    if (compressed_count != compressed_data_expected.size()) {
        printf("Mismatch in compressed count:\n");
        printf("  Expected: %zu\n", compressed_data_expected.size());
        printf("  Actual:   %u\n", compressed_count);
        correct = false;
    }

    if (correct) {
        for (size_t i = 0; i < compressed_data_expected.size(); i++) {
            if (compressed_data[i] != compressed_data_expected[i]) {
                printf("Mismatch in compressed data at index %zu:\n", i);
                printf(
                    "  Expected: 0x%02x\n",
                    static_cast<unsigned char>(compressed_data_expected[i]));
                printf(
                    "  Actual:   0x%02x\n",
                    static_cast<unsigned char>(compressed_data[i]));
                correct = false;
                break;
            }

            if (compressed_lengths[i] != compressed_lengths_expected[i]) {
                printf("Mismatch in compressed lengths at index %zu:\n", i);
                printf("  Expected: %u\n", compressed_lengths_expected[i]);
                printf("  Actual:   %u\n", compressed_lengths[i]);
                correct = false;
                break;
            }
        }
    }

    if (!correct) {
        if (raw.size() <= 1024) {
            printf("\nInput:\n");
            for (size_t i = 0; i < raw.size(); i++) {
                printf("  [%4zu] = 0x%02x\n",
                       i,
                       static_cast<unsigned char>(raw[i]));
            }

            printf("\nExpected:\n");
            for (size_t i = 0; i < compressed_data_expected.size(); i++) {
                printf(
                    "  [%4zu] = data: 0x%02x, length: %u\n",
                    i,
                    static_cast<unsigned char>(compressed_data_expected[i]),
                    compressed_lengths_expected[i]);
            }

            printf("\nActual:\n");
            if (compressed_data.size() == 0) {
                printf("  (empty)\n");
            }

            for (size_t i = 0; i < compressed_data.size(); i++) {
                printf(
                    "  [%4zu] = data: 0x%02x, length: %u\n",
                    i,
                    static_cast<unsigned char>(compressed_data[i]),
                    compressed_lengths[i]);
            }
        }

        exit(1);
    }

    if (mode == Mode::TEST) {
        return {};
    }

    double target_time_ms = 1000.0;
    double time_ms = benchmark_ms(target_time_ms, reset, f);

    CUDA_CHECK(cudaFree(raw_gpu));
    CUDA_CHECK(cudaFree(workspace));
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
        printf("  Testing compression for size %u\n", test_size);
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
    printf("  Testing compression on file 'rle_raw.bmp' (size %zu)\n", raw.size());

    auto results = run_config(Mode::BENCHMARK, raw);

    printf("  Time: %.2f ms\n", results.time_ms);

    return 0;
}