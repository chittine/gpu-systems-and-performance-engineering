#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

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

__device__ inline void cp_async4(void *smem_ptr, const void *glob_ptr) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" ::"r"(smem),
        "l"(glob_ptr),
        "n"(BYTES));
}

__device__ __forceinline__ void async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N> __device__ __forceinline__ void async_wait_pending() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Too slow to actually run!)
//
// void matmul_cpu_naive(
//     int32_t size_i,
//     int32_t size_j,
//     int32_t size_k,
//     float const *a,
//     float const *b,
//     float *c) {
//     for (int32_t i = 0; i < size_i; ++i) {
//         for (int32_t j = 0; j < size_j; ++j) {
//             float sum = 0.0;
//             for (int32_t k = 0; k < size_k; ++k) {
//                 sum += a[i * size_k + k] * b[k * size_j + j];
//             }
//             c[i * size_j + j] = sum;
//         }
//     }
// }

/// <--- your code here --->

/*
    // OPTIONAL: Uncomment this block to include your kernel implementation
    // from Lab 4 for easy comparison.

    ////////////////////////////////////////////////////////////////////////////////
    // GPU Implementation with Reuse in L1/Shmem and Registers (Baseline from Lab 4)

    #define HAS_LAB_4_BASELINE_IMPL // <~~ keep this line if you want to benchmark your
   Lab 4 kernel!

    namespace matmul_l1_reg {

    __global__ void matmul_l1_reg(
        int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
        // TODO: your GPU code here
    }

    void launch_matmul_l1_reg(
        int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
        // TODO: your CPU code here
    }

    } // namespace matmul_l1_reg
*/

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

namespace matmul_improved {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;

constexpr int TM = 4;
constexpr int TN = 4;

constexpr int THREADS_X = BN / TN;   // 16
constexpr int THREADS_Y = BM / TM;   // 16

__global__ void matmul_improved(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *__restrict__ a,
    float const *__restrict__ b,
    float *__restrict__ c)
{
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;   // 0..255

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    #pragma unroll
    for (int kk = 0; kk < size_k; kk += BK) {
        {
            const int a_row = tid / 4;      
            const int a_vec = tid % 4;      
            const int a_col = a_vec * 4;    

            const int global_a_row = block_row + a_row;
            const int global_a_col = kk + a_col;

            if (global_a_row < size_i && global_a_col + 3 < size_k) {
                float4 v = *reinterpret_cast<float4 const *>(
                    &a[global_a_row * size_k + global_a_col]);
                As[a_row][a_col + 0] = v.x;
                As[a_row][a_col + 1] = v.y;
                As[a_row][a_col + 2] = v.z;
                As[a_row][a_col + 3] = v.w;
            } else {
                #pragma unroll
                for (int t = 0; t < 4; t++) {
                    int gc = global_a_col + t;
                    As[a_row][a_col + t] =
                        (global_a_row < size_i && gc < size_k)
                            ? a[global_a_row * size_k + gc]
                            : 0.0f;
                }
            }
        }

        {
            const int b_row = tid / 16;     // 0..15
            const int b_vec = tid % 16;     // 0..15
            const int b_col = b_vec * 4;    // 0,4,8,...,60

            const int global_b_row = kk + b_row;
            const int global_b_col = block_col + b_col;

            if (global_b_row < size_k && global_b_col + 3 < size_j) {
                float4 v = *reinterpret_cast<float4 const *>(
                    &b[global_b_row * size_j + global_b_col]);
                Bs[b_row][b_col + 0] = v.x;
                Bs[b_row][b_col + 1] = v.y;
                Bs[b_row][b_col + 2] = v.z;
                Bs[b_row][b_col + 3] = v.w;
            } else {
                #pragma unroll
                for (int t = 0; t < 4; t++) {
                    int gc = global_b_col + t;
                    Bs[b_row][b_col + t] =
                        (global_b_row < size_k && gc < size_j)
                            ? b[global_b_row * size_j + gc]
                            : 0.0f;
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float a0 = As[ty * TM + 0][k];
            float a1 = As[ty * TM + 1][k];
            float a2 = As[ty * TM + 2][k];
            float a3 = As[ty * TM + 3][k];

            float b0 = Bs[k][tx * TN + 0];
            float b1 = Bs[k][tx * TN + 1];
            float b2 = Bs[k][tx * TN + 2];
            float b3 = Bs[k][tx * TN + 3];

            acc[0][0] = fmaf(a0, b0, acc[0][0]);
            acc[0][1] = fmaf(a0, b1, acc[0][1]);
            acc[0][2] = fmaf(a0, b2, acc[0][2]);
            acc[0][3] = fmaf(a0, b3, acc[0][3]);

            acc[1][0] = fmaf(a1, b0, acc[1][0]);
            acc[1][1] = fmaf(a1, b1, acc[1][1]);
            acc[1][2] = fmaf(a1, b2, acc[1][2]);
            acc[1][3] = fmaf(a1, b3, acc[1][3]);

            acc[2][0] = fmaf(a2, b0, acc[2][0]);
            acc[2][1] = fmaf(a2, b1, acc[2][1]);
            acc[2][2] = fmaf(a2, b2, acc[2][2]);
            acc[2][3] = fmaf(a2, b3, acc[2][3]);

            acc[3][0] = fmaf(a3, b0, acc[3][0]);
            acc[3][1] = fmaf(a3, b1, acc[3][1]);
            acc[3][2] = fmaf(a3, b2, acc[3][2]);
            acc[3][3] = fmaf(a3, b3, acc[3][3]);
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        const int global_row = block_row + ty * TM + i;
        if (global_row < size_i) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                const int global_col = block_col + tx * TN + j;
                if (global_col < size_j) {
                    c[global_row * size_j + global_col] = acc[i][j];
                }
            }
        }
    }
}

void launch_matmul_improved(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c)
{
    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid(
        (size_j + BN - 1) / BN,
        (size_i + BM - 1) / BM
    );

    matmul_improved<<<grid, block>>>(size_i, size_j, size_k, a, b, c);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace matmul_improved


////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation with Reduction along k

// Optimized GPU Implementation with Reduction along k

namespace matmul_improved_reduce {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;

constexpr int TM = 4;
constexpr int TN = 4;

constexpr int THREADS_X = BN / TN;   // 16
constexpr int THREADS_Y = BM / TM;   // 16

__host__ __device__ inline int choose_split_k(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k)
{
    int tiles_i = (size_i + BM - 1) / BM;
    int tiles_j = (size_j + BN - 1) / BN;
    int tiles = tiles_i * tiles_j;

    if (size_k >= 16384 && tiles < 16) return 16;
    if (size_k >= 4096  && tiles < 32) return 8;
    if (size_k >= 2048  && tiles < 48) return 4;
    return 1;
}

__global__ void matmul_splitk_partial(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    int32_t k_chunk,
    float const *__restrict__ a,
    float const *__restrict__ b,
    float *__restrict__ workspace)
{
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    const int split_idx = blockIdx.z;
    const int k_begin = split_idx * k_chunk;
    const int k_end = min(size_k, k_begin + k_chunk);

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            acc[i][j] = 0.0f;
        }
    }

    for (int kk = k_begin; kk < k_end; kk += BK) {
        #pragma unroll
        for (int load = 0; load < 4; load++) {
            int idx = tid + load * 256;
            int a_row = idx / BK;
            int a_col = idx % BK;

            int global_a_row = block_row + a_row;
            int global_a_col = kk + a_col;

            if (global_a_row < size_i && global_a_col < k_end) {
                As[a_row][a_col] = a[global_a_row * size_k + global_a_col];
            } else {
                As[a_row][a_col] = 0.0f;
            }
        }

        #pragma unroll
        for (int load = 0; load < 4; load++) {
            int idx = tid + load * 256;
            int b_row = idx / BN;
            int b_col = idx % BN;

            int global_b_row = kk + b_row;
            int global_b_col = block_col + b_col;

            if (global_b_row < k_end && global_b_col < size_j) {
                Bs[b_row][b_col] = b[global_b_row * size_j + global_b_col];
            } else {
                Bs[b_row][b_col] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float a0 = As[ty * TM + 0][k];
            float a1 = As[ty * TM + 1][k];
            float a2 = As[ty * TM + 2][k];
            float a3 = As[ty * TM + 3][k];

            float b0 = Bs[k][tx * TN + 0];
            float b1 = Bs[k][tx * TN + 1];
            float b2 = Bs[k][tx * TN + 2];
            float b3 = Bs[k][tx * TN + 3];

            acc[0][0] = fmaf(a0, b0, acc[0][0]);
            acc[0][1] = fmaf(a0, b1, acc[0][1]);
            acc[0][2] = fmaf(a0, b2, acc[0][2]);
            acc[0][3] = fmaf(a0, b3, acc[0][3]);

            acc[1][0] = fmaf(a1, b0, acc[1][0]);
            acc[1][1] = fmaf(a1, b1, acc[1][1]);
            acc[1][2] = fmaf(a1, b2, acc[1][2]);
            acc[1][3] = fmaf(a1, b3, acc[1][3]);

            acc[2][0] = fmaf(a2, b0, acc[2][0]);
            acc[2][1] = fmaf(a2, b1, acc[2][1]);
            acc[2][2] = fmaf(a2, b2, acc[2][2]);
            acc[2][3] = fmaf(a2, b3, acc[2][3]);

            acc[3][0] = fmaf(a3, b0, acc[3][0]);
            acc[3][1] = fmaf(a3, b1, acc[3][1]);
            acc[3][2] = fmaf(a3, b2, acc[3][2]);
            acc[3][3] = fmaf(a3, b3, acc[3][3]);
        }

        __syncthreads();
    }

    float *partial = workspace + (size_t)split_idx * size_i * size_j;

    int row0 = block_row + ty * TM + 0;
    int row1 = block_row + ty * TM + 1;
    int row2 = block_row + ty * TM + 2;
    int row3 = block_row + ty * TM + 3;

    int col0 = block_col + tx * TN + 0;
    int col1 = block_col + tx * TN + 1;
    int col2 = block_col + tx * TN + 2;
    int col3 = block_col + tx * TN + 3;

    if (row0 < size_i) {
        if (col0 < size_j) partial[row0 * size_j + col0] = acc[0][0];
        if (col1 < size_j) partial[row0 * size_j + col1] = acc[0][1];
        if (col2 < size_j) partial[row0 * size_j + col2] = acc[0][2];
        if (col3 < size_j) partial[row0 * size_j + col3] = acc[0][3];
    }
    if (row1 < size_i) {
        if (col0 < size_j) partial[row1 * size_j + col0] = acc[1][0];
        if (col1 < size_j) partial[row1 * size_j + col1] = acc[1][1];
        if (col2 < size_j) partial[row1 * size_j + col2] = acc[1][2];
        if (col3 < size_j) partial[row1 * size_j + col3] = acc[1][3];
    }
    if (row2 < size_i) {
        if (col0 < size_j) partial[row2 * size_j + col0] = acc[2][0];
        if (col1 < size_j) partial[row2 * size_j + col1] = acc[2][1];
        if (col2 < size_j) partial[row2 * size_j + col2] = acc[2][2];
        if (col3 < size_j) partial[row2 * size_j + col3] = acc[2][3];
    }
    if (row3 < size_i) {
        if (col0 < size_j) partial[row3 * size_j + col0] = acc[3][0];
        if (col1 < size_j) partial[row3 * size_j + col1] = acc[3][1];
        if (col2 < size_j) partial[row3 * size_j + col2] = acc[3][2];
        if (col3 < size_j) partial[row3 * size_j + col3] = acc[3][3];
    }
}

__global__ void reduce_splitk(
    int32_t size_i,
    int32_t size_j,
    int32_t split_k,
    float const *__restrict__ workspace,
    float *__restrict__ c)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= size_i || col >= size_j) return;

    float sum = 0.0f;
    size_t base = (size_t)row * size_j + col;

    for (int s = 0; s < split_k; s++) {
        sum += workspace[(size_t)s * size_i * size_j + base];
    }

    c[base] = sum;
}

size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
    int split_k = choose_split_k(size_i, size_j, size_k);
    if (split_k <= 1) return 0;
    return (size_t)split_k * size_i * size_j * sizeof(float);
}

void launch_matmul_improved_reduce(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c,
    void *workspace
) {
    int split_k = choose_split_k(size_i, size_j, size_k);

    if (split_k <= 1) {
        matmul_improved::launch_matmul_improved(size_i, size_j, size_k, a, b, c);
        return;
    }

    float *partial = reinterpret_cast<float *>(workspace);
    int k_chunk = (size_k + split_k - 1) / split_k;

    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid(
        (size_j + BN - 1) / BN,
        (size_i + BM - 1) / BM,
        split_k
    );

    matmul_splitk_partial<<<grid, block>>>(
        size_i, size_j, size_k, k_chunk, a, b, partial);
    CUDA_CHECK(cudaGetLastError());

    dim3 red_block(16, 16);
    dim3 red_grid(
        (size_j + red_block.x - 1) / red_block.x,
        (size_i + red_block.y - 1) / red_block.y
    );

    reduce_splitk<<<red_grid, red_block>>>(size_i, size_j, split_k, partial, c);
    CUDA_CHECK(cudaGetLastError());
}

}; // namespace matmul_improved_reduce
/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

std::vector<float> read_data(std::string const &path, int32_t size) {
    std::ifstream file(path, std::ios::binary);
    std::vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
    if (file.fail()) {
        std::cerr << "Failed to read " << path << std::endl;
        std::abort();
    }
    return data;
}

template <typename Reset, typename F>
double
benchmark_ms(double target_time_ms, int32_t num_iters_inner, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < num_iters_inner; ++i) {
            f();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms / num_iters_inner);
    }
    return best_time_ms;
}

struct BenchmarkConfig {
    int32_t size_i;
    int32_t size_j;
    int32_t size_k;
};

struct TestData {
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> a;
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> b;
    std::map<std::tuple<int32_t, int32_t, int32_t>, std::vector<float>> c;
};

TestData read_test_data(
    std::string const &test_data_dir,
    std::vector<BenchmarkConfig> const &configs) {
    auto data = TestData{};
    for (auto const &config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;

        auto path_prefix = test_data_dir + "/test_";

        if (data.a.find({size_i, size_k}) == data.a.end()) {
            data.a[{size_i, size_k}] = read_data(
                path_prefix + "a_" + std::to_string(size_i) + "x" +
                    std::to_string(size_k) + ".bin",
                size_i * size_k);
        }

        if (data.b.find({size_k, size_j}) == data.b.end()) {
            data.b[{size_k, size_j}] = read_data(
                path_prefix + "b_" + std::to_string(size_k) + "x" +
                    std::to_string(size_j) + ".bin",
                size_k * size_j);
        }

        if (data.c.find({size_i, size_j, size_k}) == data.c.end()) {
            data.c[{size_i, size_j, size_k}] = read_data(
                path_prefix + "c_" + std::to_string(size_i) + "x" +
                    std::to_string(size_j) + "x" + std::to_string(size_k) + ".bin",
                size_i * size_j);
        }
    }
    return data;
}

struct BenchmarkResults {
    char const *name;
    std::map<std::tuple<int32_t, int32_t, int32_t>, double> elapsed_ms;
};

enum class Phase {
    WARMUP,
    BENCHMARK,
};

template <typename Impl>
void run_config(
    Phase phase,
    TestData const &data,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {
    auto size_i = config.size_i;
    auto size_j = config.size_j;
    auto size_k = config.size_k;

    auto const &a = data.a.at({size_i, size_k});
    auto const &b = data.b.at({size_k, size_j});
    auto const &c = data.c.at({size_i, size_j, size_k});

    float *a_gpu;
    float *b_gpu;
    float *c_gpu;
    CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_gpu, size_k * size_j * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c_gpu, size_i * size_j * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        a_gpu,
        a.data(),
        size_i * size_k * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        b_gpu,
        b.data(),
        size_k * size_j * sizeof(float),
        cudaMemcpyHostToDevice));

    size_t workspace_size = Impl::get_workspace_size(size_i, size_j, size_k);
    void *workspace_gpu = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_gpu, workspace_size));
        CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
    }

    void *flush_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&flush_gpu, 1024*1024*64));
    CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));

    if (phase == Phase::BENCHMARK) {
        printf("  %6d  %6d  %6d", size_i, size_j, size_k);
    } else {
        printf("  warmup %6d  %6d  %6d", size_i, size_j, size_k);
    }

    Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu, workspace_gpu);

    std::vector<float> c_out_host(size_i * size_j);
    CUDA_CHECK(cudaMemcpy(
        c_out_host.data(),
        c_gpu,
        size_i * size_j * sizeof(float),
        cudaMemcpyDeviceToHost));

    double mse = 0.0;
    double ref_mean_square = 0.0;
    for (int32_t i = 0; i < size_i; ++i) {
        for (int32_t j = 0; j < size_j; ++j) {
            float diff = c_out_host[i * size_j + j] - c[i * size_j + j];
            mse += diff * diff;
            ref_mean_square += c[i * size_j + j] * c[i * size_j + j];
        }
    }
    mse /= size_i * size_j;
    ref_mean_square /= size_i * size_j;
    float rmse = std::sqrt(mse);
    float rel_rmse = rmse / std::sqrt(ref_mean_square);

    if (phase == Phase::BENCHMARK) {
        printf("  %8.02e", rel_rmse);
    }

    if (rel_rmse > 1e-5) {
        if (phase == Phase::BENCHMARK) {
            printf("  %9s  %7s", "-", "-");
        }
    } else {
        double target_time_ms = 40.0;
        double elapsed_ms = 0.0;
        if (phase == Phase::BENCHMARK) {
            elapsed_ms = benchmark_ms(
                target_time_ms,
                1,
                [&]() {
                    if (workspace_size > 0) {
                        CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
                    }
                    CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));
                },
                [&]() {
                    Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu, workspace_gpu);
                });
        } else {
            elapsed_ms = benchmark_ms(
                target_time_ms,
                1,
                [&]() {
                    if (workspace_size > 0) {
                        CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
                    }
                    CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));
                },
                [&]() {
                    Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu, workspace_gpu);
                }); 
        }

        if (phase == Phase::BENCHMARK) {
            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            printf("  %9.02f  %7.02f", elapsed_ms, tflop / (elapsed_ms * 1e-3));

            results.elapsed_ms[{size_i, size_j, size_k}] = elapsed_ms;
        }
    }

    printf("\n");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    CUDA_CHECK(cudaFree(flush_gpu));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaFree(workspace_gpu));
    }
}

template <typename Impl>
BenchmarkResults run_all_configs(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = BenchmarkResults{Impl::name};
    if (phase == Phase::WARMUP) {
        printf("warmup %s:\n\n", Impl::name);
    } else {
        printf("%s:\n\n", Impl::name);
        printf(
            "  %-6s  %-6s  %-6s  %-8s  %-9s  %-7s\n",
            "size_i",
            "size_j",
            "size_k",
            "RRMSE",
            "time (ms)",
            "TFLOP/s");
        printf(
            "  %-6s  %-6s  %-6s  %-8s  %-9s  %-7s\n",
            "------",
            "------",
            "------",
            "--------",
            "---------",
            "-------");
    }
    for (auto const &config : configs) {
        run_config<Impl>(phase, data, config, results);
    }
    printf("\n");
    return results;
}

#ifdef HAS_LAB_4_BASELINE_IMPL

struct MatmulL1Reg {
    constexpr static char const *name = "matmul_l1_reg";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return 0;
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_l1_reg::launch_matmul_l1_reg(size_i, size_j, size_k, a, b, c);
    }
};

#endif

struct MatmulImproved {
    constexpr static char const *name = "matmul_improved";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return 0;
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_improved::launch_matmul_improved(size_i, size_j, size_k, a, b, c);
    }
};

struct MatmulImprovedReduce {
    constexpr static char const *name = "matmul_improved_reduce";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return matmul_improved_reduce::get_workspace_size(size_i, size_j, size_k);
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_improved_reduce::launch_matmul_improved_reduce(
            size_i,
            size_j,
            size_k,
            a,
            b,
            c,
            workspace);
    }
};

std::vector<BenchmarkResults> run_all_impls(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = std::vector<BenchmarkResults>{};
#ifdef HAS_LAB_4_BASELINE_IMPL
    results.push_back(run_all_configs<MatmulL1Reg>(phase, data, configs));
#endif
    results.push_back(run_all_configs<MatmulImproved>(phase, data, configs));
    results.push_back(run_all_configs<MatmulImprovedReduce>(phase, data, configs));
    return results;
}

void write_json_results(
    std::string const &path,
    std::vector<BenchmarkResults> const &results) {
    auto file = std::ofstream(path);
    file << "{\n";
    for (int32_t i = 0; i < results.size(); ++i) {
        auto const &result = results.at(i);
        file << "  \"" << result.name << "\": [\n";
        int32_t j = 0;
        for (auto const &[config, elapsed_ms] : result.elapsed_ms) {
            auto [size_i, size_j, size_k] = config;
            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            double tflop_per_sec = tflop / (elapsed_ms * 1e-3);
            file << "    {\n";
            file << "      \"size_i\": " << size_i << ",\n";
            file << "      \"size_j\": " << size_j << ",\n";
            file << "      \"size_k\": " << size_k << ",\n";
            file << "      \"elapsed_ms\": " << elapsed_ms << ",\n";
            file << "      \"tflop_per_sec\": " << tflop_per_sec << "\n";
            file << "    }";
            if (j + 1 < result.elapsed_ms.size()) {
                file << ",";
            }
            file << "\n";
            ++j;
        }
        file << "  ]";
        if (i + 1 < results.size()) {
            file << ",";
        }
        file << "\n";
    }
    file << "}\n";
}

int main(int argc, char **argv) {
    std::string test_data_dir = ".";


    auto configs = std::vector<BenchmarkConfig>{
        {3072, 3072, 3072},
        {512, 3072, 3072},
        {256, 3072, 3072},
        {128, 3072, 3072},
        {64, 3072, 3072},
        {32, 3072, 3072},
        {16, 3072, 3072},
        {1, 3072, 3072},
        {256, 256, 256},
        {256, 256, 1024},
        {256, 256, 8192},
        {128, 128, 32768},
    };
    auto data = read_test_data(test_data_dir, configs);
    run_all_impls(Phase::WARMUP, data, configs);
    auto results = run_all_impls(Phase::BENCHMARK, data, configs);

    for (int32_t j = 1; j < results.size(); ++j) {
        for (int32_t i = j; i > 0;) {
            --i;
            auto const &first = results.at(i);
            auto const &second = results.at(j);
            printf("\nspeedups %s -> %s:\n\n", first.name, second.name);
            printf("  %-6s  %-6s  %-6s  %-7s\n", "size_i", "size_j", "size_k", "speedup");
            printf("  %-6s  %-6s  %-6s  %-7s\n", "------", "------", "------", "-------");
            for (auto const &config : configs) {
                auto size_i = config.size_i;
                auto size_j = config.size_j;
                auto size_k = config.size_k;
                printf("  %6d  %6d  %6d", size_i, size_j, size_k);
                auto it_first = first.elapsed_ms.find({size_i, size_j, size_k});
                auto it_second = second.elapsed_ms.find({size_i, size_j, size_k});
                if (it_first != first.elapsed_ms.end() &&
                    it_second != second.elapsed_ms.end()) {
                    printf("  %6.02fx", it_first->second / it_second->second);
                } else {
                    printf("  %7s", "-");
                }
                printf("\n");
            }
        }
    }

    write_json_results("out/results.json", results);

    return 0;
}


