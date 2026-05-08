#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>
#include <cmath>

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
    // from Lab 5 for easy comparison.

    ////////////////////////////////////////////////////////////////////////////////
    // Optimized GPU Implementation with Reduction along k (Baseline from Lab 5)

    #define HAS_LAB_5_BASELINE_IMPL // <~~ keep this line if you want to benchmark your Lab 5 kernel!

    namespace matmul_improved_reduce {

    // TODO: your GPU kernels here...

    size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        // TODO: your CPU code here
        return 0;
    }

    void launch_matmul_improved_reduce(
        int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a, // pointer to GPU memory
        float const *b, // pointer to GPU memory
        float *c,       // pointer to GPU memory
        void *workspace // pointer to GPU memory
    ) {
        // TODO: your CPU code here
    }

    } // namespace matmul_improved_reduce
*/

////////////////////////////////////////////////////////////////////////////////
// Tensor Core GPU Implementation


namespace matmul_tensor {

__device__ __forceinline__ uint32_t float_to_tf32(float x) {
    uint32_t out;
    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(out) : "f"(x));
    return out;
}

__device__ __forceinline__ void mma_m16n8k8_tf32(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

template <
    int BLOCK_M,
    int BLOCK_N,
    int BLOCK_K,
    int NWARPS_M,
    int NWARPS_N,
    int SMEM_A_PAD = 4,
    int SMEM_B_PAD = 4>
__global__ __launch_bounds__(NWARPS_M * NWARPS_N * 32, 2)
void matmul_tensor_kernel_t(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *__restrict__ a,
    float const *__restrict__ b,
    float *__restrict__ c
) {
    constexpr int NWARPS   = NWARPS_M * NWARPS_N;
    constexpr int WARP_M   = BLOCK_M / NWARPS_M;
    constexpr int WARP_N   = BLOCK_N / NWARPS_N;
    constexpr int MMA_ROWS = WARP_M / 16;
    constexpr int MMA_COLS = WARP_N / 8;

    static_assert(WARP_M % 16 == 0, "WARP_M must be multiple of 16");
    static_assert(WARP_N % 8 == 0, "WARP_N must be multiple of 8");

    __shared__ float As[2][BLOCK_K][BLOCK_M + SMEM_A_PAD];
    __shared__ float Bs[2][BLOCK_K][BLOCK_N + SMEM_B_PAD];

    const int tid    = threadIdx.x;
    const int lane   = tid & 31;
    const int warpId = tid >> 5;

    const int warp_m_idx = warpId / NWARPS_N;
    const int warp_n_idx = warpId % NWARPS_N;

    const int block_i = blockIdx.y * BLOCK_M;
    const int block_j = blockIdx.x * BLOCK_N;
    const int warp_i  = block_i + warp_m_idx * WARP_M;
    const int warp_j  = block_j + warp_n_idx * WARP_N;

    float acc[MMA_ROWS][MMA_COLS][4] = {};

    const int a_row_in_quad = lane >> 2; // 0..7
    const int a_col_in_quad = lane & 3;  // 0..3
    const int b_row_in_half = lane & 3;  // 0..3
    const int b_col_in_half = lane >> 2; // 0..7

    const int num_k_tiles = (size_k + BLOCK_K - 1) / BLOCK_K;

    {
        const int kk = 0;

        for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += NWARPS * 32) {
            int m  = idx / BLOCK_K;
            int k  = idx % BLOCK_K;
            int gm = block_i + m;
            int gk = kk + k;
            As[0][k][m] = (gm < size_i && gk < size_k) ? a[gm * size_k + gk] : 0.0f;
        }

        for (int idx = tid; idx < BLOCK_K * BLOCK_N; idx += NWARPS * 32) {
            int k  = idx / BLOCK_N;
            int n  = idx % BLOCK_N;
            int gk = kk + k;
            int gn = block_j + n;
            Bs[0][k][n] = (gk < size_k && gn < size_j) ? b[gk * size_j + gn] : 0.0f;
        }
    }

    __syncthreads();

    for (int tile = 0; tile < num_k_tiles; ++tile) {
        const int stage_cur  = tile & 1;
        const int stage_next = stage_cur ^ 1;
        const int kk_next    = (tile + 1) * BLOCK_K;

        if (tile + 1 < num_k_tiles) {
            for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += NWARPS * 32) {
                int m  = idx / BLOCK_K;
                int k  = idx % BLOCK_K;
                int gm = block_i + m;
                int gk = kk_next + k;
                As[stage_next][k][m] =
                    (gm < size_i && gk < size_k) ? a[gm * size_k + gk] : 0.0f;
            }

            for (int idx = tid; idx < BLOCK_K * BLOCK_N; idx += NWARPS * 32) {
                int k  = idx / BLOCK_N;
                int n  = idx % BLOCK_N;
                int gk = kk_next + k;
                int gn = block_j + n;
                Bs[stage_next][k][n] =
                    (gk < size_k && gn < size_j) ? b[gk * size_j + gn] : 0.0f;
            }
        }

        #pragma unroll
        for (int kk_micro = 0; kk_micro < BLOCK_K; kk_micro += 8) {
            uint32_t mA[MMA_ROWS][4];
            uint32_t mB[MMA_COLS][2];

            #pragma unroll
            for (int mr = 0; mr < MMA_ROWS; ++mr) {
                const int base_m = warp_m_idx * WARP_M + mr * 16;

                mA[mr][0] = float_to_tf32(
                    As[stage_cur][kk_micro + a_col_in_quad][base_m + a_row_in_quad]);
                mA[mr][1] = float_to_tf32(
                    As[stage_cur][kk_micro + a_col_in_quad][base_m + a_row_in_quad + 8]);
                mA[mr][2] = float_to_tf32(
                    As[stage_cur][kk_micro + a_col_in_quad + 4][base_m + a_row_in_quad]);
                mA[mr][3] = float_to_tf32(
                    As[stage_cur][kk_micro + a_col_in_quad + 4][base_m + a_row_in_quad + 8]);
            }

            #pragma unroll
            for (int mc = 0; mc < MMA_COLS; ++mc) {
                const int base_n = warp_n_idx * WARP_N + mc * 8;

                mB[mc][0] = float_to_tf32(
                    Bs[stage_cur][kk_micro + b_row_in_half][base_n + b_col_in_half]);
                mB[mc][1] = float_to_tf32(
                    Bs[stage_cur][kk_micro + b_row_in_half + 4][base_n + b_col_in_half]);
            }

            #pragma unroll
            for (int mr = 0; mr < MMA_ROWS; ++mr) {
                #pragma unroll
                for (int mc = 0; mc < MMA_COLS; ++mc) {
                    mma_m16n8k8_tf32(
                        acc[mr][mc][0], acc[mr][mc][1],
                        acc[mr][mc][2], acc[mr][mc][3],
                        mA[mr][0], mA[mr][1], mA[mr][2], mA[mr][3],
                        mB[mc][0], mB[mc][1],
                        acc[mr][mc][0], acc[mr][mc][1],
                        acc[mr][mc][2], acc[mr][mc][3]);
                }
            }
        }

        __syncthreads();
    }

    const int c_row_in_half = lane >> 2; // 0..7
    const int c_stripe      = lane & 3;  // 0..3

    #pragma unroll
    for (int mr = 0; mr < MMA_ROWS; ++mr) {
        #pragma unroll
        for (int mc = 0; mc < MMA_COLS; ++mc) {
            const int base_i = warp_i + mr * 16;
            const int base_j = warp_j + mc * 8;

            const int ci0 = base_i + c_row_in_half;
            const int ci1 = base_i + 8 + c_row_in_half;
            const int cj0 = base_j + 2 * c_stripe;
            const int cj1 = base_j + 2 * c_stripe + 1;

            if (ci0 < size_i && cj0 < size_j) c[ci0 * size_j + cj0] = acc[mr][mc][0];
            if (ci0 < size_i && cj1 < size_j) c[ci0 * size_j + cj1] = acc[mr][mc][1];
            if (ci1 < size_i && cj0 < size_j) c[ci1 * size_j + cj0] = acc[mr][mc][2];
            if (ci1 < size_i && cj1 < size_j) c[ci1 * size_j + cj1] = acc[mr][mc][3];
        }
    }
}

size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
    (void)size_i;
    (void)size_j;
    (void)size_k;
    return 0;
}

void launch_matmul_tensor(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c,
    void *workspace
) {
    (void)workspace;

    if (size_i >= 256) {
        constexpr int BLOCK_M  = 128;
        constexpr int BLOCK_N  = 128;
        constexpr int BLOCK_K  = 16;
        constexpr int NWARPS_M = 4;
        constexpr int NWARPS_N = 4;
        dim3 block(NWARPS_M * NWARPS_N * 32, 1, 1);
        dim3 grid(
            (size_j + BLOCK_N - 1) / BLOCK_N,
            (size_i + BLOCK_M - 1) / BLOCK_M,
            1);
        matmul_tensor_kernel_t<BLOCK_M, BLOCK_N, BLOCK_K, NWARPS_M, NWARPS_N>
            <<<grid, block>>>(size_i, size_j, size_k, a, b, c);
    } else {
        constexpr int BLOCK_M  = 64;
        constexpr int BLOCK_N  = 128;
        constexpr int BLOCK_K  = 16;
        constexpr int NWARPS_M = 2;
        constexpr int NWARPS_N = 4;
        dim3 block(NWARPS_M * NWARPS_N * 32, 1, 1);
        dim3 grid(
            (size_j + BLOCK_N - 1) / BLOCK_N,
            (size_i + BLOCK_M - 1) / BLOCK_M,
            1);
        matmul_tensor_kernel_t<BLOCK_M, BLOCK_N, BLOCK_K, NWARPS_M, NWARPS_N>
            <<<grid, block>>>(size_i, size_j, size_k, a, b, c);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace matmul_tensor



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

    if (rel_rmse > 1e-3) {
        if (phase == Phase::BENCHMARK) {
            printf("  %9s  %7s", "-", "-");
        }
    } else {
        double target_time_ms = 200.0;
        double elapsed_ms = benchmark_ms(
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
    if (workspace_size > 0) {
        CUDA_CHECK(cudaFree(workspace_gpu));
    }
    CUDA_CHECK(cudaFree(flush_gpu));
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

#ifdef HAS_LAB_5_BASELINE_IMPL

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

#endif

struct MatmulTensor {
    constexpr static char const *name = "matmul_tensor";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return matmul_tensor::get_workspace_size(size_i, size_j, size_k);
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_tensor::launch_matmul_tensor(size_i, size_j, size_k, a, b, c, workspace);
    }
};

BenchmarkResults get_cublas_fma_results() {
    // Hard-coded data collected on A4000 GPU
    return BenchmarkResults{
        "cublas_fma",
        {
            {{3072, 3072, 3072}, 3.152},
            {{2048, 3072, 3072}, 2.174},
            {{1024, 3072, 3072}, 1.090},
            {{512, 3072, 3072}, 0.559},
            {{256, 3072, 3072}, 0.356},
            {{128, 3072, 3072}, 0.256},
            {{64, 3072, 3072}, 0.194},
            {{32, 3072, 3072}, 0.181},
            {{16, 3072, 3072}, 0.181},
        }};
}

std::vector<BenchmarkResults> run_all_impls(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = std::vector<BenchmarkResults>{};
#ifdef HAS_LAB_5_BASELINE_IMPL
    results.push_back(run_all_configs<MatmulImprovedReduce>(phase, data, configs));
#endif
    results.push_back(run_all_configs<MatmulTensor>(phase, data, configs));
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

void print_speedup(
    std::vector<BenchmarkConfig> const &configs,
    BenchmarkResults const &first,
    BenchmarkResults const &second) {
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
        if (it_first != first.elapsed_ms.end() && it_second != second.elapsed_ms.end()) {
            printf("  %6.02fx", it_first->second / it_second->second);
        } else {
            printf("  %7s", "-");
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    std::string test_data_dir = ".";


    auto configs = std::vector<BenchmarkConfig>{
        {3072, 3072, 3072},
        {2048, 3072, 3072},
        {1024, 3072, 3072},
        {512, 3072, 3072},
        {256, 3072, 3072},
        {128, 3072, 3072},
        {64, 3072, 3072},
        {32, 3072, 3072},
        {16, 3072, 3072},
    };
    auto data = read_test_data(test_data_dir, configs);
    run_all_impls(Phase::WARMUP, data, configs);
    auto results = run_all_impls(Phase::BENCHMARK, data, configs);

    for (int32_t j = 1; j < results.size(); ++j) {
        for (int32_t i = j; i > 0;) {
            --i;
            print_speedup(configs, results.at(i), results.at(j));
        }
    }

    printf("\n-----------------------------------------------------------\n");
    printf("---- Comparison to non-tensor-core cuBLAS performance: ----\n");
    printf("-----------------------------------------------------------\n");

    print_speedup(configs, get_cublas_fma_results(), results.at(results.size() - 1));

    write_json_results("out/results.json", results);

    return 0;
}
