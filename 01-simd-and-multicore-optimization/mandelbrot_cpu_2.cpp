// Optional arguments:
//  -r <img_size>
//  -b <max iterations>
//  -i <implementation: {"scalar", "vector", "vector_ilp", "vector_multicore",
//  "vector_multicore_multithread", "vector_multicore_multithread_ilp", "all"}>

#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <pthread.h>
#include <thread>
#include <vector>

constexpr float window_zoom = 1.0 / 10000.0f;
constexpr float window_x = -0.743643887 - 0.5 * window_zoom;
constexpr float window_y = 0.131825904 - 0.5 * window_zoom;
constexpr uint32_t default_max_iters = 2000;

// CPU Scalar Mandelbrot set generation.
// Based on the "optimized escape time algorithm" in
// https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set
void mandelbrot_cpu_scalar(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    for (uint64_t i = 0; i < img_size; ++i) {
        for (uint64_t j = 0; j < img_size; ++j) {
            float cx = (float(j) / float(img_size)) * window_zoom + window_x;
            float cy = (float(i) / float(img_size)) * window_zoom + window_y;

            float x2 = 0.0f;
            float y2 = 0.0f;
            float w = 0.0f;
            uint32_t iters = 0;
            while (x2 + y2 <= 4.0f && iters < max_iters) {
                float x = x2 - y2 + cx;
                float y = w - (x2 + y2) + cy;
                x2 = x * x;
                y2 = y * y;
                float z = x + y;
                w = z * z;
                ++iters;
            }

            // Write result.
            out[i * img_size + j] = iters;
        }
    }
}

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

/// <--- your code here --->

/*
    // OPTIONAL: Uncomment this block to include your CPU vector implementation
    // from Lab 1 for easy comparison.
    //
    // (If you do this, you'll need to update your code to use the new constants
    // 'window_zoom', 'window_x', and 'window_y'.)

    #define HAS_VECTOR_IMPL // <~~ keep this line if you want to benchmark the vector kernel!

    ////////////////////////////////////////////////////////////////////////////////
    // Vector

    void mandelbrot_cpu_vector(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
        // your code here...
    }
*/

////////////////////////////////////////////////////////////////////////////////
// Vector + ILP

void mandelbrot_cpu_vector_ilp(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
   constexpr int VEC = 8;
    constexpr int ILP = 4;

    const __m256 four = _mm256_set1_ps(4.0f);
    const __m256i one = _mm256_set1_epi32(1);

    for (uint32_t i = 0; i < img_size; ++i) {
        // Compute cy once per row
        float cy_scalar = (float(i) / float(img_size)) * window_zoom + window_y;
        __m256 cy = _mm256_set1_ps(cy_scalar);

        for (uint32_t j = 0; j < img_size; j += ILP * VEC) {
            // Per-vector state
            __m256 x2[ILP], y2[ILP], w[ILP], cx[ILP];
            __m256i iters[ILP];
            __m256 active[ILP];

            // --- Initialization ---
            #pragma unroll
            for (int v = 0; v < ILP; ++v) {
                // Compute pixel indices for this vector: [j+v*8+0, j+v*8+1, ..., j+v*8+7]
                __m256 offsets = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0);
                __m256 pixel_indices = _mm256_set1_ps(j + v * VEC);
                pixel_indices = _mm256_add_ps(pixel_indices, offsets);
                
                // Convert to cx coordinates: cx = (pixel_index / img_size) * window_zoom + window_x
                __m256 scale = _mm256_set1_ps(window_zoom / float(img_size));
                cx[v] = _mm256_mul_ps(pixel_indices, scale);
                cx[v] = _mm256_add_ps(cx[v], _mm256_set1_ps(window_x));

                x2[v] = _mm256_setzero_ps();
                y2[v] = _mm256_setzero_ps();
                w[v]  = _mm256_setzero_ps();
                iters[v] = _mm256_setzero_si256();
                active[v] = _mm256_castsi256_ps(_mm256_set1_epi32(-1)); // all lanes active
            }

            // --- Mandelbrot iteration ---
            bool any_active = true;
            while (any_active) {
                any_active = false;

                #pragma unroll
                for (int v = 0; v < ILP; ++v) {
                    // Check if this vector still has active lanes
                    if (_mm256_movemask_ps(active[v]) == 0)
                        continue;

                    __m256 sum = _mm256_add_ps(x2[v], y2[v]);
                    __m256 mask_mag = _mm256_cmp_ps(sum, four, _CMP_LE_OS);

                    __m256i mask_iter = _mm256_cmpgt_epi32( _mm256_set1_epi32(max_iters),iters[v]);

                    __m256 mask =_mm256_and_ps(mask_mag, _mm256_castsi256_ps(mask_iter));

                    active[v] = mask;

                    if (_mm256_movemask_ps(mask) == 0)
                        continue;

                    // Mandelbrot math
                    __m256 x = _mm256_add_ps(_mm256_sub_ps(x2[v], y2[v]), cx[v]);
                    __m256 y = _mm256_add_ps(_mm256_sub_ps(w[v], sum), cy);

                    x2[v] = _mm256_mul_ps(x, x);
                    y2[v] = _mm256_mul_ps(y, y);

                    __m256 z = _mm256_add_ps(x, y);
                    w[v] = _mm256_mul_ps(z, z);

                    // iters++
                    iters[v] = _mm256_add_epi32( iters[v],_mm256_and_si256(one, _mm256_castps_si256(mask)));

                    any_active = true;
                }
            }

            // --- Store results ---
            #pragma unroll
            for (int v = 0; v < ILP; ++v) {
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i * img_size + j + v * VEC),iters[v]);
            }
        }
    }
}



constexpr int NUM_THREADS = 8; // spawn 8 threads, one per core

struct ThreadData {
    uint32_t thread_id;
    uint32_t num_threads;
    uint32_t img_size;
    uint32_t max_iters;
    uint32_t *out;
};

void* mandelbrot_worker(void* arg) {
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);
    uint32_t img_size = data->img_size;
    uint32_t max_iters = data->max_iters;
    uint32_t *out = data->out;
    uint32_t tid = data->thread_id;
    uint32_t num_threads = data->num_threads;

    for (uint32_t i = tid; i < img_size; i += num_threads) {
        float cy = (float(i) / float(img_size)) * window_zoom + window_y;

        for (uint32_t j = 0; j < img_size; j++) {
            float cx = (float(j) / float(img_size)) * window_zoom + window_x;

            float x2 = 0.0f, y2 = 0.0f, w = 0.0f;
            uint32_t iters = 0;

            while (x2 + y2 <= 4.0f && iters < max_iters) {
                float x = x2 - y2 + cx;
                float y = w - (x2 + y2) + cy;
                x2 = x * x;
                y2 = y * y;
                float z = x + y;
                w = z * z;
                ++iters;
            }

            out[i * img_size + j] = iters;
        }
    }

    return nullptr;
}


////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core

void mandelbrot_cpu_vector_multicore(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out) {
   
   pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    for (uint32_t t = 0; t < NUM_THREADS; t++) {
        thread_data[t] = { t, NUM_THREADS, img_size, max_iters, out };
        pthread_create(&threads[t], nullptr, mandelbrot_worker, &thread_data[t]);
    }

    for (uint32_t t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], nullptr);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core + Multi-thread-per-core

void mandelbrot_cpu_vector_multicore_multithread(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out) {
    const uint32_t num_threads = std::thread::hardware_concurrency() * 2; // 2 threads per core
    std::vector<std::thread> threads(num_threads);

    auto thread_work = [&](uint32_t thread_id) {
        uint32_t rows_per_thread = (img_size + num_threads - 1) / num_threads;
        uint32_t row_start = thread_id * rows_per_thread;
        uint32_t row_end = std::min(row_start + rows_per_thread, img_size);

        for (uint32_t i = row_start; i < row_end; ++i) {
            for (uint32_t j = 0; j < img_size; ++j) {
                float cx = (float(j) / img_size) * window_zoom + window_x;
                float cy = (float(i) / img_size) * window_zoom + window_y;
                float x2 = 0.0f, y2 = 0.0f, w = 0.0f;
                uint32_t iters = 0;
                while (x2 + y2 <= 4.0f && iters < max_iters) {
                    float x = x2 - y2 + cx;
                    float y = w - (x2 + y2) + cy;
                    x2 = x*x;
                    y2 = y*y;
                    w = (x+y)*(x+y);
                    ++iters;
                }
                out[i * img_size + j] = iters;
            }
        }
    };

    for (uint32_t t = 0; t < num_threads; ++t) {
        threads[t] = std::thread(thread_work, t);
    }
    for (auto &t : threads) t.join();
}

////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core + Multi-thread-per-core + ILP

void mandelbrot_cpu_vector_multicore_multithread_ilp(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out) {
    constexpr int ILP = 4; // pixels per thread per iteration
    int num_threads = std::thread::hardware_concurrency() * 2; // multi-thread per core

    std::vector<std::thread> threads(num_threads);

    auto worker = [=](int thread_id) {
        int total_threads = num_threads;
        int total_pixels = img_size * img_size;

        for (int base = thread_id * ILP; base < total_pixels; base += total_threads * ILP) {
            float cx[ILP], cy[ILP];
            float x2[ILP], y2[ILP], w[ILP];
            uint32_t iters[ILP];
            bool active[ILP];

            for (int v = 0; v < ILP; ++v) {
                int pixel = base + v;
                if (pixel >= total_pixels) {
                    active[v] = false;
                    continue;
                }

                int row = pixel / img_size;
                int col = pixel % img_size;

                cx[v] = (float(col) / img_size) * window_zoom + window_x;
                cy[v] = (float(row) / img_size) * window_zoom + window_y;

                x2[v] = 0.0f;
                y2[v] = 0.0f;
                w[v]  = 0.0f;
                iters[v] = 0;
                active[v] = true;
            }

            bool any_active = true;
            while (any_active) {
                any_active = false;
                for (int v = 0; v < ILP; ++v) {
                    if (!active[v]) continue;

                    float sum = x2[v] + y2[v];
                    if (sum > 4.0f || iters[v] >= max_iters) {
                        active[v] = false;
                        continue;
                    }

                    float x = x2[v] - y2[v] + cx[v];
                    float y = w[v] - sum + cy[v];

                    x2[v] = x * x;
                    y2[v] = y * y;
                    float z = x + y;
                    w[v] = z * z;

                    ++iters[v];
                    any_active = true;
                }
            }

            for (int v = 0; v < ILP; ++v) {
                int pixel = base + v;
                if (pixel < total_pixels) {
                    out[pixel] = iters[v];
                }
            }
        }
    };

    // Launch all threads
    for (int t = 0; t < num_threads; ++t) {
        threads[t] = std::thread(worker, t);
    }

    // Join threads
    for (auto &t : threads) t.join();
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/types.h>
#include <vector>

// Useful functions and structures.
enum MandelbrotImpl {
    SCALAR,
    VECTOR,
    VECTOR_ILP,
    VECTOR_MULTICORE,
    VECTOR_MULTICORE_MULTITHREAD,
    VECTOR_MULTICORE_MULTITHREAD_ILP,
    ALL
};

// Command-line arguments parser.
int ParseArgsAndMakeSpec(
    int argc,
    char *argv[],
    uint32_t *img_size,
    uint32_t *max_iters,
    MandelbrotImpl *impl) {
    char *implementation_str = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-r") == 0) {
            if (i + 1 < argc) {
                *img_size = atoi(argv[++i]);
                if (*img_size % 32 != 0) {
                    std::cerr << "Error: Image width must be a multiple of 32"
                              << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: No value specified for -r" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                *max_iters = atoi(argv[++i]);
            } else {
                std::cerr << "Error: No value specified for -b" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-i") == 0) {
            if (i + 1 < argc) {
                implementation_str = argv[++i];
                if (strcmp(implementation_str, "scalar") == 0) {
                    *impl = SCALAR;
                } else if (strcmp(implementation_str, "vector") == 0) {
                    *impl = VECTOR;
                } else if (strcmp(implementation_str, "vector_ilp") == 0) {
                    *impl = VECTOR_ILP;
                } else if (strcmp(implementation_str, "vector_multicore") == 0) {
                    *impl = VECTOR_MULTICORE;
                } else if (
                    strcmp(implementation_str, "vector_multicore_multithread") == 0) {
                    *impl = VECTOR_MULTICORE_MULTITHREAD;
                } else if (
                    strcmp(implementation_str, "vector_multicore_multithread_ilp") == 0) {
                    *impl = VECTOR_MULTICORE_MULTITHREAD_ILP;
                } else if (strcmp(implementation_str, "all") == 0) {
                    *impl = ALL;
                } else {
                    std::cerr << "Error: unknown implementation" << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: No value specified for -i" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Unknown flag: " << argv[i] << std::endl;
            return 1;
        }
    }
    std::cout << "Testing with image size " << *img_size << "x" << *img_size << " and "
              << *max_iters << " max iterations." << std::endl;

    return 0;
}

// Output image writers: BMP file header structure
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType{0x4D42};   // File type, always "BM"
    uint32_t fileSize{0};        // Size of the file in bytes
    uint16_t reserved1{0};       // Always 0
    uint16_t reserved2{0};       // Always 0
    uint32_t dataOffset{54};     // Start position of pixel data
    uint32_t headerSize{40};     // Size of this header (40 bytes)
    int32_t width{0};            // Image width in pixels
    int32_t height{0};           // Image height in pixels
    uint16_t planes{1};          // Number of color planes
    uint16_t bitsPerPixel{24};   // Bits per pixel (24 for RGB)
    uint32_t compression{0};     // Compression method (0 for uncompressed)
    uint32_t imageSize{0};       // Size of raw bitmap data
    int32_t xPixelsPerMeter{0};  // Horizontal resolution
    int32_t yPixelsPerMeter{0};  // Vertical resolution
    uint32_t colorsUsed{0};      // Number of colors in the color palette
    uint32_t importantColors{0}; // Number of important colors
};
#pragma pack(pop)

void writeBMP(const char *fname, uint32_t img_size, const std::vector<uint8_t> &pixels) {
    uint32_t width = img_size;
    uint32_t height = img_size;

    BMPHeader header;
    header.width = width;
    header.height = height;
    header.imageSize = width * height * 3;
    header.fileSize = header.dataOffset + header.imageSize;

    std::ofstream file(fname, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    file.write(reinterpret_cast<const char *>(pixels.data()), pixels.size());
}

std::vector<uint8_t> iters_to_colors(
    uint32_t img_size,
    uint32_t max_iters,
    const std::vector<uint32_t> &iters) {
    uint32_t width = img_size;
    uint32_t height = img_size;
    uint32_t min_iters = max_iters;
    for (uint32_t i = 0; i < img_size; i++) {
        for (uint32_t j = 0; j < img_size; j++) {
            min_iters = std::min(min_iters, iters[i * img_size + j]);
        }
    }
    float log_iters_min = log2f(static_cast<float>(min_iters));
    float log_iters_range =
        log2f(static_cast<float>(max_iters) / static_cast<float>(min_iters));
    auto pixel_data = std::vector<uint8_t>(width * height * 3);
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            uint32_t iter = iters[i * width + j];

            uint8_t r = 0, g = 0, b = 0;
            if (iter < max_iters) {
                auto log_iter = log2f(static_cast<float>(iter)) - log_iters_min;
                auto intensity = static_cast<uint8_t>(log_iter * 222 / log_iters_range);
                r = 32;
                g = 32 + intensity;
                b = 32;
            }

            auto index = (i * width + j) * 3;
            pixel_data[index] = b;
            pixel_data[index + 1] = g;
            pixel_data[index + 2] = r;
        }
    }
    return pixel_data;
}

// Benchmarking macros and configuration.
static constexpr size_t kNumOfOuterIterations = 10;
static constexpr size_t kNumOfInnerIterations = 1;
#define BENCHPRESS(func, ...) \
    do { \
        std::cout << std::endl << "Running " << #func << " ...\n"; \
        std::vector<double> times(kNumOfOuterIterations); \
        for (size_t i = 0; i < kNumOfOuterIterations; ++i) { \
            auto start = std::chrono::high_resolution_clock::now(); \
            for (size_t j = 0; j < kNumOfInnerIterations; ++j) { \
                func(__VA_ARGS__); \
            } \
            auto end = std::chrono::high_resolution_clock::now(); \
            times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start) \
                           .count() / \
                kNumOfInnerIterations; \
        } \
        std::sort(times.begin(), times.end()); \
        std::stringstream sstream; \
        sstream << std::fixed << std::setw(6) << std::setprecision(2) \
                << times[0] / 1'000'000; \
        std::cout << "  Runtime: " << sstream.str() << " ms" << std::endl; \
    } while (0)

double difference(
    uint32_t img_size,
    uint32_t max_iters,
    std::vector<uint32_t> &result,
    std::vector<uint32_t> &ref_result) {
    int64_t diff = 0;
    for (uint32_t i = 0; i < img_size; i++) {
        for (uint32_t j = 0; j < img_size; j++) {
            diff +=
                abs(int(result[i * img_size + j]) - int(ref_result[i * img_size + j]));
        }
    }
    return diff / double(img_size * img_size * max_iters);
}

void dump_image(
    const char *fname,
    uint32_t img_size,
    uint32_t max_iters,
    const std::vector<uint32_t> &iters) {
    // Dump result as an image.
    auto pixel_data = iters_to_colors(img_size, max_iters, iters);
    writeBMP(fname, img_size, pixel_data);
}

// Main function.
// Compile with:
//  g++ -march=native -O3 -Wall -Wextra -o mandelbrot mandelbrot_cpu.cc
int main(int argc, char *argv[]) {
    // Get Mandelbrot spec.
    uint32_t img_size = 1024;
    uint32_t max_iters = default_max_iters;
    enum MandelbrotImpl impl = ALL;
    if (ParseArgsAndMakeSpec(argc, argv, &img_size, &max_iters, &impl))
        return -1;

    // Allocate memory.
    std::vector<uint32_t> result(img_size * img_size);
    std::vector<uint32_t> ref_result(img_size * img_size);

    // Compute the reference solution
    mandelbrot_cpu_scalar(img_size, max_iters, ref_result.data());

    // Test the desired kernels.
    if (impl == SCALAR || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_scalar, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_scalar.bmp", img_size, max_iters, result);
    }

#ifdef HAS_VECTOR_IMPL
    if (impl == VECTOR || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_vector.bmp", img_size, max_iters, result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }
#endif

    if (impl == VECTOR_ILP || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector_ilp, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_vector_ilp.bmp", img_size, max_iters, result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    if (impl == VECTOR_MULTICORE || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector_multicore, img_size, max_iters, result.data());
        dump_image(
            "out/mandelbrot_cpu_vector_multicore.bmp",
            img_size,
            max_iters,
            result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    if (impl == VECTOR_MULTICORE_MULTITHREAD || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(
            mandelbrot_cpu_vector_multicore_multithread,
            img_size,
            max_iters,
            result.data());
        dump_image(
            "out/mandelbrot_cpu_vector_multicore_multithread.bmp",
            img_size,
            max_iters,
            result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    if (impl == VECTOR_MULTICORE_MULTITHREAD_ILP || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(
            mandelbrot_cpu_vector_multicore_multithread_ilp,
            img_size,
            max_iters,
            result.data());
        dump_image(
            "out/mandelbrot_cpu_vector_multicore_multithread_ilp.bmp",
            img_size,
            max_iters,
            result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    return 0;
}
