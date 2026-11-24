/*
 * Matmul micro-benchmark on the layer shapes of Qwen2.5-1.5B (random weights).
 *
 *   bench_kernels [threads] [iters]
 *
 * Prints a markdown table: per weight format and shape, the single-token
 * (decode) matvec time and effective weight bandwidth, and the 64-token
 * (prefill) GFLOP/s. Decode is memory-bound, so GB/s is the number to watch.
 */
#include "../include/tinyinfer.h"
#include "../src/platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 0x1234567;
static float frand(void) {
    rs ^= rs << 13;
    rs ^= rs >> 7;
    rs ^= rs << 17;
    return (float)(rs >> 40) / 16777216.0f - 0.5f;
}

int main(int argc, char **argv) {
    int threads = argc > 1 ? atoi(argv[1]) : 0;
    int iters = argc > 2 ? atoi(argv[2]) : 20;
    if (threads > 0) ti_set_threads(threads);
    struct { const char *name; int rows, cols; } shapes[] = {
        {"attn_q / attn_output 1536x1536", 1536, 1536},
        {"attn_k / attn_v 256x1536", 256, 1536},
        {"ffn_gate / ffn_up 8960x1536", 8960, 1536},
        {"ffn_down 1536x8960", 1536, 8960},
    };
    int types[] = {TI_TYPE_F32, TI_TYPE_F16, TI_TYPE_Q8_0, TI_TYPE_Q4_0};
    const char *tnames[] = {"F32", "F16", "Q8_0", "Q4_0"};
    printf("simd=%s threads=%d iters=%d\n\n", ti_simd_path(), ti_get_threads(), iters);
    printf("| format | shape | decode ms | decode GB/s | prefill-64 GFLOP/s |\n");
    printf("|---|---|---|---|---|\n");
    const int B = 64;
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        int R = shapes[s].rows, K = shapes[s].cols;
        float *row = (float *)malloc(sizeof(float) * K);
        float *x = (float *)malloc(sizeof(float) * K * B);
        float *y = (float *)malloc(sizeof(float) * R * B);
        for (int i = 0; i < K * B; i++) x[i] = frand();
        for (size_t t = 0; t < sizeof types / sizeof types[0]; t++) {
            size_t rb = ti_type_row_size(types[t], K);
            char *w = (char *)malloc(rb * (size_t)R);
            for (int r = 0; r < R; r++) {
                for (int k = 0; k < K; k++) row[k] = frand();
                ti_quantize_row(types[t], row, w + rb * r, K);
            }
            ti_matmul(types[t], w, x, y, R, K, 1); /* warm-up */
            double t0 = ti_time_sec();
            for (int i = 0; i < iters; i++) ti_matmul(types[t], w, x, y, R, K, 1);
            double dt = (ti_time_sec() - t0) / iters;
            int it2 = iters / 4 > 0 ? iters / 4 : 1;
            double t1 = ti_time_sec();
            for (int i = 0; i < it2; i++) ti_matmul(types[t], w, x, y, R, K, B);
            double dt2 = (ti_time_sec() - t1) / it2;
            printf("| %s | %s | %.3f | %.1f | %.1f |\n", tnames[t], shapes[s].name, dt * 1e3,
                   (double)rb * R / dt / 1e9, 2.0 * R * K * B / dt2 / 1e9);
            fflush(stdout);
            free(w);
        }
        free(row);
        free(x);
        free(y);
    }
    return 0;
}
