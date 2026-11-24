/*
 * Dependency-free C unit tests. Runs anywhere the core compiles, including
 * cross-compiled aarch64 binaries under qemu in CI (which exercises the NEON
 * kernels against the scalar reference).
 *
 *   test_core                        kernel / quant / sampler / tokenizer tests
 *   test_core model.gguf logits.bin  + compares ti_forward against reference
 *                                     logits for tokens {5, 77, 300, 12}
 */
#include "../include/tinyinfer.h"
#include "../src/kernels.h"
#include "../src/quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, ...)                                     \
    do {                                                     \
        if (cond) {                                          \
            g_pass++;                                        \
        } else {                                             \
            g_fail++;                                        \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);      \
            printf(__VA_ARGS__);                             \
            printf("\n");                                    \
        }                                                    \
    } while (0)

static uint64_t rs = 88172645463325252ull;
static float frand(void) { /* approx N(0,1) via sum of uniforms */
    float s = 0;
    for (int i = 0; i < 4; i++) {
        rs ^= rs << 13;
        rs ^= rs >> 7;
        rs ^= rs << 17;
        s += (float)(rs >> 40) / 16777216.0f;
    }
    return (s - 2.0f) * 1.7320508f;
}

static void test_fp16(void) {
    int bad = 0;
    for (uint32_t h = 0; h < 65536; h++) {
        if (((h >> 10) & 0x1f) == 0x1f && (h & 0x3ff)) continue; /* NaN */
        float f = ti_h2f((uint16_t)h);
        float g = ti_h2f_soft((uint16_t)h);
        if (ti_f2h(f) != h || memcmp(&f, &g, 4) != 0) bad++;
    }
    CHECK(bad == 0, "fp16 roundtrip mismatches: %d", bad);
    CHECK(ti_f2h(65520.0f) == 0x7c00, "65520 should overflow to inf");
    CHECK(ti_f2h(65504.0f) == 0x7bff, "max half");
    CHECK(ti_f2h(1.0f + 1.0f / 2048) == 0x3c00, "tie rounds to even (down)");
    CHECK(ti_f2h(1.0f + 3.0f / 2048) == 0x3c02, "tie rounds to even (up)");
    CHECK(ti_f2h(5.9604645e-08f) == 0x0001, "smallest subnormal");
}

static void test_quant_bounds(void) {
    enum { N = 32 * 64 };
    static float x[N], y[N];
    static block_q8_0 q8[N / 32];
    static block_q4_0 q4[N / 32];
    for (int i = 0; i < N; i++) x[i] = frand();
    ti_quantize_row_q8_0(x, q8, N);
    ti_dequantize_row_q8_0(q8, y, N);
    int bad = 0;
    for (int b = 0; b < N / 32; b++) {
        float amax = 0;
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[b * 32 + j]));
        float d = amax / 127, bound = d / 2 + 127 * d / 2048 + 1e-9f;
        for (int j = 0; j < 32; j++)
            if (fabsf(x[b * 32 + j] - y[b * 32 + j]) > bound) bad++;
    }
    CHECK(bad == 0, "q8_0 error bound violated %d times", bad);
    ti_quantize_row_q4_0(x, q4, N);
    ti_dequantize_row_q4_0(q4, y, N);
    bad = 0;
    for (int b = 0; b < N / 32; b++) {
        float amax = 0;
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[b * 32 + j]));
        float d = amax / 8, bound = d + 15 * d / 2048 + 1e-9f;
        for (int j = 0; j < 32; j++)
            if (fabsf(x[b * 32 + j] - y[b * 32 + j]) > bound) bad++;
    }
    CHECK(bad == 0, "q4_0 error bound violated %d times", bad);
}

static void test_dots(void) {
    enum { N = 32 * 48 };
    static float w[N], x[N], wd[N], xd[N];
    static block_q8_0 wq8[N / 32], xq8[N / 32];
    static block_q4_0 wq4[N / 32];
    static uint16_t wh[N];
    for (int i = 0; i < N; i++) {
        w[i] = frand();
        x[i] = frand();
        wh[i] = ti_f2h(w[i]);
    }
    ti_quantize_row_q8_0(w, wq8, N);
    ti_quantize_row_q8_0(x, xq8, N);
    ti_quantize_row_q4_0(w, wq4, N);
    for (int n = 32; n <= N; n += 32 * 7) {
        double ref = 0;
        for (int i = 0; i < n; i++) ref += (double)w[i] * x[i];
        float got = ti_dot_f32(w, x, n);
        CHECK(fabs(got - ref) < 1e-3 * sqrt((double)n), "dot_f32 n=%d %f vs %f", n, got, ref);
        ref = 0;
        for (int i = 0; i < n; i++) ref += (double)ti_h2f(wh[i]) * x[i];
        got = ti_dot_f16_f32(wh, x, n);
        CHECK(fabs(got - ref) < 1e-3 * sqrt((double)n), "dot_f16 n=%d %f vs %f", n, got, ref);

        ti_dequantize_row_q8_0(wq8, wd, n);
        ti_dequantize_row_q8_0(xq8, xd, n);
        ref = 0;
        for (int i = 0; i < n; i++) ref += (double)wd[i] * xd[i];
        float s = ti_dot_q8_0_q8_0_scalar(wq8, xq8, n);
        got = ti_dot_q8_0_q8_0(wq8, xq8, n);
        CHECK(fabs(s - ref) < 1e-3 * sqrt((double)n), "q8 scalar n=%d %f vs %f", n, s, ref);
        CHECK(fabs(got - s) < 1e-4 * sqrt((double)n), "q8 simd n=%d %f vs scalar %f", n, got, s);

        ti_dequantize_row_q4_0(wq4, wd, n);
        ref = 0;
        for (int i = 0; i < n; i++) ref += (double)wd[i] * xd[i];
        s = ti_dot_q4_0_q8_0_scalar(wq4, xq8, n);
        got = ti_dot_q4_0_q8_0(wq4, xq8, n);
        CHECK(fabs(s - ref) < 1e-3 * sqrt((double)n), "q4 scalar n=%d %f vs %f", n, s, ref);
        CHECK(fabs(got - s) < 1e-4 * sqrt((double)n), "q4 simd n=%d %f vs scalar %f", n, got, s);

        /* 4-token blocked kernel vs single dots (use w as a 2nd..4th "token") */
        const block_q8_0 *xs[4] = {xq8, wq8, xq8, wq8};
        float o8[4], o4[4];
        ti_dot_q_x4(TI_TYPE_Q8_0, wq8, xs, n, o8);
        ti_dot_q_x4(TI_TYPE_Q4_0, wq4, xs, n, o4);
        for (int k = 0; k < 4; k++) {
            float e8 = ti_dot_q8_0_q8_0_scalar(wq8, xs[k], n);
            float e4 = ti_dot_q4_0_q8_0_scalar(wq4, xs[k], n);
            CHECK(fabs(o8[k] - e8) < 1e-4 * sqrt((double)n) * (1 + fabs(e8)), "q8 x4 k=%d", k);
            CHECK(fabs(o4[k] - e4) < 1e-4 * sqrt((double)n) * (1 + fabs(e4)), "q4 x4 k=%d", k);
        }
    }
}

static void test_matmul_threads(void) {
    enum { R = 97, K = 256, T = 3 };
    static float w[R * K], x[T * K], y1[T * R], y4[T * R];
    static block_q4_0 wq[R * K / 32];
    for (int i = 0; i < R * K; i++) w[i] = frand();
    for (int i = 0; i < T * K; i++) x[i] = frand();
    for (int r = 0; r < R; r++) ti_quantize_row_q4_0(w + r * K, wq + r * (K / 32), K);
    ti_set_threads(1);
    ti_matmul(TI_TYPE_Q4_0, wq, x, y1, R, K, T);
    ti_set_threads(4);
    ti_matmul(TI_TYPE_Q4_0, wq, x, y4, R, K, T);
    CHECK(memcmp(y1, y4, sizeof y1) == 0, "matmul differs between 1 and 4 threads");
    ti_matmul(TI_TYPE_F32, w, x, y4, R, K, T);
    double maxe = 0;
    for (int t = 0; t < T; t++)
        for (int r = 0; r < R; r++) {
            double ref = 0;
            for (int k = 0; k < K; k++) ref += (double)w[r * K + k] * x[t * K + k];
            maxe = fmax(maxe, fabs(ref - y4[t * R + r]));
        }
    CHECK(maxe < 1e-3, "f32 matmul max err %g", maxe);
}

static void test_sampler(void) {
    float l[5] = {0.1f, 2.0f, -1.0f, 1.9f, 0.0f};
    ti_sampler_params p = ti_sampler_default();
    uint64_t rng = 1;
    p.temperature = 0;
    CHECK(ti_sample(l, 5, &p, NULL, 0, &rng) == 1, "greedy");
    float l2[5] = {0.1f, 2.0f, -1.0f, 1.9f, 0.0f};
    int32_t hist[1] = {1};
    p.repeat_penalty = 2.0f;
    CHECK(ti_sample(l2, 5, &p, hist, 1, &rng) == 3, "repetition penalty flips argmax");
    p = ti_sampler_default();
    p.top_k = 2;
    int bad = 0;
    for (int i = 0; i < 500; i++) {
        float l3[5] = {0.1f, 2.0f, -1.0f, 1.9f, 0.0f};
        int s = ti_sample(l3, 5, &p, NULL, 0, &rng);
        if (s != 1 && s != 3) bad++;
    }
    CHECK(bad == 0, "top-k=2 produced %d out-of-set samples", bad);
}

static void test_pretok(void) {
    const char *s = "Hello world's  test\n\n 123!!";
    const char *expect[] = {"Hello", " world", "'s", " ", " test", "\n\n", " ", "1", "2", "3", "!!"};
    int ends[32];
    int n = ti_pretokenize(s, (int)strlen(s), ends, 32);
    int ok = n == 11;
    for (int i = 0, prev = 0; ok && i < n; prev = ends[i], i++)
        ok = (int)strlen(expect[i]) == ends[i] - prev && !memcmp(s + prev, expect[i], strlen(expect[i]));
    CHECK(ok, "pretokenize basic split (n=%d)", n);
}

static int test_model(const char *model, const char *expected) {
    ti_model *m = ti_model_load(model, 64, 0);
    CHECK(m != NULL, "load %s: %s", model, ti_last_error());
    if (!m) return 1;
    const ti_config *c = ti_model_config(m);
    int32_t toks[4] = {5, 77, 300, 12};
    size_t nv = (size_t)c->vocab_size * 4;
    float *got = (float *)malloc(nv * sizeof(float)), *ref = (float *)malloc(nv * sizeof(float));
    FILE *f = fopen(expected, "rb");
    CHECK(f != NULL, "open %s", expected);
    if (!f) return 1;
    size_t rd = fread(ref, sizeof(float), nv, f);
    fclose(f);
    CHECK(rd == nv, "expected logits file has %zu floats, need %zu", rd, nv);
    CHECK(ti_forward(m, toks, 4, 0, got, 1) == 0, "forward failed: %s", ti_last_error());
    double mean = 0, var = 0, maxe = 0;
    for (size_t i = 0; i < nv; i++) mean += ref[i];
    mean /= (double)nv;
    for (size_t i = 0; i < nv; i++) var += (ref[i] - mean) * (ref[i] - mean);
    double sd = sqrt(var / (double)nv);
    for (size_t i = 0; i < nv; i++) maxe = fmax(maxe, fabs(got[i] - ref[i]));
    printf("forward vs reference: max|diff|/std = %.3g\n", maxe / sd);
    CHECK(maxe / sd < 2e-2, "forward logits differ from reference: %g", maxe / sd);
    free(got);
    free(ref);
    ti_model_free(m);
    return 0;
}

int main(int argc, char **argv) {
    printf("tinyinfer test_core (simd path: %s)\n", ti_simd_path());
    test_fp16();
    test_quant_bounds();
    test_dots();
    test_matmul_threads();
    test_sampler();
    test_pretok();
    if (argc >= 3) test_model(argv[1], argv[2]);
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
