/*
 * Matmul kernels.
 *
 *   y[t][r] = sum_c W[r][c] * x[t][c]
 *
 * W is a GGUF tensor with ne = [cols, rows] (row-major, `cols` contiguous).
 * For quantised weights the activations x are first quantised to Q8_0 so the
 * inner loop is an int8 x int8 (or int4 x int8) dot product with one float
 * multiply per 32-element block:
 *
 *   sum_b  d_w[b] * d_x[b] * sum_{j<32} q_w[b][j] * q_x[b][j]
 *
 * SIMD paths:
 *   __ARM_NEON (+ __ARM_FEATURE_DOTPROD on M1: SDOT)  -> Apple Silicon
 *   __AVX2__ + __FMA__ (+ __F16C__)                   -> x86-64 dev boxes / CI
 *   scalar fallback                                    -> everything else
 *
 * Rows are distributed over the thread pool in chunks; each row is reused for
 * every token of the batch while it is hot in L1 (prefill speed-up).
 */
#include "kernels.h"
#include "platform.h"

#include <stdlib.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define TI_AVX2 1
#endif

/* ----------------------------------------------------------- scalar ------ */
float ti_dot_q8_0_q8_0_scalar(const block_q8_0 *w, const block_q8_0 *x, int n) {
    const int nb = n / TI_QK;
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        int isum = 0;
        for (int j = 0; j < TI_QK; j++) isum += (int)w[b].qs[j] * (int)x[b].qs[j];
        sum += ti_h2f(w[b].d) * ti_h2f(x[b].d) * (float)isum;
    }
    return sum;
}

float ti_dot_q4_0_q8_0_scalar(const block_q4_0 *w, const block_q8_0 *x, int n) {
    const int nb = n / TI_QK;
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        int isum = 0;
        for (int j = 0; j < TI_QK / 2; j++) {
            int lo = (w[b].qs[j] & 0x0f) - 8;
            int hi = (w[b].qs[j] >> 4) - 8;
            isum += lo * (int)x[b].qs[j] + hi * (int)x[b].qs[j + TI_QK / 2];
        }
        sum += ti_h2f(w[b].d) * ti_h2f(x[b].d) * (float)isum;
    }
    return sum;
}

#ifdef TI_AVX2
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, sh);
    sh = _mm_movehl_ps(sh, lo);
    lo = _mm_add_ss(lo, sh);
    return _mm_cvtss_f32(lo);
}

/* int8 x int8 -> 8 x int32 partial sums using the sign trick:
 * maddubs needs an unsigned operand, so use |a| and sign(b, a). */
static inline __m256 mul_sum_i8_f32(__m256i a, __m256i b) {
    __m256i ax = _mm256_sign_epi8(a, a);
    __m256i sy = _mm256_sign_epi8(b, a);
    __m256i d16 = _mm256_maddubs_epi16(ax, sy);
    __m256i d32 = _mm256_madd_epi16(d16, _mm256_set1_epi16(1));
    return _mm256_cvtepi32_ps(d32);
}
#endif

/* ------------------------------------------------------------- f32 ------- */
float ti_dot_f32(const float *a, const float *b, int n) {
    int i = 0;
    float sum = 0.0f;
#if defined(__ARM_NEON)
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
    for (; i + 8 <= n; i += 8) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    sum = vaddvq_f32(vaddq_f32(s0, s1));
#elif defined(TI_AVX2)
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    sum = hsum256(_mm256_add_ps(s0, s1));
#endif
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

/* ------------------------------------------------------------- f16 ------- */
float ti_dot_f16_f32(const uint16_t *w, const float *x, int n) {
    int i = 0;
    float sum = 0.0f;
#if defined(__ARM_NEON) && defined(__aarch64__)
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
    for (; i + 8 <= n; i += 8) {
        float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(w + i));
        s0 = vfmaq_f32(s0, vcvt_f32_f16(vget_low_f16(h)), vld1q_f32(x + i));
        s1 = vfmaq_f32(s1, vcvt_high_f32_f16(h), vld1q_f32(x + i + 4));
    }
    sum = vaddvq_f32(vaddq_f32(s0, s1));
#elif defined(TI_AVX2) && defined(__F16C__)
    __m256 s0 = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 wf = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i)));
        s0 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x + i), s0);
    }
    sum = hsum256(s0);
#endif
    for (; i < n; i++) sum += ti_h2f(w[i]) * x[i];
    return sum;
}

float ti_dot_bf16_f32(const uint16_t *w, const float *x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += ti_bf2f(w[i]) * x[i];
    return sum;
}

/* ------------------------------------------------------------ q8_0 ------- */
float ti_dot_q8_0_q8_0(const block_q8_0 *w, const block_q8_0 *x, int n) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    const int nb = n / TI_QK;
    float32x4_t acc = vdupq_n_f32(0);
    for (int b = 0; b < nb; b++) {
        int8x16_t w0 = vld1q_s8(w[b].qs), w1 = vld1q_s8(w[b].qs + 16);
        int8x16_t x0 = vld1q_s8(x[b].qs), x1 = vld1q_s8(x[b].qs + 16);
#if defined(__ARM_FEATURE_DOTPROD)
        int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, x0), w1, x1);
#else
        int16x8_t p0 = vmull_s8(vget_low_s8(w0), vget_low_s8(x0));
        p0 = vmlal_s8(p0, vget_high_s8(w0), vget_high_s8(x0));
        int16x8_t p1 = vmull_s8(vget_low_s8(w1), vget_low_s8(x1));
        p1 = vmlal_s8(p1, vget_high_s8(w1), vget_high_s8(x1));
        int32x4_t p = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
#endif
        acc = vmlaq_n_f32(acc, vcvtq_f32_s32(p), ti_h2f(w[b].d) * ti_h2f(x[b].d));
    }
    return vaddvq_f32(acc);
#elif defined(TI_AVX2)
    const int nb = n / TI_QK;
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        __m256i a = _mm256_loadu_si256((const __m256i *)w[b].qs);
        __m256i c = _mm256_loadu_si256((const __m256i *)x[b].qs);
        __m256 d = _mm256_set1_ps(ti_h2f(w[b].d) * ti_h2f(x[b].d));
        acc = _mm256_fmadd_ps(d, mul_sum_i8_f32(a, c), acc);
    }
    return hsum256(acc);
#else
    return ti_dot_q8_0_q8_0_scalar(w, x, n);
#endif
}

/* ------------------------------------------------------------ q4_0 ------- */
float ti_dot_q4_0_q8_0(const block_q4_0 *w, const block_q8_0 *x, int n) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    const int nb = n / TI_QK;
    const uint8x16_t m4 = vdupq_n_u8(0x0f);
    const int8x16_t s8 = vdupq_n_s8(8);
    float32x4_t acc = vdupq_n_f32(0);
    for (int b = 0; b < nb; b++) {
        uint8x16_t v = vld1q_u8(w[b].qs);
        int8x16_t w0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, m4)), s8);
        int8x16_t w1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), s8);
        int8x16_t x0 = vld1q_s8(x[b].qs), x1 = vld1q_s8(x[b].qs + 16);
#if defined(__ARM_FEATURE_DOTPROD)
        int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, x0), w1, x1);
#else
        int16x8_t p0 = vmull_s8(vget_low_s8(w0), vget_low_s8(x0));
        p0 = vmlal_s8(p0, vget_high_s8(w0), vget_high_s8(x0));
        int16x8_t p1 = vmull_s8(vget_low_s8(w1), vget_low_s8(x1));
        p1 = vmlal_s8(p1, vget_high_s8(w1), vget_high_s8(x1));
        int32x4_t p = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
#endif
        acc = vmlaq_n_f32(acc, vcvtq_f32_s32(p), ti_h2f(w[b].d) * ti_h2f(x[b].d));
    }
    return vaddvq_f32(acc);
#elif defined(TI_AVX2)
    const int nb = n / TI_QK;
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m256i s8 = _mm256_set1_epi8(8);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        __m128i v = _mm_loadu_si128((const __m128i *)w[b].qs);
        __m128i lo = _mm_and_si128(v, m4);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), m4);
        __m256i a = _mm256_sub_epi8(_mm256_set_m128i(hi, lo), s8);
        __m256i c = _mm256_loadu_si256((const __m256i *)x[b].qs);
        __m256 d = _mm256_set1_ps(ti_h2f(w[b].d) * ti_h2f(x[b].d));
        acc = _mm256_fmadd_ps(d, mul_sum_i8_f32(a, c), acc);
    }
    return hsum256(acc);
#else
    return ti_dot_q4_0_q8_0_scalar(w, x, n);
#endif
}

/* ------------------------------------------- 4-token blocked kernels ----- */
/* Prefill: one weight block is loaded/unpacked once and multiplied with the
 * matching block of 4 different tokens (4 independent accumulators). This
 * quarters the weight traffic and the q4 unpacking work per FLOP. */
void ti_dot_q_x4(int type, const void *wv, const block_q8_0 *const x[4], int n, float out[4]) {
    const int nb = n / TI_QK;
#if defined(__ARM_NEON) && defined(__aarch64__)
    float32x4_t acc[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
    const uint8x16_t m4 = vdupq_n_u8(0x0f);
    const int8x16_t s8 = vdupq_n_s8(8);
    for (int b = 0; b < nb; b++) {
        int8x16_t w0, w1;
        float dw;
        if (type == TI_TYPE_Q8_0) {
            const block_q8_0 *w = (const block_q8_0 *)wv + b;
            w0 = vld1q_s8(w->qs);
            w1 = vld1q_s8(w->qs + 16);
            dw = ti_h2f(w->d);
        } else {
            const block_q4_0 *w = (const block_q4_0 *)wv + b;
            uint8x16_t v = vld1q_u8(w->qs);
            w0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, m4)), s8);
            w1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), s8);
            dw = ti_h2f(w->d);
        }
        for (int k = 0; k < 4; k++) {
            const block_q8_0 *xb = x[k] + b;
            int8x16_t x0 = vld1q_s8(xb->qs), x1 = vld1q_s8(xb->qs + 16);
#if defined(__ARM_FEATURE_DOTPROD)
            int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, x0), w1, x1);
#else
            int16x8_t p0 = vmull_s8(vget_low_s8(w0), vget_low_s8(x0));
            p0 = vmlal_s8(p0, vget_high_s8(w0), vget_high_s8(x0));
            int16x8_t p1 = vmull_s8(vget_low_s8(w1), vget_low_s8(x1));
            p1 = vmlal_s8(p1, vget_high_s8(w1), vget_high_s8(x1));
            int32x4_t p = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
#endif
            acc[k] = vmlaq_n_f32(acc[k], vcvtq_f32_s32(p), dw * ti_h2f(xb->d));
        }
    }
    for (int k = 0; k < 4; k++) out[k] = vaddvq_f32(acc[k]);
#elif defined(TI_AVX2)
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                     _mm256_setzero_ps()};
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m256i s8 = _mm256_set1_epi8(8);
    for (int b = 0; b < nb; b++) {
        __m256i a;
        float dw;
        if (type == TI_TYPE_Q8_0) {
            const block_q8_0 *w = (const block_q8_0 *)wv + b;
            a = _mm256_loadu_si256((const __m256i *)w->qs);
            dw = ti_h2f(w->d);
        } else {
            const block_q4_0 *w = (const block_q4_0 *)wv + b;
            __m128i v = _mm_loadu_si128((const __m128i *)w->qs);
            __m128i lo = _mm_and_si128(v, m4);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), m4);
            a = _mm256_sub_epi8(_mm256_set_m128i(hi, lo), s8);
            dw = ti_h2f(w->d);
        }
        const __m256i ax = _mm256_sign_epi8(a, a);
        for (int k = 0; k < 4; k++) {
            const block_q8_0 *xb = x[k] + b;
            __m256i c = _mm256_loadu_si256((const __m256i *)xb->qs);
            __m256i d16 = _mm256_maddubs_epi16(ax, _mm256_sign_epi8(c, a));
            __m256 p = _mm256_cvtepi32_ps(_mm256_madd_epi16(d16, _mm256_set1_epi16(1)));
            acc[k] = _mm256_fmadd_ps(_mm256_set1_ps(dw * ti_h2f(xb->d)), p, acc[k]);
        }
    }
    for (int k = 0; k < 4; k++) out[k] = hsum256(acc[k]);
#else
    (void)nb;
    for (int k = 0; k < 4; k++)
        out[k] = type == TI_TYPE_Q8_0 ? ti_dot_q8_0_q8_0((const block_q8_0 *)wv, x[k], n)
                                      : ti_dot_q4_0_q8_0((const block_q4_0 *)wv, x[k], n);
#endif
}

TI_API const char *ti_simd_path(void) {
#if defined(__ARM_NEON) && defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
    return "neon-dotprod";
#elif defined(__ARM_NEON) && defined(__aarch64__)
    return "neon";
#elif defined(TI_AVX2) && defined(__F16C__)
    return "avx2+f16c";
#elif defined(TI_AVX2)
    return "avx2";
#else
    return "scalar";
#endif
}

/* ------------------------------------------------------------ matmul ----- */
typedef struct {
    int type;
    const void *w;
    const float *x;      /* f32 activations [n_tok][cols] */
    const void *xq;      /* q8_0 activations [n_tok][cols/32] */
    float *y;            /* [n_tok][rows] */
    int rows, cols, n_tok;
    size_t w_row_bytes, xq_row_bytes;
} matmul_job;

static void matmul_task(void *vctx, int r0, int r1) {
    const matmul_job *j = (const matmul_job *)vctx;
    for (int r = r0; r < r1; r++) {
        const void *wr = (const char *)j->w + (size_t)r * j->w_row_bytes;
        int t = 0;
        if (j->type == TI_TYPE_Q8_0 || j->type == TI_TYPE_Q4_0) {
            for (; t + 4 <= j->n_tok; t += 4) {
                const block_q8_0 *xs[4];
                float o[4];
                for (int k = 0; k < 4; k++)
                    xs[k] = (const block_q8_0 *)((const char *)j->xq + (t + k) * j->xq_row_bytes);
                ti_dot_q_x4(j->type, wr, xs, j->cols, o);
                for (int k = 0; k < 4; k++) j->y[(size_t)(t + k) * j->rows + r] = o[k];
            }
        }
        for (; t < j->n_tok; t++) {
            const float *xt = j->x + (size_t)t * j->cols;
            float v;
            switch (j->type) {
            case TI_TYPE_F32: v = ti_dot_f32((const float *)wr, xt, j->cols); break;
            case TI_TYPE_F16: v = ti_dot_f16_f32((const uint16_t *)wr, xt, j->cols); break;
            case TI_TYPE_BF16: v = ti_dot_bf16_f32((const uint16_t *)wr, xt, j->cols); break;
            case TI_TYPE_Q8_0:
                v = ti_dot_q8_0_q8_0((const block_q8_0 *)wr,
                                     (const block_q8_0 *)((const char *)j->xq + t * j->xq_row_bytes),
                                     j->cols);
                break;
            case TI_TYPE_Q4_0:
                v = ti_dot_q4_0_q8_0((const block_q4_0 *)wr,
                                     (const block_q8_0 *)((const char *)j->xq + t * j->xq_row_bytes),
                                     j->cols);
                break;
            default: v = 0.0f; break;
            }
            j->y[(size_t)t * j->rows + r] = v;
        }
    }
}

typedef struct {
    const float *x;
    void *xq;
    int cols;
    size_t row_bytes;
} quant_job;

static void quant_task(void *vctx, int t0, int t1) {
    const quant_job *q = (const quant_job *)vctx;
    for (int t = t0; t < t1; t++)
        ti_quantize_row_q8_0(q->x + (size_t)t * q->cols,
                             (block_q8_0 *)((char *)q->xq + t * q->row_bytes), q->cols);
}

/* Not re-entrant: one scratch buffer for the quantised activations. */
static void *g_xq = NULL;
static size_t g_xq_cap = 0;

TI_API void ti_matmul(int type, const void *w, const float *x, float *y, int rows, int cols,
                      int n_tok) {
    matmul_job j;
    j.type = type;
    j.w = w;
    j.x = x;
    j.xq = NULL;
    j.y = y;
    j.rows = rows;
    j.cols = cols;
    j.n_tok = n_tok;
    j.w_row_bytes = ti_type_row_size(type, cols);
    j.xq_row_bytes = ti_type_row_size(TI_TYPE_Q8_0, cols);
    if (type == TI_TYPE_Q8_0 || type == TI_TYPE_Q4_0) {
        size_t need = j.xq_row_bytes * (size_t)n_tok;
        if (need > g_xq_cap) {
            free(g_xq);
            g_xq = malloc(need);
            g_xq_cap = g_xq ? need : 0;
            if (!g_xq) return;
        }
        quant_job q = {x, g_xq, cols, j.xq_row_bytes};
        ti_parallel_for(n_tok, 1, quant_task, &q);
        j.xq = g_xq;
    }
    int nth = ti_pool_threads();
    int grain = rows / (nth * 8);
    if (grain < 4) grain = 4;
    ti_parallel_for(rows, grain, matmul_task, &j);
}
