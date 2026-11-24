/*
 * Q8_0 / Q4_0 block quantisation (reference scalar implementations).
 *
 * Q8_0:  d = max|x| / 127,  q = round(x / d)            in [-127, 127]
 * Q4_0:  m = x with the largest |x| (sign kept), d = m / -8,
 *        q = clamp(trunc(x / d + 8.5), 0, 15),  x' = d * (q - 8)
 *
 * Using the signed max for Q4_0 maps the extreme value exactly onto -8 and
 * gives the asymmetric range [-8, 7] one extra level on the side that matters.
 * Error bounds (tested): |x - x'| <= d/2 for Q8_0, <= |d|/2 (+ clamp) for Q4_0.
 */
#include "quant.h"

#include <math.h>
#include <stdlib.h>

void ti_quantize_row_q8_0(const float *x, block_q8_0 *y, int64_t n) {
    const int64_t nb = n / TI_QK;
    for (int64_t i = 0; i < nb; i++) {
        const float *xb = x + i * TI_QK;
        float amax = 0.0f;
        for (int j = 0; j < TI_QK; j++) {
            float a = fabsf(xb[j]);
            if (a > amax) amax = a;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = ti_f2h(d);
        for (int j = 0; j < TI_QK; j++) y[i].qs[j] = (int8_t)roundf(xb[j] * id);
    }
}

void ti_quantize_row_q4_0(const float *x, block_q4_0 *y, int64_t n) {
    const int64_t nb = n / TI_QK;
    for (int64_t i = 0; i < nb; i++) {
        const float *xb = x + i * TI_QK;
        float amax = 0.0f, maxv = 0.0f;
        for (int j = 0; j < TI_QK; j++) {
            float a = fabsf(xb[j]);
            if (a > amax) {
                amax = a;
                maxv = xb[j];
            }
        }
        const float d = maxv / -8.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = ti_f2h(d);
        for (int j = 0; j < TI_QK / 2; j++) {
            float x0 = xb[j] * id, x1 = xb[j + TI_QK / 2] * id;
            int q0 = (int)(x0 + 8.5f), q1 = (int)(x1 + 8.5f);
            if (q0 > 15) q0 = 15;
            if (q1 > 15) q1 = 15;
            if (q0 < 0) q0 = 0;
            if (q1 < 0) q1 = 0;
            y[i].qs[j] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}

void ti_dequantize_row_q8_0(const block_q8_0 *x, float *y, int64_t n) {
    const int64_t nb = n / TI_QK;
    for (int64_t i = 0; i < nb; i++) {
        const float d = ti_h2f(x[i].d);
        for (int j = 0; j < TI_QK; j++) y[i * TI_QK + j] = d * (float)x[i].qs[j];
    }
}

void ti_dequantize_row_q4_0(const block_q4_0 *x, float *y, int64_t n) {
    const int64_t nb = n / TI_QK;
    for (int64_t i = 0; i < nb; i++) {
        const float d = ti_h2f(x[i].d);
        for (int j = 0; j < TI_QK / 2; j++) {
            y[i * TI_QK + j] = d * (float)((x[i].qs[j] & 0x0f) - 8);
            y[i * TI_QK + j + TI_QK / 2] = d * (float)((x[i].qs[j] >> 4) - 8);
        }
    }
}

TI_API size_t ti_type_row_size(int type, int64_t n) {
    switch (type) {
    case TI_TYPE_F32: return (size_t)n * 4;
    case TI_TYPE_F16:
    case TI_TYPE_BF16: return (size_t)n * 2;
    case TI_TYPE_Q8_0: return (size_t)(n / TI_QK) * sizeof(block_q8_0);
    case TI_TYPE_Q4_0: return (size_t)(n / TI_QK) * sizeof(block_q4_0);
    default: return 0;
    }
}

TI_API void ti_quantize_row(int type, const float *x, void *out, int64_t n) {
    switch (type) {
    case TI_TYPE_F32: memcpy(out, x, (size_t)n * 4); break;
    case TI_TYPE_F16: {
        uint16_t *o = (uint16_t *)out;
        for (int64_t i = 0; i < n; i++) o[i] = ti_f2h(x[i]);
        break;
    }
    case TI_TYPE_Q8_0: ti_quantize_row_q8_0(x, (block_q8_0 *)out, n); break;
    case TI_TYPE_Q4_0: ti_quantize_row_q4_0(x, (block_q4_0 *)out, n); break;
    default: break;
    }
}

TI_API void ti_dequantize_row(int type, const void *x, float *out, int64_t n) {
    switch (type) {
    case TI_TYPE_F32: memcpy(out, x, (size_t)n * 4); break;
    case TI_TYPE_F16: {
        const uint16_t *h = (const uint16_t *)x;
        for (int64_t i = 0; i < n; i++) out[i] = ti_h2f(h[i]);
        break;
    }
    case TI_TYPE_BF16: {
        const uint16_t *h = (const uint16_t *)x;
        for (int64_t i = 0; i < n; i++) out[i] = ti_bf2f(h[i]);
        break;
    }
    case TI_TYPE_Q8_0: ti_dequantize_row_q8_0((const block_q8_0 *)x, out, n); break;
    case TI_TYPE_Q4_0: ti_dequantize_row_q4_0((const block_q4_0 *)x, out, n); break;
    default: break;
    }
}

TI_API uint16_t ti_fp32_to_fp16(float f) { return ti_f2h(f); }
TI_API float ti_fp16_to_fp32(uint16_t h) { return ti_h2f(h); }
