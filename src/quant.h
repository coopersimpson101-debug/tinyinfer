/* Block quantisation formats (bit-compatible with ggml / GGUF). */
#ifndef TI_QUANT_H
#define TI_QUANT_H

#include <stdint.h>
#include <string.h>
#include "../include/tinyinfer.h"

/* Q8_0: 32 weights -> fp16 scale d + 32 x int8.  x ~= d * q  (34 bytes) */
typedef struct {
    uint16_t d;
    int8_t qs[TI_QK];
} block_q8_0;

/* Q4_0: 32 weights -> fp16 scale d + 16 bytes of nibbles.  x ~= d * (q - 8).
 * Element j (0..15) is the low nibble of qs[j], element j+16 the high nibble. */
typedef struct {
    uint16_t d;
    uint8_t qs[TI_QK / 2];
} block_q4_0;

_Static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");
_Static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must be 18 bytes");

/* IEEE-754 half <-> single, implemented with integer ops (round to nearest
 * even), so results match numpy's float16 casts bit for bit. */
static inline float ti_h2f_soft(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else { /* subnormal: normalise */
            int e = -1;
            do {
                e++;
                man <<= 1;
            } while (!(man & 0x400));
            man &= 0x3ff;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* half -> float is exact, so the hardware instruction (F16C / ARMv8 FCVT) gives
 * identical results; it matters because the kernels convert two scales per
 * 32-element block. */
#if defined(__F16C__)
#include <immintrin.h>
static inline float ti_h2f(uint16_t h) { return _cvtsh_ss(h); }
#elif defined(__aarch64__)
static inline float ti_h2f(uint16_t h) {
    __fp16 f;
    memcpy(&f, &h, 2);
    return (float)f;
}
#else
#define ti_h2f ti_h2f_soft
#endif

static inline uint16_t ti_f2h(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t absx = x & 0x7fffffffu;
    if (absx >= 0x7f800000u) /* inf / nan */
        return (uint16_t)(sign | 0x7c00 | (absx > 0x7f800000u ? 0x200 : 0));
    if (absx >= 0x477ff000u) /* rounds to >= 65520 -> inf */
        return (uint16_t)(sign | 0x7c00);
    if (absx < 0x38800000u) { /* result is subnormal or zero (|x| < 2^-14) */
        if (absx < 0x33000000u) return (uint16_t)sign; /* < 2^-25 rounds to 0 */
        uint32_t e = absx >> 23;
        uint32_t m = (absx & 0x7fffff) | 0x800000;
        uint32_t shift = 126 - e; /* 14..24 */
        uint32_t r = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (r & 1))) r++;
        return (uint16_t)(sign | r);
    }
    /* normal */
    uint32_t e = (absx >> 23) - 112;
    uint32_t m = absx & 0x7fffff;
    uint32_t r = (e << 10) | (m >> 13);
    uint32_t rem = m & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) r++;
    return (uint16_t)(sign | r);
}

static inline float ti_bf2f(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

void ti_quantize_row_q8_0(const float *x, block_q8_0 *y, int64_t n);
void ti_quantize_row_q4_0(const float *x, block_q4_0 *y, int64_t n);
void ti_dequantize_row_q8_0(const block_q8_0 *x, float *y, int64_t n);
void ti_dequantize_row_q4_0(const block_q4_0 *x, float *y, int64_t n);

#endif
