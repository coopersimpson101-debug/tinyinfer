#ifndef TI_KERNELS_H
#define TI_KERNELS_H

#include "quant.h"

/* Single dot products (SIMD-dispatched at compile time). n = element count. */
float ti_dot_f32(const float *a, const float *b, int n);
float ti_dot_f16_f32(const uint16_t *w, const float *x, int n);
float ti_dot_bf16_f32(const uint16_t *w, const float *x, int n);
float ti_dot_q8_0_q8_0(const block_q8_0 *w, const block_q8_0 *x, int n);
float ti_dot_q4_0_q8_0(const block_q4_0 *w, const block_q8_0 *x, int n);

/* out[k] = dot(w, x[k]) for 4 tokens; type is TI_TYPE_Q8_0 or TI_TYPE_Q4_0 */
void ti_dot_q_x4(int type, const void *w, const block_q8_0 *const x[4], int n, float out[4]);

/* Portable scalar versions, always compiled; used by tests to check SIMD. */
float ti_dot_q8_0_q8_0_scalar(const block_q8_0 *w, const block_q8_0 *x, int n);
float ti_dot_q4_0_q8_0_scalar(const block_q4_0 *w, const block_q8_0 *x, int n);

#endif
