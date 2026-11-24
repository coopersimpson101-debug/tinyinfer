// tinyinfer Metal kernels (decode path: one token per command buffer).
//
// NOTE: written for Apple Silicon but NOT compiled or run in this repo's CI
// (no macOS hardware was available while writing it). See README.
//
// Conventions match the CPU engine:
//   matvec:  y[r] = sum_c W[r][c] * x[c], one SIMD-group (32 lanes) per row,
//            8 rows per threadgroup, lanes stride over 32-element blocks.
//   rope:    NeoX layout, pair (i, i + hd/2), angle = pos * inv_freq[i]
//   attn:    one threadgroup per query head, scores in a device scratch row.
#include <metal_stdlib>
using namespace metal;

#define QK 32
#define ROWS_PER_TG 8

struct block_q8_0 { half d; char qs[QK]; };        // 34 bytes, same as ggml
struct block_q4_0 { half d; uchar qs[QK / 2]; }; // 18 bytes, same as ggml

struct MatvecArgs { int rows; int cols; };

kernel void matvec_f32(device const float *w [[buffer(0)]],
                       device const float *x [[buffer(1)]],
                       device float *y [[buffer(2)]],
                       constant MatvecArgs &a [[buffer(3)]],
                       uint tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    const int r = int(tg) * ROWS_PER_TG + int(sg);
    if (r >= a.rows) return;
    device const float4 *wr = (device const float4 *)(w + (ulong)r * a.cols);
    device const float4 *x4 = (device const float4 *)x;
    float sum = 0.0f;
    for (int i = int(lane); i < a.cols / 4; i += 32) sum += dot(wr[i], x4[i]);
    sum = simd_sum(sum);
    if (lane == 0) y[r] = sum;
}

kernel void matvec_f16(device const half *w [[buffer(0)]],
                       device const float *x [[buffer(1)]],
                       device float *y [[buffer(2)]],
                       constant MatvecArgs &a [[buffer(3)]],
                       uint tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    const int r = int(tg) * ROWS_PER_TG + int(sg);
    if (r >= a.rows) return;
    device const half4 *wr = (device const half4 *)(w + (ulong)r * a.cols);
    device const float4 *x4 = (device const float4 *)x;
    float sum = 0.0f;
    for (int i = int(lane); i < a.cols / 4; i += 32) sum += dot(float4(wr[i]), x4[i]);
    sum = simd_sum(sum);
    if (lane == 0) y[r] = sum;
}

kernel void matvec_q8_0(device const block_q8_0 *w [[buffer(0)]],
                        device const float *x [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant MatvecArgs &a [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const int r = int(tg) * ROWS_PER_TG + int(sg);
    if (r >= a.rows) return;
    const int nb = a.cols / QK;
    device const block_q8_0 *wr = w + (ulong)r * nb;
    float sum = 0.0f;
    for (int b = int(lane); b < nb; b += 32) {
        device const float4 *xb = (device const float4 *)(x + b * QK);
        float s = 0.0f;
        for (int j = 0; j < QK / 4; j++) {
            float4 q = float4(wr[b].qs[4 * j], wr[b].qs[4 * j + 1], wr[b].qs[4 * j + 2], wr[b].qs[4 * j + 3]);
            s += dot(q, xb[j]);
        }
        sum += s * float(wr[b].d);
    }
    sum = simd_sum(sum);
    if (lane == 0) y[r] = sum;
}

kernel void matvec_q4_0(device const block_q4_0 *w [[buffer(0)]],
                        device const float *x [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant MatvecArgs &a [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const int r = int(tg) * ROWS_PER_TG + int(sg);
    if (r >= a.rows) return;
    const int nb = a.cols / QK;
    device const block_q4_0 *wr = w + (ulong)r * nb;
    float sum = 0.0f;
    for (int b = int(lane); b < nb; b += 32) {
        device const float *xb = x + b * QK;
        float s = 0.0f;
        for (int j = 0; j < QK / 2; j++) {
            const uchar v = wr[b].qs[j];
            s += float(int(v & 0x0F) - 8) * xb[j] + float(int(v >> 4) - 8) * xb[j + QK / 2];
        }
        sum += s * float(wr[b].d);
    }
    sum = simd_sum(sum);
    if (lane == 0) y[r] = sum;
}

struct NormArgs { int n; float eps; };

// single threadgroup; o = x / sqrt(mean(x^2) + eps) * w
kernel void rmsnorm(device const float *x [[buffer(0)]],
                    device const float *w [[buffer(1)]],
                    device float *o [[buffer(2)]],
                    constant NormArgs &a [[buffer(3)]],
                    uint tid [[thread_position_in_threadgroup]],
                    uint tpg [[threads_per_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[32];
    float ss = 0.0f;
    for (int i = int(tid); i < a.n; i += int(tpg)) ss += x[i] * x[i];
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = lane < (tpg + 31) / 32 ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float s = rsqrt(partial[0] / float(a.n) + a.eps);
    for (int i = int(tid); i < a.n; i += int(tpg)) o[i] = x[i] * s * w[i];
}

kernel void add_inplace(device float *x [[buffer(0)]],
                        device const float *y [[buffer(1)]],
                        constant int &n [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    if (int(i) < n) x[i] += y[i];
}

kernel void silu_mul(device float *g [[buffer(0)]],
                     device const float *u [[buffer(1)]],
                     constant int &n [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (int(i) < n) {
        const float v = g[i];
        g[i] = v / (1.0f + exp(-v)) * u[i];
    }
}

struct RopeArgs { int n_heads; int head_dim; int pos; };

// x: [n_heads][head_dim]; one thread per (head, i < head_dim/2)
kernel void rope_neox(device float *x [[buffer(0)]],
                      device const float *inv_freq [[buffer(1)]],
                      constant RopeArgs &a [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    const int half_d = a.head_dim / 2;
    const int h = int(gid) / half_d, i = int(gid) % half_d;
    if (h >= a.n_heads) return;
    const float ang = float(a.pos) * inv_freq[i];
    const float c = precise::cos(ang), s = precise::sin(ang);
    device float *p = x + h * a.head_dim;
    const float x0 = p[i], x1 = p[i + half_d];
    p[i] = x0 * c - x1 * s;
    p[i + half_d] = x0 * s + x1 * c;
}

struct KVArgs { int kv_dim; int pos; };

kernel void kv_store(device const float *k [[buffer(0)]],
                     device const float *v [[buffer(1)]],
                     device float *kc [[buffer(2)]],
                     device float *vc [[buffer(3)]],
                     constant KVArgs &a [[buffer(4)]],
                     uint i [[thread_position_in_grid]]) {
    if (int(i) < a.kv_dim) {
        kc[(ulong)a.pos * a.kv_dim + i] = k[i];
        vc[(ulong)a.pos * a.kv_dim + i] = v[i];
    }
}

struct AttnArgs { int n_heads; int n_kv_heads; int head_dim; int pos; int max_ctx; float scale; };

// one threadgroup per query head
kernel void attention(device const float *q [[buffer(0)]],
                      device const float *kc [[buffer(1)]],
                      device const float *vc [[buffer(2)]],
                      device float *scores [[buffer(3)]],
                      device float *out [[buffer(4)]],
                      constant AttnArgs &a [[buffer(5)]],
                      uint h [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint tpg [[threads_per_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    const int hd = a.head_dim, kvd = a.n_kv_heads * hd;
    const int kvh = int(h) / (a.n_heads / a.n_kv_heads);
    device const float *qh = q + h * hd;
    device float *sc = scores + (ulong)h * a.max_ctx;
    const int n = a.pos + 1;
    const uint nsg = (tpg + 31) / 32;

    for (int t = int(tid); t < n; t += int(tpg)) {
        device const float *k = kc + (ulong)t * kvd + kvh * hd;
        float s = 0.0f;
        for (int i = 0; i < hd; i++) s += qh[i] * k[i];
        sc[t] = s * a.scale;
    }
    threadgroup_barrier(mem_flags::mem_device);

    float m = -INFINITY;
    for (int t = int(tid); t < n; t += int(tpg)) m = max(m, sc[t]);
    m = simd_max(m);
    if (lane == 0) red[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = lane < nsg ? red[lane] : -INFINITY;
        v = simd_max(v);
        if (lane == 0) red[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float l = 0.0f;
    for (int t = int(tid); t < n; t += int(tpg)) {
        const float e = exp(sc[t] - m);
        sc[t] = e;
        l += e;
    }
    l = simd_sum(l);
    if (lane == 0) red[sg] = l;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (sg == 0) {
        float v = lane < nsg ? red[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) red[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / red[0];

    for (int d = int(tid); d < hd; d += int(tpg)) {
        float acc = 0.0f;
        for (int t = 0; t < n; t++) acc += sc[t] * vc[(ulong)t * kvd + kvh * hd + d];
        out[h * hd + d] = acc * inv;
    }
}
