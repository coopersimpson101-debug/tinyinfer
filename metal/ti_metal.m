// Metal host code for tinyinfer.
//
// NOTE: this file targets macOS / Apple Silicon and could not be compiled or
// run where the repo was written (Windows + Linux CI only). It is built by
// `make metal`; treat it as unverified until the steps in the README's
// "Hardware validation" section have been run on an M1.
//
// Design
// - Weights are not copied: the GGUF mmap is wrapped once with
//   newBufferWithBytesNoCopy (unified memory) and every tensor is addressed as
//   (buffer, byte offset). Small f32 vectors (norms, biases) are copied.
// - One command buffer per token; every op of every layer is encoded into a
//   single serial compute encoder, so there is exactly one CPU<->GPU sync
//   per generated token.
// - KV cache and activations live in shared-storage buffers.
// - Compiled with manual reference counting (-fno-objc-arc) because the
//   Metal objects live in plain C structs; ti_metal_free releases them.
#if __has_feature(objc_arc)
#error "compile ti_metal.m with -fno-objc-arc"
#endif
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ti_metal.h"
#include "../src/platform.h"

#include <math.h>
#include <string.h>
#include <unistd.h>

static const char kKernelSource[] = {
#include "kernels_metal.inc"
    , 0};

typedef struct { int rows, cols; } MatvecArgs;
typedef struct { int n; float eps; } NormArgs;
typedef struct { int n_heads, head_dim, pos; } RopeArgs;
typedef struct { int kv_dim, pos; } KVArgs;
typedef struct { int n_heads, n_kv_heads, head_dim, pos, max_ctx; float scale; } AttnArgs;

typedef struct {
    id<MTLBuffer> buf;
    NSUInteger off;
    int type, rows, cols;
} MTensor;

typedef struct {
    MTensor wq, wk, wv, wo, w_gate, w_up, w_down;
    id<MTLBuffer> attn_norm, ffn_norm, bq, bk, bv;
} MLayer;

typedef struct {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> mv_f32, mv_f16, mv_q8, mv_q4, rmsnorm, add, silu_mul, rope, kv_store,
        attention;
    id<MTLBuffer> weights; /* wraps the whole GGUF mapping */
    MLayer *layers;
    MTensor output;
    id<MTLBuffer> out_norm, inv_freq;
    id<MTLBuffer> x, xb, q, k, v, att, hb, hb2, tmp, logits, scores, kc, vc;
    int n_layers;
} MetalCtx;

#define REL(x)              do {                        [(x) release];          (x) = nil;          } while (0)

static id<MTLComputePipelineState> make_pipe(MetalCtx *c, id<MTLLibrary> lib, NSString *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) {
        ti_set_error("Metal: kernel %s not found", name.UTF8String);
        return nil;
    }
    id<MTLComputePipelineState> p = [c->dev newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    if (!p) ti_set_error("Metal: pipeline %s: %s", name.UTF8String, err.localizedDescription.UTF8String);
    return p;
}

static int wrap_tensor(MetalCtx *c, const ti_model *m, const ti_tensor *t, MTensor *out) {
    const uint8_t *base = (const uint8_t *)m->gguf.map.data;
    const uint8_t *p = (const uint8_t *)t->data;
    if (p < base || p + t->nbytes > base + m->gguf.map.size) {
        ti_set_error("Metal: tensor not inside the GGUF mapping");
        return -1;
    }
    if (t->ne0 % 32 != 0) {
        ti_set_error("Metal: matrix columns must be a multiple of 32");
        return -1;
    }
    out->buf = c->weights;
    out->off = (NSUInteger)(p - base);
    out->type = t->type;
    out->rows = (int)t->ne1;
    out->cols = (int)t->ne0;
    return 0;
}

static id<MTLBuffer> copy_vec(MetalCtx *c, const float *v, int n) {
    if (!v) return nil;
    return [c->dev newBufferWithBytes:v length:sizeof(float) * (NSUInteger)n
                              options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> scratch(MetalCtx *c, size_t floats) {
    return [c->dev newBufferWithLength:sizeof(float) * floats options:MTLResourceStorageModeShared];
}

void *ti_metal_init(ti_model *m) {
    @autoreleasepool {
        MetalCtx *c = (MetalCtx *)calloc(1, sizeof(MetalCtx));
        c->dev = MTLCreateSystemDefaultDevice();
        if (!c->dev) {
            ti_set_error("Metal: no GPU device");
            free(c);
            return NULL;
        }
        c->queue = [c->dev newCommandQueue];
        NSError *err = nil;
        NSString *src = [NSString stringWithUTF8String:kKernelSource];
        MTLCompileOptions *opt = [MTLCompileOptions new];
        id<MTLLibrary> lib = [c->dev newLibraryWithSource:src options:opt error:&err];
        [opt release];
        if (!lib) {
            ti_set_error("Metal: shader compile failed: %s", err.localizedDescription.UTF8String);
            ti_metal_free(c);
            return NULL;
        }
        c->mv_f32 = make_pipe(c, lib, @"matvec_f32");
        c->mv_f16 = make_pipe(c, lib, @"matvec_f16");
        c->mv_q8 = make_pipe(c, lib, @"matvec_q8_0");
        c->mv_q4 = make_pipe(c, lib, @"matvec_q4_0");
        c->rmsnorm = make_pipe(c, lib, @"rmsnorm");
        c->add = make_pipe(c, lib, @"add_inplace");
        c->silu_mul = make_pipe(c, lib, @"silu_mul");
        c->rope = make_pipe(c, lib, @"rope_neox");
        c->kv_store = make_pipe(c, lib, @"kv_store");
        c->attention = make_pipe(c, lib, @"attention");
        [lib release];
        if (!c->mv_f32 || !c->mv_f16 || !c->mv_q8 || !c->mv_q4 || !c->rmsnorm || !c->add ||
            !c->silu_mul || !c->rope || !c->kv_store || !c->attention) {
            ti_metal_free(c);
            return NULL;
        }

        /* zero-copy view of the whole mmap (page aligned, length rounded up) */
        const size_t page = (size_t)getpagesize();
        size_t len = (m->gguf.map.size + page - 1) / page * page;
        c->weights = [c->dev newBufferWithBytesNoCopy:m->gguf.map.data
                                               length:len
                                              options:MTLResourceStorageModeShared
                                          deallocator:nil];
        if (!c->weights) {
            ti_set_error("Metal: could not wrap the model mapping (%zu bytes; max buffer %llu)",
                         len, (unsigned long long)c->dev.maxBufferLength);
            ti_metal_free(c);
            return NULL;
        }

        const ti_config *cf = &m->cfg;
        for (int l = 0; l < cf->n_layers; l++) {
            const ti_layer *L = &m->layers[l];
            if (L->wq.type == TI_TYPE_BF16) {
                ti_set_error("Metal: BF16 weights are not supported (use f16/q8_0/q4_0)");
                ti_metal_free(c);
                return NULL;
            }
        }
        c->n_layers = cf->n_layers;
        c->layers = (MLayer *)calloc((size_t)cf->n_layers, sizeof(MLayer));
        for (int l = 0; l < cf->n_layers; l++) {
            const ti_layer *L = &m->layers[l];
            MLayer *G = &c->layers[l];
            if (wrap_tensor(c, m, &L->wq, &G->wq) || wrap_tensor(c, m, &L->wk, &G->wk) ||
                wrap_tensor(c, m, &L->wv, &G->wv) || wrap_tensor(c, m, &L->wo, &G->wo) ||
                wrap_tensor(c, m, &L->w_gate, &G->w_gate) || wrap_tensor(c, m, &L->w_up, &G->w_up) ||
                wrap_tensor(c, m, &L->w_down, &G->w_down)) {
                ti_metal_free(c);
                return NULL;
            }
            G->attn_norm = copy_vec(c, L->attn_norm, cf->dim);
            G->ffn_norm = copy_vec(c, L->ffn_norm, cf->dim);
            G->bq = copy_vec(c, L->bq, cf->n_heads * cf->head_dim);
            G->bk = copy_vec(c, L->bk, cf->n_kv_heads * cf->head_dim);
            G->bv = copy_vec(c, L->bv, cf->n_kv_heads * cf->head_dim);
        }
        if (wrap_tensor(c, m, &m->output, &c->output)) {
            ti_metal_free(c);
            return NULL;
        }
        c->out_norm = copy_vec(c, m->out_norm, cf->dim);
        c->inv_freq = copy_vec(c, m->rope_inv_freq, cf->head_dim / 2);

        const size_t D = (size_t)cf->dim, QD = (size_t)cf->n_heads * cf->head_dim;
        const size_t KVD = (size_t)cf->n_kv_heads * cf->head_dim, F = (size_t)cf->ffn_dim;
        c->x = scratch(c, D);
        c->xb = scratch(c, D);
        c->tmp = scratch(c, D);
        c->q = scratch(c, QD);
        c->att = scratch(c, QD);
        c->k = scratch(c, KVD);
        c->v = scratch(c, KVD);
        c->hb = scratch(c, F);
        c->hb2 = scratch(c, F);
        c->logits = scratch(c, (size_t)cf->vocab_size);
        c->scores = scratch(c, (size_t)cf->n_heads * cf->max_ctx);
        c->kc = scratch(c, (size_t)cf->n_layers * cf->max_ctx * KVD);
        c->vc = scratch(c, (size_t)cf->n_layers * cf->max_ctx * KVD);
        if (!c->kc || !c->vc || !c->scores) {
            ti_set_error("Metal: out of GPU memory for the KV cache");
            ti_metal_free(c);
            return NULL;
        }
        return c;
    }
}

void ti_metal_free(void *vctx) {
    MetalCtx *c = (MetalCtx *)vctx;
    if (!c) return;
    if (c->layers) {
        for (int l = 0; l < c->n_layers; l++) {
            MLayer *L = &c->layers[l];
            REL(L->attn_norm); REL(L->ffn_norm); REL(L->bq); REL(L->bk); REL(L->bv);
            /* MTensor.buf aliases c->weights (not owned) */
        }
        free(c->layers);
    }
    REL(c->mv_f32); REL(c->mv_f16); REL(c->mv_q8); REL(c->mv_q4); REL(c->rmsnorm);
    REL(c->add); REL(c->silu_mul); REL(c->rope); REL(c->kv_store); REL(c->attention);
    REL(c->out_norm); REL(c->inv_freq);
    REL(c->x); REL(c->xb); REL(c->q); REL(c->k); REL(c->v); REL(c->att); REL(c->hb);
    REL(c->hb2); REL(c->tmp); REL(c->logits); REL(c->scores); REL(c->kc); REL(c->vc);
    REL(c->weights);
    REL(c->queue);
    REL(c->dev);
    free(c);
}

/* ------------------------------------------------------------ encoding --- */
static void enc_matvec(MetalCtx *c, id<MTLComputeCommandEncoder> e, const MTensor *w,
                       id<MTLBuffer> x, id<MTLBuffer> y) {
    id<MTLComputePipelineState> p;
    switch (w->type) {
    case TI_TYPE_F32: p = c->mv_f32; break;
    case TI_TYPE_F16: p = c->mv_f16; break;
    case TI_TYPE_Q8_0: p = c->mv_q8; break;
    default: p = c->mv_q4; break;
    }
    MatvecArgs a = {w->rows, w->cols};
    [e setComputePipelineState:p];
    [e setBuffer:w->buf offset:w->off atIndex:0];
    [e setBuffer:x offset:0 atIndex:1];
    [e setBuffer:y offset:0 atIndex:2];
    [e setBytes:&a length:sizeof a atIndex:3];
    NSUInteger groups = ((NSUInteger)w->rows + 7) / 8;
    [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void enc_elementwise(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p,
                            id<MTLBuffer> a, id<MTLBuffer> b, int n) {
    [e setComputePipelineState:p];
    [e setBuffer:a offset:0 atIndex:0];
    [e setBuffer:b offset:0 atIndex:1];
    [e setBytes:&n length:sizeof n atIndex:2];
    [e dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void enc_rmsnorm(MetalCtx *c, id<MTLComputeCommandEncoder> e, id<MTLBuffer> x,
                        id<MTLBuffer> w, id<MTLBuffer> o, int n, float eps) {
    NormArgs a = {n, eps};
    [e setComputePipelineState:c->rmsnorm];
    [e setBuffer:x offset:0 atIndex:0];
    [e setBuffer:w offset:0 atIndex:1];
    [e setBuffer:o offset:0 atIndex:2];
    [e setBytes:&a length:sizeof a atIndex:3];
    NSUInteger tpg = MIN((NSUInteger)1024, c->rmsnorm.maxTotalThreadsPerThreadgroup);
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

static void enc_rope(MetalCtx *c, id<MTLComputeCommandEncoder> e, id<MTLBuffer> x, int n_heads,
                     int hd, int pos) {
    RopeArgs a = {n_heads, hd, pos};
    [e setComputePipelineState:c->rope];
    [e setBuffer:x offset:0 atIndex:0];
    [e setBuffer:c->inv_freq offset:0 atIndex:1];
    [e setBytes:&a length:sizeof a atIndex:2];
    [e dispatchThreads:MTLSizeMake((NSUInteger)(n_heads * hd / 2), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

int ti_metal_forward(void *vctx, ti_model *m, int token, int pos, float *logits) {
    MetalCtx *c = (MetalCtx *)vctx;
    const ti_config *cf = &m->cfg;
    const int D = cf->dim, hd = cf->head_dim, QD = cf->n_heads * hd, KVD = cf->n_kv_heads * hd;
    const int F = cf->ffn_dim;
    if (token < 0 || token >= cf->vocab_size) {
        ti_set_error("token id %d out of range", token);
        return -1;
    }
    @autoreleasepool {
        /* embedding lookup on the CPU straight into the shared buffer */
        size_t rb = ti_type_row_size(m->tok_embd.type, D);
        ti_dequantize_row(m->tok_embd.type, (const char *)m->tok_embd.data + rb * (size_t)token,
                          (float *)c->x.contents, D);

        id<MTLCommandBuffer> cb = [c->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder]; /* serial dispatch */
        const size_t layer_kv = (size_t)cf->max_ctx * KVD * sizeof(float);
        for (int l = 0; l < cf->n_layers; l++) {
            MLayer *L = &c->layers[l];
            enc_rmsnorm(c, e, c->x, L->attn_norm, c->xb, D, cf->rms_eps);
            enc_matvec(c, e, &L->wq, c->xb, c->q);
            enc_matvec(c, e, &L->wk, c->xb, c->k);
            enc_matvec(c, e, &L->wv, c->xb, c->v);
            if (L->bq) {
                enc_elementwise(e, c->add, c->q, L->bq, QD);
                enc_elementwise(e, c->add, c->k, L->bk, KVD);
                enc_elementwise(e, c->add, c->v, L->bv, KVD);
            }
            enc_rope(c, e, c->q, cf->n_heads, hd, pos);
            enc_rope(c, e, c->k, cf->n_kv_heads, hd, pos);

            KVArgs ka = {KVD, pos};
            [e setComputePipelineState:c->kv_store];
            [e setBuffer:c->k offset:0 atIndex:0];
            [e setBuffer:c->v offset:0 atIndex:1];
            [e setBuffer:c->kc offset:layer_kv * l atIndex:2];
            [e setBuffer:c->vc offset:layer_kv * l atIndex:3];
            [e setBytes:&ka length:sizeof ka atIndex:4];
            [e dispatchThreads:MTLSizeMake((NSUInteger)KVD, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

            AttnArgs aa = {cf->n_heads, cf->n_kv_heads, hd, pos, cf->max_ctx, 1.0f / sqrtf((float)hd)};
            [e setComputePipelineState:c->attention];
            [e setBuffer:c->q offset:0 atIndex:0];
            [e setBuffer:c->kc offset:layer_kv * l atIndex:1];
            [e setBuffer:c->vc offset:layer_kv * l atIndex:2];
            [e setBuffer:c->scores offset:0 atIndex:3];
            [e setBuffer:c->att offset:0 atIndex:4];
            [e setBytes:&aa length:sizeof aa atIndex:5];
            NSUInteger atpg = MIN((NSUInteger)256, c->attention.maxTotalThreadsPerThreadgroup);
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)cf->n_heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(atpg, 1, 1)];

            enc_matvec(c, e, &L->wo, c->att, c->tmp);
            enc_elementwise(e, c->add, c->x, c->tmp, D);
            enc_rmsnorm(c, e, c->x, L->ffn_norm, c->xb, D, cf->rms_eps);
            enc_matvec(c, e, &L->w_gate, c->xb, c->hb);
            enc_matvec(c, e, &L->w_up, c->xb, c->hb2);
            enc_elementwise(e, c->silu_mul, c->hb, c->hb2, F);
            enc_matvec(c, e, &L->w_down, c->hb, c->tmp);
            enc_elementwise(e, c->add, c->x, c->tmp, D);
        }
        if (logits) {
            enc_rmsnorm(c, e, c->x, c->out_norm, c->xb, D, cf->rms_eps);
            enc_matvec(c, e, &c->output, c->xb, c->logits);
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            ti_set_error("Metal: command buffer failed: %s",
                         cb.error.localizedDescription.UTF8String);
            return -1;
        }
        if (logits) memcpy(logits, c->logits.contents, sizeof(float) * (size_t)cf->vocab_size);
    }
    return 0;
}
