/* Internal model structures (shared with the Metal backend). */
#ifndef TI_MODEL_H
#define TI_MODEL_H

#include "../include/tinyinfer.h"
#include "gguf.h"
#include "tokenizer.h"
#include "kvcache.h"

#define TI_MAX_SEQS 16

typedef struct {
    int type;          /* ti_type */
    const void *data;  /* points into the mmap (or an owned copy) */
    int64_t ne0, ne1;  /* ne0 = columns (contiguous), ne1 = rows */
    size_t nbytes;
} ti_tensor;

typedef struct {
    const float *attn_norm, *ffn_norm;
    const float *bq, *bk, *bv; /* NULL if the model has no QKV bias */
    ti_tensor wq, wk, wv, wo, w_gate, w_up, w_down;
} ti_layer;

struct ti_model {
    gguf_file gguf;
    ti_config cfg;
    ti_tokenizer tok;
    ti_tensor tok_embd, output;
    const float *out_norm;
    ti_layer *layers;

    /* KV cache: [n_layers][max_ctx][n_kv_heads * head_dim], f32 */
    float *k_cache, *v_cache;
    /* paged KV cache (ti_kv_enable_paged): replaces k_cache/v_cache when set */
    int paged;
    ti_kv_pool pool;
    ti_kv_seq seqs[TI_MAX_SEQS];
    float *rope_inv_freq; /* [head_dim / 2] */

    /* activations for a batch of up to max_batch tokens */
    int max_batch;
    float *x, *xb, *q, *k, *v, *att, *hb, *hb2, *tmp;

    /* f32 copies of small vectors that were stored as f16/bf16 */
    void **owned;
    int n_owned, cap_owned;

    void *metal; /* ti_metal_ctx*, NULL when running on the CPU */
};

#endif
