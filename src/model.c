/*
 * Qwen2 transformer: loading from GGUF and the forward pass.
 *
 * Per layer (pre-norm residual block):
 *   h  = x + Wo * Attn(RoPE(Wq*n + bq), RoPE(Wk*n + bk), Wv*n + bv),  n = RMSNorm(x)
 *   x' = h + Wdown * (SiLU(Wgate*m) . (Wup*m)),                         m = RMSNorm(h)
 * logits = Wout * RMSNorm(x_L)   (Wout = token_embd when embeddings are tied)
 *
 * RMSNorm(x) = x / sqrt(mean(x^2) + eps) * g
 * RoPE (NeoX layout, as used by Qwen2 in llama.cpp): for i < d/2 the pair
 *   (x[i], x[i + d/2]) is rotated by angle pos * theta^(-2i/d).
 * GQA: query head h reads KV head h / (n_heads / n_kv_heads).
 * Attention uses a single-pass "online softmax" so no score buffer is needed:
 *   m' = max(m, s), l' = l*e^(m-m') + e^(s-m'), acc' = acc*e^(m-m') + e^(s-m') v
 */
#include "model.h"
#include "kernels.h"
#include "platform.h"
#include "../include/tinyinfer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TI_METAL
#include "../metal/ti_metal.h"
#endif

#define TI_MAX_BATCH 64

static int own(ti_model *m, void *p) {
    if (m->n_owned == m->cap_owned) {
        int nc = m->cap_owned ? m->cap_owned * 2 : 64;
        void **n = (void **)realloc(m->owned, sizeof(void *) * (size_t)nc);
        if (!n) return -1;
        m->owned = n;
        m->cap_owned = nc;
    }
    m->owned[m->n_owned++] = p;
    return 0;
}

static int supported_matmul_type(int t) {
    return t == TI_TYPE_F32 || t == TI_TYPE_F16 || t == TI_TYPE_BF16 || t == TI_TYPE_Q8_0 ||
           t == TI_TYPE_Q4_0;
}

static int get_tensor(ti_model *m, const char *name, ti_tensor *out, int64_t ne0, int64_t ne1,
                      int required) {
    const gguf_tensor *t = gguf_find_tensor(&m->gguf, name);
    memset(out, 0, sizeof *out);
    if (!t) {
        if (required) ti_set_error("missing tensor '%s'", name);
        return required ? -1 : 1;
    }
    if (!supported_matmul_type((int)t->type)) {
        ti_set_error("tensor '%s' has unsupported ggml type %u (supported: F32, F16, BF16, "
                     "Q8_0, Q4_0; re-quantise with tinyinfer-quantize)", name, t->type);
        return -1;
    }
    if ((int64_t)t->ne[0] != ne0 || (ne1 > 0 && (int64_t)t->ne[1] != ne1)) {
        ti_set_error("tensor '%s' has shape [%llu, %llu], expected [%lld, %lld]", name,
                     (unsigned long long)t->ne[0], (unsigned long long)t->ne[1], (long long)ne0,
                     (long long)ne1);
        return -1;
    }
    out->type = (int)t->type;
    out->data = t->data;
    out->ne0 = (int64_t)t->ne[0];
    out->ne1 = (int64_t)t->ne[1];
    out->nbytes = t->nbytes;
    return 0;
}

/* 1-D f32 vector (norm weights, biases); converts other types to an owned copy. */
static const float *get_vec(ti_model *m, const char *name, int64_t n, int required, int *err) {
    ti_tensor t;
    int r = get_tensor(m, name, &t, n, 0, required);
    if (r != 0) {
        if (r < 0) *err = 1;
        return NULL;
    }
    if (t.type == TI_TYPE_F32) return (const float *)t.data;
    float *c = (float *)malloc(sizeof(float) * (size_t)n);
    if (!c || own(m, c) != 0) {
        free(c);
        *err = 1;
        return NULL;
    }
    ti_dequantize_row(t.type, t.data, c, n);
    return c;
}

TI_API const ti_config *ti_model_config(const ti_model *m) { return &m->cfg; }

TI_API void ti_model_free(ti_model *m) {
    if (!m) return;
#ifdef TI_METAL
    if (m->metal) ti_metal_free(m->metal);
#endif
    ti_tokenizer_free(&m->tok);
    if (m->paged) {
        for (int i = 0; i < TI_MAX_SEQS; i++) ti_kv_seq_release(&m->pool, &m->seqs[i]);
        ti_kv_pool_destroy(&m->pool);
    }
    for (int i = 0; i < m->n_owned; i++) free(m->owned[i]);
    free(m->owned);
    free(m->layers);
    free(m->k_cache);
    free(m->v_cache);
    free(m->rope_inv_freq);
    free(m->x); free(m->xb); free(m->q); free(m->k); free(m->v);
    free(m->att); free(m->hb); free(m->hb2); free(m->tmp);
    gguf_close(&m->gguf);
    free(m);
}

TI_API ti_model *ti_model_load(const char *path, int ctx_len, int n_threads) {
    ti_model *m = (ti_model *)calloc(1, sizeof(ti_model));
    if (!m) return NULL;
    if (n_threads > 0) ti_set_threads(n_threads);
    if (gguf_open(&m->gguf, path) != 0) {
        free(m);
        return NULL;
    }
    gguf_file *g = &m->gguf;
    char arch[64] = "";
    gguf_get_str(g, "general.architecture", arch, sizeof arch);
    if (strcmp(arch, "qwen2") != 0) {
        ti_set_error("unsupported architecture '%s' (tinyinfer implements qwen2)", arch);
        goto fail;
    }
    ti_config *c = &m->cfg;
    c->n_layers = (int)gguf_get_int(g, "qwen2.block_count", 0);
    c->dim = (int)gguf_get_int(g, "qwen2.embedding_length", 0);
    c->ffn_dim = (int)gguf_get_int(g, "qwen2.feed_forward_length", 0);
    c->n_heads = (int)gguf_get_int(g, "qwen2.attention.head_count", 0);
    c->n_kv_heads = (int)gguf_get_int(g, "qwen2.attention.head_count_kv", c->n_heads);
    c->rope_theta = (float)gguf_get_float(g, "qwen2.rope.freq_base", 10000.0);
    c->rms_eps = (float)gguf_get_float(g, "qwen2.attention.layer_norm_rms_epsilon", 1e-6);
    c->train_ctx = (int)gguf_get_int(g, "qwen2.context_length", 2048);
    if (c->n_layers <= 0 || c->dim <= 0 || c->ffn_dim <= 0 || c->n_heads <= 0 ||
        c->n_kv_heads <= 0 || c->n_heads % c->n_kv_heads != 0) {
        ti_set_error("invalid qwen2 hyper-parameters in GGUF");
        goto fail;
    }
    c->head_dim = (int)gguf_get_int(g, "qwen2.attention.key_length", c->dim / c->n_heads);
    if (c->head_dim % 2 || c->head_dim > 1024) {
        ti_set_error("head_dim must be even and <= 1024");
        goto fail;
    }
    c->max_ctx = ctx_len > 0 ? ctx_len : (c->train_ctx < 2048 ? c->train_ctx : 2048);

    const gguf_tensor *te = gguf_find_tensor(g, "token_embd.weight");
    if (!te) {
        ti_set_error("missing tensor 'token_embd.weight'");
        goto fail;
    }
    c->vocab_size = (int)te->ne[1];
    const int D = c->dim, QD = c->n_heads * c->head_dim, KVD = c->n_kv_heads * c->head_dim;
    const int F = c->ffn_dim, V = c->vocab_size;
    if (get_tensor(m, "token_embd.weight", &m->tok_embd, D, V, 1)) goto fail;
    int err = 0;
    m->out_norm = get_vec(m, "output_norm.weight", D, 1, &err);
    int r = get_tensor(m, "output.weight", &m->output, D, V, 0);
    if (r < 0) goto fail;
    c->tied_embeddings = (r == 1);
    if (c->tied_embeddings) m->output = m->tok_embd;

    m->layers = (ti_layer *)calloc((size_t)c->n_layers, sizeof(ti_layer));
    if (!m->layers) goto oom;
    c->has_qkv_bias = 1;
    for (int l = 0; l < c->n_layers && !err; l++) {
        ti_layer *L = &m->layers[l];
        char nm[96];
#define T2(field, suffix, ne0, ne1)                                        \
    snprintf(nm, sizeof nm, "blk.%d.%s", l, suffix);                       \
    if (get_tensor(m, nm, &L->field, ne0, ne1, 1)) goto fail;
#define V1(field, suffix, n, req)                                          \
    snprintf(nm, sizeof nm, "blk.%d.%s", l, suffix);                       \
    L->field = get_vec(m, nm, n, req, &err);
        V1(attn_norm, "attn_norm.weight", D, 1);
        V1(ffn_norm, "ffn_norm.weight", D, 1);
        V1(bq, "attn_q.bias", QD, 0);
        V1(bk, "attn_k.bias", KVD, 0);
        V1(bv, "attn_v.bias", KVD, 0);
        if (!L->bq) c->has_qkv_bias = 0;
        T2(wq, "attn_q.weight", D, QD);
        T2(wk, "attn_k.weight", D, KVD);
        T2(wv, "attn_v.weight", D, KVD);
        T2(wo, "attn_output.weight", QD, D);
        T2(w_gate, "ffn_gate.weight", D, F);
        T2(w_up, "ffn_up.weight", D, F);
        T2(w_down, "ffn_down.weight", F, D);
#undef T2
#undef V1
    }
    if (err) goto fail;

    if (ti_tokenizer_load(&m->tok, g) != 0) goto fail;

    /* KV cache + scratch */
    size_t kv = (size_t)c->n_layers * (size_t)c->max_ctx * (size_t)KVD;
    m->k_cache = (float *)calloc(kv, sizeof(float));
    m->v_cache = (float *)calloc(kv, sizeof(float));
    m->rope_inv_freq = (float *)malloc(sizeof(float) * (size_t)(c->head_dim / 2));
    m->max_batch = TI_MAX_BATCH;
    const size_t B = (size_t)m->max_batch;
    m->x = (float *)malloc(sizeof(float) * B * D);
    m->xb = (float *)malloc(sizeof(float) * B * D);
    m->tmp = (float *)malloc(sizeof(float) * B * D);
    m->q = (float *)malloc(sizeof(float) * B * QD);
    m->att = (float *)malloc(sizeof(float) * B * QD);
    m->k = (float *)malloc(sizeof(float) * B * KVD);
    m->v = (float *)malloc(sizeof(float) * B * KVD);
    m->hb = (float *)malloc(sizeof(float) * B * F);
    m->hb2 = (float *)malloc(sizeof(float) * B * F);
    if (!m->k_cache || !m->v_cache || !m->rope_inv_freq || !m->x || !m->xb || !m->tmp ||
        !m->q || !m->att || !m->k || !m->v || !m->hb || !m->hb2)
        goto oom;
    for (int i = 0; i < c->head_dim / 2; i++)
        m->rope_inv_freq[i] =
            (float)pow((double)c->rope_theta, -2.0 * (double)i / (double)c->head_dim);
    return m;
oom:
    ti_set_error("out of memory (KV cache for ctx=%d?)", m->cfg.max_ctx);
fail:
    ti_model_free(m);
    return NULL;
}

/* ------------------------------------------------------------ helpers ---- */
static void rmsnorm(float *o, const float *x, const float *w, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float s = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * s * w[i];
}

static void rope_neox(float *v, int n_heads, int hd, const float *cs, const float *sn) {
    const int half = hd / 2;
    for (int h = 0; h < n_heads; h++) {
        float *p = v + h * hd;
        for (int i = 0; i < half; i++) {
            float a = p[i], b = p[i + half];
            p[i] = a * cs[i] - b * sn[i];
            p[i + half] = a * sn[i] + b * cs[i];
        }
    }
}

typedef struct {
    const ti_model *m;
    int layer, n, pos;
    const ti_kv_seq *seq; /* paged mode only */
} attn_job;

/* online-softmax update of (mx, l, out) with one key/value row */
static inline void attn_accum(const float *q, const float *k, const float *v, int hd, float scale,
                              float *mx, float *l, float *out) {
    float sc = ti_dot_f32(q, k, hd) * scale;
    if (sc > *mx) {
        float r = expf(*mx - sc); /* exp(-inf) = 0 on the first step */
        *l *= r;
        for (int i = 0; i < hd; i++) out[i] *= r;
        *mx = sc;
    }
    float w = expf(sc - *mx);
    *l += w;
    for (int i = 0; i < hd; i++) out[i] += w * v[i];
}

/* one task = one (token, head) pair */
static void attn_task(void *vctx, int s, int e) {
    const attn_job *j = (const attn_job *)vctx;
    const ti_config *c = &j->m->cfg;
    const int hd = c->head_dim, KVD = c->n_kv_heads * hd, QD = c->n_heads * hd;
    const int group = c->n_heads / c->n_kv_heads;
    const float scale = 1.0f / sqrtf((float)hd);
    const float *kc = j->seq ? NULL : j->m->k_cache + (size_t)j->layer * c->max_ctx * KVD;
    const float *vc = j->seq ? NULL : j->m->v_cache + (size_t)j->layer * c->max_ctx * KVD;
    for (int task = s; task < e; task++) {
        int t = task / c->n_heads, h = task % c->n_heads;
        int kvh = h / group;
        const float *q = j->m->q + (size_t)t * QD + h * hd;
        float *out = j->m->att + (size_t)t * QD + h * hd;
        int last = j->pos + t; /* causal: attend to positions 0..last */
        float mx = -INFINITY, l = 0.0f;
        for (int i = 0; i < hd; i++) out[i] = 0.0f;
        if (!j->seq) {
            for (int p = 0; p <= last; p++)
                attn_accum(q, kc + (size_t)p * KVD + kvh * hd, vc + (size_t)p * KVD + kvh * hd, hd,
                           scale, &mx, &l, out);
        } else { /* paged: walk the block table, same visiting order as above */
            const ti_kv_pool *P = &j->m->pool;
            const int bs = P->block_size;
            for (int b = 0; b * bs <= last; b++) {
                const size_t base = ti_kv_offset(P, j->seq->table[b], j->layer, 0) + (size_t)kvh * hd;
                const int n_slot = (last - b * bs + 1) < bs ? (last - b * bs + 1) : bs;
                for (int sl = 0; sl < n_slot; sl++)
                    attn_accum(q, P->k + base + (size_t)sl * KVD, P->v + base + (size_t)sl * KVD, hd,
                               scale, &mx, &l, out);
            }
        }
        float inv = 1.0f / l;
        for (int i = 0; i < hd; i++) out[i] *= inv;
    }
}

static void mm(const ti_tensor *w, const float *x, float *y, int n_tok) {
    ti_matmul(w->type, w->data, x, y, (int)w->ne1, (int)w->ne0, n_tok);
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

/* forward for n <= max_batch tokens; writes logits for all n tokens (all=1) or
 * only for the last one */
static int forward_chunk(ti_model *m, ti_kv_seq *seq, const int32_t *tokens, int n, int pos,
                         float *logits, int all) {
    const ti_config *c = &m->cfg;
    const int D = c->dim, hd = c->head_dim, QD = c->n_heads * hd, KVD = c->n_kv_heads * hd;
    const int F = c->ffn_dim;
    for (int t = 0; t < n; t++) {
        if (tokens[t] < 0 || tokens[t] >= c->vocab_size) {
            ti_set_error("token id %d out of range", tokens[t]);
            return -1;
        }
        size_t rb = ti_type_row_size(m->tok_embd.type, D);
        ti_dequantize_row(m->tok_embd.type, (const char *)m->tok_embd.data + rb * tokens[t],
                          m->x + (size_t)t * D, D);
    }
    float cs[512], sn[512]; /* head_dim/2 <= 512 */
    for (int l = 0; l < c->n_layers; l++) {
        const ti_layer *L = &m->layers[l];
        for (int t = 0; t < n; t++)
            rmsnorm(m->xb + (size_t)t * D, m->x + (size_t)t * D, L->attn_norm, D, c->rms_eps);
        mm(&L->wq, m->xb, m->q, n);
        mm(&L->wk, m->xb, m->k, n);
        mm(&L->wv, m->xb, m->v, n);
        float *kc = seq ? NULL : m->k_cache + (size_t)l * c->max_ctx * KVD;
        float *vc = seq ? NULL : m->v_cache + (size_t)l * c->max_ctx * KVD;
        for (int t = 0; t < n; t++) {
            float *q = m->q + (size_t)t * QD, *k = m->k + (size_t)t * KVD, *v = m->v + (size_t)t * KVD;
            if (L->bq) for (int i = 0; i < QD; i++) q[i] += L->bq[i];
            if (L->bk) for (int i = 0; i < KVD; i++) k[i] += L->bk[i];
            if (L->bv) for (int i = 0; i < KVD; i++) v[i] += L->bv[i];
            const int p = pos + t;
            for (int i = 0; i < hd / 2; i++) {
                double a = (double)p * (double)m->rope_inv_freq[i];
                cs[i] = (float)cos(a);
                sn[i] = (float)sin(a);
            }
            rope_neox(q, c->n_heads, hd, cs, sn);
            rope_neox(k, c->n_kv_heads, hd, cs, sn);
            float *kd, *vd;
            if (seq) {
                size_t off = ti_kv_offset(&m->pool, seq->table[p / m->pool.block_size], l,
                                          p % m->pool.block_size);
                kd = m->pool.k + off;
                vd = m->pool.v + off;
            } else {
                kd = kc + (size_t)p * KVD;
                vd = vc + (size_t)p * KVD;
            }
            memcpy(kd, k, sizeof(float) * KVD);
            memcpy(vd, v, sizeof(float) * KVD);
        }
        attn_job aj = {m, l, n, pos, seq};
        ti_parallel_for(n * c->n_heads, 1, attn_task, &aj);
        mm(&L->wo, m->att, m->tmp, n);
        for (size_t i = 0; i < (size_t)n * D; i++) m->x[i] += m->tmp[i];

        for (int t = 0; t < n; t++)
            rmsnorm(m->xb + (size_t)t * D, m->x + (size_t)t * D, L->ffn_norm, D, c->rms_eps);
        mm(&L->w_gate, m->xb, m->hb, n);
        mm(&L->w_up, m->xb, m->hb2, n);
        for (size_t i = 0; i < (size_t)n * F; i++) m->hb[i] = silu(m->hb[i]) * m->hb2[i];
        mm(&L->w_down, m->hb, m->tmp, n);
        for (size_t i = 0; i < (size_t)n * D; i++) m->x[i] += m->tmp[i];
    }
    if (!logits) return 0;
    int t0 = all ? 0 : n - 1;
    for (int t = t0; t < n; t++)
        rmsnorm(m->xb + (size_t)t * D, m->x + (size_t)t * D, m->out_norm, D, c->rms_eps);
    mm(&m->output, m->xb + (size_t)t0 * D, logits, n - t0);
    return 0;
}

TI_API int ti_forward_seq(ti_model *m, int seq_id, const int32_t *tokens, int n, int pos,
                          float *logits, int all_logits) {
    if (n <= 0) return 0;
    if (pos < 0 || pos + n > m->cfg.max_ctx) {
        ti_set_error("context overflow: pos %d + %d tokens > ctx %d", pos, n, m->cfg.max_ctx);
        return -1;
    }
    ti_kv_seq *seq = NULL;
    if (m->paged) {
        if (seq_id < 0 || seq_id >= TI_MAX_SEQS) {
            ti_set_error("sequence id %d out of range [0, %d)", seq_id, TI_MAX_SEQS);
            return -1;
        }
        seq = &m->seqs[seq_id];
        if (pos > seq->len) {
            ti_set_error("sequence %d has %d cached positions; cannot write at %d", seq_id,
                         seq->len, pos);
            return -1;
        }
        /* writing at pos discards everything cached from pos on (rollback) */
        ti_kv_seq_truncate(&m->pool, seq, pos);
        if (ti_kv_seq_prepare(&m->pool, seq, pos, pos + n) != 0) return -1;
    } else if (seq_id != 0) {
        ti_set_error("sequence ids other than 0 need the paged KV cache (ti_kv_enable_paged)");
        return -1;
    }
#ifdef TI_METAL
    if (m->metal) {
        const int V = m->cfg.vocab_size;
        for (int i = 0; i < n; i++) {
            float *out = all_logits ? logits + (size_t)i * V : (i == n - 1 ? logits : NULL);
            if (ti_metal_forward(m->metal, m, tokens[i], pos + i, out) != 0) return -1;
        }
        return 0;
    }
#endif
    const int V = m->cfg.vocab_size;
    for (int s = 0; s < n; s += m->max_batch) {
        int cnt = n - s < m->max_batch ? n - s : m->max_batch;
        int last_chunk = (s + cnt == n);
        float *out = NULL;
        if (all_logits) out = logits + (size_t)s * V;
        else if (last_chunk) out = logits;
        if (forward_chunk(m, seq, tokens + s, cnt, pos + s, out, all_logits) != 0) return -1;
        if (seq) seq->len = pos + s + cnt;
    }
    return 0;
}

TI_API int ti_forward(ti_model *m, const int32_t *tokens, int n, int pos, float *logits,
                      int all_logits) {
    return ti_forward_seq(m, 0, tokens, n, pos, logits, all_logits);
}

/* ------------------------------------------------------- paged KV cache --- */
TI_API int ti_kv_enable_paged(ti_model *m, int block_size, int n_blocks) {
    if (m->metal) {
        ti_set_error("the paged KV cache is implemented for the CPU backend only");
        return -1;
    }
    if (m->paged) {
        ti_set_error("paged KV cache already enabled");
        return -1;
    }
    const ti_config *c = &m->cfg;
    if (block_size <= 0) block_size = 16;
    if (n_blocks <= 0) n_blocks = (c->max_ctx + block_size - 1) / block_size;
    if (ti_kv_pool_init(&m->pool, n_blocks, block_size, c->n_layers,
                        c->n_kv_heads * c->head_dim) != 0)
        return -1;
    free(m->k_cache);
    free(m->v_cache);
    m->k_cache = m->v_cache = NULL;
    memset(m->seqs, 0, sizeof m->seqs);
    m->paged = 1;
    return 0;
}

TI_API int ti_kv_is_paged(const ti_model *m) { return m->paged; }

TI_API int ti_kv_seq_trim(ti_model *m, int seq_id, int n_pos) {
    if (!m->paged) return 0; /* contiguous cache: stale positions are simply overwritten */
    if (seq_id < 0 || seq_id >= TI_MAX_SEQS) return -1;
    ti_kv_seq_truncate(&m->pool, &m->seqs[seq_id], n_pos);
    return 0;
}

TI_API int ti_kv_seq_copy(ti_model *m, int dst, int src) {
    if (!m->paged || dst < 0 || src < 0 || dst >= TI_MAX_SEQS || src >= TI_MAX_SEQS) {
        ti_set_error("ti_kv_seq_copy needs the paged KV cache and valid sequence ids");
        return -1;
    }
    return ti_kv_seq_fork(&m->pool, &m->seqs[dst], &m->seqs[src]);
}

TI_API int ti_kv_seq_len(const ti_model *m, int seq_id) {
    if (!m->paged || seq_id < 0 || seq_id >= TI_MAX_SEQS) return -1;
    return m->seqs[seq_id].len;
}

TI_API int ti_kv_blocks_free(const ti_model *m) { return m->paged ? m->pool.n_free : -1; }
TI_API int ti_kv_blocks_total(const ti_model *m) { return m->paged ? m->pool.n_blocks : -1; }

/* ----------------------------------------------------------- tokenizer --- */
TI_API int ti_tokenize(const ti_model *m, const char *text, int32_t *out, int max_tokens,
                       int parse_special) {
    return ti_tok_encode(&m->tok, text, out, max_tokens, parse_special);
}

TI_API int ti_detokenize(const ti_model *m, const int32_t *tokens, int n, char *out,
                         int max_bytes) {
    return ti_tok_decode(&m->tok, tokens, n, out, max_bytes);
}

TI_API int ti_token_id(const ti_model *m, const char *piece) {
    return ti_tok_lookup(&m->tok, piece, (int)strlen(piece));
}

TI_API int ti_eos_id(const ti_model *m) { return m->tok.eos_id; }
TI_API int ti_bos_id(const ti_model *m) { return m->tok.bos_id; }
