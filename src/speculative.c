/*
 * Speculative decoding (Leviathan, Kalman & Matias 2023; Chen et al. 2023).
 *
 * State between rounds: n committed tokens x_0..x_{n-1}; both KV caches hold
 * positions 0..n-2 and x_{n-1} is "pending" (not yet fed to either model).
 *
 * One round with lookahead k:
 *   draft : feed x_{n-1}, d_1, .., d_{k-1} one at a time; d_i ~ q_i
 *   target: one batched forward over [x_{n-1}, d_1..d_k] -> p_1..p_{k+1}
 *   accept: for i = 1..k, keep d_i with probability min(1, p_i(d_i)/q_i(d_i));
 *           at the first rejection emit t ~ norm(max(0, p_i - q_i)) and stop;
 *           if all k were kept emit a bonus t ~ p_{k+1}.
 *   commit: d_1..d_a, t  (a + 1 tokens for one target call)
 *
 * Why the output is distributed exactly as p: for a drafted token x,
 *   P(emit x) = q(x) min(1, p(x)/q(x)) + (1 - sum_y min(q(y), p(y))) * r(x)
 *             = min(p(x), q(x)) + max(0, p(x) - q(x)) = p(x),
 * because the rejection mass 1 - sum_y min(p, q) equals sum_y max(0, p - q),
 * the normaliser of the residual r. Positions after a rejection are re-drawn
 * next round from the target-conditioned state, so the whole sequence has the
 * target's law. With greedy decoding p and q are one-hot, so the rule keeps
 * d_i iff it equals the target argmax: output is token-for-token identical to
 * greedy decoding with the target alone.
 *
 * Rolling back: after a rejection the caches contain entries for positions
 * past the committed prefix. The contiguous cache just overwrites them next
 * round; the paged cache frees their blocks (ti_forward_seq truncates).
 */
#include "../include/tinyinfer.h"
#include "model.h"
#include "platform.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint64_t xs64(uint64_t *s) {
    uint64_t x = *s ? *s : 0x9E3779B97F4A7C15ull;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1Dull;
}

static double unit(uint64_t *s) { return (double)(xs64(s) >> 11) * (1.0 / 9007199254740992.0); }

typedef struct {
    float v;
    int id;
} cand;

static int cmp_desc(const void *a, const void *b) {
    float x = ((const cand *)a)->v, y = ((const cand *)b)->v;
    if (x != y) return (x < y) - (x > y);
    return ((const cand *)a)->id - ((const cand *)b)->id;
}

/* a ranks below b in cmp_desc order (lower value, or equal value and larger id) */
static int worse(const cand *a, const cand *b) {
    return a->v < b->v || (a->v == b->v && a->id > b->id);
}

static void heap_down(cand *h, int n, int i) { /* min-heap on rank: h[0] is the worst kept */
    for (;;) {
        int l = 2 * i + 1, r = l + 1, s = i;
        if (l < n && worse(&h[l], &h[s])) s = l;
        if (r < n && worse(&h[r], &h[s])) s = r;
        if (s == i) return;
        cand t = h[i];
        h[i] = h[s];
        h[s] = t;
        i = s;
    }
}

static int argmax(const float *x, int n) {
    int b = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[b]) b = i;
    return b;
}

TI_API void ti_sampler_probs(const float *logits, int n, const ti_sampler_params *p, float *probs) {
    memset(probs, 0, sizeof(float) * (size_t)n);
    if (p->temperature <= 0.0f) {
        probs[argmax(logits, n)] = 1.0f;
        return;
    }
    cand *c = (cand *)malloc(sizeof(cand) * (size_t)n);
    if (!c) {
        probs[argmax(logits, n)] = 1.0f;
        return;
    }
    for (int i = 0; i < n; i++) {
        c[i].v = logits[i] / p->temperature;
        c[i].id = i;
    }
    int nc = n;
    const int need_sort = (p->top_k > 0 && p->top_k < n) || (p->top_p > 0.0f && p->top_p < 1.0f);
    if (p->top_k > 0 && p->top_k < n) {
        /* top-k by a size-k min-heap (O(V log k)), then sort only the k survivors */
        const int k = p->top_k;
        for (int i = k / 2 - 1; i >= 0; i--) heap_down(c, k, i);
        for (int i = k; i < n; i++)
            if (worse(&c[0], &c[i])) {
                c[0] = c[i];
                heap_down(c, k, 0);
            }
        nc = k;
        qsort(c, (size_t)k, sizeof(cand), cmp_desc);
    } else if (need_sort) {
        qsort(c, (size_t)n, sizeof(cand), cmp_desc);
    }
    float mx = c[0].v;
    if (!need_sort)
        for (int i = 1; i < nc; i++)
            if (c[i].v > mx) mx = c[i].v;
    double z = 0.0;
    for (int i = 0; i < nc; i++) z += exp((double)(c[i].v - mx));
    if (p->top_p > 0.0f && p->top_p < 1.0f) {
        double cum = 0.0;
        int keep = nc;
        for (int i = 0; i < nc; i++) {
            cum += exp((double)(c[i].v - mx)) / z;
            if (cum >= p->top_p) {
                keep = i + 1;
                break;
            }
        }
        nc = keep;
        z = 0.0;
        for (int i = 0; i < nc; i++) z += exp((double)(c[i].v - mx));
    }
    for (int i = 0; i < nc; i++) probs[c[i].id] = (float)(exp((double)(c[i].v - mx)) / z);
    free(c);
}

TI_API int ti_sample_from(const float *probs, int n, uint64_t *rng) {
    double mass = 0.0;
    for (int i = 0; i < n; i++) mass += probs[i];
    double r = unit(rng) * mass, acc = 0.0;
    int last = -1;
    for (int i = 0; i < n; i++) {
        if (probs[i] <= 0.0f) continue;
        acc += probs[i];
        last = i;
        if (r < acc) return i;
    }
    return last >= 0 ? last : 0;
}

TI_API int ti_spec_accept(const float *p, const float *q, int n, int x, float *scratch,
                          uint64_t *rng, int *accepted) {
    /* accept with probability min(1, p(x)/q(x)); written as u * q < p to avoid 0/0 */
    double u = unit(rng);
    if (q[x] > 0.0f && u * (double)q[x] < (double)p[x]) {
        *accepted = 1;
        return x;
    }
    *accepted = 0;
    double z = 0.0;
    for (int i = 0; i < n; i++) {
        float d = p[i] - q[i];
        scratch[i] = d > 0.0f ? d : 0.0f;
        z += scratch[i];
    }
    if (z <= 0.0) return ti_sample_from(p, n, rng); /* p == q up to rounding */
    return ti_sample_from(scratch, n, rng);
}

static int is_eos(const ti_model *m, int t) {
    return t == m->tok.eos_id || (m->tok.eot_id >= 0 && t == m->tok.eot_id);
}

TI_API int ti_generate(ti_model *m, const int32_t *prompt, int n_prompt, int n_predict,
                       const ti_sampler_params *sp, int stop_at_eos, int32_t *out,
                       ti_spec_stats *st) {
    const int V = m->cfg.vocab_size;
    ti_spec_stats s0;
    memset(&s0, 0, sizeof s0);
    if (n_prompt <= 0 || n_prompt + n_predict > m->cfg.max_ctx) {
        ti_set_error("prompt (%d) + n_predict (%d) must fit in ctx %d", n_prompt, n_predict,
                     m->cfg.max_ctx);
        return -1;
    }
    float *logits = (float *)malloc(sizeof(float) * (size_t)V);
    float *probs = (float *)malloc(sizeof(float) * (size_t)V);
    if (!logits || !probs) {
        free(logits);
        free(probs);
        return -1;
    }
    uint64_t rng = sp->seed ? sp->seed : 1;
    double t0 = ti_time_sec();
    int rc = ti_forward(m, prompt, n_prompt, 0, logits, 0);
    s0.t_prefill = ti_time_sec() - t0;
    int n = 0, pos = n_prompt;
    while (rc == 0 && n < n_predict) {
        ti_sampler_probs(logits, V, sp, probs);
        int t = ti_sample_from(probs, V, &rng);
        out[n++] = t;
        if ((stop_at_eos && is_eos(m, t)) || n == n_predict) break;
        rc = ti_forward(m, &out[n - 1], 1, pos++, logits, 0);
        s0.rounds++;
    }
    s0.generated = n;
    s0.t_total = s0.t_target = ti_time_sec() - t0;
    if (st) *st = s0;
    free(logits);
    free(probs);
    return rc == 0 ? n : -1;
}

TI_API int ti_spec_generate(ti_model *tg, ti_model *dr, const int32_t *prompt, int n_prompt,
                            int n_predict, int k, const ti_sampler_params *sp, int stop_at_eos,
                            int32_t *out, ti_spec_stats *st) {
    const int V = tg->cfg.vocab_size;
    if (dr->cfg.vocab_size != V) {
        ti_set_error("draft vocab (%d) != target vocab (%d)", dr->cfg.vocab_size, V);
        return -1;
    }
    if (k < 1) k = 1;
    if (k > 32) k = 32;
    int ctx = tg->cfg.max_ctx < dr->cfg.max_ctx ? tg->cfg.max_ctx : dr->cfg.max_ctx;
    if (n_prompt <= 0 || n_prompt + n_predict > ctx) {
        ti_set_error("prompt (%d) + n_predict (%d) must fit in ctx %d", n_prompt, n_predict, ctx);
        return -1;
    }
    ti_spec_stats s0;
    memset(&s0, 0, sizeof s0);
    const size_t Vz = (size_t)V;
    float *tl = (float *)malloc(sizeof(float) * Vz * (size_t)(k + 1)); /* target logits */
    float *dl = (float *)malloc(sizeof(float) * Vz);                   /* draft logits */
    float *q = (float *)malloc(sizeof(float) * Vz * (size_t)k);        /* draft probs */
    float *pp = (float *)malloc(sizeof(float) * Vz);
    float *scratch = (float *)malloc(sizeof(float) * Vz);
    int32_t *buf = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n_prompt + n_predict + k + 2));
    if (!tl || !dl || !q || !pp || !scratch || !buf) {
        ti_set_error("out of memory in speculative decoding");
        free(tl); free(dl); free(q); free(pp); free(scratch); free(buf);
        return -1;
    }
    uint64_t rng = sp->seed ? sp->seed : 1;
    double T0 = ti_time_sec(), t;
    int rc = 0;
    memcpy(buf, prompt, sizeof(int32_t) * (size_t)n_prompt);
    int n = n_prompt; /* committed tokens */
    if (n_prompt > 1) {
        t = ti_time_sec();
        rc = ti_forward(tg, buf, n_prompt - 1, 0, tl, 0);
        s0.t_target += ti_time_sec() - t;
        t = ti_time_sec();
        if (!rc) rc = ti_forward(dr, buf, n_prompt - 1, 0, dl, 0);
        s0.t_draft += ti_time_sec() - t;
    }
    s0.t_prefill = ti_time_sec() - T0;
    int done = 0;
    while (!rc && !done && n - n_prompt < n_predict) {
        int remaining = n_predict - (n - n_prompt);
        int kk = k < remaining - 1 ? k : remaining - 1; /* the round emits <= kk + 1 tokens */
        if (kk < 0) kk = 0;
        /* 1. draft proposes kk tokens */
        t = ti_time_sec();
        for (int i = 0; i < kk && !rc; i++) {
            rc = ti_forward(dr, &buf[n - 1 + i], 1, n - 1 + i, dl, 0);
            if (rc) break;
            ti_sampler_probs(dl, V, sp, q + (size_t)i * Vz);
            buf[n + i] = ti_sample_from(q + (size_t)i * Vz, V, &rng);
        }
        s0.t_draft += ti_time_sec() - t;
        if (rc) break;
        /* 2. target scores [x_{n-1}, d_1..d_kk] in one call */
        t = ti_time_sec();
        rc = ti_forward(tg, &buf[n - 1], kk + 1, n - 1, tl, 1);
        s0.t_target += ti_time_sec() - t;
        if (rc) break;
        s0.rounds++;
        s0.drafted += kk;
        /* 3. accept / reject */
        int a = 0, emitted = -1;
        for (int i = 0; i < kk; i++) {
            ti_sampler_probs(tl + (size_t)i * Vz, V, sp, pp);
            int acc;
            int tok = ti_spec_accept(pp, q + (size_t)i * Vz, V, buf[n + i], scratch, &rng, &acc);
            if (!acc) {
                emitted = tok;
                break;
            }
            a++;
        }
        if (emitted < 0) { /* all accepted: bonus token from p_{kk+1} */
            ti_sampler_probs(tl + (size_t)kk * Vz, V, sp, pp);
            emitted = ti_sample_from(pp, V, &rng);
        }
        s0.accepted += a;
        buf[n + a] = emitted;
        /* 4. commit, stopping at end-of-sequence */
        for (int i = 0; i <= a; i++) {
            out[n - n_prompt] = buf[n];
            n++;
            if (stop_at_eos && is_eos(tg, buf[n - 1])) {
                done = 1;
                break;
            }
        }
        if (done || n - n_prompt >= n_predict) break;
        /* 5. keep the draft cache in step: if every draft token was accepted the draft
         *    has not yet seen d_kk (now at position n-2) */
        if (a == kk && kk > 0) {
            t = ti_time_sec();
            rc = ti_forward(dr, &buf[n - 2], 1, n - 2, dl, 0);
            s0.t_draft += ti_time_sec() - t;
        }
    }
    s0.generated = n - n_prompt;
    s0.t_total = ti_time_sec() - T0;
    if (st) *st = s0;
    free(tl); free(dl); free(q); free(pp); free(scratch); free(buf);
    return rc ? -1 : s0.generated;
}
