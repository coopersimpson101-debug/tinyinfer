/*
 * Sampling: repetition penalty -> greedy | (temperature -> top-k -> top-p -> draw).
 *
 * Repetition penalty (CTRL / llama.cpp convention): for every distinct token in
 * the last `repeat_last_n` tokens, logit /= penalty if logit > 0 else *= penalty.
 * top-k keeps the k largest logits (min-heap, O(V log k)); top-p keeps the
 * smallest prefix of the sorted distribution whose mass reaches p. Without
 * top-k, candidates with p_i < (1 - top_p) / (V - 1) are dropped before the
 * sort: they can never be inside the nucleus, which avoids sorting 150k items.
 */
#include "../include/tinyinfer.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float logit;
    int id;
} cand;

TI_API ti_sampler_params ti_sampler_default(void) {
    ti_sampler_params p;
    p.temperature = 0.7f;
    p.top_k = 40;
    p.top_p = 0.9f;
    p.repeat_penalty = 1.1f;
    p.repeat_last_n = 64;
    p.seed = 0x9E3779B97F4A7C15ull;
    return p;
}

static uint64_t xorshift64s(uint64_t *s) {
    uint64_t x = *s ? *s : 0x9E3779B97F4A7C15ull;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1Dull;
}

static float rand_unit(uint64_t *s) { return (float)(xorshift64s(s) >> 40) / 16777216.0f; }

static int cmp_desc(const void *a, const void *b) {
    float x = ((const cand *)a)->logit, y = ((const cand *)b)->logit;
    return (x < y) - (x > y);
}

static void heap_sift_down(cand *h, int n, int i) {
    for (;;) {
        int l = 2 * i + 1, r = l + 1, s = i;
        if (l < n && h[l].logit < h[s].logit) s = l;
        if (r < n && h[r].logit < h[s].logit) s = r;
        if (s == i) return;
        cand t = h[i];
        h[i] = h[s];
        h[s] = t;
        i = s;
    }
}

static int argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

TI_API int ti_sample(float *logits, int n, const ti_sampler_params *p, const int32_t *history,
                     int n_history, uint64_t *rng) {
    if (p->repeat_penalty != 1.0f && p->repeat_last_n > 0 && history && n_history > 0) {
        int start = n_history > p->repeat_last_n ? n_history - p->repeat_last_n : 0;
        unsigned char *seen = (unsigned char *)calloc((size_t)n, 1);
        for (int i = start; i < n_history; i++) {
            int id = history[i];
            if (id < 0 || id >= n || (seen && seen[id])) continue;
            if (seen) seen[id] = 1;
            logits[id] = logits[id] > 0 ? logits[id] / p->repeat_penalty
                                        : logits[id] * p->repeat_penalty;
        }
        free(seen);
    }
    if (p->temperature <= 0.0f) return argmax(logits, n);

    for (int i = 0; i < n; i++) logits[i] /= p->temperature;
    float mx = logits[argmax(logits, n)];

    cand *c = (cand *)malloc(sizeof(cand) * (size_t)n);
    if (!c) return argmax(logits, n);
    int nc = 0;
    if (p->top_k > 0 && p->top_k < n) {
        int k = p->top_k;
        for (int i = 0; i < n; i++) {
            if (nc < k) {
                c[nc].logit = logits[i];
                c[nc].id = i;
                nc++;
                if (nc == k)
                    for (int j = k / 2 - 1; j >= 0; j--) heap_sift_down(c, k, j);
            } else if (logits[i] > c[0].logit) {
                c[0].logit = logits[i];
                c[0].id = i;
                heap_sift_down(c, k, 0);
            }
        }
    } else if (p->top_p > 0.0f && p->top_p < 1.0f && n > 1) {
        double z = 0.0;
        for (int i = 0; i < n; i++) z += exp((double)(logits[i] - mx));
        double cutoff = (1.0 - p->top_p) / (double)(n - 1);
        for (int i = 0; i < n; i++) {
            if (exp((double)(logits[i] - mx)) / z >= cutoff) {
                c[nc].logit = logits[i];
                c[nc].id = i;
                nc++;
            }
        }
        if (nc == 0) {
            free(c);
            return argmax(logits, n);
        }
    } else {
        for (int i = 0; i < n; i++) {
            c[i].logit = logits[i];
            c[i].id = i;
        }
        nc = n;
    }
    qsort(c, (size_t)nc, sizeof(cand), cmp_desc);

    /* softmax over candidates (stored in .logit as probabilities) */
    const float top = c[0].logit; /* saved: the loop below overwrites c[0] */
    double z = 0.0;
    for (int i = 0; i < nc; i++) z += exp((double)(c[i].logit - top));
    for (int i = 0; i < nc; i++) c[i].logit = (float)(exp((double)(c[i].logit - top)) / z);

    if (p->top_p > 0.0f && p->top_p < 1.0f) {
        double cum = 0.0;
        int keep = nc;
        for (int i = 0; i < nc; i++) {
            cum += c[i].logit;
            if (cum >= p->top_p) {
                keep = i + 1;
                break;
            }
        }
        nc = keep;
    }
    double mass = 0.0;
    for (int i = 0; i < nc; i++) mass += c[i].logit;
    double r = (double)rand_unit(rng) * mass, acc = 0.0;
    int out = c[nc - 1].id;
    for (int i = 0; i < nc; i++) {
        acc += c[i].logit;
        if (r < acc) {
            out = c[i].id;
            break;
        }
    }
    free(c);
    return out;
}
