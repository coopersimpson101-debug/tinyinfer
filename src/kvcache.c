/* Paged KV cache: block pool, free-list allocator, per-sequence block tables. */
#include "kvcache.h"
#include "platform.h"

#include <stdlib.h>
#include <string.h>

int ti_kv_pool_init(ti_kv_pool *p, int n_blocks, int block_size, int n_layers, int kv_dim) {
    memset(p, 0, sizeof *p);
    if (n_blocks <= 0 || block_size <= 0 || n_layers <= 0 || kv_dim <= 0) {
        ti_set_error("invalid paged KV cache geometry");
        return -1;
    }
    p->block_size = block_size;
    p->n_blocks = n_blocks;
    p->n_layers = n_layers;
    p->kv_dim = kv_dim;
    size_t n = (size_t)n_blocks * n_layers * block_size * kv_dim;
    p->k = (float *)calloc(n, sizeof(float));
    p->v = (float *)calloc(n, sizeof(float));
    p->free_list = (int *)malloc(sizeof(int) * (size_t)n_blocks);
    p->refcount = (int *)calloc((size_t)n_blocks, sizeof(int));
    if (!p->k || !p->v || !p->free_list || !p->refcount) {
        ti_kv_pool_destroy(p);
        ti_set_error("out of memory allocating %d KV blocks", n_blocks);
        return -1;
    }
    /* stack ordered so that block 0 is handed out first */
    for (int i = 0; i < n_blocks; i++) p->free_list[i] = n_blocks - 1 - i;
    p->n_free = n_blocks;
    return 0;
}

void ti_kv_pool_destroy(ti_kv_pool *p) {
    free(p->k);
    free(p->v);
    free(p->free_list);
    free(p->refcount);
    memset(p, 0, sizeof *p);
}

int ti_kv_block_alloc(ti_kv_pool *p) {
    if (p->n_free == 0) return -1;
    int b = p->free_list[--p->n_free];
    p->refcount[b] = 1;
    return b;
}

void ti_kv_block_release(ti_kv_pool *p, int b) {
    if (b < 0 || b >= p->n_blocks || p->refcount[b] <= 0) return;
    if (--p->refcount[b] == 0) p->free_list[p->n_free++] = b;
}

static int seq_grow(ti_kv_seq *s, int need) {
    if (need <= s->cap) return 0;
    int nc = s->cap ? s->cap : 8;
    while (nc < need) nc *= 2;
    int *t = (int *)realloc(s->table, sizeof(int) * (size_t)nc);
    if (!t) return -1;
    s->table = t;
    s->cap = nc;
    return 0;
}

static void copy_block(ti_kv_pool *p, int dst, int src) {
    size_t n = (size_t)p->n_layers * p->block_size * p->kv_dim;
    memcpy(p->k + ti_kv_offset(p, dst, 0, 0), p->k + ti_kv_offset(p, src, 0, 0), n * sizeof(float));
    memcpy(p->v + ti_kv_offset(p, dst, 0, 0), p->v + ti_kv_offset(p, src, 0, 0), n * sizeof(float));
}

int ti_kv_seq_prepare(ti_kv_pool *p, ti_kv_seq *s, int first_write, int n_pos) {
    const int bs = p->block_size;
    int need = (n_pos + bs - 1) / bs;
    int first_blk = first_write / bs;
    /* count blocks we must obtain, so failure leaves the sequence untouched */
    int want = need > s->n_blocks ? need - s->n_blocks : 0;
    for (int i = first_blk; i < s->n_blocks && i < need; i++)
        if (p->refcount[s->table[i]] > 1) want++;
    if (want > p->n_free) {
        ti_set_error("paged KV cache exhausted: need %d more blocks, %d free (of %d)", want,
                     p->n_free, p->n_blocks);
        return -1;
    }
    if (seq_grow(s, need) != 0) {
        ti_set_error("out of memory growing a block table");
        return -1;
    }
    for (int i = first_blk; i < s->n_blocks && i < need; i++) { /* copy-on-write */
        int b = s->table[i];
        if (p->refcount[b] > 1) {
            int nb = ti_kv_block_alloc(p);
            copy_block(p, nb, b);
            ti_kv_block_release(p, b);
            s->table[i] = nb;
        }
    }
    while (s->n_blocks < need) s->table[s->n_blocks++] = ti_kv_block_alloc(p);
    return 0;
}

void ti_kv_seq_truncate(ti_kv_pool *p, ti_kv_seq *s, int n_pos) {
    if (n_pos < 0) n_pos = 0;
    int keep = (n_pos + p->block_size - 1) / p->block_size;
    while (s->n_blocks > keep) ti_kv_block_release(p, s->table[--s->n_blocks]);
    if (s->len > n_pos) s->len = n_pos;
}

void ti_kv_seq_release(ti_kv_pool *p, ti_kv_seq *s) {
    ti_kv_seq_truncate(p, s, 0);
    free(s->table);
    memset(s, 0, sizeof *s);
}

int ti_kv_seq_fork(ti_kv_pool *p, ti_kv_seq *dst, const ti_kv_seq *src) {
    if (dst == src) return 0;
    ti_kv_seq_truncate(p, dst, 0);
    if (seq_grow(dst, src->n_blocks) != 0) return -1;
    for (int i = 0; i < src->n_blocks; i++) {
        dst->table[i] = src->table[i];
        p->refcount[src->table[i]]++;
    }
    dst->n_blocks = src->n_blocks;
    dst->len = src->len;
    return 0;
}
