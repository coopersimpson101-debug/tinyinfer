/*
 * Paged KV cache (vLLM-style "PagedAttention" memory layout, Kwon et al. 2023).
 *
 * Keys and values live in a pool of fixed-size blocks; a block holds
 * `block_size` consecutive positions for every layer:
 *
 *     pool.k[block][layer][slot][kv_dim]      (same for pool.v)
 *
 * Each sequence owns a block table: logical block i (positions
 * [i*bs, (i+1)*bs)) -> physical block id. Position p of a sequence is stored
 * at block table[p / bs], slot p % bs. Blocks are reference counted, so a
 * sequence can be forked cheaply (prefix sharing); writing into a shared block
 * first copies it (copy-on-write).
 */
#ifndef TI_KVCACHE_H
#define TI_KVCACHE_H

#include <stddef.h>

typedef struct ti_kv_pool {
    int block_size, n_blocks, n_layers, kv_dim;
    float *k, *v;
    int *free_list; /* stack of free block ids */
    int n_free;
    int *refcount;
} ti_kv_pool;

typedef struct ti_kv_seq {
    int *table; /* logical -> physical block id */
    int n_blocks, cap;
    int len; /* number of valid positions */
} ti_kv_seq;

int ti_kv_pool_init(ti_kv_pool *p, int n_blocks, int block_size, int n_layers, int kv_dim);
void ti_kv_pool_destroy(ti_kv_pool *p);
int ti_kv_block_alloc(ti_kv_pool *p);            /* -1 when the pool is exhausted */
void ti_kv_block_release(ti_kv_pool *p, int b);  /* drops one reference */

/* Make positions [0, n_pos) writable for `s`: allocates missing blocks and
 * un-shares (copy-on-write) shared blocks that overlap [first_write, n_pos).
 * Returns 0, or -1 if the pool ran out of blocks (s is left unchanged in size). */
int ti_kv_seq_prepare(ti_kv_pool *p, ti_kv_seq *s, int first_write, int n_pos);
/* Keep only positions [0, n_pos); blocks past the end go back to the pool. */
void ti_kv_seq_truncate(ti_kv_pool *p, ti_kv_seq *s, int n_pos);
void ti_kv_seq_release(ti_kv_pool *p, ti_kv_seq *s);
/* dst becomes a copy of src that shares all of src's blocks. */
int ti_kv_seq_fork(ti_kv_pool *p, ti_kv_seq *dst, const ti_kv_seq *src);

static inline size_t ti_kv_offset(const ti_kv_pool *p, int block, int layer, int slot) {
    return (((size_t)block * (size_t)p->n_layers + (size_t)layer) * (size_t)p->block_size +
            (size_t)slot) * (size_t)p->kv_dim;
}

#endif
