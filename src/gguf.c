/*
 * GGUF v3 parser.
 *
 * Layout (all little endian):
 *   u32 magic "GGUF" | u32 version | u64 n_tensors | u64 n_kv
 *   n_kv x { gguf_string key | u32 value_type | value }
 *   n_tensors x { gguf_string name | u32 n_dims | u64 ne[n_dims] | u32 ggml_type | u64 offset }
 *   padding to `general.alignment` (default 32)
 *   tensor data (each tensor at data_offset + offset)
 * gguf_string = u64 length + bytes (no NUL). Arrays = u32 elem_type, u64 n, elems.
 *
 * The file is memory mapped; tensors point straight into the mapping, so
 * loading a 1.5 GB model costs no copies and pages fault in lazily.
 */
#include "gguf.h"
#include "../include/tinyinfer.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    const uint8_t *p, *end;
    int err;
} reader;

static int need(reader *r, size_t n) {
    if (r->err || (size_t)(r->end - r->p) < n) {
        r->err = 1;
        return 0;
    }
    return 1;
}

static uint32_t rd_u32(reader *r) {
    uint32_t v = 0;
    if (need(r, 4)) {
        memcpy(&v, r->p, 4);
        r->p += 4;
    }
    return v;
}

static uint64_t rd_u64(reader *r) {
    uint64_t v = 0;
    if (need(r, 8)) {
        memcpy(&v, r->p, 8);
        r->p += 8;
    }
    return v;
}

static gguf_str rd_str(reader *r) {
    gguf_str s = {0, NULL};
    s.len = rd_u64(r);
    if (need(r, (size_t)s.len)) {
        s.s = (const char *)r->p;
        r->p += s.len;
    }
    return s;
}

size_t gguf_scalar_size(uint32_t t) {
    switch (t) {
    case GGUF_U8: case GGUF_I8: case GGUF_BOOL: return 1;
    case GGUF_U16: case GGUF_I16: return 2;
    case GGUF_U32: case GGUF_I32: case GGUF_F32: return 4;
    case GGUF_U64: case GGUF_I64: case GGUF_F64: return 8;
    default: return 0;
    }
}

static void rd_scalar(reader *r, uint32_t t, gguf_kv *kv) {
    size_t n = gguf_scalar_size(t);
    if (!need(r, n)) return;
    const uint8_t *p = r->p;
    switch (t) {
    case GGUF_U8: kv->v.u64 = p[0]; break;
    case GGUF_BOOL: kv->v.u64 = p[0] != 0; break;
    case GGUF_I8: kv->v.i64 = (int8_t)p[0]; break;
    case GGUF_U16: { uint16_t x; memcpy(&x, p, 2); kv->v.u64 = x; break; }
    case GGUF_I16: { int16_t x; memcpy(&x, p, 2); kv->v.i64 = x; break; }
    case GGUF_U32: { uint32_t x; memcpy(&x, p, 4); kv->v.u64 = x; break; }
    case GGUF_I32: { int32_t x; memcpy(&x, p, 4); kv->v.i64 = x; break; }
    case GGUF_F32: { float x; memcpy(&x, p, 4); kv->v.f64 = x; break; }
    case GGUF_U64: { uint64_t x; memcpy(&x, p, 8); kv->v.u64 = x; break; }
    case GGUF_I64: { int64_t x; memcpy(&x, p, 8); kv->v.i64 = x; break; }
    case GGUF_F64: { double x; memcpy(&x, p, 8); kv->v.f64 = x; break; }
    default: r->err = 1; return;
    }
    r->p += n;
}

const uint8_t *gguf_read_arr_str(const uint8_t *p, gguf_str *out) {
    uint64_t len;
    memcpy(&len, p, 8);
    out->len = len;
    out->s = (const char *)(p + 8);
    return p + 8 + len;
}

static size_t tensor_nbytes(uint32_t type, const uint64_t *ne, uint32_t n_dims) {
    uint64_t rows = 1;
    for (uint32_t i = 1; i < n_dims; i++) rows *= ne[i];
    size_t row = ti_type_row_size((int)type, (int64_t)ne[0]);
    return row * (size_t)rows;
}

int gguf_open(gguf_file *g, const char *path) {
    memset(g, 0, sizeof *g);
    if (ti_mmap_open(&g->map, path) != 0) return -1;
    reader r = {(const uint8_t *)g->map.data, (const uint8_t *)g->map.data + g->map.size, 0};
    uint32_t magic = rd_u32(&r);
    if (magic != 0x46554747u) { /* "GGUF" */
        ti_set_error("'%s' is not a GGUF file (bad magic)", path);
        goto fail;
    }
    g->version = rd_u32(&r);
    if (g->version != 3 && g->version != 2) {
        ti_set_error("unsupported GGUF version %u (need 2 or 3)", g->version);
        goto fail;
    }
    g->n_tensors = rd_u64(&r);
    g->n_kv = rd_u64(&r);
    if (r.err || g->n_kv > 1u << 20 || g->n_tensors > 1u << 20) {
        ti_set_error("corrupt GGUF header");
        goto fail;
    }
    g->kv = (gguf_kv *)calloc(g->n_kv ? g->n_kv : 1, sizeof(gguf_kv));
    g->tensors = (gguf_tensor *)calloc(g->n_tensors ? g->n_tensors : 1, sizeof(gguf_tensor));
    if (!g->kv || !g->tensors) {
        ti_set_error("out of memory");
        goto fail;
    }
    g->kv_begin = (size_t)(r.p - (const uint8_t *)g->map.data);
    g->alignment = 32;
    for (uint64_t i = 0; i < g->n_kv && !r.err; i++) {
        gguf_kv *kv = &g->kv[i];
        kv->raw_begin = (size_t)(r.p - (const uint8_t *)g->map.data);
        kv->key = rd_str(&r);
        kv->type = rd_u32(&r);
        if (kv->type == GGUF_STRING) {
            kv->v.str = rd_str(&r);
        } else if (kv->type == GGUF_ARRAY) {
            kv->arr_type = rd_u32(&r);
            kv->arr_len = rd_u64(&r);
            kv->arr_data = r.p;
            if (kv->arr_type == GGUF_STRING) {
                for (uint64_t k = 0; k < kv->arr_len && !r.err; k++) (void)rd_str(&r);
            } else {
                size_t es = gguf_scalar_size(kv->arr_type);
                if (!es || !need(&r, es * kv->arr_len)) {
                    r.err = 1;
                    break;
                }
                r.p += es * kv->arr_len;
            }
        } else {
            rd_scalar(&r, kv->type, kv);
        }
        kv->raw_end = (size_t)(r.p - (const uint8_t *)g->map.data);
        if (!r.err && gguf_str_eq(kv->key, "general.alignment") && kv->v.u64 > 0)
            g->alignment = (uint32_t)kv->v.u64;
    }
    g->kv_end = (size_t)(r.p - (const uint8_t *)g->map.data);
    for (uint64_t i = 0; i < g->n_tensors && !r.err; i++) {
        gguf_tensor *t = &g->tensors[i];
        t->name = rd_str(&r);
        t->n_dims = rd_u32(&r);
        if (t->n_dims > 4) {
            r.err = 1;
            break;
        }
        for (int d = 0; d < 4; d++) t->ne[d] = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) t->ne[d] = rd_u64(&r);
        t->type = rd_u32(&r);
        t->offset = rd_u64(&r);
    }
    if (r.err) {
        ti_set_error("truncated or corrupt GGUF metadata");
        goto fail;
    }
    size_t off = (size_t)(r.p - (const uint8_t *)g->map.data);
    size_t a = g->alignment;
    g->data_offset = (off + a - 1) / a * a;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->tensors[i];
        t->nbytes = tensor_nbytes(t->type, t->ne, t->n_dims);
        size_t start = g->data_offset + (size_t)t->offset;
        t->data = (const uint8_t *)g->map.data + start;
        if (t->nbytes && start + t->nbytes > g->map.size) {
            ti_set_error("tensor '%.*s' extends past end of file", (int)t->name.len, t->name.s);
            goto fail;
        }
    }
    return 0;
fail:
    gguf_close(g);
    return -1;
}

void gguf_close(gguf_file *g) {
    free(g->kv);
    free(g->tensors);
    ti_mmap_close(&g->map);
    memset(g, 0, sizeof *g);
}

int gguf_str_eq(gguf_str s, const char *c) {
    size_t n = strlen(c);
    return s.len == n && memcmp(s.s, c, n) == 0;
}

const gguf_kv *gguf_find_kv(const gguf_file *g, const char *key) {
    for (uint64_t i = 0; i < g->n_kv; i++)
        if (gguf_str_eq(g->kv[i].key, key)) return &g->kv[i];
    return NULL;
}

const gguf_tensor *gguf_find_tensor(const gguf_file *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (gguf_str_eq(g->tensors[i].name, name)) return &g->tensors[i];
    return NULL;
}

int64_t gguf_get_int(const gguf_file *g, const char *key, int64_t def) {
    const gguf_kv *kv = gguf_find_kv(g, key);
    if (!kv) return def;
    switch (kv->type) {
    case GGUF_U8: case GGUF_U16: case GGUF_U32: case GGUF_U64: case GGUF_BOOL:
        return (int64_t)kv->v.u64;
    case GGUF_I8: case GGUF_I16: case GGUF_I32: case GGUF_I64: return kv->v.i64;
    case GGUF_F32: case GGUF_F64: return (int64_t)kv->v.f64;
    default: return def;
    }
}

double gguf_get_float(const gguf_file *g, const char *key, double def) {
    const gguf_kv *kv = gguf_find_kv(g, key);
    if (!kv) return def;
    if (kv->type == GGUF_F32 || kv->type == GGUF_F64) return kv->v.f64;
    if (kv->type == GGUF_STRING || kv->type == GGUF_ARRAY) return def;
    return (double)gguf_get_int(g, key, 0);
}

int gguf_get_str(const gguf_file *g, const char *key, char *buf, size_t cap) {
    const gguf_kv *kv = gguf_find_kv(g, key);
    if (!kv || kv->type != GGUF_STRING || cap == 0) return -1;
    size_t n = kv->v.str.len < cap - 1 ? (size_t)kv->v.str.len : cap - 1;
    memcpy(buf, kv->v.str.s, n);
    buf[n] = 0;
    return 0;
}
