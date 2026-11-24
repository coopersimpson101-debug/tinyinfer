/* Minimal GGUF v3 reader (also accepts v2, which has the same layout). */
#ifndef TI_GGUF_H
#define TI_GGUF_H

#include <stddef.h>
#include <stdint.h>
#include "platform.h"

enum gguf_vtype {
    GGUF_U8 = 0, GGUF_I8 = 1, GGUF_U16 = 2, GGUF_I16 = 3, GGUF_U32 = 4, GGUF_I32 = 5,
    GGUF_F32 = 6, GGUF_BOOL = 7, GGUF_STRING = 8, GGUF_ARRAY = 9, GGUF_U64 = 10,
    GGUF_I64 = 11, GGUF_F64 = 12,
};

typedef struct {
    uint64_t len;
    const char *s; /* points into the mapping, NOT NUL terminated */
} gguf_str;

typedef struct {
    gguf_str key;
    uint32_t type;
    /* scalar values */
    union {
        uint64_t u64;
        int64_t i64;
        double f64;
        gguf_str str;
    } v;
    /* arrays */
    uint32_t arr_type;
    uint64_t arr_len;
    const uint8_t *arr_data; /* raw pointer to the first element */
    size_t raw_begin, raw_end; /* byte range of the whole entry in the file */
} gguf_kv;

typedef struct {
    gguf_str name;
    uint32_t n_dims;
    uint64_t ne[4];
    uint32_t type;
    uint64_t offset; /* relative to data section */
    const void *data;
    size_t nbytes;
} gguf_tensor;

typedef struct {
    ti_mmap map;
    uint32_t version;
    uint64_t n_kv, n_tensors;
    gguf_kv *kv;
    gguf_tensor *tensors;
    uint32_t alignment;
    size_t data_offset;
    size_t kv_begin, kv_end; /* byte range of the raw KV section (for rewriting) */
} gguf_file;

int gguf_open(gguf_file *g, const char *path);
void gguf_close(gguf_file *g);

const gguf_kv *gguf_find_kv(const gguf_file *g, const char *key);
const gguf_tensor *gguf_find_tensor(const gguf_file *g, const char *name);
int gguf_str_eq(gguf_str s, const char *c);

/* typed getters with defaults (integers of any width are accepted) */
int64_t gguf_get_int(const gguf_file *g, const char *key, int64_t def);
double gguf_get_float(const gguf_file *g, const char *key, double def);
/* copies into buf (NUL terminated); returns 0 if found */
int gguf_get_str(const gguf_file *g, const char *key, char *buf, size_t cap);

/* Array of strings iteration: returns pointer after the element. */
const uint8_t *gguf_read_arr_str(const uint8_t *p, gguf_str *out);
size_t gguf_scalar_size(uint32_t type);

#endif
