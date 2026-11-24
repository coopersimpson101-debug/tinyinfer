/*
 * tinyinfer-quantize: rewrite a GGUF file with every 2-D weight matrix
 * converted to Q8_0 / Q4_0 / F16 / F32 ("pure" quantisation: unlike
 * llama.cpp's default Q4_0 recipe the output/embedding tensors are not
 * promoted to K-quants, so both engines can read the result).
 *
 *   tinyinfer-quantize in.gguf out.gguf q4_0
 *
 * Metadata is copied byte-for-byte except general.file_type, which is
 * updated so tools like llama-bench report the right type.
 */
#include "../include/tinyinfer.h"
#include "gguf.h"
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int wr(FILE *f, const void *p, size_t n) { return fwrite(p, 1, n, f) == n ? 0 : -1; }
static int wr_u32(FILE *f, uint32_t v) { return wr(f, &v, 4); }
static int wr_u64(FILE *f, uint64_t v) { return wr(f, &v, 8); }
static int pad_to(FILE *f, long long pos, size_t a) {
    static const char z[64] = {0};
    size_t p = (size_t)(((pos + (long long)a - 1) / (long long)a) * (long long)a - pos);
    return wr(f, z, p);
}

static int should_quantize(const gguf_tensor *t) {
    if (t->n_dims != 2 || t->ne[0] % TI_QK != 0) return 0;
    if (t->name.len < 7 || memcmp(t->name.s + t->name.len - 7, ".weight", 7) != 0) return 0;
    for (uint64_t i = 0; i + 4 <= t->name.len; i++) /* skip *norm* tensors */
        if (memcmp(t->name.s + i, "norm", 4) == 0) return 0;
    return t->type == TI_TYPE_F32 || t->type == TI_TYPE_F16 || t->type == TI_TYPE_BF16 ||
           t->type == TI_TYPE_Q8_0 || t->type == TI_TYPE_Q4_0;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: tinyinfer-quantize in.gguf out.gguf {q4_0|q8_0|f16|f32}\n");
        return 1;
    }
    int type, ftype;
    if (!strcmp(argv[3], "q4_0")) { type = TI_TYPE_Q4_0; ftype = 2; }
    else if (!strcmp(argv[3], "q8_0")) { type = TI_TYPE_Q8_0; ftype = 7; }
    else if (!strcmp(argv[3], "f16")) { type = TI_TYPE_F16; ftype = 1; }
    else if (!strcmp(argv[3], "f32")) { type = TI_TYPE_F32; ftype = 0; }
    else { fprintf(stderr, "unknown type %s\n", argv[3]); return 1; }

    gguf_file g;
    if (gguf_open(&g, argv[1]) != 0) {
        fprintf(stderr, "error: %s\n", ti_last_error());
        return 1;
    }
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        const gguf_tensor *t = &g.tensors[i];
        if (!should_quantize(t) && ti_type_row_size((int)t->type, (int64_t)t->ne[0]) == 0) {
            fprintf(stderr, "error: tensor %.*s has unsupported type %u\n", (int)t->name.len,
                    t->name.s, t->type);
            return 1;
        }
    }
    FILE *f = fopen(argv[2], "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
    const size_t A = g.alignment;
    const uint8_t *base = (const uint8_t *)g.map.data;
    int rc = 0;
    rc |= wr_u32(f, 0x46554747u);
    rc |= wr_u32(f, 3);
    rc |= wr_u64(f, g.n_tensors);
    rc |= wr_u64(f, g.n_kv);
    for (uint64_t i = 0; i < g.n_kv; i++) {
        const gguf_kv *kv = &g.kv[i];
        if (gguf_str_eq(kv->key, "general.file_type")) {
            rc |= wr_u64(f, kv->key.len);
            rc |= wr(f, kv->key.s, (size_t)kv->key.len);
            rc |= wr_u32(f, GGUF_U32);
            rc |= wr_u32(f, (uint32_t)ftype);
        } else {
            rc |= wr(f, base + kv->raw_begin, kv->raw_end - kv->raw_begin);
        }
    }
    /* tensor infos with new types and offsets */
    uint32_t *types = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)g.n_tensors);
    size_t off = 0;
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        const gguf_tensor *t = &g.tensors[i];
        types[i] = should_quantize(t) ? (uint32_t)type : t->type;
        rc |= wr_u64(f, t->name.len);
        rc |= wr(f, t->name.s, (size_t)t->name.len);
        rc |= wr_u32(f, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++) rc |= wr_u64(f, t->ne[d]);
        rc |= wr_u32(f, types[i]);
        rc |= wr_u64(f, off);
        size_t rows = (size_t)(t->ne[1] * t->ne[2] * t->ne[3]);
        size_t nb = ti_type_row_size((int)types[i], (int64_t)t->ne[0]) * rows;
        off = (off + nb + A - 1) / A * A;
    }
    rc |= pad_to(f, ftell(f), A);
    float *row = NULL;
    void *qrow = NULL;
    size_t row_cap = 0;
    size_t total_in = 0, total_out = 0;
    for (uint64_t i = 0; i < g.n_tensors && !rc; i++) {
        const gguf_tensor *t = &g.tensors[i];
        size_t rows = (size_t)(t->ne[1] * t->ne[2] * t->ne[3]);
        size_t n0 = (size_t)t->ne[0];
        size_t in_rb = ti_type_row_size((int)t->type, (int64_t)n0);
        size_t out_rb = ti_type_row_size((int)types[i], (int64_t)n0);
        if (types[i] == t->type) {
            rc |= wr(f, t->data, t->nbytes);
        } else {
            if (n0 > row_cap) {
                free(row);
                free(qrow);
                row = (float *)malloc(sizeof(float) * n0);
                qrow = malloc(sizeof(float) * n0);
                row_cap = n0;
            }
            for (size_t r = 0; r < rows && !rc; r++) {
                ti_dequantize_row((int)t->type, (const uint8_t *)t->data + r * in_rb, row, (int64_t)n0);
                ti_quantize_row((int)types[i], row, qrow, (int64_t)n0);
                rc |= wr(f, qrow, out_rb);
            }
        }
        total_in += t->nbytes;
        total_out += out_rb * rows;
        rc |= pad_to(f, (long long)(out_rb * rows), A);
        fprintf(stderr, "\r[%llu/%llu] %-40.*s", (unsigned long long)i + 1,
                (unsigned long long)g.n_tensors, (int)t->name.len, t->name.s);
    }
    fclose(f);
    free(row);
    free(qrow);
    free(types);
    gguf_close(&g);
    if (rc) {
        fprintf(stderr, "\nerror writing %s\n", argv[2]);
        return 1;
    }
    fprintf(stderr, "\nwrote %s: %.1f MB -> %.1f MB\n", argv[2], total_in / 1e6, total_out / 1e6);
    return 0;
}
