/*
 * Byte-level BPE (GPT-2 / Qwen2 flavour).
 *
 * Encoding pipeline
 *   1. split out special tokens (<|im_start|>, <|endoftext|>, ...) verbatim
 *   2. pre-tokenise each remaining span with a hand-written matcher for the
 *      Qwen2 regex
 *        (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}
 *        | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
 *      Unicode categories are approximated with range tables (see is_letter);
 *      exact for ASCII, Latin, Greek, Cyrillic, CJK, kana, Hangul and emoji,
 *      approximate for scripts with combining marks (e.g. Devanagari).
 *   3. map every byte of a pre-token to a printable code point (GPT-2's
 *      bytes_to_unicode table), so that " hello" becomes "Ġhello"
 *   4. repeatedly merge the adjacent pair with the lowest merge rank
 *   5. look up each resulting symbol in the vocabulary
 * Decoding just reverses the byte mapping.
 */
#include "tokenizer.h"
#include "platform.h"
#include "../include/tinyinfer.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------ hash map --- */
static uint64_t fnv1a(const char *s, uint32_t n) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i < n; i++) {
        h ^= (uint8_t)s[i];
        h *= 1099511628211ull;
    }
    return h;
}

static int hmap_init(ti_hmap *m, uint32_t n) {
    uint32_t cap = 16;
    while (cap < n * 2u) cap <<= 1;
    m->slots = (ti_hslot *)calloc(cap, sizeof(ti_hslot));
    m->cap = m->slots ? cap : 0;
    return m->slots ? 0 : -1;
}

static void hmap_put(ti_hmap *m, const char *k, uint32_t len, int32_t v) {
    uint32_t i = (uint32_t)fnv1a(k, len) & (m->cap - 1);
    while (m->slots[i].key) {
        if (m->slots[i].len == len && memcmp(m->slots[i].key, k, len) == 0) return; /* keep first */
        i = (i + 1) & (m->cap - 1);
    }
    m->slots[i].key = k;
    m->slots[i].len = len;
    m->slots[i].val = v;
}

static int32_t hmap_get(const ti_hmap *m, const char *k, uint32_t len) {
    if (!m->cap) return -1;
    uint32_t i = (uint32_t)fnv1a(k, len) & (m->cap - 1);
    while (m->slots[i].key) {
        if (m->slots[i].len == len && memcmp(m->slots[i].key, k, len) == 0) return m->slots[i].val;
        i = (i + 1) & (m->cap - 1);
    }
    return -1;
}

/* --------------------------------------------------------------- utf-8 --- */
static int utf8_decode(const unsigned char *s, int n, int *cp) {
    unsigned c = s[0];
    if (c < 0x80) { *cp = (int)c; return 1; }
    if ((c >> 5) == 0x6 && n >= 2 && (s[1] & 0xc0) == 0x80) {
        *cp = (int)(((c & 0x1f) << 6) | (s[1] & 0x3f));
        return 2;
    }
    if ((c >> 4) == 0xe && n >= 3 && (s[1] & 0xc0) == 0x80 && (s[2] & 0xc0) == 0x80) {
        *cp = (int)(((c & 0x0f) << 12) | ((s[1] & 0x3f) << 6) | (s[2] & 0x3f));
        return 3;
    }
    if ((c >> 3) == 0x1e && n >= 4 && (s[1] & 0xc0) == 0x80 && (s[2] & 0xc0) == 0x80 &&
        (s[3] & 0xc0) == 0x80) {
        *cp = (int)(((c & 0x07) << 18) | ((s[1] & 0x3f) << 12) | ((s[2] & 0x3f) << 6) |
                    (s[3] & 0x3f));
        return 4;
    }
    *cp = 0xFFFD; /* invalid byte: treat as one "other" code point */
    return 1;
}

static int utf8_encode(int cp, char *out) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char)(0xc0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3f));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xe0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3f));
        out[2] = (char)(0x80 | (cp & 0x3f));
        return 3;
    }
    out[0] = (char)(0xf0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3f));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3f));
    out[3] = (char)(0x80 | (cp & 0x3f));
    return 4;
}

/* ------------------------------------------------- unicode categories ---- */
static int is_space(int c) {
    return c == ' ' || (c >= 0x09 && c <= 0x0d) || c == 0x85 || c == 0xa0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f ||
           c == 0x205f || c == 0x3000;
}

static int is_number(int c) {
    if (c < 0x80) return c >= '0' && c <= '9';
    return c == 0xb2 || c == 0xb3 || c == 0xb9 || (c >= 0xbc && c <= 0xbe) ||
           (c >= 0x660 && c <= 0x669) || (c >= 0x6f0 && c <= 0x6f9) ||
           (c >= 0x966 && c <= 0x96f) || c == 0x2070 || (c >= 0x2074 && c <= 0x2079) ||
           (c >= 0x2080 && c <= 0x2089) || (c >= 0x2150 && c <= 0x2182) ||
           (c >= 0x2185 && c <= 0x2189) || (c >= 0x2460 && c <= 0x249b) ||
           (c >= 0x24ea && c <= 0x24ff) || (c >= 0x2776 && c <= 0x2793) || c == 0x3007 ||
           (c >= 0x3021 && c <= 0x3029) || (c >= 0xff10 && c <= 0xff19);
}

static int is_letter(int c) {
    if (c < 0x80) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    if (c < 0x100) return c == 0xaa || c == 0xb5 || c == 0xba || (c >= 0xc0 && c != 0xd7 && c != 0xf7);
    if (is_number(c) || is_space(c)) return 0;
    /* non-letter blocks (punctuation, symbols, marks, emoji, private use) */
    if (c >= 0x2c2 && c <= 0x2c5) return 0;
    if (c >= 0x2d2 && c <= 0x2df) return 0;
    if (c >= 0x2e5 && c <= 0x2eb) return 0;
    if (c >= 0x2ef && c <= 0x36f) return 0; /* modifier symbols + combining marks */
    if (c == 0x375 || c == 0x37e || c == 0x384 || c == 0x385 || c == 0x387) return 0;
    if (c >= 0x482 && c <= 0x489) return 0;
    if (c >= 0x591 && c <= 0x5cf) return 0; /* Hebrew points + punctuation */
    if (c >= 0x600 && c <= 0x61f) return 0;
    if (c >= 0x64b && c <= 0x66d) return 0;
    if (c >= 0x2000 && c <= 0x206f) return 0; /* general punctuation */
    if (c >= 0x20a0 && c <= 0x20ff) return 0; /* currency, combining for symbols */
    if (c >= 0x2100 && c <= 0x214f) {          /* letterlike: only some are letters */
        return c == 0x2102 || c == 0x2107 || (c >= 0x210a && c <= 0x2113) || c == 0x2115 ||
               (c >= 0x2119 && c <= 0x211d) || c == 0x2124 || c == 0x2126 || c == 0x2128 ||
               (c >= 0x212a && c <= 0x212d) || (c >= 0x212f && c <= 0x2139) ||
               (c >= 0x213c && c <= 0x213f) || (c >= 0x2145 && c <= 0x2149) || c == 0x214e;
    }
    if (c >= 0x2190 && c <= 0x2bff) return 0; /* arrows, maths, box drawing, dingbats */
    if (c >= 0x2e00 && c <= 0x2e7f) return 0;
    if (c >= 0x2ff0 && c <= 0x3004) return 0;
    if (c >= 0x3008 && c <= 0x3020) return 0; /* CJK brackets & punctuation */
    if (c >= 0x302a && c <= 0x3030) return 0;
    if (c >= 0x3036 && c <= 0x303a) return 0;
    if (c >= 0x303d && c <= 0x303f) return 0;
    if (c >= 0x3099 && c <= 0x309c) return 0;
    if (c == 0x30a0 || c == 0x30fb) return 0;
    if (c >= 0x3200 && c <= 0x33ff) return 0; /* enclosed CJK, compatibility */
    if (c >= 0x4dc0 && c <= 0x4dff) return 0;
    if (c >= 0xd800 && c <= 0xf8ff) return 0; /* surrogates, private use */
    if (c >= 0xfe00 && c <= 0xfe6f) return 0; /* variation selectors, forms */
    if (c >= 0xff00 && c <= 0xff20) return 0;
    if (c >= 0xff3b && c <= 0xff40) return 0;
    if (c >= 0xff5b && c <= 0xff65) return 0;
    if (c >= 0xffe0 && c <= 0xffff) return 0;
    if (c >= 0x1d000 && c <= 0x1d3ff) return 0; /* musical symbols */
    if (c >= 0x1f000 && c <= 0x1faff) return 0; /* emoji & pictographs */
    if (c >= 0xe0000) return 0;                 /* tags, variation sel. supplement */
    return 1;
}

static int is_nl(int c) { return c == '\r' || c == '\n'; }
static int is_other(int c) { return !is_space(c) && !is_letter(c) && !is_number(c); }
static int lower_ascii(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

/* Pre-tokenise text[0..len): writes byte end-offsets of each piece. */
TI_API int ti_pretokenize(const char *text, int len, int *ends, int max_ends) {
    const unsigned char *s = (const unsigned char *)text;
    /* decode once into code points with byte offsets */
    int *cps = (int *)malloc(sizeof(int) * (size_t)(len + 1));
    int *offs = (int *)malloc(sizeof(int) * (size_t)(len + 2));
    if (!cps || !offs) {
        free(cps);
        free(offs);
        return -1;
    }
    int n = 0;
    for (int b = 0; b < len;) {
        int cp;
        int k = utf8_decode(s + b, len - b, &cp);
        cps[n] = cp;
        offs[n] = b;
        n++;
        b += k;
    }
    offs[n] = len;
    int count = 0;
    int i = 0;
#define CP(k) ((k) < n ? cps[k] : -1)
    while (i < n) {
        int c = cps[i], j = -1;
        /* 1. contractions */
        if (c == '\'' && i + 1 < n) {
            int c1 = lower_ascii(CP(i + 1)), c2 = lower_ascii(CP(i + 2));
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') j = i + 2;
            else if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') ||
                     (c1 == 'l' && c2 == 'l'))
                j = i + 3;
        }
        /* 2. [^\r\n\p{L}\p{N}]?\p{L}+ */
        if (j < 0) {
            int k = -1;
            if (is_letter(c)) k = i;
            else if (!is_nl(c) && !is_number(c) && i + 1 < n && is_letter(cps[i + 1])) k = i + 1;
            if (k >= 0) {
                while (k < n && is_letter(cps[k])) k++;
                j = k;
            }
        }
        /* 3. \p{N} (single digit) */
        if (j < 0 && is_number(c)) j = i + 1;
        /* 4.  ?[^\s\p{L}\p{N}]+[\r\n]* */
        if (j < 0) {
            int k = i;
            if (c == ' ' && i + 1 < n && is_other(cps[i + 1])) k = i + 1;
            if (k < n && is_other(cps[k])) {
                while (k < n && is_other(cps[k])) k++;
                while (k < n && is_nl(cps[k])) k++;
                j = k;
            }
        }
        /* 5-7. whitespace */
        if (j < 0 && is_space(c)) {
            int e = i;
            while (e < n && is_space(cps[e])) e++;
            int last_nl = -1;
            for (int k = i; k < e; k++)
                if (is_nl(cps[k])) last_nl = k;
            if (last_nl >= 0) j = last_nl + 1;          /* \s*[\r\n]+ */
            else if (e == n) j = e;                      /* \s+(?!\S) at end */
            else if (e - i >= 2) j = e - 1;              /* \s+(?!\S) */
            else j = e;                                  /* \s+ */
        }
        if (j < 0) j = i + 1;
        if (count < max_ends) ends[count] = offs[j];
        count++;
        i = j;
    }
#undef CP
    free(cps);
    free(offs);
    return count;
}

/* ---------------------------------------------------------------- load --- */
static void build_byte_map(ti_tokenizer *t) {
    int n = 0;
    for (int i = 0; i < 512; i++) t->cp_to_byte[i] = -1;
    for (int b = 0; b < 256; b++) {
        int keep = (b >= '!' && b <= '~') || (b >= 0xa1 && b <= 0xac) || (b >= 0xae && b <= 0xff);
        t->byte_to_cp[b] = keep ? b : 256 + n++;
    }
    for (int b = 0; b < 256; b++) t->cp_to_byte[t->byte_to_cp[b]] = b;
}

static const ti_tokenizer *g_sort_tok;
static int cmp_special(const void *a, const void *b) {
    int ia = *(const int *)a, ib = *(const int *)b;
    return g_sort_tok->piece_len[ib] - g_sort_tok->piece_len[ia];
}

int ti_tokenizer_load(ti_tokenizer *t, const gguf_file *g) {
    memset(t, 0, sizeof *t);
    build_byte_map(t);
    const gguf_kv *tok = gguf_find_kv(g, "tokenizer.ggml.tokens");
    if (!tok || tok->type != GGUF_ARRAY || tok->arr_type != GGUF_STRING) {
        ti_set_error("GGUF has no tokenizer.ggml.tokens");
        return -1;
    }
    char model[64] = {0};
    if (gguf_get_str(g, "tokenizer.ggml.model", model, sizeof model) == 0 &&
        strcmp(model, "gpt2") != 0) {
        ti_set_error("unsupported tokenizer model '%s' (only byte-level BPE 'gpt2')", model);
        return -1;
    }
    t->n_vocab = (int)tok->arr_len;
    t->pieces = (char **)calloc((size_t)t->n_vocab, sizeof(char *));
    t->piece_len = (int *)calloc((size_t)t->n_vocab, sizeof(int));
    t->token_type = (int *)calloc((size_t)t->n_vocab, sizeof(int));
    if (!t->pieces || !t->piece_len || !t->token_type ||
        hmap_init(&t->vocab, (uint32_t)t->n_vocab) != 0)
        goto oom;
    const uint8_t *p = tok->arr_data;
    for (int i = 0; i < t->n_vocab; i++) {
        gguf_str s;
        p = gguf_read_arr_str(p, &s);
        t->pieces[i] = (char *)malloc(s.len + 1);
        if (!t->pieces[i]) goto oom;
        memcpy(t->pieces[i], s.s, s.len);
        t->pieces[i][s.len] = 0;
        t->piece_len[i] = (int)s.len;
        t->token_type[i] = 1;
        hmap_put(&t->vocab, t->pieces[i], (uint32_t)s.len, i);
    }
    const gguf_kv *tt = gguf_find_kv(g, "tokenizer.ggml.token_type");
    if (tt && tt->type == GGUF_ARRAY && (tt->arr_type == GGUF_I32 || tt->arr_type == GGUF_U32) &&
        (int)tt->arr_len == t->n_vocab) {
        for (int i = 0; i < t->n_vocab; i++) {
            int32_t v;
            memcpy(&v, tt->arr_data + 4 * (size_t)i, 4);
            t->token_type[i] = v;
        }
    } else { /* heuristic: <|...|> are specials */
        for (int i = 0; i < t->n_vocab; i++) {
            int L = t->piece_len[i];
            if (L >= 4 && strncmp(t->pieces[i], "<|", 2) == 0 &&
                strcmp(t->pieces[i] + L - 2, "|>") == 0)
                t->token_type[i] = 3;
        }
    }
    /* merges */
    const gguf_kv *mg = gguf_find_kv(g, "tokenizer.ggml.merges");
    if (mg && mg->type == GGUF_ARRAY && mg->arr_type == GGUF_STRING) {
        size_t total = 0;
        const uint8_t *q = mg->arr_data;
        for (uint64_t i = 0; i < mg->arr_len; i++) {
            gguf_str s;
            q = gguf_read_arr_str(q, &s);
            total += s.len;
        }
        t->merge_store = (char *)malloc(total + 1);
        if (!t->merge_store || hmap_init(&t->merges, (uint32_t)mg->arr_len + 1) != 0) goto oom;
        q = mg->arr_data;
        size_t off = 0;
        for (uint64_t i = 0; i < mg->arr_len; i++) {
            gguf_str s;
            q = gguf_read_arr_str(q, &s);
            memcpy(t->merge_store + off, s.s, s.len);
            hmap_put(&t->merges, t->merge_store + off, (uint32_t)s.len, (int32_t)i);
            off += s.len;
        }
    }
    /* specials, longest first so that greedy matching prefers the longest */
    t->specials = (int *)malloc(sizeof(int) * (size_t)t->n_vocab);
    if (!t->specials) goto oom;
    for (int i = 0; i < t->n_vocab; i++)
        if ((t->token_type[i] == 3 || t->token_type[i] == 4) && t->piece_len[i] > 0)
            t->specials[t->n_specials++] = i;
    g_sort_tok = t;
    qsort(t->specials, (size_t)t->n_specials, sizeof(int), cmp_special);

    t->bos_id = (int)gguf_get_int(g, "tokenizer.ggml.bos_token_id", -1);
    t->eos_id = (int)gguf_get_int(g, "tokenizer.ggml.eos_token_id", -1);
    t->eot_id = ti_tok_lookup(t, "<|im_end|>", 10);
    return 0;
oom:
    ti_set_error("out of memory loading tokenizer");
    ti_tokenizer_free(t);
    return -1;
}

void ti_tokenizer_free(ti_tokenizer *t) {
    if (t->pieces)
        for (int i = 0; i < t->n_vocab; i++) free(t->pieces[i]);
    free(t->pieces);
    free(t->piece_len);
    free(t->token_type);
    free(t->vocab.slots);
    free(t->merges.slots);
    free(t->merge_store);
    free(t->specials);
    memset(t, 0, sizeof *t);
}

int ti_tok_lookup(const ti_tokenizer *t, const char *s, int len) {
    return hmap_get(&t->vocab, s, (uint32_t)len);
}

/* ---------------------------------------------------------------- BPE ---- */
typedef struct {
    int32_t *out;
    int max, count;
} emitter;

static void emit(emitter *e, int32_t id) {
    if (e->count < e->max) e->out[e->count] = id;
    e->count++;
}

/* BPE-encode one pre-token (raw bytes). */
static int bpe_word(const ti_tokenizer *t, const unsigned char *w, int wl, emitter *e) {
    /* byte-level mapped string: each byte -> 1 or 2 utf-8 bytes */
    char *buf = (char *)malloc((size_t)wl * 2 + 1);
    int *start = (int *)malloc(sizeof(int) * (size_t)(wl + 1));
    int *slen = (int *)malloc(sizeof(int) * (size_t)(wl + 1));
    char *key = (char *)malloc((size_t)wl * 2 + 2);
    if (!buf || !start || !slen || !key) {
        free(buf); free(start); free(slen); free(key);
        return -1;
    }
    int bl = 0, ns = 0;
    for (int i = 0; i < wl; i++) {
        start[ns] = bl;
        slen[ns] = utf8_encode(t->byte_to_cp[w[i]], buf + bl);
        bl += slen[ns];
        ns++;
    }
    if (t->merges.cap) {
        for (;;) {
            int best = -1, best_rank = 0x7fffffff;
            for (int i = 0; i + 1 < ns; i++) {
                int kl = slen[i] + 1 + slen[i + 1];
                memcpy(key, buf + start[i], (size_t)slen[i]);
                key[slen[i]] = ' ';
                memcpy(key + slen[i] + 1, buf + start[i + 1], (size_t)slen[i + 1]);
                int r = hmap_get(&t->merges, key, (uint32_t)kl);
                if (r >= 0 && r < best_rank) {
                    best_rank = r;
                    best = i;
                }
            }
            if (best < 0) break;
            slen[best] += slen[best + 1]; /* symbols are contiguous in buf */
            for (int i = best + 1; i + 1 < ns; i++) {
                start[i] = start[i + 1];
                slen[i] = slen[i + 1];
            }
            ns--;
        }
    }
    for (int i = 0; i < ns; i++) {
        int id = hmap_get(&t->vocab, buf + start[i], (uint32_t)slen[i]);
        if (id >= 0) {
            emit(e, id);
            continue;
        }
        /* fall back to single byte-symbols */
        int p = start[i], end = start[i] + slen[i];
        while (p < end) {
            int cp;
            int k = utf8_decode((const unsigned char *)buf + p, end - p, &cp);
            int bid = hmap_get(&t->vocab, buf + p, (uint32_t)k);
            if (bid >= 0) emit(e, bid);
            p += k;
        }
    }
    free(buf); free(start); free(slen); free(key);
    return 0;
}

static int encode_span(const ti_tokenizer *t, const char *s, int len, emitter *e) {
    if (len <= 0) return 0;
    int cap = len + 1;
    int *ends = (int *)malloc(sizeof(int) * (size_t)cap);
    if (!ends) return -1;
    int n = ti_pretokenize(s, len, ends, cap);
    int prev = 0;
    for (int i = 0; i < n; i++) {
        if (bpe_word(t, (const unsigned char *)s + prev, ends[i] - prev, e) != 0) {
            free(ends);
            return -1;
        }
        prev = ends[i];
    }
    free(ends);
    return 0;
}

int ti_tok_encode(const ti_tokenizer *t, const char *text, int32_t *out, int max_tokens,
                  int parse_special) {
    emitter e = {out, max_tokens, 0};
    int len = (int)strlen(text);
    int seg = 0, i = 0;
    while (i < len) {
        int hit = -1;
        if (parse_special) {
            for (int k = 0; k < t->n_specials; k++) {
                int id = t->specials[k], L = t->piece_len[id];
                if (t->pieces[id][0] == text[i] && L <= len - i &&
                    memcmp(text + i, t->pieces[id], (size_t)L) == 0) {
                    hit = id;
                    break;
                }
            }
        }
        if (hit >= 0) {
            if (encode_span(t, text + seg, i - seg, &e) != 0) return -1;
            emit(&e, hit);
            i += t->piece_len[hit];
            seg = i;
        } else {
            i++;
        }
    }
    if (encode_span(t, text + seg, len - seg, &e) != 0) return -1;
    return e.count;
}

int ti_tok_piece_bytes(const ti_tokenizer *t, int id, char *out, int cap) {
    if (id < 0 || id >= t->n_vocab) return 0;
    const char *p = t->pieces[id];
    int L = t->piece_len[id], n = 0;
    if (t->token_type[id] == 3 || t->token_type[id] == 4) {
        for (int i = 0; i < L; i++) {
            if (n < cap) out[n] = p[i];
            n++;
        }
        return n;
    }
    for (int i = 0; i < L;) {
        int cp;
        int k = utf8_decode((const unsigned char *)p + i, L - i, &cp);
        if (cp < 512 && t->cp_to_byte[cp] >= 0) {
            if (n < cap) out[n] = (char)t->cp_to_byte[cp];
            n++;
        } else {
            for (int j = 0; j < k; j++) {
                if (n < cap) out[n] = p[i + j];
                n++;
            }
        }
        i += k;
    }
    return n;
}

int ti_tok_decode(const ti_tokenizer *t, const int32_t *tokens, int n, char *out, int max_bytes) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        int room = max_bytes - 1 - total;
        if (room < 0) room = 0;
        total += ti_tok_piece_bytes(t, tokens[i], out ? out + total : NULL, out ? room : 0);
    }
    if (out && max_bytes > 0) out[total < max_bytes - 1 ? total : max_bytes - 1] = 0;
    return total;
}
