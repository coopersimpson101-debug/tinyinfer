/* Byte-level BPE tokenizer (GPT-2 / Qwen2 style) loaded from GGUF metadata. */
#ifndef TI_TOKENIZER_H
#define TI_TOKENIZER_H

#include <stdint.h>
#include "gguf.h"

typedef struct {
    const char *key;
    uint32_t len;
    int32_t val;
} ti_hslot;

typedef struct {
    ti_hslot *slots;
    uint32_t cap; /* power of two */
} ti_hmap;

typedef struct ti_tokenizer {
    int n_vocab;
    char **pieces;     /* NUL-terminated copies */
    int *piece_len;
    int *token_type;   /* 1 normal, 3 control, 4 user-defined, ... */
    ti_hmap vocab;     /* piece -> id */
    ti_hmap merges;    /* "left right" -> rank */
    char *merge_store; /* backing storage for merge keys */
    int *specials;     /* ids of control/user-defined tokens, longest first */
    int n_specials;
    int byte_to_cp[256];
    int cp_to_byte[512];
    int bos_id, eos_id, eot_id; /* eot = <|im_end|> if present */
} ti_tokenizer;

int ti_tokenizer_load(ti_tokenizer *t, const gguf_file *g);
void ti_tokenizer_free(ti_tokenizer *t);

int ti_tok_encode(const ti_tokenizer *t, const char *text, int32_t *out, int max_tokens,
                  int parse_special);
int ti_tok_decode(const ti_tokenizer *t, const int32_t *tokens, int n, char *out, int max_bytes);
/* bytes of a single token (for streaming output); returns length */
int ti_tok_piece_bytes(const ti_tokenizer *t, int id, char *out, int cap);
int ti_tok_lookup(const ti_tokenizer *t, const char *s, int len);

#endif
