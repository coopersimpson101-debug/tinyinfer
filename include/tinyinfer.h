/*
 * tinyinfer - a from-scratch LLM inference engine (C11, no dependencies).
 *
 * Public C API. Everything the CLI, the Python test-suite (via ctypes) and the
 * Metal backend need is declared here.
 */
#ifndef TINYINFER_H
#define TINYINFER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(TI_BUILD_DLL)
#define TI_API __declspec(dllexport)
#else
#define TI_API
#endif

/* ggml / GGUF tensor type ids (subset we implement). */
enum ti_type {
    TI_TYPE_F32 = 0,
    TI_TYPE_F16 = 1,
    TI_TYPE_Q4_0 = 2,
    TI_TYPE_Q8_0 = 8,
    TI_TYPE_BF16 = 30,
};

#define TI_QK 32 /* block size for Q4_0 / Q8_0 */

typedef struct ti_model ti_model;

typedef struct ti_config {
    int n_layers;
    int dim;          /* embedding_length */
    int ffn_dim;      /* feed_forward_length */
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int vocab_size;
    int max_ctx;      /* context length actually allocated for the KV cache */
    int train_ctx;    /* context_length from the file */
    float rope_theta;
    float rms_eps;
    int tied_embeddings;
    int has_qkv_bias;
} ti_config;

/* ---- model ---------------------------------------------------------------- */
/* ctx_len <= 0 -> min(train_ctx, 2048). n_threads <= 0 -> number of cores. */
TI_API ti_model *ti_model_load(const char *path, int ctx_len, int n_threads);
TI_API void ti_model_free(ti_model *m);
TI_API const ti_config *ti_model_config(const ti_model *m);
TI_API const char *ti_last_error(void);

/*
 * Run the transformer on n tokens placed at positions [pos, pos+n).
 * logits: vocab_size floats for the last token, or n*vocab_size floats when
 * all_logits != 0. Returns 0 on success.
 */
TI_API int ti_forward(ti_model *m, const int32_t *tokens, int n, int pos,
                      float *logits, int all_logits);

/* ---- paged KV cache ------------------------------------------------------
 * Switch the model from one contiguous [layers][ctx][kv] cache to a pool of
 * n_blocks blocks of block_size positions (block_size <= 0 -> 16; n_blocks <= 0
 * -> enough for one max_ctx sequence). Up to 16 sequences then share the pool
 * via ti_forward_seq. Writing at position p of a sequence discards its cached
 * positions >= p (this is how rejected draft tokens are rolled back).
 * CPU backend only. */
TI_API int ti_kv_enable_paged(ti_model *m, int block_size, int n_blocks);
TI_API int ti_kv_is_paged(const ti_model *m);
TI_API int ti_forward_seq(ti_model *m, int seq_id, const int32_t *tokens, int n, int pos,
                          float *logits, int all_logits);
TI_API int ti_kv_seq_trim(ti_model *m, int seq_id, int n_pos); /* keep [0, n_pos) */
TI_API int ti_kv_seq_copy(ti_model *m, int dst, int src);      /* share blocks (CoW) */
TI_API int ti_kv_seq_len(const ti_model *m, int seq_id);
TI_API int ti_kv_blocks_free(const ti_model *m);
TI_API int ti_kv_blocks_total(const ti_model *m);

/* ---- tokenizer ----------------------------------------------------------- */
/* Returns the number of tokens written (or needed if > max_tokens), <0 on error. */
TI_API int ti_tokenize(const ti_model *m, const char *text, int32_t *out,
                       int max_tokens, int parse_special);
/* Writes the UTF-8 bytes of the tokens (NUL terminated). Returns byte count needed. */
TI_API int ti_detokenize(const ti_model *m, const int32_t *tokens, int n,
                         char *out, int max_bytes);
TI_API int ti_token_id(const ti_model *m, const char *piece); /* -1 if absent */
TI_API int ti_eos_id(const ti_model *m);
TI_API int ti_bos_id(const ti_model *m);
/* Split text into pre-tokens (exposed for tests). Writes byte offsets of the
 * end of each pre-token into ends[]; returns count. */
TI_API int ti_pretokenize(const char *text, int len, int *ends, int max_ends);

/* ---- quantisation -------------------------------------------------------- */
TI_API size_t ti_type_row_size(int type, int64_t n); /* bytes for n elements */
TI_API void ti_quantize_row(int type, const float *x, void *out, int64_t n);
TI_API void ti_dequantize_row(int type, const void *x, float *out, int64_t n);
TI_API uint16_t ti_fp32_to_fp16(float f);
TI_API float ti_fp16_to_fp32(uint16_t h);

/* y[t][r] = sum_c W[r][c] * x[t][c]   (W stored row-major in `type`)
 * Activations are quantised to Q8_0 internally for Q8_0/Q4_0 weights. */
TI_API void ti_matmul(int type, const void *w, const float *x, float *y,
                      int rows, int cols, int n_tok);
TI_API void ti_set_threads(int n_threads);
TI_API int ti_get_threads(void);
/* Which SIMD path was compiled in: "neon-dotprod", "neon", "avx2", "scalar". */
TI_API const char *ti_simd_path(void);

/* ---- sampling ------------------------------------------------------------ */
typedef struct ti_sampler_params {
    float temperature;      /* <= 0 -> greedy */
    int top_k;              /* <= 0 -> disabled */
    float top_p;            /* >= 1 -> disabled */
    float repeat_penalty;   /* 1 -> disabled */
    int repeat_last_n;      /* window of previous tokens penalised */
    uint64_t seed;
} ti_sampler_params;

TI_API ti_sampler_params ti_sampler_default(void);
/* Samples a token. `logits` is modified in place (penalty / temperature).
 * rng_state is a xorshift64* state (non-zero). */
TI_API int ti_sample(float *logits, int n_vocab, const ti_sampler_params *p,
                     const int32_t *history, int n_history, uint64_t *rng_state);

/* ---- speculative decoding --------------------------------------------------
 * Draft-then-verify decoding (Leviathan et al. 2023; Chen et al. 2023). The
 * draft proposes k tokens x_i ~ q_i; the target scores all of them in one
 * batched forward; x_i is kept with probability min(1, p_i(x_i) / q_i(x_i)),
 * and the first rejected position is resampled from norm(max(0, p_i - q_i)).
 * If all k are accepted a bonus token is drawn from p_{k+1}. The output
 * sequence is distributed exactly as sampling from the target alone. */
typedef struct ti_spec_stats {
    int rounds;       /* target verification calls */
    int drafted;      /* draft tokens proposed */
    int accepted;     /* draft tokens accepted */
    int generated;    /* tokens emitted */
    double t_prefill, t_draft, t_target, t_total; /* seconds (t_total includes prefill) */
} ti_spec_stats;

/* Dense next-token distribution under (temperature, top_k, top_p); greedy
 * (temperature <= 0) gives a one-hot argmax. Repetition penalty is ignored. */
TI_API void ti_sampler_probs(const float *logits, int n, const ti_sampler_params *p, float *probs);
/* Draw from a dense distribution with the xorshift64* state. */
TI_API int ti_sample_from(const float *probs, int n, uint64_t *rng);
/* The acceptance rule for one drafted token. Returns the emitted token and
 * sets *accepted. `scratch` must hold n floats. */
TI_API int ti_spec_accept(const float *p, const float *q, int n, int draft_token, float *scratch,
                          uint64_t *rng, int *accepted);
/* Plain autoregressive generation through the same sampling path (baseline).
 * Writes up to n_predict tokens to out, returns the count (<0 on error). */
TI_API int ti_generate(ti_model *m, const int32_t *prompt, int n_prompt, int n_predict,
                       const ti_sampler_params *p, int stop_at_eos, int32_t *out,
                       ti_spec_stats *stats);
TI_API int ti_spec_generate(ti_model *target, ti_model *draft, const int32_t *prompt,
                            int n_prompt, int n_predict, int k, const ti_sampler_params *p,
                            int stop_at_eos, int32_t *out, ti_spec_stats *stats);

#ifdef __cplusplus
}
#endif
#endif
