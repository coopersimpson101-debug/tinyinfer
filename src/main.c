/*
 * tinyinfer CLI.
 *
 *   tinyinfer -m model.gguf -p "prompt" -n 128 [--chat] [--temp 0.7] ...
 *   tinyinfer -m model.gguf --bench 512,128      (llama-bench style pp/tg)
 *
 * Generated text goes to stdout; model info and timings go to stderr.
 */
#include "../include/tinyinfer.h"
#include "model.h"
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#ifdef TI_METAL
#include "../metal/ti_metal.h"
#endif

static void usage(void) {
    fprintf(stderr,
            "usage: tinyinfer -m model.gguf [options]\n"
            "  -p, --prompt TEXT       prompt (default: \"Once upon a time\")\n"
            "  -f, --file PATH         read the prompt from a file\n"
            "  -n, --n-predict N       tokens to generate (default 128)\n"
            "  -t, --threads N         CPU threads (default: all cores)\n"
            "  -c, --ctx N             context length / KV cache size (default min(train, 2048))\n"
            "  --temp F                temperature, 0 = greedy (default 0.7)\n"
            "  --top-k N               (default 40, 0 = off)\n"
            "  --top-p F               (default 0.9, 1 = off)\n"
            "  --repeat-penalty F      (default 1.1, 1 = off)\n"
            "  --repeat-last-n N       (default 64)\n"
            "  -s, --seed N            RNG seed\n"
            "  --chat                  wrap the prompt in the Qwen chat template\n"
            "  --system TEXT           system prompt used with --chat\n"
            "  --ignore-eos            keep generating after end-of-sequence\n"
            "  --bench PP,TG           benchmark: prefill PP tokens, generate TG tokens\n"
            "  -r, --reps N            repetitions for --bench (default 3)\n"
            "  --metal                 run on the GPU (macOS builds with TI_METAL)\n"
            "  --dump-logits FILE      write the prompt's last-position logits (float32)\n"
            "  -v, --verbose           print model information\n");
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1);
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) {
        free(b);
        b = NULL;
    }
    if (b) b[n] = 0;
    fclose(f);
    return b;
}

static const char *type_name(int t) {
    switch (t) {
    case TI_TYPE_F32: return "F32";
    case TI_TYPE_F16: return "F16";
    case TI_TYPE_BF16: return "BF16";
    case TI_TYPE_Q8_0: return "Q8_0";
    case TI_TYPE_Q4_0: return "Q4_0";
    default: return "?";
    }
}

static int is_stop(const ti_model *m, int tok) {
    return tok == m->tok.eos_id || tok == m->tok.eot_id ||
           tok == ti_token_id(m, "<|endoftext|>");
}

static int run_bench(ti_model *m, int pp, int tg, int reps) {
    const ti_config *c = ti_model_config(m);
    /* like llama-bench: ppN = N-token prompt from an empty cache, tgN = N
     * single-token decode steps, also starting from an empty cache */
    int need = pp > tg ? pp : tg;
    if (need > c->max_ctx) {
        fprintf(stderr, "bench needs ctx >= %d (use -c)\n", need);
        return 1;
    }
    float *logits = (float *)malloc(sizeof(float) * (size_t)c->vocab_size);
    int32_t *toks = (int32_t *)malloc(sizeof(int32_t) * (size_t)(pp > 0 ? pp : 1));
    if (!logits || !toks) return 1;
    uint64_t s = 1234567;
    for (int i = 0; i < pp; i++) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        toks[i] = (int32_t)((s >> 33) % (uint64_t)c->vocab_size);
    }
    double best_pp = 0, best_tg = 0, sum_pp = 0, sum_tg = 0;
    for (int r = 0; r < reps; r++) {
        double t0 = ti_time_sec();
        if (pp > 0 && ti_forward(m, toks, pp, 0, logits, 0) != 0) return 1;
        double t1 = ti_time_sec();
        int32_t tok = 0;
        for (int i = 0; i < tg; i++) {
            if (ti_forward(m, &tok, 1, i, logits, 0) != 0) return 1;
            int best = 0;
            for (int v = 1; v < c->vocab_size; v++)
                if (logits[v] > logits[best]) best = v;
            tok = best;
        }
        double t2 = ti_time_sec();
        double pps = pp > 0 ? pp / (t1 - t0) : 0, tgs = tg > 0 ? tg / (t2 - t1) : 0;
        sum_pp += pps;
        sum_tg += tgs;
        if (pps > best_pp) best_pp = pps;
        if (tgs > best_tg) best_tg = tgs;
        fprintf(stderr, "rep %d: pp%d %.2f tok/s, tg%d %.2f tok/s\n", r, pp, pps, tg, tgs);
    }
    printf("| backend | threads | test | t/s (mean) | t/s (best) |\n|---|---|---|---|---|\n");
    const char *be = m->metal ? "metal" : ti_simd_path();
    if (pp > 0) printf("| %s | %d | pp%d | %.2f | %.2f |\n", be, ti_get_threads(), pp, sum_pp / reps, best_pp);
    if (tg > 0) printf("| %s | %d | tg%d | %.2f | %.2f |\n", be, ti_get_threads(), tg, sum_tg / reps, best_tg);
    free(logits);
    free(toks);
    return 0;
}

int main(int argc, char **argv) {
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
    SetConsoleOutputCP(CP_UTF8);
#endif
    const char *model_path = NULL, *prompt = "Once upon a time", *prompt_file = NULL;
    const char *system = "You are a helpful assistant.";
    const char *dump_logits = NULL;
    int n_predict = 128, threads = 0, ctx = 0, chat = 0, ignore_eos = 0, verbose = 0;
    int bench_pp = -1, bench_tg = -1, reps = 3, use_metal = 0;
    ti_sampler_params sp = ti_sampler_default();
    int seed_set = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
#define NEXT() (i + 1 < argc ? argv[++i] : (usage(), exit(1), (char *)NULL))
        if (!strcmp(a, "-m") || !strcmp(a, "--model")) model_path = NEXT();
        else if (!strcmp(a, "-p") || !strcmp(a, "--prompt")) prompt = NEXT();
        else if (!strcmp(a, "-f") || !strcmp(a, "--file")) prompt_file = NEXT();
        else if (!strcmp(a, "-n") || !strcmp(a, "--n-predict")) n_predict = atoi(NEXT());
        else if (!strcmp(a, "-t") || !strcmp(a, "--threads")) threads = atoi(NEXT());
        else if (!strcmp(a, "-c") || !strcmp(a, "--ctx")) ctx = atoi(NEXT());
        else if (!strcmp(a, "--temp")) sp.temperature = (float)atof(NEXT());
        else if (!strcmp(a, "--top-k")) sp.top_k = atoi(NEXT());
        else if (!strcmp(a, "--top-p")) sp.top_p = (float)atof(NEXT());
        else if (!strcmp(a, "--repeat-penalty")) sp.repeat_penalty = (float)atof(NEXT());
        else if (!strcmp(a, "--repeat-last-n")) sp.repeat_last_n = atoi(NEXT());
        else if (!strcmp(a, "-s") || !strcmp(a, "--seed")) { sp.seed = strtoull(NEXT(), NULL, 10); seed_set = 1; }
        else if (!strcmp(a, "--chat")) chat = 1;
        else if (!strcmp(a, "--system")) system = NEXT();
        else if (!strcmp(a, "--ignore-eos")) ignore_eos = 1;
        else if (!strcmp(a, "-r") || !strcmp(a, "--reps")) reps = atoi(NEXT());
        else if (!strcmp(a, "--metal")) use_metal = 1;
        else if (!strcmp(a, "-v") || !strcmp(a, "--verbose")) verbose = 1;
        else if (!strcmp(a, "--dump-logits")) dump_logits = NEXT();
        else if (!strcmp(a, "--bench")) {
            const char *b = NEXT();
            if (sscanf(b, "%d,%d", &bench_pp, &bench_tg) != 2) { usage(); return 1; }
        } else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(); return 0; }
        else { fprintf(stderr, "unknown argument '%s'\n", a); usage(); return 1; }
#undef NEXT
    }
    if (!model_path) { usage(); return 1; }
    if (!seed_set) sp.seed = (uint64_t)(ti_time_sec() * 1e6) | 1;
    if (reps < 1) reps = 1;
    if (bench_pp >= 0 && ctx <= 0) ctx = (bench_pp > bench_tg ? bench_pp : bench_tg) + 1;

    double tl0 = ti_time_sec();
    ti_model *m = ti_model_load(model_path, ctx, threads);
    if (!m) {
        fprintf(stderr, "error: %s\n", ti_last_error());
        return 1;
    }
    double tl1 = ti_time_sec();
    const ti_config *c = ti_model_config(m);
    if (use_metal) {
#ifdef TI_METAL
        m->metal = ti_metal_init(m);
        if (!m->metal) {
            fprintf(stderr, "error: Metal init failed: %s\n", ti_last_error());
            ti_model_free(m);
            return 1;
        }
#else
        fprintf(stderr, "error: this binary was built without Metal (make metal on macOS)\n");
        ti_model_free(m);
        return 1;
#endif
    }
    if (verbose || bench_pp >= 0) {
        fprintf(stderr,
                "[tinyinfer] qwen2: layers=%d dim=%d ffn=%d heads=%d kv_heads=%d head_dim=%d "
                "vocab=%d ctx=%d rope_theta=%g tied=%d bias=%d weights=%s simd=%s threads=%d "
                "backend=%s load=%.0fms\n",
                c->n_layers, c->dim, c->ffn_dim, c->n_heads, c->n_kv_heads, c->head_dim,
                c->vocab_size, c->max_ctx, c->rope_theta, c->tied_embeddings, c->has_qkv_bias,
                type_name(m->layers[0].wq.type), ti_simd_path(), ti_get_threads(),
                m->metal ? "metal" : "cpu", (tl1 - tl0) * 1e3);
    }
    if (bench_pp >= 0) {
        int rc = run_bench(m, bench_pp, bench_tg, reps);
        ti_model_free(m);
        return rc;
    }

    char *file_text = NULL;
    if (prompt_file) {
        file_text = read_file(prompt_file);
        if (!file_text) {
            fprintf(stderr, "error: cannot read %s\n", prompt_file);
            return 1;
        }
        prompt = file_text;
    }
    char *full = NULL;
    if (chat) {
        size_t n = strlen(prompt) + strlen(system) + 128;
        full = (char *)malloc(n);
        snprintf(full, n,
                 "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n"
                 "<|im_start|>assistant\n",
                 system, prompt);
        prompt = full;
    }
    int max_tok = (int)strlen(prompt) + 8;
    int32_t *toks = (int32_t *)malloc(sizeof(int32_t) * (size_t)(max_tok + n_predict + 1));
    int n_prompt = ti_tokenize(m, prompt, toks, max_tok, 1);
    if (n_prompt <= 0) {
        fprintf(stderr, "error: empty prompt after tokenisation\n");
        return 1;
    }
    if (n_prompt + n_predict > c->max_ctx) {
        n_predict = c->max_ctx - n_prompt;
        if (n_predict <= 0) {
            fprintf(stderr, "error: prompt (%d tokens) exceeds context %d\n", n_prompt, c->max_ctx);
            return 1;
        }
    }
    float *logits = (float *)malloc(sizeof(float) * (size_t)c->vocab_size);
    if (!chat) {
        char *echo = (char *)malloc(strlen(prompt) * 2 + 16);
        ti_detokenize(m, toks, n_prompt, echo, (int)strlen(prompt) * 2 + 16);
        fputs(echo, stdout);
        fflush(stdout);
        free(echo);
    }
    double t0 = ti_time_sec();
    if (ti_forward(m, toks, n_prompt, 0, logits, 0) != 0) {
        fprintf(stderr, "error: %s\n", ti_last_error());
        return 1;
    }
    double t1 = ti_time_sec();
    if (dump_logits) { /* last-position prompt logits as raw float32 (for tests) */
        FILE *df = fopen(dump_logits, "wb");
        if (df) {
            fwrite(logits, sizeof(float), (size_t)c->vocab_size, df);
            fclose(df);
        }
    }
    uint64_t rng = sp.seed;
    int n_tok = n_prompt, n_gen = 0, n_decode = 0;
    char piece[256];
    for (; n_gen < n_predict; n_gen++) {
        int next = ti_sample(logits, c->vocab_size, &sp, toks, n_tok, &rng);
        toks[n_tok++] = next;
        if (!ignore_eos && is_stop(m, next)) {
            n_gen++;
            break;
        }
        int nb = ti_tok_piece_bytes(&m->tok, next, piece, (int)sizeof piece);
        fwrite(piece, 1, (size_t)(nb < (int)sizeof piece ? nb : (int)sizeof piece), stdout);
        fflush(stdout);
        if (n_gen + 1 == n_predict) {
            n_gen++;
            break;
        }
        if (ti_forward(m, &toks[n_tok - 1], 1, n_tok - 1, logits, 0) != 0) {
            fprintf(stderr, "\nerror: %s\n", ti_last_error());
            return 1;
        }
        n_decode++;
    }
    double t2 = ti_time_sec();
    fputc('\n', stdout);
    double pp_ms = (t1 - t0) * 1e3, tg_ms = (t2 - t1) * 1e3;
    fprintf(stderr,
            "\n[tinyinfer] prefill: %d tokens in %.1f ms (%.2f tok/s)\n"
            "[tinyinfer] decode:  %d tokens in %.1f ms (%.2f tok/s)\n",
            n_prompt, pp_ms, n_prompt / (t1 - t0), n_decode, tg_ms,
            n_decode > 0 ? n_decode / (t2 - t1) : 0.0);
    free(logits);
    free(toks);
    free(full);
    free(file_text);
    ti_model_free(m);
    return 0;
}
