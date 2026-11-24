/* Platform layer: file mapping, timers, a tiny thread pool, error strings. */
#ifndef TI_PLATFORM_H
#define TI_PLATFORM_H

#include <stddef.h>
#include <stdint.h>

typedef struct ti_mmap {
    void *data;
    size_t size;
    void *handle_file; /* Windows only */
    void *handle_map;  /* Windows only */
} ti_mmap;

int ti_mmap_open(ti_mmap *m, const char *path);
void ti_mmap_close(ti_mmap *m);

double ti_time_sec(void);
int ti_num_cores(void);

void ti_set_error(const char *fmt, ...);

/* parallel for: fn(ctx, start, end) over [0, n) in chunks of `grain`. The
 * calling thread participates. Safe to call with a single thread. */
typedef void (*ti_task_fn)(void *ctx, int start, int end);
void ti_parallel_for(int n, int grain, ti_task_fn fn, void *ctx);
int ti_pool_threads(void);

#endif
