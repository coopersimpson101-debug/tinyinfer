/*
 * Platform layer.
 *
 * Thread pool design: N-1 persistent workers + the calling thread. A job is
 * published by bumping an atomic generation counter (odd while the job slot
 * is being rewritten, even once published); workers spin briefly on it (cheap
 * wake-up for the ~200 matmuls per decoded token) and then fall back to
 * sleeping on a condition variable. Work is handed out dynamically with an
 * atomic chunk counter, and the caller waits for *items done*, not for every
 * worker to check in, so a descheduled worker never stalls a matmul.
 *
 * POSIX builds use pthreads; Windows uses native threads + CONDITION_VARIABLE
 * (no winpthreads dependency).
 */
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#endif
#endif
#include "platform.h"
#include "../include/tinyinfer.h"

#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#define TI_PAUSE() _mm_pause()
#elif defined(__aarch64__)
#define TI_PAUSE() __asm__ __volatile__("yield")
#else
#define TI_PAUSE() ((void)0)
#endif

/* ---------------------------------------------------------------- errors -- */
static char g_err[512];

void ti_set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof g_err, fmt, ap);
    va_end(ap);
}

TI_API const char *ti_last_error(void) { return g_err; }

/* ------------------------------------------------------------------ mmap -- */
#ifdef _WIN32
int ti_mmap_open(ti_mmap *m, const char *path) {
    memset(m, 0, sizeof *m);
    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE) {
        ti_set_error("cannot open '%s'", path);
        return -1;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(f, &sz) || sz.QuadPart == 0) {
        CloseHandle(f);
        ti_set_error("cannot stat '%s'", path);
        return -1;
    }
    HANDLE map = CreateFileMappingA(f, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!map) {
        CloseHandle(f);
        ti_set_error("CreateFileMapping failed for '%s'", path);
        return -1;
    }
    void *p = MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0);
    if (!p) {
        CloseHandle(map);
        CloseHandle(f);
        ti_set_error("MapViewOfFile failed for '%s'", path);
        return -1;
    }
    m->data = p;
    m->size = (size_t)sz.QuadPart;
    m->handle_file = f;
    m->handle_map = map;
    return 0;
}

void ti_mmap_close(ti_mmap *m) {
    if (m->data) UnmapViewOfFile(m->data);
    if (m->handle_map) CloseHandle((HANDLE)m->handle_map);
    if (m->handle_file) CloseHandle((HANDLE)m->handle_file);
    memset(m, 0, sizeof *m);
}
#else
int ti_mmap_open(ti_mmap *m, const char *path) {
    memset(m, 0, sizeof *m);
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        ti_set_error("cannot open '%s'", path);
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size == 0) {
        close(fd);
        ti_set_error("cannot stat '%s'", path);
        return -1;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        ti_set_error("mmap failed for '%s'", path);
        return -1;
    }
    m->data = p;
    m->size = (size_t)st.st_size;
    return 0;
}

void ti_mmap_close(ti_mmap *m) {
    if (m->data) munmap(m->data, m->size);
    memset(m, 0, sizeof *m);
}
#endif

/* ----------------------------------------------------------------- timer -- */
double ti_time_sec(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    LARGE_INTEGER t;
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
#endif
}

int ti_num_cores(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

/* ------------------------------------------------------------ threadpool -- */
#define TI_MAX_THREADS 64
#define TI_SPIN_ITERS 20000

#ifdef _WIN32
typedef HANDLE ti_thread_t;
typedef CRITICAL_SECTION ti_mutex_t;
typedef CONDITION_VARIABLE ti_cond_t;
#define ti_mutex_init(m) InitializeCriticalSection(m)
#define ti_mutex_lock(m) EnterCriticalSection(m)
#define ti_mutex_unlock(m) LeaveCriticalSection(m)
#define ti_cond_init(c) InitializeConditionVariable(c)
#define ti_cond_wait(c, m) SleepConditionVariableCS(c, m, INFINITE)
#define ti_cond_broadcast(c) WakeAllConditionVariable(c)
#define ti_yield() SwitchToThread()
#else
typedef pthread_t ti_thread_t;
typedef pthread_mutex_t ti_mutex_t;
typedef pthread_cond_t ti_cond_t;
#define ti_mutex_init(m) pthread_mutex_init(m, NULL)
#define ti_mutex_lock(m) pthread_mutex_lock(m)
#define ti_mutex_unlock(m) pthread_mutex_unlock(m)
#define ti_cond_init(c) pthread_cond_init(c, NULL)
#define ti_cond_wait(c, m) pthread_cond_wait(c, m)
#define ti_cond_broadcast(c) pthread_cond_broadcast(c)
#define ti_yield() sched_yield()
#endif

typedef struct {
    int n_workers; /* excluding the caller */
    int requested; /* total threads requested (>=1) */
    ti_thread_t threads[TI_MAX_THREADS];
    ti_mutex_t mu;
    ti_cond_t cv;
    atomic_int sleepers;
    /* even = job published, odd = main thread is (re)writing the job */
    atomic_uint generation;
    atomic_int active; /* workers currently inside a job */
    atomic_int next;   /* next item index to hand out */
    atomic_int done;   /* items completed */
    atomic_int quit;
    /* current job (only written while generation is odd and active == 0) */
    ti_task_fn fn;
    void *ctx;
    int n, grain;
} ti_pool;

static ti_pool g_pool;
static int g_pool_ready = 0;
static int g_threads_wanted = 0; /* 0 = auto */

static void run_chunks(ti_task_fn fn, void *ctx, int n, int grain) {
    for (;;) {
        int s = atomic_fetch_add_explicit(&g_pool.next, grain, memory_order_relaxed);
        if (s >= n) break;
        int e = s + grain < n ? s + grain : n;
        fn(ctx, s, e);
        atomic_fetch_add_explicit(&g_pool.done, e - s, memory_order_release);
    }
}

#ifdef _WIN32
static DWORD WINAPI worker_main(LPVOID arg)
#else
static void *worker_main(void *arg)
#endif
{
    (void)arg;
    unsigned seen = 0;
    for (;;) {
        unsigned g;
        int spins = 0;
        /* wait for a new, fully published (even) generation */
        while (((g = atomic_load(&g_pool.generation)) == seen || (g & 1u)) &&
               !atomic_load(&g_pool.quit)) {
            if (++spins < TI_SPIN_ITERS) {
                TI_PAUSE();
                continue;
            }
            ti_mutex_lock(&g_pool.mu);
            atomic_fetch_add(&g_pool.sleepers, 1);
            while (((g = atomic_load(&g_pool.generation)) == seen || (g & 1u)) &&
                   !atomic_load(&g_pool.quit))
                ti_cond_wait(&g_pool.cv, &g_pool.mu);
            atomic_fetch_sub(&g_pool.sleepers, 1);
            ti_mutex_unlock(&g_pool.mu);
            spins = 0;
        }
        if (atomic_load(&g_pool.quit)) break;
        /* Dekker-style handshake with ti_parallel_for: announce ourselves, then
         * re-check that the job did not change before touching its fields. */
        atomic_fetch_add(&g_pool.active, 1);
        if (atomic_load(&g_pool.generation) == g)
            run_chunks(g_pool.fn, g_pool.ctx, g_pool.n, g_pool.grain);
        atomic_fetch_sub(&g_pool.active, 1);
        seen = g;
    }
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

static void pool_shutdown(void) {
    if (!g_pool_ready) return;
    atomic_store(&g_pool.quit, 1);
    ti_mutex_lock(&g_pool.mu);
    ti_cond_broadcast(&g_pool.cv);
    ti_mutex_unlock(&g_pool.mu);
    for (int i = 0; i < g_pool.n_workers; i++) {
#ifdef _WIN32
        WaitForSingleObject(g_pool.threads[i], INFINITE);
        CloseHandle(g_pool.threads[i]);
#else
        pthread_join(g_pool.threads[i], NULL);
#endif
    }
    g_pool_ready = 0;
}

static void pool_start(int total) {
    if (total < 1) total = 1;
    if (total > TI_MAX_THREADS) total = TI_MAX_THREADS;
    memset(&g_pool, 0, sizeof g_pool);
    ti_mutex_init(&g_pool.mu);
    ti_cond_init(&g_pool.cv);
    g_pool.requested = total;
    g_pool.n_workers = 0;
    for (int i = 0; i < total - 1; i++) {
#ifdef _WIN32
        HANDLE h = CreateThread(NULL, 0, worker_main, NULL, 0, NULL);
        if (!h) break;
        g_pool.threads[i] = h;
#else
        if (pthread_create(&g_pool.threads[i], NULL, worker_main, NULL) != 0) break;
#endif
        g_pool.n_workers++;
    }
    g_pool_ready = 1;
}

static void pool_ensure(void) {
    int want = g_threads_wanted > 0 ? g_threads_wanted : ti_num_cores();
    if (g_pool_ready && g_pool.requested == want) return;
    if (g_pool_ready) {
        pool_shutdown();
    } else {
        static int registered = 0;
        if (!registered) {
            atexit(pool_shutdown);
            registered = 1;
        }
    }
    pool_start(want);
}

TI_API void ti_set_threads(int n_threads) {
    g_threads_wanted = n_threads;
    if (g_pool_ready) pool_ensure();
}

TI_API int ti_get_threads(void) {
    pool_ensure();
    return g_pool.n_workers + 1;
}

int ti_pool_threads(void) { return ti_get_threads(); }

void ti_parallel_for(int n, int grain, ti_task_fn fn, void *ctx) {
    if (n <= 0) return;
    if (grain < 1) grain = 1;
    pool_ensure();
    if (g_pool.n_workers == 0 || n <= grain) {
        fn(ctx, 0, n);
        return;
    }
    /* 1. mark the job slot as being rewritten (odd generation) */
    atomic_fetch_add(&g_pool.generation, 1);
    /* 2. wait for stragglers still inside the previous job (they find no work
     *    left and leave almost immediately) */
    int spins = 0;
    while (atomic_load(&g_pool.active) != 0) {
        if (++spins > TI_SPIN_ITERS) ti_yield();
        else TI_PAUSE();
    }
    g_pool.fn = fn;
    g_pool.ctx = ctx;
    g_pool.n = n;
    g_pool.grain = grain;
    atomic_store(&g_pool.next, 0);
    atomic_store(&g_pool.done, 0);
    /* 3. publish (even generation) */
    atomic_fetch_add(&g_pool.generation, 1);
    if (atomic_load(&g_pool.sleepers) > 0) {
        ti_mutex_lock(&g_pool.mu);
        ti_cond_broadcast(&g_pool.cv);
        ti_mutex_unlock(&g_pool.mu);
    }
    run_chunks(fn, ctx, n, grain);
    /* 4. wait until every item is finished (not for every worker to check in:
     *    a descheduled worker must not stall the whole forward pass) */
    spins = 0;
    while (atomic_load_explicit(&g_pool.done, memory_order_acquire) < n) {
        if (++spins < TI_SPIN_ITERS) TI_PAUSE();
        else ti_yield();
    }
}
