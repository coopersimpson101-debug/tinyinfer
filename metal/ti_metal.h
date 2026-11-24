/* Metal (Apple GPU) backend. Only compiled on macOS with -DTI_METAL. */
#ifndef TI_METAL_H
#define TI_METAL_H

#include "../src/model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Upload/wrap the weights and build pipelines. Returns NULL on failure
 * (message in ti_last_error()). */
void *ti_metal_init(ti_model *m);
void ti_metal_free(void *ctx);

/* One token at position pos. Writes vocab_size logits if logits != NULL. */
int ti_metal_forward(void *ctx, ti_model *m, int token, int pos, float *logits);

#ifdef __cplusplus
}
#endif
#endif
