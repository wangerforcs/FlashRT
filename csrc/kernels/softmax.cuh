// ================================================================
// FlashVLA — Softmax kernel (FP16)
// Port of pi05 softmax_bf16_kernel. In-place row-wise softmax.
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

void softmax_fp16(__half* data, int rows, int cols, cudaStream_t stream = 0);

// Softmax with state token masking: first `mask_rows` rows have cols [mask_start, cols) set to -inf.
// Eliminates separate mask kernel launch. Used by Pi0 state-masked attention.
// mask_rows: first N rows get mask_start applied
// mask_start: state token's key limit (enc_seq+1)
// pad_start: actual S_kv before padding (for pad column masking on all rows)
void softmax_state_masked_fp16(__half* data, int rows, int cols,
                                int mask_rows, int mask_start, int pad_start,
                                cudaStream_t stream = 0);
