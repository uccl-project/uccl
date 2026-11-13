// NFH: lossless BF16 codec for the P2P compressed-write path.
//
// The sign + 7-bit mantissa byte of every value is stored verbatim (the split
// plane, sent first). Exponents are coded with a length-limited canonical
// Huffman code over a 16-value window of the buffer's most frequent exponents
// plus one escape symbol followed by the raw exponent byte. One warp decodes
// one chunk of B values; the 32 lane bit streams of a chunk are stored
// interleaved in the order the decoder refills them, so each chunk needs one
// 32-bit offset and no per-lane state. The chunk size B (2048..16384) is a
// function of n, so small messages still spread over enough warps.
//
// Compressed layout for n BF16 values (byte offsets from the buffer start):
//   [0, n)          split plane: (x >> 8 & 0x80) | (x & 0x7f) per value
//   H = align16(n)  64-byte header (magic, n, window, 17 code lengths)
//   H + 64          (C + 1) uint32 chunk word offsets, C = n / B
//   ...             exponent bytes of the n - B * C tail values
//   align16(...)    interleaved code words
// A compressed size of exactly 2n means the stored form: [split | exponents].
// The encoder falls back to it whenever the coded form would not be smaller,
// so the output never exceeds the input size.
#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  NFH_OK = 0,
  NFH_ERR_ARG = 1,
  NFH_ERR_CUDA = 2,
  NFH_ERR_FORMAT = 3,
};

typedef struct nfh_encoder nfh_encoder;
typedef struct nfh_decoder nfh_decoder;

// Encoder / decoder state for the current device. Kernel grids are sized for
// the SMs of the context current at creation (a green context's partition).
// The encoder owns a fixed workspace (about 300 MB) independent of the
// message size.
int nfh_encoder_create(nfh_encoder** enc);
void nfh_encoder_destroy(nfh_encoder* enc);
int nfh_decoder_create(nfh_decoder** dec);
void nfh_decoder_destroy(nfh_decoder* dec);

// Phase 1: write the split plane of n values to out[0, n). Asynchronous.
int nfh_split(nfh_encoder* enc, void const* in, size_t n, void* out,
              cudaStream_t stream);

// Phase 2: write everything after the split plane. out must hold 2n bytes
// and be 16-byte aligned. Synchronizes the stream; *out_bytes receives the
// total compressed size including the split plane (always <= 2n).
int nfh_encode(nfh_encoder* enc, void const* in, size_t n, void* out,
               cudaStream_t stream, size_t* out_bytes);

// nfh_encode in two halves: begin queues the work (it does not read the split
// plane, so it may run on another stream concurrently with nfh_split); end
// synchronizes that stream and returns the size. One encode in flight per
// encoder; in must stay unchanged until end returns.
int nfh_encode_begin(nfh_encoder* enc, void const* in, size_t n, void* out,
                     cudaStream_t stream);
int nfh_encode_end(nfh_encoder* enc, cudaStream_t stream, size_t* out_bytes);

// Write the stored form's exponent plane to out[n, 2n) (after nfh_split the
// buffer then holds the 2n-byte stored form). Asynchronous.
int nfh_store(nfh_encoder* enc, void const* in, size_t n, void* out,
              cudaStream_t stream);

// Decode in_bytes of compressed data (16-byte aligned) into n values.
// Asynchronous. A malformed header leaves the output undefined and stores
// NFH_ERR_FORMAT into *status (device-writable, may be NULL); *status is
// never written on success.
int nfh_decode(nfh_decoder* dec, void const* in, size_t in_bytes, void* out,
               size_t n, cudaStream_t stream, int* status);

char const* nfh_error_string(int err);

// Testing hook: force the chunk size (power of two in [2048, 16384]; 0 restores
// the size rule). Both ends must use the same setting.
void nfh_set_chunk_override(int chunk);

#ifdef __cplusplus
}
#endif
