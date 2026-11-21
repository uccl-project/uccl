#include "nfh_codec.h"
#include <cub/device/device_scan.cuh>
#include <algorithm>
#include <cstring>
#include <new>
#include <cuda.h>

namespace {

constexpr int P = 12;  // LUT peek bits
constexpr int LUTN = 1 << P;
constexpr int NSYM = 17;  // 16 window exponents + escape
constexpr int ESC = 16;
constexpr int MAXLEN = 10;         // code length limit (<= P)
constexpr int BMAX = 16384;        // values per chunk (one warp), largest
constexpr int BMIN = 2048;         // smallest chunk
constexpr int WPB = 4;             // encoder warps per block
constexpr int DWPB = 8;            // decoder warps per block (share one LUT)
constexpr int REFILL = 32;         // refill one word when fewer bits remain
constexpr int NLOOK = 3;           // lookups per step
constexpr int GATE = 18;           // most bits one lookup consumes
constexpr int RROWS = 8;           // decoder ring rows per warp
constexpr int PFG = 4;             // split-plane rows prefetched ahead
constexpr size_t SEGV = 1u << 26;  // values per encoder segment
constexpr size_t TARGET_CHUNKS = 4096;  // chunk count the size rule aims for
constexpr int RULE_MIN =
    4096;  // smallest chunk the size rule picks (2048 costs ~0.3 bit/w)
constexpr int HIST_SHIFT = 4;  // histogram samples 1 of 2^shift 4 KB blocks
constexpr uint32_t kMagic = 0x3148464eu;  // "NFH1"

struct Header {
  uint32_t magic, version;
  uint64_t n;
  uint32_t chunk, nchunks, tail, lo;
  uint8_t len[NSYM];
  uint8_t pad[64 - 32 - NSYM];
};
static_assert(sizeof(Header) == 64, "header is 64 bytes");

// Slot words per chunk: per-lane bound (every value at the longest code) + 2,
// odd.
__host__ __device__ inline int sw_of(int B) {
  return 32 * ((((B / 32) * (MAXLEN + 8) + 31) / 32 + 2) | 1);
}

int g_chunk_override = 0;

// Chunk size from n alone (sender and receiver agree without exchanging it):
// the largest power of two in [RULE_MIN, BMAX] that still yields TARGET_CHUNKS
// chunks, so small messages keep enough warps busy.
inline int chunk_for(size_t n) {
  if (g_chunk_override) return g_chunk_override;
  int B = BMAX;
  while (B > RULE_MIN && n / (size_t)B < TARGET_CHUNKS) B >>= 1;
  return B;
}

__host__ __device__ inline size_t hdr_off(size_t n) {
  return (n + 15) & ~size_t(15);
}
__host__ __device__ inline size_t tab_off(size_t n) {
  return hdr_off(n) + sizeof(Header);
}
__host__ __device__ inline size_t tail_off(size_t n, size_t C) {
  return tab_off(n) + (C + 1) * 4;
}
__host__ __device__ inline size_t words_off(size_t n, size_t C, size_t tail) {
  return (tail_off(n, C) + tail + 15) & ~size_t(15);
}

__device__ __forceinline__ uint32_t lane_lt() {
  uint32_t m;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
  return m;
}
__device__ __forceinline__ uint32_t split_byte(uint32_t x) {
  return ((x >> 8) & 0x80u) | (x & 0x7Fu);
}
__device__ __forceinline__ uint16_t join(uint32_t r, uint32_t e) {
  return (uint16_t)(((r & 0x80u) << 8) | ((e & 0xFFu) << 7) | (r & 0x7Fu));
}

// ---------------- code tables, built on the device from the header
// ---------------- LUT entry meta: [0,16) 4 window symbols | [16,19) count |
// [19,23) total bits | [23,27) first length | [27] escape. The LUT word .x
// holds the decoded exponent bytes (zero beyond the count).
struct Tables {
  uint8_t ord[NSYM], idx[MAXLEN + 1], cnt[MAXLEN + 1];
  uint16_t first[MAXLEN + 1], code[NSYM];
  int8_t tsym[1 << MAXLEN];
  uint8_t tlen[1 << MAXLEN];
};

__device__ bool header_ok(Header const* h, size_t n, int B, size_t C,
                          size_t tail) {
  if (h->magic != kMagic || h->version != 1 || h->n != n ||
      h->chunk != (uint32_t)B || h->nchunks != C || h->tail != tail ||
      h->lo > 240)
    return false;
  uint32_t kraft = 0;
  for (int s = 0; s < NSYM; ++s) {
    int l = h->len[s];
    if (l < 1 || l > MAXLEN) return false;
    kraft += 1u << (MAXLEN - l);
  }
  return kraft == (1u << MAXLEN);  // complete prefix code
}

// All threads of the block call this; ends with the tables ready in shared
// memory.
__device__ void build_tables(uint8_t const* len, Tables& t) {
  if (threadIdx.x == 0) {
    for (int l = 0; l <= MAXLEN; ++l) t.cnt[l] = 0;
    for (int s = 0; s < NSYM; ++s) t.cnt[len[s]]++;
    int k = 0;
    uint32_t code = 0;
    for (int l = 1; l <= MAXLEN; ++l) {
      code = (code + t.cnt[l - 1]) << 1;
      t.first[l] = (uint16_t)code;
      t.idx[l] = (uint8_t)k;
      for (int s = 0; s < NSYM; ++s)
        if (len[s] == l) {
          t.code[s] = (uint16_t)(code + (k - t.idx[l]));
          t.ord[k++] = (uint8_t)s;
        }
    }
  }
  __syncthreads();
  for (int x = threadIdx.x; x < (1 << MAXLEN); x += blockDim.x) {
    for (int l = 1; l <= MAXLEN; ++l) {
      uint32_t c = (uint32_t)x >> (MAXLEN - l);
      if (c >= t.first[l] && c < (uint32_t)t.first[l] + t.cnt[l]) {
        t.tsym[x] = (int8_t)t.ord[t.idx[l] + c - t.first[l]];
        t.tlen[x] = (uint8_t)l;
        break;
      }
    }
  }
  __syncthreads();
}

__device__ uint2 lut_entry(uint32_t w, Tables const& t, uint32_t lo) {
  int q = 0, k = 0, len1 = 0;
  uint32_t syms = 0, bytes = 0;
  bool esc = false;
  while (k < 4) {
    uint32_t x = ((w << q) & (LUTN - 1)) >> (P - MAXLEN);
    int hit = t.tsym[x], hl = t.tlen[x];
    if (hl > P - q) break;
    if (hit == ESC) {
      if (k == 0) {
        esc = true;
        len1 = hl;
      }
      break;
    }
    syms |= (uint32_t)hit << (4 * k);
    bytes |= ((uint32_t)hit + lo) << (8 * k);
    if (k == 0) len1 = hl;
    q += hl;
    k++;
  }
  uint32_t meta = syms | ((uint32_t)k << 16) | ((uint32_t)q << 19) |
                  ((uint32_t)len1 << 23) | ((uint32_t)esc << 27);
  return make_uint2(bytes, meta);
}

// Shared by the encoder's replay and the decoder: bits q and values c of one
// step.
__device__ __forceinline__ void step_plan(uint32_t ent, int r, int& q, int& c,
                                          bool& esc) {
  esc = (ent >> 27) & 1;
  if (esc) {
    q = ((ent >> 23) & 15) + 8;
    c = 1;
    return;
  }
  c = (ent >> 16) & 7;
  q = (ent >> 19) & 15;
  if (c > r) {
    c = 1;
    q = (ent >> 23) & 15;
  }
}

// ---------------- simple kernels ----------------
__global__ void split_vec_kernel(uint4 const* __restrict__ in, size_t n8,
                                 uint2* __restrict__ raw) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n8;
       i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = in[i];
    uint32_t w[4] = {v.x, v.y, v.z, v.w}, r[2] = {0, 0};
#pragma unroll
    for (int k = 0; k < 8; ++k)
      r[k >> 2] |= split_byte((w[k >> 1] >> (16 * (k & 1))) & 0xFFFFu)
                   << (8 * (k & 3));
    raw[i] = make_uint2(r[0], r[1]);
  }
}

__global__ void split_scalar_kernel(uint16_t const* __restrict__ in, size_t lo,
                                    size_t n, uint8_t* __restrict__ raw) {
  for (size_t i = lo + blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    raw[i] = (uint8_t)split_byte(in[i]);
}

// Exponent bytes of values [lo, n) to dst[0, n - lo).
__global__ void exp_kernel(uint16_t const* __restrict__ in, size_t lo, size_t n,
                           uint8_t* __restrict__ dst) {
  for (size_t i = lo + blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    dst[i - lo] = (uint8_t)(in[i] >> 7);
}

// out[i] = join(raw[i], exp[i - lo]) for i in [lo, n).
__global__ void join_kernel(uint8_t const* __restrict__ raw,
                            uint8_t const* __restrict__ exp, size_t lo,
                            size_t n, uint16_t* __restrict__ out) {
  for (size_t i = lo + blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    out[i] = join(raw[i], exp[i - lo]);
}

__global__ void set_status_kernel(int* status, int v) { *status = v; }

__global__ void hist_kernel(uint4 const* __restrict__ in, size_t n4,
                            unsigned long long* __restrict__ h) {
  __shared__ unsigned int sh[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) sh[i] = 0;
  __syncthreads();
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;;
       t += (size_t)gridDim.x * blockDim.x) {
    size_t i = ((t >> 8) << (8 + HIST_SHIFT)) | (t & 255);
    if (i >= n4) break;
    uint4 v = in[i];
    uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      atomicAdd(&sh[(w[j] >> 7) & 0xFF], 1u);
      atomicAdd(&sh[(w[j] >> 23) & 0xFF], 1u);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 256; i += blockDim.x)
    if (sh[i]) atomicAdd(&h[i], (unsigned long long)sh[i]);
}

// ---------------- encoder ----------------
// Window, code lengths (length-limited Huffman over 17 symbols) and encoder
// tables, all on the device from the sampled histogram. Writes the header into
// the output and tab[0, LUTN) LUT meta, tab[LUTN, LUTN + 256) per-exponent
// code | len << 24 (escape: esc code << 8 | e, len + 8). One block.
// Length-limited Huffman over the 17 symbols, one warp: lane i < 17 is leaf i.
// Each merge takes the two lightest live nodes; a leaf's length is the number
// of merges its subtree took part in. Over-long codes raise rare weights and
// retry.
__device__ void huffman_lengths_warp(float const* f0, uint8_t* out) {
  int lane = threadIdx.x & 31;
  float tot = 0;
  for (int i = 0; i < NSYM; ++i) tot += f0[i] > 0 ? f0[i] : 0.5f;
  float f = lane < NSYM ? (f0[lane] > 0 ? f0[lane] : 0.5f) : 0.f;
  for (int iter = 0; iter < 64; ++iter) {
    float w = lane < NSYM ? f : __int_as_float(0x7f800000);
    int grp = lane, depth = 0;
    for (int step = 0; step < NSYM - 1; ++step) {
      uint32_t key = __float_as_uint(w);
      uint32_t m1 = __reduce_min_sync(0xffffffffu, key);
      int a = __ffs(__ballot_sync(0xffffffffu, key == m1)) - 1;
      uint32_t key2 = lane == a ? 0xffffffffu : key;
      uint32_t m2 = __reduce_min_sync(0xffffffffu, key2);
      int b = __ffs(__ballot_sync(0xffffffffu, key2 == m2)) - 1;
      float wa = __shfl_sync(0xffffffffu, w, a),
            wb = __shfl_sync(0xffffffffu, w, b);
      if (lane == a) w = wa + wb;
      if (lane == b) w = __int_as_float(0x7f800000);
      if (lane < NSYM && (grp == a || grp == b)) {
        depth++;
        grp = a;
      }
    }
    int mx =
        (int)__reduce_max_sync(0xffffffffu, lane < NSYM ? (unsigned)depth : 0u);
    if (mx <= MAXLEN) {
      if (lane < NSYM) out[lane] = (uint8_t)depth;
      return;
    }
    f = fmaxf(f, tot / (float)(1 << (MAXLEN - 1)) * (1.f + iter));
  }
  // Not reached (the floor makes the code fit within a few rounds); stay
  // complete anyway.
  if (lane < NSYM) out[lane] = lane == ESC ? 1 : 5;
}

__global__ void __launch_bounds__(256)
    codebook_kernel(unsigned long long const* __restrict__ hist, size_t n,
                    int B, size_t C, size_t tail, Header* __restrict__ hdr_out,
                    uint32_t* __restrict__ tab) {
  __shared__ Tables t;
  __shared__ Header h;
  __shared__ unsigned long long hs[256], ws[256];
  __shared__ float fs[NSYM];
  int tid = threadIdx.x, lane = tid & 31;
  hs[tid] = hist[tid];
  __syncthreads();
  unsigned long long w = 0;
  if (tid <= 240)
    for (int j = 0; j < 16; ++j) w += hs[tid + j];
  ws[tid] = w;
  __syncthreads();
  if (tid < 32) {
    // window = first argmax of the 16-wide sums
    unsigned long long bw = ws[lane];
    int bl = lane;
    for (int l = lane + 32; l <= 240; l += 32)
      if (ws[l] > bw) {
        bw = ws[l];
        bl = l;
      }
    unsigned long long tot = 0;
    for (int i = lane; i < 256; i += 32) tot += hs[i];
    for (int o = 16; o; o >>= 1) {
      unsigned long long ow = __shfl_xor_sync(0xffffffffu, bw, o);
      int ol = __shfl_xor_sync(0xffffffffu, bl, o);
      if (ow > bw || (ow == bw && ol < bl)) {
        bw = ow;
        bl = ol;
      }
      tot += __shfl_xor_sync(0xffffffffu, tot, o);
    }
    if (lane < 16) fs[lane] = (float)hs[bl + lane];
    if (lane == 0) fs[ESC] = (float)(tot - bw);
    __syncwarp();
    huffman_lengths_warp(fs, h.len);
    if (lane == 0) {
      h.magic = kMagic;
      h.version = 1;
      h.n = n;
      h.chunk = (uint32_t)B;
      h.nchunks = (uint32_t)C;
      h.tail = (uint32_t)tail;
      h.lo = (uint32_t)bl;
      for (int i = 0; i < (int)sizeof(h.pad); ++i) h.pad[i] = 0;
    }
  }
  __syncthreads();
  if (tid < (int)(sizeof(Header) / 4))
    reinterpret_cast<uint32_t*>(hdr_out)[tid] =
        reinterpret_cast<uint32_t const*>(&h)[tid];
  build_tables(h.len, t);
  uint32_t lo = h.lo;
  for (int i = tid; i < LUTN; i += blockDim.x) tab[i] = lut_entry(i, t, lo).y;
  for (int e = tid; e < 256; e += blockDim.x) {
    int sy = e - (int)lo;
    tab[LUTN + e] = (sy >= 0 && sy < 16)
                        ? ((uint32_t)t.code[sy] | ((uint32_t)h.len[sy] << 24))
                        : ((((uint32_t)t.code[ESC] << 8) | (uint32_t)e) |
                           ((uint32_t)(h.len[ESC] + 8) << 24));
  }
}

// Phase 1: every lane codes its 512 values into a private word stream (scratch,
// interleaved [word][lane]). Phase 2: replay the decoder's schedule and emit
// the words in the order the decoder will refill them.
__global__ void __launch_bounds__(WPB * 32)
    encode_kernel(uint16_t const* __restrict__ in,
                  uint32_t const* __restrict__ tab, int B, size_t ch0,
                  size_t nch, uint32_t* __restrict__ scratch,
                  uint32_t* __restrict__ slots, uint32_t* __restrict__ cnt) {
  int const NPL = B >> 5, G = NPL >> 3, SW = sw_of(B);
  __shared__ uint32_t lut[LUTN];
  __shared__ uint32_t ctab[256];
  for (int i = threadIdx.x; i < LUTN; i += blockDim.x) lut[i] = tab[i];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) ctab[i] = tab[LUTN + i];
  __syncthreads();
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t* sc = scratch + (blockIdx.x * (size_t)WPB + warp) * SW;
  for (size_t ch = blockIdx.x * (size_t)WPB + warp; ch < nch;
       ch += (size_t)gridDim.x * WPB) {
    uint4 const* src = (uint4 const*)(in + (ch0 + ch) * B);
    uint32_t* out = slots + ch * (size_t)SW;
    unsigned long long acc = 0;
    int ab = 0, nw = 0;
    for (int j = 0; j < G; ++j) {
      uint4 v = src[j * 32 + lane];
      uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        uint32_t x = (w[k >> 1] >> (16 * (k & 1))) & 0xFFFF;
        uint32_t c = ctab[(x >> 7) & 0xFF];
        int l = c >> 24;
        acc = (acc << l) | (c & 0xFFFFFF);
        ab += l;
        if (ab >= 32) {
          ab -= 32;
          sc[nw * 32 + lane] = (uint32_t)(acc >> ab);
          nw++;
        }
      }
    }
    if (ab) {
      sc[nw * 32 + lane] = (uint32_t)(acc << (32 - ab));
      nw++;
    }
    __syncwarp();
    // Phase 2: the decoder's schedule, with the same 64-bit bit buffer.
    int r = NPL, nb = 0, rdw = 0, done_row = 0;
    unsigned long long buf = 0;
    uint32_t base = 0;
    while (__any_sync(0xffffffffu, r > 0)) {
      int gjl = (NPL - r) >> 3;
      bool run = r > 0 && (gjl - done_row) < RROWS - 1;
      bool need = run && nb < REFILL;
      uint32_t vote = __ballot_sync(0xffffffffu, need);
      if (need) {
        uint32_t wd = rdw < nw ? sc[rdw * 32 + lane] : 0u;
        rdw++;
        out[base + __popc(vote & lane_lt())] = wd;
        buf |= (unsigned long long)wd << (32 - nb);
        nb += 32;
      }
      base += __popc(vote);
      if (run) {
#pragma unroll
        for (int s = 0; s < NLOOK; ++s) {
          if (r == 0 || nb < GATE) break;
          int q, c;
          bool esc;
          step_plan(lut[(uint32_t)(buf >> (64 - P))], r, q, c, esc);
          buf <<= q;
          nb -= q;
          r -= c;
        }
      }
      done_row =
          (int)__reduce_min_sync(0xffffffffu, (unsigned)((NPL - r) >> 3));
    }
    if (lane == 0) cnt[ch] = base;
    __syncwarp();
  }
}

// Copy each chunk's words to carry + offl[ch]; drop (and flag) anything past
// max_words.
__global__ void compact_kernel(uint32_t const* __restrict__ slots, int SW,
                               uint32_t const* __restrict__ cnt,
                               uint32_t const* __restrict__ offl, size_t nch,
                               uint32_t const* __restrict__ state,
                               uint32_t* __restrict__ words,
                               uint32_t* __restrict__ off_out, size_t max_words,
                               uint32_t* __restrict__ overflow) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wpb = blockDim.x >> 5;
  uint64_t carry = state[0];
  for (size_t ch = blockIdx.x * (size_t)wpb + warp; ch < nch;
       ch += (size_t)gridDim.x * wpb) {
    uint64_t o = carry + offl[ch];
    uint32_t c = cnt[ch];
    if (lane == 0) off_out[ch] = (uint32_t)o;
    if (o + c > max_words) {
      if (lane == 0) atomicOr(overflow, 1u);
      continue;
    }
    uint32_t const* s = slots + ch * (size_t)SW;
    for (uint32_t i = lane; i < c; i += 32) words[o + i] = s[i];
  }
}

__global__ void carry_kernel(uint32_t const* offl, uint32_t const* cnt,
                             size_t nch, uint32_t* state, uint32_t* off_end) {
  uint64_t t = (uint64_t)state[0] + offl[nch - 1] + cnt[nch - 1];
  if (t > 0xFFFFFFFFull) {
    state[1] = 1;
    t = 0xFFFFFFFFull;
  }
  state[0] = (uint32_t)t;
  *off_end = (uint32_t)t;
}

// ---------------- decoder ----------------
__device__ __forceinline__ uint4 combine8(unsigned long long e8, uint2 r8) {
  uint32_t E[2] = {(uint32_t)e8, (uint32_t)(e8 >> 32)}, R[2] = {r8.x, r8.y},
           o[4];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    uint32_t hi = (R[h] & 0x80808080u) | ((E[h] >> 1) & 0x7F7F7F7Fu);
    uint32_t lo = ((E[h] << 7) & 0x80808080u) | (R[h] & 0x7F7F7F7Fu);
    uint32_t a, b;
    asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(a) : "r"(lo), "r"(hi));
    asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(b) : "r"(lo), "r"(hi));
    o[2 * h] = a;
    o[2 * h + 1] = b;
  }
  return make_uint4(o[0], o[1], o[2], o[3]);
}

// sched[0]: next chunk, sched[1]: finished blocks. The last block resets both,
// so consecutive launches on one stream start from zero.
template <bool ALIGNED_OUT>
__global__ void __launch_bounds__(DWPB * 32)
    decode_kernel(uint8_t const* __restrict__ in, size_t n, int B,
                  size_t in_bytes, uint32_t* __restrict__ sched,
                  int* __restrict__ status, uint16_t* __restrict__ outp) {
  int const NPL = B >> 5, G = NPL >> 3;
  extern __shared__ uint2 lut[];  // LUTN entries
  __shared__ uint2 ringx[DWPB][RROWS][32];
  __shared__ Tables t;
  __shared__ uint8_t len_s[NSYM];
  __shared__ uint32_t lo_s;
  __shared__ int ok_s;
  size_t C = n / B, tail = n - C * B;
  Header const* h = (Header const*)(in + hdr_off(n));
  uint32_t const* off = (uint32_t const*)(in + tab_off(n));
  uint32_t const* words = (uint32_t const*)(in + words_off(n, C, tail));
  if (threadIdx.x == 0) {
    bool g = C > 0 && header_ok(h, n, B, C, tail) &&
             words_off(n, C, tail) + (size_t)off[C] * 4 == in_bytes;
    ok_s = g;
    if (!g && blockIdx.x == 0 && status) *status = NFH_ERR_FORMAT;
    if (g) lo_s = h->lo;
  }
  if (threadIdx.x < NSYM) len_s[threadIdx.x] = h->len[threadIdx.x];
  __syncthreads();
  if (!ok_s) return;  // uniform across blocks: no block touches sched
  build_tables(len_s, t);
  for (int i = threadIdx.x; i < LUTN; i += blockDim.x)
    lut[i] = lut_entry(i, t, lo_s);
  __syncthreads();
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (;;) {
    uint32_t ch = 0;
    if (lane == 0) ch = atomicAdd(&sched[0], 1u);
    ch = __shfl_sync(0xffffffffu, ch, 0);
    if (ch >= C) break;
    uint32_t const* src = words + off[ch];
    uint2 const* rw = (uint2 const*)(in + (size_t)ch * B);
    uint16_t* o16 = outp + (size_t)ch * B;
    int r = NPL, nb = 0, pc = 0, gj = 0, done_row = 0;
    unsigned long long buf = 0, p0 = 0, p1 = 0;
    uint32_t base = 0;
#pragma unroll
    for (int a = 0; a < PFG; ++a)
      asm volatile("prefetch.global.L1 [%0];" ::"l"(rw + a * 32 + lane));
    while (__any_sync(0xffffffffu, r > 0)) {
      bool run = r > 0 && (gj - done_row) < RROWS - 1;
      bool need = run && nb < REFILL;
      uint32_t vote = __ballot_sync(0xffffffffu, need);
      if (need) {
        buf |= (unsigned long long)src[base + __popc(vote & lane_lt())]
               << (32 - nb);
        nb += 32;
      }
      base += __popc(vote);
      if (lane == 0)
        asm volatile("prefetch.global.L1 [%0];" ::"l"(src + base + 64));
      if (run) {
#pragma unroll
        for (int s = 0; s < NLOOK; ++s) {
          if (r == 0 || nb < GATE) break;
          uint2 e2 = lut[(uint32_t)(buf >> (64 - P))];
          uint32_t ent = e2.y;
          int q, c;
          bool esc;
          step_plan(ent, r, q, c, esc);
          uint32_t e4 = e2.x;
          if (esc) {
            int l1 = (ent >> 23) & 15;
            e4 = (uint32_t)(buf >> (64 - l1 - 8)) & 0xFFu;
          } else if (c < (int)((ent >> 16) & 7)) {
            e4 &= 0xFFu;
          }
          p0 |= (unsigned long long)e4 << (8 * pc);  // pc <= 7
          if (pc > 4) p1 |= (unsigned long long)e4 >> (64 - 8 * pc);
          pc += c;
          buf <<= q;
          nb -= q;
          r -= c;
          if (pc >= 8) {
            ringx[warp][gj % RROWS][lane] =
                make_uint2((uint32_t)p0, (uint32_t)(p0 >> 32));
            gj++;
            p0 = p1;
            p1 = 0;
            pc -= 8;
          }
        }
      }
      int mn = (int)__reduce_min_sync(0xffffffffu, (unsigned)gj);
      __syncwarp();
      for (; done_row < mn; ++done_row) {
        uint2 e = ringx[warp][done_row % RROWS][lane];
        uint4 v = combine8(((unsigned long long)e.y << 32) | e.x,
                           rw[done_row * 32 + lane]);
        uint16_t* d = o16 + (size_t)(done_row * 32 + lane) * 8;
        if (ALIGNED_OUT) {
          *(uint4*)d = v;
        } else {
          uint32_t vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            d[2 * k] = (uint16_t)vv[k];
            d[2 * k + 1] = (uint16_t)(vv[k] >> 16);
          }
        }
        if (done_row + PFG < G)
          asm volatile("prefetch.global.L1 [%0];" ::"l"(
              rw + (done_row + PFG) * 32 + lane));
      }
      __syncwarp();
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(&sched[1], 1u) == gridDim.x - 1) {
      sched[0] = 0;
      sched[1] = 0;
      __threadfence();
    }
  }
}

// SMs the current context may use: a green context (an SM partition, as set up
// for UCCL_DIETGPU_SM_USAGE) reports the full device in the attribute.
int sm_count(int dev, int* out) {
  if (cudaDeviceGetAttribute(out, cudaDevAttrMultiProcessorCount, dev) !=
      cudaSuccess)
    return -1;
#if CUDA_VERSION >= 12040
  CUcontext ctx = nullptr;
  CUdevResource res;
  if (cuCtxGetCurrent(&ctx) == CUDA_SUCCESS && ctx &&
      cuCtxGetDevResource(ctx, &res, CU_DEV_RESOURCE_TYPE_SM) == CUDA_SUCCESS &&
      res.sm.smCount > 0 && (int)res.sm.smCount < *out)
    *out = (int)res.sm.smCount;
#endif
  return 0;
}

int launch_grid(int sms, size_t work, int per_block, int per_sm = 8) {
  size_t g = (work + per_block - 1) / per_block;
  return (int)std::max<size_t>(1, std::min<size_t>(g, (size_t)sms * per_sm));
}

}  // namespace

struct nfh_encoder {
  int sms = 0, max_grid = 0;
  size_t max_chunks = 0;  // chunks per segment at BMIN
  unsigned long long* d_hist = nullptr;
  uint32_t *d_slots = nullptr, *d_scratch = nullptr, *d_cnt = nullptr,
           *d_offl = nullptr, *d_state = nullptr, *d_tab = nullptr;
  void* d_cub = nullptr;
  size_t cub_bytes = 0;
  uint32_t* h_state = nullptr;
  // in-flight encode between nfh_encode_begin and nfh_encode_end
  uint16_t const* in = nullptr;
  uint8_t* out = nullptr;
  size_t n = 0, W = 0;
  int pending = 0;  // 0 none, 1 coded (check size), 2 stored already launched
};

struct nfh_decoder {
  int sms = 0, blocks_per_sm = 0;
  uint32_t* d_sched = nullptr;
};

#define NFH_CK(x)                                \
  do {                                           \
    if ((x) != cudaSuccess) return NFH_ERR_CUDA; \
  } while (0)

extern "C" {

void nfh_set_chunk_override(int chunk) {
  g_chunk_override =
      (chunk >= BMIN && chunk <= BMAX && (chunk & (chunk - 1)) == 0) ? chunk
                                                                     : 0;
}

char const* nfh_error_string(int err) {
  switch (err) {
    case NFH_OK:
      return "ok";
    case NFH_ERR_ARG:
      return "invalid argument";
    case NFH_ERR_CUDA:
      return "CUDA error";
    case NFH_ERR_FORMAT:
      return "malformed NFH stream";
    default:
      return "unknown NFH error";
  }
}

void nfh_encoder_destroy(nfh_encoder* e) {
  if (!e) return;
  cudaFree(e->d_hist);
  cudaFree(e->d_slots);
  cudaFree(e->d_scratch);
  cudaFree(e->d_cnt);
  cudaFree(e->d_offl);
  cudaFree(e->d_state);
  cudaFree(e->d_cub);
  cudaFree(e->d_tab);
  cudaFreeHost(e->h_state);
  delete e;
}

int nfh_encoder_create(nfh_encoder** out) {
  if (!out) return NFH_ERR_ARG;
  *out = nullptr;
  auto* e = new (std::nothrow) nfh_encoder();
  if (!e) return NFH_ERR_ARG;
  int dev = 0, occ = 0;
  auto fail = [&]() {
    nfh_encoder_destroy(e);
    return NFH_ERR_CUDA;
  };
  if (cudaGetDevice(&dev) != cudaSuccess || sm_count(dev, &e->sms) != 0 ||
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, encode_kernel,
                                                    WPB * 32, 0) != cudaSuccess)
    return fail();
  // Chunks are strided over the encoder's warps, so a grid of only the
  // resident blocks leaves a tail when the context has few SMs; size it for
  // the whole device and let blocks queue instead.
  int dev_sms = 0;
  if (cudaDeviceGetAttribute(&dev_sms, cudaDevAttrMultiProcessorCount, dev) !=
      cudaSuccess)
    return fail();
  e->max_grid = std::max(dev_sms, e->sms) * std::max(occ, 1);
  e->max_chunks = SEGV / BMIN;
  size_t slot_words = 0, scratch_words = 0;
  for (int b = BMIN; b <= BMAX; b <<= 1) {
    size_t chunks = SEGV / b,
           grid = std::min<size_t>(e->max_grid, (chunks + WPB - 1) / WPB);
    slot_words = std::max(slot_words, chunks * sw_of(b));
    scratch_words = std::max(scratch_words, grid * WPB * sw_of(b));
  }
  if (cudaMalloc(&e->d_hist, 256 * 8) != cudaSuccess ||
      cudaMalloc(&e->d_slots, slot_words * 4) != cudaSuccess ||
      cudaMalloc(&e->d_scratch, scratch_words * 4) != cudaSuccess ||
      cudaMalloc(&e->d_cnt, e->max_chunks * 4) != cudaSuccess ||
      cudaMalloc(&e->d_offl, e->max_chunks * 4) != cudaSuccess ||
      cudaMalloc(&e->d_state, 8) != cudaSuccess ||
      cudaMalloc(&e->d_tab, (LUTN + 256) * 4) != cudaSuccess ||
      cub::DeviceScan::ExclusiveSum(nullptr, e->cub_bytes, e->d_cnt, e->d_offl,
                                    (int)e->max_chunks) != cudaSuccess ||
      cudaMalloc(&e->d_cub, std::max<size_t>(e->cub_bytes, 1)) != cudaSuccess ||
      cudaMallocHost(&e->h_state, 8) != cudaSuccess)
    return fail();
  *out = e;
  return NFH_OK;
}

void nfh_decoder_destroy(nfh_decoder* d) {
  if (!d) return;
  cudaFree(d->d_sched);
  delete d;
}

int nfh_decoder_create(nfh_decoder** out) {
  if (!out) return NFH_ERR_ARG;
  *out = nullptr;
  auto* d = new (std::nothrow) nfh_decoder();
  if (!d) return NFH_ERR_ARG;
  int dev = 0, occ = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || sm_count(dev, &d->sms) != 0 ||
      cudaFuncSetAttribute(decode_kernel<true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           LUTN * 8) != cudaSuccess ||
      cudaFuncSetAttribute(decode_kernel<false>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           LUTN * 8) != cudaSuccess ||
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occ, decode_kernel<true>, DWPB * 32, LUTN * 8) != cudaSuccess ||
      cudaMalloc(&d->d_sched, 8) != cudaSuccess ||
      cudaMemset(d->d_sched, 0, 8) != cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess) {
    nfh_decoder_destroy(d);
    return NFH_ERR_CUDA;
  }
  d->blocks_per_sm = std::max(occ, 1);
  *out = d;
  return NFH_OK;
}

int nfh_split(nfh_encoder* e, void const* in, size_t n, void* out,
              cudaStream_t s) {
  if (!e || (!in && n) || (!out && n)) return NFH_ERR_ARG;
  if (n == 0) return NFH_OK;
  auto const* x = static_cast<uint16_t const*>(in);
  auto* raw = static_cast<uint8_t*>(out);
  size_t lo = 0;
  if ((reinterpret_cast<uintptr_t>(in) & 15) == 0 &&
      (reinterpret_cast<uintptr_t>(out) & 7) == 0) {
    size_t n8 = n / 8;
    if (n8)
      split_vec_kernel<<<launch_grid(e->sms, n8, 256), 256, 0, s>>>(
          (uint4 const*)x, n8, (uint2*)raw);
    lo = n8 * 8;
  }
  if (lo < n)
    split_scalar_kernel<<<launch_grid(e->sms, n - lo, 256), 256, 0, s>>>(
        x, lo, n, raw);
  NFH_CK(cudaGetLastError());
  return NFH_OK;
}

int nfh_encode_begin(nfh_encoder* e, void const* in, size_t n, void* out,
                     cudaStream_t s) {
  if (!e || (!in && n) || (!out && n) ||
      (reinterpret_cast<uintptr_t>(out) & 15) || e->pending)
    return NFH_ERR_ARG;
  auto const* x = static_cast<uint16_t const*>(in);
  auto* o = static_cast<uint8_t*>(out);
  e->in = x;
  e->out = o;
  e->n = n;
  if (n == 0) {
    e->pending = 3;
    return NFH_OK;
  }
  int B = chunk_for(n);
  size_t C = n / B, tail = n - C * B, W = words_off(n, C, tail);
  e->W = W;
  if (C == 0 || (reinterpret_cast<uintptr_t>(in) & 15) || W >= 2 * n) {
    exp_kernel<<<launch_grid(e->sms, n, 256), 256, 0, s>>>(x, 0, n, o + n);
    NFH_CK(cudaGetLastError());
    e->pending = 2;
    return NFH_OK;
  }
  size_t max_words = (2 * n - W) / 4;
  int SW = sw_of(B);
  size_t seg = SEGV / B;
  NFH_CK(cudaMemsetAsync(e->d_hist, 0, 256 * 8, s));
  hist_kernel<<<launch_grid(e->sms, n / 8 >> HIST_SHIFT, 256), 256, 0, s>>>(
      (uint4 const*)x, n / 8, e->d_hist);
  codebook_kernel<<<1, 256, 0, s>>>(e->d_hist, n, B, C, tail,
                                    (Header*)(o + hdr_off(n)), e->d_tab);
  if (tail)
    exp_kernel<<<launch_grid(e->sms, tail, 256), 256, 0, s>>>(
        x, C * B, n, o + tail_off(n, C));
  uint32_t* off_out = (uint32_t*)(o + tab_off(n));
  uint32_t* words = (uint32_t*)(o + W);
  NFH_CK(cudaMemsetAsync(e->d_state, 0, 8, s));
  for (size_t ch0 = 0; ch0 < C; ch0 += seg) {
    size_t nch = std::min<size_t>(seg, C - ch0);
    int g = (int)std::min<size_t>(e->max_grid, (nch + WPB - 1) / WPB);
    encode_kernel<<<g, WPB * 32, 0, s>>>(x, e->d_tab, B, ch0, nch, e->d_scratch,
                                         e->d_slots, e->d_cnt);
    NFH_CK(cub::DeviceScan::ExclusiveSum(e->d_cub, e->cub_bytes, e->d_cnt,
                                         e->d_offl, (int)nch, s));
    compact_kernel<<<launch_grid(e->sms, nch, 8), 256, 0, s>>>(
        e->d_slots, SW, e->d_cnt, e->d_offl, nch, e->d_state, words,
        off_out + ch0, max_words, e->d_state + 1);
    carry_kernel<<<1, 1, 0, s>>>(e->d_offl, e->d_cnt, nch, e->d_state,
                                 off_out + ch0 + nch);
  }
  NFH_CK(cudaMemcpyAsync(e->h_state, e->d_state, 8, cudaMemcpyDeviceToHost, s));
  NFH_CK(cudaGetLastError());
  e->pending = 1;
  return NFH_OK;
}

int nfh_encode_end(nfh_encoder* e, cudaStream_t s, size_t* out_bytes) {
  if (!e || !out_bytes || !e->pending) return NFH_ERR_ARG;
  int mode = e->pending;
  e->pending = 0;
  *out_bytes = 0;
  if (mode == 3) return NFH_OK;
  NFH_CK(cudaStreamSynchronize(s));
  size_t n = e->n;
  if (mode == 1) {
    size_t total = e->W + (size_t)e->h_state[0] * 4;
    if (!e->h_state[1] && total < 2 * n) {
      *out_bytes = total;
      return NFH_OK;
    }
    exp_kernel<<<launch_grid(e->sms, n, 256), 256, 0, s>>>(e->in, 0, n,
                                                           e->out + n);
    NFH_CK(cudaGetLastError());
    NFH_CK(cudaStreamSynchronize(s));
  }
  *out_bytes = 2 * n;
  return NFH_OK;
}

int nfh_store(nfh_encoder* e, void const* in, size_t n, void* out,
              cudaStream_t s) {
  if (!e || (!in && n) || (!out && n)) return NFH_ERR_ARG;
  if (n == 0) return NFH_OK;
  exp_kernel<<<launch_grid(e->sms, n, 256), 256, 0, s>>>(
      static_cast<uint16_t const*>(in), 0, n, static_cast<uint8_t*>(out) + n);
  NFH_CK(cudaGetLastError());
  return NFH_OK;
}

int nfh_encode(nfh_encoder* e, void const* in, size_t n, void* out,
               cudaStream_t s, size_t* out_bytes) {
  if (!out_bytes) return NFH_ERR_ARG;
  int r = nfh_encode_begin(e, in, n, out, s);
  if (r != NFH_OK) return r;
  return nfh_encode_end(e, s, out_bytes);
}

int nfh_decode(nfh_decoder* d, void const* in, size_t in_bytes, void* out,
               size_t n, cudaStream_t s, int* status) {
  if (!d || (!in && n) || (!out && n) ||
      (reinterpret_cast<uintptr_t>(in) & 15) ||
      (reinterpret_cast<uintptr_t>(out) & 1))
    return NFH_ERR_ARG;
  if (n == 0) return NFH_OK;
  auto const* src = static_cast<uint8_t const*>(in);
  auto* y = static_cast<uint16_t*>(out);
  if (in_bytes == 2 * n) {  // stored form
    join_kernel<<<launch_grid(d->sms, n, 256), 256, 0, s>>>(src, src + n, 0, n,
                                                            y);
    NFH_CK(cudaGetLastError());
    return NFH_OK;
  }
  int B = chunk_for(n);
  size_t C = n / B, tail = n - C * B;
  if (C == 0 || in_bytes > 2 * n || in_bytes < words_off(n, C, tail)) {
    if (status) set_status_kernel<<<1, 1, 0, s>>>(status, NFH_ERR_FORMAT);
    NFH_CK(cudaGetLastError());
    return NFH_OK;
  }
  int g = (int)std::min<size_t>((size_t)d->sms * d->blocks_per_sm,
                                (C + DWPB - 1) / DWPB);
  if ((reinterpret_cast<uintptr_t>(out) & 15) == 0)
    decode_kernel<true><<<g, DWPB * 32, LUTN * 8, s>>>(src, n, B, in_bytes,
                                                       d->d_sched, status, y);
  else
    decode_kernel<false><<<g, DWPB * 32, LUTN * 8, s>>>(src, n, B, in_bytes,
                                                        d->d_sched, status, y);
  if (tail)
    join_kernel<<<launch_grid(d->sms, tail, 256), 256, 0, s>>>(
        src, src + tail_off(n, C), C * B, n, y);
  NFH_CK(cudaGetLastError());
  return NFH_OK;
}

}  // extern "C"
