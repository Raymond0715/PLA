#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {
// Four warps, each computing 16 horizontal 2x4 groups, for four row pairs.
constexpr int WARPS = 4;
constexpr int GROUPS = WARPS * 16;
constexpr int OUT_W = GROUPS * 4;
constexpr int OUT_H = 8;
constexpr int ROW_PAIRS = OUT_H / 2;
constexpr int UNION_H = OUT_H + 6;
constexpr int UNION_W = OUT_W + 6;
// Logical column 3 is the first aligned global vector. Shift it to
// shared column 8 so both sides of cp.async are 16-byte aligned.
constexpr int SHIFT = 5;
constexpr int PITCH = ((UNION_W + SHIFT + 15) / 16) * 16;

__device__ __forceinline__ void copy_async_16(__half* dst, const __half* src) {
#if __CUDA_ARCH__ >= 800
  unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::
               "r"(address), "l"(src) : "memory");
#else
  *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
#endif
}

__device__ __forceinline__ void commit_async() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

__device__ __forceinline__ void wait_async() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

template<bool INTERIOR>
__device__ __forceinline__ void load_union(__half* tile, const __half* x,
                                          int64_t base, int bx, int by,
                                          int h, int w) {
  constexpr int VECTORS = OUT_W / 8;
  for (int task = threadIdx.x; task < UNION_H * VECTORS; task += WARPS * 32) {
    const int sy = task / VECTORS;
    const int sx = 3 + (task % VECTORS) * 8;
    const int iy = by + sy - 3, ix = bx + sx - 3;
    const int64_t offset = base + int64_t(iy) * w + ix;
    __half* dst = tile + sy * PITCH + sx + SHIFT;
    if (INTERIOR || (iy >= 0 && iy < h && ix >= 0 && ix + 7 < w &&
                     ((reinterpret_cast<uintptr_t>(x) + offset * sizeof(__half)) & 15) == 0)) {
      copy_async_16(dst, x + offset);
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        dst[j] = (iy >= 0 && iy < h && ix + j >= 0 && ix + j < w)
                   ? x[base + int64_t(iy) * w + ix + j] : __float2half_rn(0.0f);
    }
  }
  // Only six halo values per row need scalar loads; the 256 middle values
  // use 32 aligned copies. No lane writes an async destination twice.
  for (int task = threadIdx.x; task < UNION_H * 6; task += WARPS * 32) {
    const int sy = task / 6, item = task % 6;
    const int sx = item < 3 ? item : OUT_W + item;
    const int iy = by + sy - 3, ix = bx + sx - 3;
    tile[sy * PITCH + sx + SHIFT] =
        (INTERIOR || (iy >= 0 && iy < h && ix >= 0 && ix < w))
          ? x[base + int64_t(iy) * w + ix] : __float2half_rn(0.0f);
  }
  commit_async();
}

__device__ __forceinline__ unsigned pack(__half a, __half b) {
  return unsigned(__half_as_ushort(a)) | (unsigned(__half_as_ushort(b)) << 16);
}

__device__ __forceinline__ unsigned input_pair(const __half* tile,
                                               int group, int row_pair, int k) {
  const int c = group * 4 + SHIFT;
  // Consecutive K elements can cross the ten-column window boundary.
  return pack(tile[(row_pair * 2 + k / 10) * PITCH + c + k % 10],
              tile[(row_pair * 2 + (k + 1) / 10) * PITCH + c + (k + 1) % 10]);
}

__device__ __forceinline__ void mma(float* d, unsigned a0, unsigned a1,
                                    unsigned a2, unsigned a3,
                                    unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void store_pair(__half* y, int64_t batch_base,
                                          int h, int w, int row, int col,
                                          float d0, float d1, __half bias) {
  if (row < h && col < w) {
    // Same cast-before-bias convention as the Triton candidates.
    const int64_t offset = batch_base + int64_t(row) * w + col;
    const __half lo = __hadd(__float2half_rn(d0), bias);
    const __half hi = __hadd(__float2half_rn(d1), bias);
    if (col + 1 < w && (reinterpret_cast<uintptr_t>(y + offset) & 3) == 0)
      *reinterpret_cast<__half2*>(y + offset) = __halves2half2(lo, hi);
    else {
      y[offset] = lo;
      if (col + 1 < w) y[offset + 1] = hi;
    }
  }
}

template<bool INTERIOR>
__global__ void stencil_tensorcore(const __half* __restrict__ x,
                                    const __half* __restrict__ weight,
                                    const __half* __restrict__ bias,
                                    __half* __restrict__ y, int h, int w) {
  __shared__ __align__(32) __half tile[UNION_H * PITCH];
  __shared__ __align__(32) __half weights[8 * 80];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bx = blockIdx.x * OUT_W;
  const int by = blockIdx.y * OUT_H;
  const int64_t batch_base = int64_t(blockIdx.z) * h * w;
  const bool interior = bx >= 3 && bx + OUT_W + 3 <= w &&
                        by >= 3 && by + OUT_H + 3 <= h && (w & 7) == 0 &&
                        (reinterpret_cast<uintptr_t>(x + batch_base) & 15) == 0;
  if (interior) load_union<true>(tile, x, batch_base, bx, by, h, w);
  else load_union<false>(tile, x, batch_base, bx, by, h, w);
  // Rebuild from live parameters so .to(), load_state_dict and weight updates
  // cannot leave a stale packed weight. Reused across all warps and row pairs.
  for (int i = tid; i < 8 * 80; i += WARPS * 32) {
    const int m = i / 80, k = i % 80;
    const int ky = k / 10 - m / 4, kx = k % 10 - m % 4;
    weights[i] = (ky >= 0 && ky < 7 && kx >= 0 && kx < 7)
                   ? weight[ky * 7 + kx] : __float2half_rn(0.0f);
  }
  // Weight preparation overlaps the outstanding input copies.
  wait_async();
  __syncthreads();
  const int g = lane >> 2;
  const int t = lane & 3;
  const int group0 = warp * 16 + g;
  const int group1 = group0 + 8;
  float accum[ROW_PAIRS][4] = {};
#pragma unroll
  for (int kb = 0; kb < 80; kb += 16) {
    const int k = kb + 2 * t;
    const unsigned b0 = pack(weights[g * 80 + k], weights[g * 80 + k + 1]);
    const unsigned b1 = pack(weights[g * 80 + k + 8], weights[g * 80 + k + 9]);
    // PTX m16n8k16 A fragment order: (row g,k), (row g+8,k),
    // (row g,k+8), (row g+8,k+8). N corresponds to 8 true outputs.
#pragma unroll
    for (int pair = 0; pair < ROW_PAIRS; ++pair) {
      mma(accum[pair], input_pair(tile, group0, pair, k), input_pair(tile, group1, pair, k),
          input_pair(tile, group0, pair, k + 8), input_pair(tile, group1, pair, k + 8), b0, b1);
    }
  }
  const __half b = bias[0];
  const int out_row = by + t / 2;
  const int subcol = 2 * (t % 2);
#pragma unroll
  for (int pair = 0; pair < ROW_PAIRS; ++pair) {
    store_pair(y, batch_base, h, w, out_row + 2 * pair, bx + 4 * group0 + subcol,
               accum[pair][0], accum[pair][1], b);
    store_pair(y, batch_base, h, w, out_row + 2 * pair, bx + 4 * group1 + subcol,
               accum[pair][2], accum[pair][3], b);
  }
}

__global__ void stencil_float(const float* x, const float* weight,
                              const float* bias, float* y, int h, int w) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (col >= w) return;
  const int64_t base = int64_t(blockIdx.z) * h * w;
  float sum = 0;
#pragma unroll
  for (int ky = 0; ky < 7; ++ky) {
#pragma unroll
    for (int kx = 0; kx < 7; ++kx) {
      int iy = row + ky - 3, ix = col + kx - 3;
      if (iy >= 0 && iy < h && ix >= 0 && ix < w)
        sum = fmaf(x[base + int64_t(iy) * w + ix], weight[ky * 7 + kx], sum);
    }
  }
  y[base + int64_t(row) * w + col] = sum + bias[0];
}
} // namespace

void launch_stencil(const void* x, const void* weight, const void* bias, void* y,
                     int n, int h, int w, bool fp16, cudaStream_t stream) {
  if (fp16) {
    // Bounds are handled in the shared-input load. A single launch includes
    // borders; the template leaves room for separately dispatched interiors.
    stencil_tensorcore<false><<<dim3((w + OUT_W - 1) / OUT_W, (h + OUT_H - 1) / OUT_H, n),
                                WARPS * 32, 0, stream>>>(
        static_cast<const __half*>(x), static_cast<const __half*>(weight),
        static_cast<const __half*>(bias), static_cast<__half*>(y), h, w);
  } else {
    stencil_float<<<dim3((w + 255) / 256, h, n), 256, 0, stream>>>(
        static_cast<const float*>(x), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(y), h, w);
  }
}
