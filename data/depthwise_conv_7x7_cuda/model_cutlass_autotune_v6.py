"""Fixed-shape fused CUTLASS depthwise 7x7 implementation.

The FP16 path computes a 16x256 output tile with four warps, stages weights in
shared memory, and specializes the 2048x2048 interior and boundary grids.
"""

import hashlib
import importlib.util
import os
import tempfile
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def _find_cutlass_include() -> Path:
    """Locate CUTLASS even when KernelBench imports a copy from /tmp."""
    candidates = []
    if os.environ.get("CUTLASS_PATH"):
        candidates.append(Path(os.environ["CUTLASS_PATH"]) / "include")
    # KernelBench's Triton/CuTe loader copies this module to a shallow /tmp
    # path, so deriving the repository root from __file__ is not reliable.
    cwd = Path.cwd()
    candidates += [
        cwd / "env_pla/lib/python3.11/site-packages/tvm/3rdparty/cutlass/include",
        cwd / "env_pla/lib/python3.11/site-packages/tilelang/3rdparty/cutlass/include",
        Path("/home/raymond/Projects/cutlass/cutlass_origin/include"),
    ]
    for package in ("tvm", "tilelang"):
        spec = importlib.util.find_spec(package)
        if spec is not None and spec.origin is not None:
            package_dir = Path(spec.origin).resolve().parent
            candidates.append(package_dir / "3rdparty/cutlass/include")
    for candidate in candidates:
        if (candidate / "cutlass/arch/mma.h").is_file():
            return candidate
    raise RuntimeError(
        "CUTLASS headers were not found; set CUTLASS_PATH to the CUTLASS source root"
    )


_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

void launch_depthwise_7x7_cutlass(const void*, const void*, void*, int, int,
                                  int, int, bool, cudaStream_t);

torch::Tensor depthwise_7x7_cuda(torch::Tensor input, torch::Tensor weight) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "input and weight must be CUDA tensors");
  TORCH_CHECK(input.dim() == 4, "input must have shape [N,C,H,W]");
  TORCH_CHECK(weight.dim() == 3 && weight.size(0) == input.size(1) &&
                  weight.size(1) == 8 && weight.size(2) == 80,
              "preprocessed weight must have shape [C,8,80]");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "dtype mismatch");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 ||
                  input.scalar_type() == torch::kFloat32,
              "only float16 and float32 are supported");
  TORCH_CHECK(input.device() == weight.device(), "device mismatch");
  TORCH_CHECK(input.size(0) <= 65535 && input.size(1) <= 65535,
              "N and C must fit CUDA grid dimensions");
  const c10::cuda::CUDAGuard guard(input.device());
  input = input.contiguous();
  weight = weight.contiguous();
  auto output = torch::empty_like(input);
  if (input.numel() == 0) return output;
  launch_depthwise_7x7_cutlass(
      input.data_ptr(), weight.data_ptr(), output.data_ptr(),
      static_cast<int>(input.size(0)), static_cast<int>(input.size(1)),
      static_cast<int>(input.size(2)), static_cast<int>(input.size(3)),
      input.scalar_type() == torch::kFloat16, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &depthwise_7x7_cuda, "CUTLASS-register depthwise 7x7 (CUDA)");
}
"""


_CUDA_SOURCE = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cutlass/arch/mma.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/half.h>
#include <cutlass/layout/matrix.h>

namespace {
constexpr int R = 3;
constexpr int OUT_H = 16;
constexpr int Y_GROUPS = OUT_H / 4;
constexpr int BLOCK_M = 256;
constexpr int BLOCK_N = 128;
constexpr int TILE_N = 128;
constexpr int BLOCK_K = 16;
constexpr int NUM_STAGES = 2;
constexpr int OUT_W = 2 * TILE_N;
constexpr int IN_W = OUT_W + 6;
constexpr int M = 8;
constexpr int K = 80;
constexpr int WARPS = 4;
constexpr int N_PER_WARP = TILE_N / WARPS;
constexpr int N_TILES_PER_WARP = N_PER_WARP / 16;
constexpr int INPUTS_PER_K_SLICE = 2 * N_PER_WARP + 6;
constexpr int CACHE_SLOTS = (INPUTS_PER_K_SLICE + 63) / 64;
constexpr int SHARED_SHIFT = 5;
constexpr int STAGE_W = 272;
constexpr int K_STAGES = K / BLOCK_K;
constexpr int FIXED_H = 2048;
constexpr int FIXED_W = 2048;
constexpr int FIXED_X_TILES = FIXED_W / OUT_W;
constexpr int FIXED_Y_TILES = FIXED_H / OUT_H;

static_assert(BLOCK_M == 256 && BLOCK_N == TILE_N && BLOCK_K == 16,
              "unsupported specialized tile");
static_assert(NUM_STAGES == 2 && TILE_N % WARPS == 0,
              "unsupported autotune configuration");

using Mma = cutlass::arch::Mma<
    cutlass::gemm::GemmShape<16, 8, 16>, 32,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>;

__device__ __forceinline__ void copy_async_16(void* dst, const void* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
               "r"(smem), "l"(src) : "memory");
#else
  *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
#endif
}

__device__ __forceinline__ void copy_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" :: : "memory");
#endif
}

__device__ __forceinline__ void copy_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" :: : "memory");
#endif
}

// Sixteen aligned vectors cover logical x=[3,130]. Halo x=[0,2] and tail
// x=[131,133] are scalar. INTERIOR removes all input bounds checks.
template <bool INTERIOR>
__device__ __forceinline__ void load_stage(
    __half input_stage[NUM_STAGES][Y_GROUPS][2][STAGE_W], int buffer, int stage,
    __half weight_stage[NUM_STAGES][M][BLOCK_K], const __half* input,
    const __half* packed_weight, int64_t plane, int channel, int tile_y,
    int tile_x, int height, int width) {
  const int tid = threadIdx.x;
  constexpr int VECTORS_PER_ROW = OUT_W / 8;
  constexpr int ROWS_PER_STAGE = 2 * Y_GROUPS;
  for (int task = tid; task < ROWS_PER_STAGE * VECTORS_PER_ROW;
       task += blockDim.x) {
    const int logical_row = task / VECTORS_PER_ROW;
    const int y_group = logical_row >> 1;
    const int row = logical_row & 1;
    const int vector = task % VECTORS_PER_ROW;
    const int sx = R + 8 * vector;
    const int gy = tile_y + 4 * y_group + 2 * stage + row - R;
    const int gx = tile_x + sx - R;
    __half* dst = &input_stage[buffer][y_group][row][sx + SHARED_SHIFT];
    const int64_t src_offset = plane + static_cast<int64_t>(gy) * width + gx;
    if (INTERIOR || (gy >= 0 && gy < height && gx >= 0 && gx + 7 < width &&
                     (reinterpret_cast<uintptr_t>(input + src_offset) & 15) == 0)) {
      copy_async_16(dst, input + src_offset);
    } else {
      #pragma unroll
      for (int i = 0; i < 8; ++i)
        dst[i] = (gy >= 0 && gy < height && gx + i >= 0 && gx + i < width)
                     ? input[plane + static_cast<int64_t>(gy) * width + gx + i]
                     : __float2half(0.0f);
    }
  }
  // Fill logical x=-1..2 and the three right-halo values.
  constexpr int SCALARS_PER_ROW = 7;
  if (tid < ROWS_PER_STAGE * SCALARS_PER_ROW) {
    const int logical_row = tid / SCALARS_PER_ROW;
    const int y_group = logical_row >> 1;
    const int row = logical_row & 1;
    const int item = tid % SCALARS_PER_ROW;
    const int sx = item < 4 ? item - 1 : OUT_W + R + item - 4;
    const int gy = tile_y + 4 * y_group + 2 * stage + row - R;
    const int gx = tile_x + sx - R;
    input_stage[buffer][y_group][row][sx + SHARED_SHIFT] =
        (sx >= 0 && (INTERIOR ||
         (gy >= 0 && gy < height && gx >= 0 && gx < width)))
            ? input[plane + static_cast<int64_t>(gy) * width + gx]
            : __float2half(0.0f);
  }
  // Sixteen threads cooperatively load the 8x16 weight slice with 16-byte
  // copies. It is reused by all four warps and all vertical output groups.
  if (tid < M * 2) {
    const int m = tid >> 1;
    const int k8 = (tid & 1) * 8;
    copy_async_16(&weight_stage[buffer][m][k8],
                  packed_weight + channel * M * K + m * K +
                      stage * BLOCK_K + k8);
  }
  copy_async_commit();
}

// Lane L owns aligned half2 pairs (-1+2L, 2L) and (63+2L, 64+2L).
// Shuffle the containing pair, then select its low/high half.
__device__ __forceinline__ cutlass::half_t select_half(uint32_t pair, int x) {
  return cutlass::half_t::bitcast((x & 1) ? static_cast<uint16_t>(pair)
                                           : static_cast<uint16_t>(pair >> 16));
}

__device__ __forceinline__ cutlass::half_t warp_get_slot0(
    const uint32_t values[CACHE_SLOTS], int x) {
  const int source_lane = ((x + 1) >> 1) & 31;
  return select_half(__shfl_sync(0xffffffffu, values[0], source_lane), x);
}

__device__ __forceinline__ cutlass::half_t warp_get_mixed(
    const uint32_t values[CACHE_SLOTS], int x) {
  const int source_lane = ((x + 1) >> 1) & 31;
  const uint32_t slot0 = __shfl_sync(0xffffffffu, values[0], source_lane);
  const uint32_t slot1 = __shfl_sync(0xffffffffu, values[1], source_lane);
  return select_half(((x + 1) >> 6) == 0 ? slot0 : slot1, x);
}

template <int MODE>
__global__ void cutlass_register_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ packed_weight,
    __half* __restrict__ output, int height, int width) {
  __shared__ __align__(32)
      __half input_stage[NUM_STAGES][Y_GROUPS][2][STAGE_W];
  __shared__ __align__(32) __half weight_stage[NUM_STAGES][M][BLOCK_K];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int batch = blockIdx.z;
  const int channel = blockIdx.y;
  int tile_ix, tile_iy;
  if (MODE == 1) {
    tile_ix = blockIdx.x % (FIXED_X_TILES - 2) + 1;
    tile_iy = blockIdx.x / (FIXED_X_TILES - 2) + 1;
  } else if (MODE == 2) {
    const int edge = blockIdx.x;
    if (edge < FIXED_X_TILES) {
      tile_ix = edge;
      tile_iy = 0;
    } else if (edge < 2 * FIXED_X_TILES) {
      tile_ix = edge - FIXED_X_TILES;
      tile_iy = FIXED_Y_TILES - 1;
    } else {
      const int side = edge - 2 * FIXED_X_TILES;
      tile_ix = (side & 1) ? FIXED_X_TILES - 1 : 0;
      tile_iy = (side >> 1) + 1;
    }
  } else {
    const int width_tiles = (width + OUT_W - 1) / OUT_W;
    tile_ix = blockIdx.x % width_tiles;
    tile_iy = blockIdx.x / width_tiles;
  }
  const int tile_x = tile_ix * OUT_W;
  const int tile_y = tile_iy * OUT_H;
  const int64_t plane = (static_cast<int64_t>(batch) * gridDim.y + channel) *
                        height * width;
  const bool interior = MODE == 1 ||
                        (tile_x >= R && tile_x + OUT_W + R <= width &&
                        tile_y >= R && tile_y + OUT_H + R <= height &&
                        (width & 7) == 0 &&
                        (reinterpret_cast<uintptr_t>(input + plane +
                         static_cast<int64_t>(tile_y - R) * width + tile_x) &
                         15) == 0);

  if (interior)
    load_stage<true>(input_stage, 0, 0, weight_stage, input, packed_weight,
                     plane, channel, tile_y, tile_x, height, width);
  else
    load_stage<false>(input_stage, 0, 0, weight_stage, input, packed_weight,
                      plane, channel, tile_y, tile_x, height, width);
  copy_async_wait();
  __syncthreads();

  Mma mma;
  // Two warps cover all 64 output-column pairs for width 128.
  Mma::FragmentC accum[Y_GROUPS][N_TILES_PER_WARP];
#pragma unroll
  for (int yg = 0; yg < Y_GROUPS; ++yg)
#pragma unroll
    for (int nt = 0; nt < N_TILES_PER_WARP; ++nt)
#pragma unroll
      for (int i = 0; i < Mma::FragmentC::kElements; ++i)
        accum[yg][nt][i] = 0.0f;

  const int group = lane >> 2;       // selects rows 0..7 and 8..15
  const int thread = lane & 3;       // selects a pair along K / output m
#pragma unroll
  for (int stage = 0; stage < K_STAGES; ++stage) {
    const int read_buffer = stage & 1;
    if (stage + 1 < K_STAGES) {
      if (interior)
        load_stage<true>(input_stage, read_buffer ^ 1, stage + 1, weight_stage,
                         input, packed_weight, plane, channel, tile_y, tile_x,
                         height, width);
      else
        load_stage<false>(input_stage, read_buffer ^ 1, stage + 1, weight_stage,
                          input, packed_weight, plane, channel, tile_y, tile_x,
                          height, width);
    }

    Mma::FragmentB bf;
    bf[0] = cutlass::half_t(weight_stage[read_buffer][group][2 * thread]);
    bf[1] = cutlass::half_t(weight_stage[read_buffer][group][2 * thread + 1]);
    bf[2] = cutlass::half_t(weight_stage[read_buffer][group][2 * thread + 8]);
    bf[3] = cutlass::half_t(weight_stage[read_buffer][group][2 * thread + 9]);

#pragma unroll
    for (int yg = 0; yg < Y_GROUPS; ++yg) {
      uint32_t row0[CACHE_SLOTS] = {0, 0};
      uint32_t row1[CACHE_SLOTS] = {0, 0};
#pragma unroll
      for (int slot = 0; slot < CACHE_SLOTS; ++slot) {
        const int pair_x = 2 * lane - 1 + 64 * slot;
        if (pair_x < INPUTS_PER_K_SLICE) {
          const int physical_x =
              warp * 2 * N_PER_WARP + pair_x + SHARED_SHIFT;
          row0[slot] = reinterpret_cast<const uint32_t*>(
              &input_stage[read_buffer][yg][0][physical_x])[0];
          row1[slot] = reinterpret_cast<const uint32_t*>(
              &input_stage[read_buffer][yg][1][physical_x])[0];
        }
      }

#pragma unroll
      for (int nt = 0; nt < N_TILES_PER_WARP; ++nt) {
        Mma::FragmentA af;
        const int x0 = 2 * (nt * 16 + group) + 2 * thread;
        const int x1 = x0 + 16;
        af[0] = warp_get_slot0(row0, x0);
        af[1] = warp_get_slot0(row0, x0 + 1);
        af[4] = warp_get_slot0(row1, x0);
        af[5] = warp_get_slot0(row1, x0 + 1);
        if (nt == 0) {
          af[2] = warp_get_slot0(row0, x1);
          af[3] = warp_get_slot0(row0, x1 + 1);
          af[6] = warp_get_slot0(row1, x1);
          af[7] = warp_get_slot0(row1, x1 + 1);
        } else {
          af[2] = warp_get_mixed(row0, x1);
          af[3] = warp_get_mixed(row0, x1 + 1);
          af[6] = warp_get_mixed(row1, x1);
          af[7] = warp_get_mixed(row1, x1 + 1);
        }
        mma(accum[yg][nt], af, bf, accum[yg][nt]);
      }
    }
    if (stage + 1 < K_STAGES) {
      copy_async_wait();
      __syncthreads();
    }
  }

  // Each pair of accumulator values maps to adjacent output x positions.
#pragma unroll
  for (int nt = 0; nt < N_TILES_PER_WARP; ++nt) {
#pragma unroll
    for (int yg = 0; yg < Y_GROUPS; ++yg) {
#pragma unroll
      for (int row_group = 0; row_group < 2; ++row_group) {
        const int n = warp * N_PER_WARP + nt * 16 + group + row_group * 8;
        const int oy = tile_y + 4 * yg + thread;
        const int ox = tile_x + 2 * n;
        const int64_t offset = plane + static_cast<int64_t>(oy) * width + ox;
        const __half lo = __float2half_rn(accum[yg][nt][2 * row_group]);
        const __half hi = __float2half_rn(accum[yg][nt][2 * row_group + 1]);
        if (oy < height && ox + 1 < width &&
            (reinterpret_cast<uintptr_t>(output + offset) & 3) == 0) {
          *reinterpret_cast<__half2*>(output + offset) =
              __halves2half2(lo, hi);
        } else {
          if (oy < height && ox < width) output[offset] = lo;
          if (oy < height && ox + 1 < width) output[offset + 1] = hi;
        }
      }
    }
  }
}

template <typename T> __device__ __forceinline__ float to_float(T x);
template <> __device__ __forceinline__ float to_float(float x) { return x; }
template <> __device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
template <typename T> __device__ __forceinline__ T from_float(float x);
template <> __device__ __forceinline__ float from_float(float x) { return x; }
template <> __device__ __forceinline__ __half from_float(float x) { return __float2half_rn(x); }

template <typename T>
__global__ void scalar_kernel(const T* input, const T* weight, T* output,
                              int channels, int height, int width) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int nc = blockIdx.z;
  if (x >= width || y >= height) return;
  const int channel = nc % channels;
  const int64_t plane = static_cast<int64_t>(nc) * height * width;
  float sum = 0.0f;
#pragma unroll
  for (int ky = 0; ky < 7; ++ky)
#pragma unroll
    for (int kx = 0; kx < 7; ++kx) {
      const int iy = y + ky - R, ix = x + kx - R;
      if (iy >= 0 && iy < height && ix >= 0 && ix < width)
        sum = fmaf(to_float(input[plane + static_cast<int64_t>(iy) * width + ix]),
                   to_float(weight[channel * M * K + ky * 8 + kx]), sum);
    }
  output[plane + static_cast<int64_t>(y) * width + x] = from_float<T>(sum);
}
}  // namespace

void launch_depthwise_7x7_cutlass(const void* input, const void* weight,
                                  void* output, int batches, int channels,
                                  int h, int w, bool fp16, cudaStream_t stream) {
  if (fp16) {
    const auto* in = static_cast<const __half*>(input);
    const auto* wt = static_cast<const __half*>(weight);
    auto* out = static_cast<__half*>(output);
    if (h == FIXED_H && w == FIXED_W) {
      const dim3 interior_grid(
          (FIXED_X_TILES - 2) * (FIXED_Y_TILES - 2), channels, batches);
      cutlass_register_kernel<1><<<interior_grid, 32 * WARPS, 0, stream>>>(
          in, wt, out, h, w);
      const dim3 boundary_grid(
          2 * FIXED_X_TILES + 2 * (FIXED_Y_TILES - 2), channels, batches);
      cutlass_register_kernel<2><<<boundary_grid, 32 * WARPS, 0, stream>>>(
          in, wt, out, h, w);
    } else {
      const dim3 grid(((w + OUT_W - 1) / OUT_W) *
                          ((h + OUT_H - 1) / OUT_H),
                      channels, batches);
      cutlass_register_kernel<0><<<grid, 32 * WARPS, 0, stream>>>(
          in, wt, out, h, w);
    }
  } else {
    const dim3 block(16, 16);
    const dim3 grid((w + 15) / 16, (h + 15) / 16, batches * channels);
    scalar_kernel<<<grid, block, 0, stream>>>(static_cast<const float*>(input),
        static_cast<const float*>(weight), static_cast<float*>(output),
        channels, h, w);
  }
}
"""


_CUTLASS_INCLUDE = _find_cutlass_include()
_source_hash = hashlib.sha256(
    (_CPP_SOURCE + _CUDA_SOURCE + str(_CUTLASS_INCLUDE)).encode()
).hexdigest()[:12]
_source_dir = Path(tempfile.gettempdir()) / f"depthwise_conv_7x7_cutlass_{_source_hash}"
_source_dir.mkdir(parents=True, exist_ok=True)
_cpp_path = _source_dir / "binding.cpp"
_cuda_path = _source_dir / "kernel.cu"
if not _cpp_path.exists() or _cpp_path.read_text() != _CPP_SOURCE:
    _cpp_path.write_text(_CPP_SOURCE)
if not _cuda_path.exists() or _cuda_path.read_text() != _CUDA_SOURCE:
    _cuda_path.write_text(_CUDA_SOURCE)

_extension = load(
    name=f"depthwise_conv_7x7_cutlass_ext_{_source_hash}",
    sources=[str(_cpp_path), str(_cuda_path)],
    extra_include_paths=[str(_CUTLASS_INCLUDE)],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    verbose=False,
)


def preprocess_weight(weight: torch.Tensor) -> torch.Tensor:
    """Pack [C,1,7,7] into the scheduled logical matrix A[C,8,80]."""
    m = torch.arange(8, device=weight.device)[:, None]
    k = torch.arange(80, device=weight.device)[None, :]
    ky = k // 8 - m // 2
    kx = k % 8 - m % 2
    valid = (ky >= 0) & (ky < 7) & (kx >= 0) & (kx < 7)
    values = weight.detach()[:, 0, ky.clamp(0, 6), kx.clamp(0, 6)]
    return torch.where(valid[None], values, torch.zeros_like(values)).contiguous()


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 1, bias: bool = False):
        super().__init__()
        if kernel_size != 7 or stride != 1 or padding != 3 or bias:
            raise ValueError("only kernel_size=7, stride=1, padding=3, bias=False is supported")
        self.conv2d = torch.nn.Conv2d(
            in_channels, in_channels, 7, stride=1, padding=3,
            groups=in_channels, bias=False)
        self._packed_weight = None
        self._packed_weight_key = None

    def _get_packed_weight(self) -> torch.Tensor:
        weight = self.conv2d.weight
        key = (weight._version, weight.data_ptr(), weight.device, weight.dtype)
        if self._packed_weight is None or self._packed_weight_key != key:
            self._packed_weight = preprocess_weight(weight)
            self._packed_weight_key = key
        return self._packed_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _extension.forward(x, self._get_packed_weight())


def get_inputs():
    return [torch.rand(1, 32, 128, 128, device="cuda")]


def get_init_inputs():
    return [32, 7, 1, 3]
