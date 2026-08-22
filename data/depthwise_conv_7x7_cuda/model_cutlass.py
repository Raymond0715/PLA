"""CUTLASS register-fragment implementation of depthwise 7x7 convolution."""

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
constexpr int OUT_H = 4;
constexpr int OUT_W = 128;
constexpr int IN_H = 10;
constexpr int IN_W = 134;
constexpr int M = 8;
constexpr int K = 80;
constexpr int WARPS = 2;

using Mma = cutlass::arch::Mma<
    cutlass::gemm::GemmShape<16, 8, 16>, 32,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>;

// Each lane owns x=lane, lane+32 and lane+64.  Shuffle first moves the
// requested original input to the consuming lane; bitcast then forms a
// CUTLASS register fragment without a shared-memory B matrix.
__device__ __forceinline__ cutlass::half_t warp_get(
    const uint32_t values[3], int x) {
  const int source_lane = x & 31;
  // Every source lane must present the same cache slot to a given shuffle.
  // Selecting values[x >> 5] before shuffling is incorrect when destination
  // lanes request different slots: the source lane would select using its own
  // x rather than the destination lane's x.
  const uint32_t slot0 = __shfl_sync(0xffffffffu, values[0], source_lane);
  const uint32_t slot1 = __shfl_sync(0xffffffffu, values[1], source_lane);
  const uint32_t slot2 = __shfl_sync(0xffffffffu, values[2], source_lane);
  const uint32_t bits = (x < 32) ? slot0 : ((x < 64) ? slot1 : slot2);
  return cutlass::half_t::bitcast(static_cast<uint16_t>(bits));
}

__global__ void cutlass_register_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ packed_weight,
    __half* __restrict__ output, int height, int width) {
  __shared__ __align__(32) __half original[IN_H][IN_W];
  __shared__ __align__(32) __half a[M][K];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int batch = blockIdx.z;
  const int channel = blockIdx.y;
  const int width_tiles = (width + OUT_W - 1) / OUT_W;
  const int tile_x = (blockIdx.x % width_tiles) * OUT_W;
  const int tile_y = (blockIdx.x / width_tiles) * OUT_H;
  const int64_t plane = (static_cast<int64_t>(batch) * gridDim.y + channel) *
                        height * width;

  for (int idx = tid; idx < IN_H * IN_W; idx += blockDim.x) {
    const int sy = idx / IN_W, sx = idx % IN_W;
    const int gy = tile_y + sy - R, gx = tile_x + sx - R;
    original[sy][sx] = (gy >= 0 && gy < height && gx >= 0 && gx < width)
        ? input[plane + static_cast<int64_t>(gy) * width + gx]
        : __float2half(0.0f);
  }
  for (int idx = tid; idx < M * K; idx += blockDim.x)
    a[0][idx] = packed_weight[channel * M * K + idx];
  __syncthreads();

  Mma mma;
  // Compute the transposed product B^T[32,80] * A^T[80,8].  Each warp
  // therefore needs two native 16x8 accumulators instead of four padded
  // 16x8 accumulators for the non-transposed product.
  Mma::FragmentC accum[2];
#pragma unroll
  for (int nt = 0; nt < 2; ++nt)
#pragma unroll
    for (int i = 0; i < Mma::FragmentC::kElements; ++i)
      accum[nt][i] = 0.0f;

  const int group = lane >> 2;       // selects rows 0..7 and 8..15
  const int thread = lane & 3;       // selects a pair along K / output m

#pragma unroll
  for (int kb = 0; kb < K; kb += 16) {
    // A K=16 slice refers to exactly two original rows and 70 unique x
    // positions per warp.  Load those once into registers before expansion.
    uint32_t row0[3] = {0, 0, 0};
    uint32_t row1[3] = {0, 0, 0};
#pragma unroll
    for (int slot = 0; slot < 3; ++slot) {
      const int x = lane + 32 * slot;
      if (x < 70) {
        row0[slot] = reinterpret_cast<const uint16_t*>(
            &original[kb / 8][warp * 64 + x])[0];
        row1[slot] = reinterpret_cast<const uint16_t*>(
            &original[kb / 8 + 1][warp * 64 + x])[0];
      }
    }

#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      Mma::FragmentA af;
      // Row-major MMA A is the activation matrix B^T[16,16].  The PTX
      // fragment gives each lane the low-K pair for rows `group` and
      // `group + 8`, followed by their high-K pairs.
      const int x0 = 2 * (nt * 16 + group) + 2 * thread;
      const int x1 = x0 + 16;
      af[0] = warp_get(row0, x0);
      af[1] = warp_get(row0, x0 + 1);
      af[2] = warp_get(row0, x1);
      af[3] = warp_get(row0, x1 + 1);
      af[4] = warp_get(row1, x0);
      af[5] = warp_get(row1, x0 + 1);
      af[6] = warp_get(row1, x1);
      af[7] = warp_get(row1, x1 + 1);

      Mma::FragmentB bf;
      // Column-major MMA B is packed_weight^T[16,8].  Its underlying
      // address is still a[group][k], so no new weight preprocessing is
      // required.
      bf[0] = cutlass::half_t(a[group][kb + 2 * thread]);
      bf[1] = cutlass::half_t(a[group][kb + 2 * thread + 1]);
      bf[2] = cutlass::half_t(a[group][kb + 2 * thread + 8]);
      bf[3] = cutlass::half_t(a[group][kb + 2 * thread + 9]);
      mma(accum[nt], af, bf, accum[nt]);
    }
  }

  // The accumulator is C^T.  Convert its two row groups back to C[m,n]
  // while writing directly to NCHW output; no physical transpose is needed.
#pragma unroll
  for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
    for (int row_group = 0; row_group < 2; ++row_group) {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const int n = warp * 32 + nt * 16 + group + row_group * 8;
        const int m = 2 * thread + pair;
        const int oy = tile_y + m / 2;
        const int ox = tile_x + 2 * n + m % 2;
        if (oy < height && ox < width)
          output[plane + static_cast<int64_t>(oy) * width + ox] =
              __float2half_rn(accum[nt][2 * row_group + pair]);
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
    const dim3 grid(((w + OUT_W - 1) / OUT_W) * ((h + OUT_H - 1) / OUT_H),
                    channels, batches);
    cutlass_register_kernel<<<grid, 32 * WARPS, 0, stream>>>(
        static_cast<const __half*>(input), static_cast<const __half*>(weight),
        static_cast<__half*>(output), h, w);
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
    extra_cuda_cflags=["-O3", "--use_fast_math"],
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
