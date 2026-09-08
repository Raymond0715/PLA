#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <limits>

void launch_stencil(const void*, const void*, const void*, void*,
                    int, int, int, bool, cudaStream_t);

torch::Tensor stencil_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && b.is_cuda(), "all tensors must be CUDA");
  TORCH_CHECK(x.dim() == 4 && x.size(1) == 1, "expected [N,1,H,W]");
  TORCH_CHECK(w.numel() == 49 && b.numel() == 1, "expected 7x7 weight and scalar bias");
  TORCH_CHECK(x.device() == w.device() && x.device() == b.device(), "device mismatch");
  TORCH_CHECK(x.scalar_type() == w.scalar_type() && x.scalar_type() == b.scalar_type(), "dtype mismatch");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kFloat32,
              "supported dtypes: float16, float32");
  TORCH_CHECK(x.size(0) <= 65535 && x.size(2) <= std::numeric_limits<int>::max() - 16 &&
              x.size(3) <= std::numeric_limits<int>::max() - 512, "shape too large");
  c10::cuda::CUDAGuard guard(x.device());
  if (x.scalar_type() == torch::kFloat16) {
    const auto* prop = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(prop->major >= 8, "FP16 Tensor Core path requires SM80+");
  }
  x = x.contiguous(); w = w.contiguous(); b = b.contiguous();
  auto y = torch::empty_like(x);
  if (!x.numel()) return y;
  launch_stencil(x.data_ptr(), w.data_ptr(), b.data_ptr(), y.data_ptr(),
                 static_cast<int>(x.size(0)), static_cast<int>(x.size(2)),
                 static_cast<int>(x.size(3)), x.scalar_type() == torch::kFloat16,
                 at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &stencil_forward, "Shared-input Tensor Core 7x7 stencil");
}
