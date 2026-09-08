#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <limits>

void launch_direct(const void*, const void*, const void*, void*, int, int, int,
                    int64_t, int64_t, int64_t, int, cudaStream_t);

torch::Tensor forward(torch::Tensor x, torch::Tensor w,
                       const std::optional<torch::Tensor>& bias) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "input and weight must be CUDA tensors");
  TORCH_CHECK((x.dim()==3 || x.dim()==4) && x.size(-3)==1, "expected [N,1,H,W] or [1,H,W]");
  TORCH_CHECK(w.numel()==49, "expected one 7x7 filter");
  TORCH_CHECK(x.device()==w.device() && x.scalar_type()==w.scalar_type(), "weight device/dtype mismatch");
  const auto type=x.scalar_type();
  TORCH_CHECK(type==torch::kFloat16 || type==torch::kBFloat16 ||
              type==torch::kFloat32 || type==torch::kFloat64, "unsupported dtype");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->device()==x.device() &&
                bias->scalar_type()==type && bias->numel()==1, "bias device/dtype/shape mismatch");
  }
  const int64_t h=x.size(-2), width=x.size(-1), n=x.dim()==4?x.size(0):1;
  TORCH_CHECK(h>0 && width>0, "spatial dimensions must be nonempty");
  TORCH_CHECK(n<=65535 && h<=65535*8 && width<=std::numeric_limits<int>::max()-512,
              "shape exceeds supported grid dimensions");
  const c10::cuda::CUDAGuard guard(x.device());
  auto y=torch::empty(x.sizes(),x.options());
  if (!n) return y;
  w=w.contiguous();
  auto b=bias.has_value()?bias->contiguous():torch::Tensor();
  const int dtype=type==torch::kFloat16?0:type==torch::kFloat32?1:type==torch::kFloat64?2:3;
  launch_direct(x.data_ptr(),w.data_ptr(),b.defined()?b.data_ptr():nullptr,y.data_ptr(),
                n,h,width,x.dim()==4?x.stride(0):0,x.stride(-2),x.stride(-1),dtype,
                at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("forward",&forward,"Register-reuse direct 7x7 convolution");
}
