#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>

namespace {
#ifndef STENCIL_ROWS
#define STENCIL_ROWS 8
#endif
constexpr int ROWS=STENCIL_ROWS;
static_assert(ROWS==8 || ROWS==16, "supported row tiles: 8,16");
constexpr int WIDTH=512;
constexpr int THREADS=128;
constexpr int COLS=WIDTH/THREADS;

template<typename T> using Acc=typename std::conditional<std::is_same<T,double>::value,double,float>::type;
template<typename T> __device__ __forceinline__ Acc<T> convert(T x) { return static_cast<Acc<T>>(x); }
template<> __device__ __forceinline__ float convert(__half x) { return __half2float(x); }
template<> __device__ __forceinline__ float convert(__nv_bfloat16 x) { return __bfloat162float(x); }
template<typename T> __device__ __forceinline__ T cast_output(Acc<T> x) { return static_cast<T>(x); }
template<> __device__ __forceinline__ __half cast_output<__half>(float x) { return __float2half_rn(x); }
template<> __device__ __forceinline__ __nv_bfloat16 cast_output<__nv_bfloat16>(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float madd(float x,float w,float a) { return __fmaf_rn(x,w,a); }
__device__ __forceinline__ double madd(double x,double w,double a) { return __fma_rn(x,w,a); }

// The compile-time input-row index makes all sixteen applicability checks
// constant, matching the Triton tl.static_range / a0...a15 implementation.
template<int IR, typename T, bool INTERIOR>
__device__ __forceinline__ void accumulate_rows(const T* x,const T* weight,
                       Acc<T> (&a)[ROWS][COLS],int r,int c,int h,int width,
                       int64_t sn,int64_t sh,int64_t sw) {
  const int iy=r+IR-3;
#pragma unroll
  for (int kc=0;kc<7;++kc) {
    Acc<T> value[COLS];
#pragma unroll
    for (int j=0;j<COLS;++j) {
      const int ix=c+j+kc-3;
      value[j]=Acc<T>(0);
      if (INTERIOR || (iy>=0 && iy<h && ix>=0 && ix<width))
        value[j]=convert(x[int64_t(blockIdx.z)*sn+int64_t(iy)*sh+int64_t(ix)*sw]);
    }
#pragma unroll
    for (int row=0;row<ROWS;++row) {
      if (IR>=row && IR<row+7) {
        const Acc<T> w=convert(weight[(IR-row)*7+kc]);
#pragma unroll
        for (int j=0;j<COLS;++j) a[row][j]=madd(value[j],w,a[row][j]);
      }
    }
  }
  if constexpr (IR<ROWS+5)
    accumulate_rows<IR+1,T,INTERIOR>(x,weight,a,r,c,h,width,sn,sh,sw);
}

template<typename T,bool INTERIOR>
__device__ __forceinline__ void compute(const T* x,const T* weight,const T* bias,
                         T* y,int h,int width,int64_t sn,int64_t sh,int64_t sw) {
  const int r=blockIdx.y*ROWS;
  const int c=blockIdx.x*WIDTH+threadIdx.x*COLS;
  Acc<T> a[ROWS][COLS]={};
  accumulate_rows<0,T,INTERIOR>(x,weight,a,r,c,h,width,sn,sh,sw);
  const Acc<T> b=bias?convert(bias[0]):Acc<T>(0);
#pragma unroll
  for (int row=0;row<ROWS;++row) {
#pragma unroll
    for (int j=0;j<COLS;++j) {
      const int ix=c+j, iy=r+row;
      if (iy<h && ix<width)
        y[(int64_t(blockIdx.z)*h+iy)*width+ix]=cast_output<T>(a[row][j]+b);
    }
  }
}

template<typename T>
__global__ void stencil_direct(const T* __restrict__ x,const T* __restrict__ weight,
                       const T* __restrict__ bias,T* __restrict__ y,
                       int h,int width,int64_t sn,int64_t sh,int64_t sw) {
  const int r=blockIdx.y*ROWS,c=blockIdx.x*WIDTH;
  if (r>=3 && r+ROWS+2<h && c>=3 && c+WIDTH+2<width)
    compute<T,true>(x,weight,bias,y,h,width,sn,sh,sw);
  else
    compute<T,false>(x,weight,bias,y,h,width,sn,sh,sw);
}

// Four adjacent FP16 values, represented as an aligned 64-bit transaction.
union alignas(8) Half4 {
  unsigned long long bits;
  __half lane[4];
};

constexpr int UNION_H=ROWS+6;
constexpr int UNION_W=518;
constexpr int SHIFT=5;
constexpr int PITCH=528;
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
  constexpr int VECTORS = WIDTH / 8;
  for (int task = threadIdx.x; task < UNION_H * VECTORS; task += THREADS) {
    const int sy = task / VECTORS;
    const int sx = 3 + (task % VECTORS) * 8;
    const int iy = by + sy - 3, ix = bx + sx - 3;
    const int64_t offset = base + int64_t(iy) * w + ix;
    __half* dst = tile + sy * PITCH + sx + SHIFT;
    if ((INTERIOR || (iy >= 0 && iy < h && ix >= 0 && ix + 7 < w)) &&
                     ((reinterpret_cast<uintptr_t>(x) + offset * sizeof(__half)) & 15) == 0) {
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
  for (int task = threadIdx.x; task < UNION_H * 6; task += THREADS) {
    const int sy = task / 6, item = task % 6;
    const int sx = item < 3 ? item : WIDTH + item;
    const int iy = by + sy - 3, ix = bx + sx - 3;
    tile[sy * PITCH + sx + SHIFT] =
        (INTERIOR || (iy >= 0 && iy < h && ix >= 0 && ix < w))
          ? x[base + int64_t(iy) * w + ix] : __float2half_rn(0.0f);
  }
  commit_async();
}

// One row's ten values are reused by four adjacent output columns.
template<int IR>
__device__ __forceinline__ void accumulate_shared(const __half* tile,
    const __half* __restrict__ weight, float (&a)[ROWS][4]) {
    // Aligned half2 loads: scalar endpoints plus four packed middle pairs.
    // Logical union starts at odd physical column SHIFT=5.
    float v[10];
    const __half* row_ptr=tile+IR*PITCH+SHIFT+threadIdx.x*4;
    v[0]=__half2float(row_ptr[0]);
#pragma unroll
    for (int p=0;p<4;++p) {
      const float2 pair=__half22float2(*reinterpret_cast<const __half2*>(row_ptr+1+p*2));
      v[1+p*2]=pair.x;
      v[2+p*2]=pair.y;
    }
    v[9]=__half2float(row_ptr[9]);
#pragma unroll
    for (int kc = 0; kc < 7; ++kc) {
#pragma unroll
      for (int row = 0; row < ROWS; ++row) {
        if (IR >= row && IR < row+7) {
          const float w = __half2float(weight[(IR-row)*7+kc]);
#pragma unroll
          for (int j = 0; j < 4; ++j) a[row][j] = __fmaf_rn(v[kc+j], w, a[row][j]);
        }
      }
    }
  if constexpr (IR<ROWS+5) accumulate_shared<IR+1>(tile,weight,a);
}

template<int FIXED_W, bool INTERIOR>
__device__ __forceinline__ void stencil_half_contiguous(const __half* __restrict__ x,
                  const __half* __restrict__ weight, const __half* __restrict__ bias,
                  __half* __restrict__ y, int h, int runtime_w, __half* tile) {
  const int width = FIXED_W ? FIXED_W : runtime_w;
  const int r = blockIdx.y * ROWS, c = blockIdx.x * WIDTH + threadIdx.x * 4;
  const int64_t batch = int64_t(blockIdx.z) * h * width;
  load_union<INTERIOR>(tile,x,batch,blockIdx.x*WIDTH,r,h,width);
  wait_async();
  __syncthreads();
  float a[ROWS][4] = {};
  accumulate_shared<0>(tile,weight,a);
  const float b = bias ? __half2float(bias[0]) : 0.f;
#pragma unroll
  for (int row = 0; row < ROWS; ++row) {
    if (r+row < h && c < width) {
      Half4 pack;
#pragma unroll
      for (int j = 0; j < 4; ++j) pack.lane[j] = __float2half_rn(a[row][j]+b);
      *reinterpret_cast<unsigned long long*>(y + batch + int64_t(r+row)*width+c) = pack.bits;
    }
  }
}

// Select the bounds-free body inside one kernel launch. The device compiler
// specializes both bodies and eliminates FIXED_W arithmetic where possible.
template<int FIXED_W>
__global__ void stencil_half_fast(const __half* x, const __half* weight,
                    const __half* bias, __half* y, int h, int width) {
  __shared__ __align__(16) __half tile[UNION_H*PITCH];
  const int w = FIXED_W ? FIXED_W : width;
  const int r = blockIdx.y*ROWS, c = blockIdx.x*WIDTH;
  if (r >= 3 && r+ROWS+2 < h && c >= 4 && c+WIDTH+3 < w)
    stencil_half_contiguous<FIXED_W,true>(x,weight,bias,y,h,w,tile);
  else
    stencil_half_contiguous<FIXED_W,false>(x,weight,bias,y,h,w,tile);
}

template<typename T>
void dispatch(const void* x,const void* w,const void* b,void* y,int n,int h,int width,
              int64_t sn,int64_t sh,int64_t sw,cudaStream_t stream) {
  stencil_direct<T><<<dim3((width+WIDTH-1)/WIDTH,(h+ROWS-1)/ROWS,n),THREADS,0,stream>>>(
       static_cast<const T*>(x),static_cast<const T*>(w),static_cast<const T*>(b),
       static_cast<T*>(y),h,width,sn,sh,sw);
}
} // namespace

void launch_direct(const void* x,const void* w,const void* b,void* y,int n,int h,int width,
                   int64_t sn,int64_t sh,int64_t sw,int type,cudaStream_t stream) {
  if (type==0 && sw==1 && sh==width && (n==1 || sn==int64_t(h)*width)
      && width%4==0 && reinterpret_cast<uintptr_t>(x)%8==0) {
    const dim3 grid((width+WIDTH-1)/WIDTH,(h+ROWS-1)/ROWS,n);
    if (width==10240)
      stencil_half_fast<10240><<<grid,THREADS,0,stream>>>(static_cast<const __half*>(x),static_cast<const __half*>(w),static_cast<const __half*>(b),static_cast<__half*>(y),h,width);
    else
      stencil_half_fast<0><<<grid,THREADS,0,stream>>>(static_cast<const __half*>(x),static_cast<const __half*>(w),static_cast<const __half*>(b),static_cast<__half*>(y),h,width);
  }
  else if (type==0) dispatch<__half>(x,w,b,y,n,h,width,sn,sh,sw,stream);
  else if (type==1) dispatch<float>(x,w,b,y,n,h,width,sn,sh,sw,stream);
  else if (type==2) dispatch<double>(x,w,b,y,n,h,width,sn,sh,sw,stream);
  else dispatch<__nv_bfloat16>(x,w,b,y,n,h,width,sn,sh,sw,stream);
}
