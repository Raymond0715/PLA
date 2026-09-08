# CUDA Tensor Core stencil v2

Preserves the original `stencil_cuda/` implementation. `model.py` is a
self-contained KernelBench entry point with unchanged `ModelNew()` and
`forward(x)` interfaces and the same `conv1_weight` / `conv1_bias` parameters.

## Change

Each four-warp CTA computes **8x256 outputs**, rather than 4x256. The 14x262
unique input region is staged in shared memory using aligned 16-byte cp.async
copies plus scalar halos. Four vertical row-pair accumulators reuse the staged
input and the same B fragment over five K=16 MMA steps. Weight parameters are
read on each invocation; FP16 cast-before-bias rounding is unchanged. FP32
retains the scalar correctness fallback. No backward is provided.

For an interior tile, logical input elements per output fall from 2.5586 to
1.7910 (30% reduction). For 10240x10240 the CTA count falls from 102400 to 51200.
This reduces cache/load and per-CTA preparation work, not necessarily DRAM
bytes: the original implementation already approached minimal DRAM traffic.

SM89 offline compilation: 40 registers/thread, 8896 bytes static shared memory,
one CTA barrier, no spills. The previous kernel used 6720 bytes shared memory.
The larger shared allocation can reduce residency; performance must be measured.

CPU register-fragment, output-coverage and async-load alignment tests and full
standalone PyTorch extension compilation passed. GPU correctness and timing
remain to be validated on the RTX 4070.

## Run

From this directory in env_pla:

```bash
python test_stencil.py
```

From PolyLLMAgent:

```bash
python run_generated_presburger_triton.py \
  --kernel-src-path ../PolyLLMAgentTriton/stencil_cuda_v2/model.py
```

The test covers odd heights around the new eight-row tile, width tails,
non-contiguous inputs, FP16/FP32, parameter updates, current-stream use and empty
inputs. It reports a skip if CUDA is unavailable. `--cpu-only` runs only the CPU
index/fragment checks.

Edit the C++/CUDA files then run `python generate_model.py` to regenerate the
embedded sources. Source hashes isolate the compiled extension from the original
version. For offline builds set `TORCH_CUDA_ARCH_LIST=8.9`.
