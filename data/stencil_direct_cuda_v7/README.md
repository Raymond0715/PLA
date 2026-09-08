# Direct CUDA v7

Based on v4 (one cooperative union load, no software pipeline). Preserves ModelNew(), conv1 parameters, FP32 accumulation order and bias-before-FP16-cast behavior.

Default CTA: 8×512 outputs, 128 threads, four adjacent columns per thread. Each thread holds 32 accumulators instead of 64. The shared input union is 14×518 with physical pitch 528. Ten shared FP16 values per row are read as two scalar endpoints and four aligned half2 pairs. This changes load representation, not arithmetic precision. Bank-conflict elimination is not claimed. Weight loading remains compiler-managed; no speculative all-weight register preload is added.

Set STENCIL_DIRECT_V7_ROWS=16 before process startup for a control that retains the v4 tile and adds the half2 read representation. The setting is part of the extension cache key. No autotuning or hidden benchmarking occurs. FP32/BF16/FP64 and noncontiguous input use the generic fallback. No backward is provided.

Offline sm_89 fast-path compilation: 8 rows uses 48 registers, 14784 bytes shared memory; 16 rows uses 80 registers, 23232 bytes shared memory. Both have zero spills. Smaller tiles incur more halo duplication and CTA overhead, so GPU speedup is not guaranteed.

From PolyLLMAgent:

```sh
python run_generated_presburger_triton.py --kernel-src-path ../PolyLLMAgentTriton/stencil_direct_cuda_v7/model.py
STENCIL_DIRECT_V7_ROWS=16 python run_generated_presburger_triton.py --kernel-src-path ../PolyLLMAgentTriton/stencil_direct_cuda_v7/model.py
```

Run `python stencil_direct_cuda_v7/test_stencil.py` from PolyLLMAgentTriton (repeat with the 16-row environment setting on GPU). CPU checks cover reduction order, both tile heights, union coverage, half2 alignment, and parameter interface. GPU numerical execution and performance require target-device validation.

Regenerate standalone model after native changes with `python stencil_direct_cuda_v7/generate_model.py`.
