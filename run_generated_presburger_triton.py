"""Run the generated Triton 7x7 stencil against the PyTorch reference."""

from pathlib import Path
import os

import torch

from scripts import run_and_check

# torch.backends.cudnn.allow_tf32 = False
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.set_float32_matmul_precision("highest")


PROJECT_ROOT = Path(__file__).resolve().parent


def check_triton():
    """Check correctness and performance of the generated Triton operator."""
    args = [
        "ref_origin=local",
        "ref_arch_src_path=data/depthwise_conv_7x7.py",
        # "kernel_src_path=data/depthwise_conv_7x7_loop_2026_08_15_17_44_09/map2mm_idx0_sample1_S0P3-4_S0P6-2_fix.py",
        "kernel_src_path=data/depthwise_conv_7x7_cuda/model_cutlass.py",
        # "kernel_src_path=data/depthwise_conv_7x7_cuda/model_v1.py",
        # "ref_arch_src_path=data/stencil_7x7.py",
        # "kernel_src_path=data/stencil_7x7_loop_2026_08_07_11_07_01/map2mm_idx0_sample0_S0P0-2_S0P3-8_fix.py",
        # "kernel_src_path=data/stencil_7x7_cuda/model_v1.py",
        # "kernel_src_path=../PolyLLMAgentTriton/triton_convstencil_1.py",
        "gpu=A100",
        "gpu_arch=['Ampere']",
        "backend=triton",
        "precision=fp16",
    ]

    # run_and_check resolves source paths relative to the working directory.
    previous_cwd = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        return run_and_check.main(args)
    finally:
        os.chdir(previous_cwd)


def main():
    (
        kernel_eval_result,
        ref_exec_eager_time,
        ref_exec_compile_time,
        kernel_exec_time,
    ) = check_triton()

    print(f"kernel_eval_result: {kernel_eval_result}")
    print(f"ref_exec_eager_time: {ref_exec_eager_time}")
    print(f"ref_exec_compile_time: {ref_exec_compile_time}")
    print(f"kernel_exec_time: {kernel_exec_time}")


if __name__ == "__main__":
    main()
