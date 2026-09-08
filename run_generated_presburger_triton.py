"""Run a generated Triton or CUDA ModelNew operator against the PyTorch reference."""

from pathlib import Path
import argparse
import json
import os

import torch

from scripts import run_and_check

# torch.backends.cudnn.allow_tf32 = False
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.set_float32_matmul_precision("highest")


PROJECT_ROOT = Path(__file__).resolve().parent


def check_triton(ref_arch_src_path=None, kernel_src_path=None, backend="triton"):
    """Check correctness and performance of the generated ModelNew operator."""
    args = [
        "ref_origin=local",
        # "ref_arch_src_path=data/depthwise_conv_7x7.py",
        # "kernel_src_path=data/depthwise_conv_7x7_loop_2026_08_15_17_44_09/map2mm_idx0_sample1_S0P3-4_S0P6-2_fix.py",
        # "kernel_src_path=data/depthwise_conv_7x7_loop_2026_08_25_18_37_52/map2mm_idx0_sample1_S0P3-4_S0P6-2_fix.py",
        # "kernel_src_path=data/depthwise_conv_7x7_loop_2026_08_30_12_32_19/map2mm_idx0_sample1_S0P3-4_S0P6-2.py",
        # "kernel_src_path=data/depthwise_conv_7x7_cuda/model_cutlass.py",
        # "kernel_src_path=data/depthwise_conv_7x7_cuda/model_cutlass_autotune_v7.py",
        # "kernel_src_path=benchmark/MetaSchedule/data/depthwise_1_8_2048_2048_k7_float16_metaschedule.py",
        # "kernel_src_path=data/depthwise_conv_7x7_cuda/model_v1.py",
        "ref_arch_src_path=data/stencil_7x7.py",
        # "kernel_src_path=data/stencil_7x7_loop_2026_09_05_18_28_45/map2mm_idx12_sample0_S0P3-2_S0P6-4_perf_candidate_codex_v6.py",
        # "ref_arch_src_path=data/stencil_3x3.py",
        # "kernel_src_path=data/stencil_3x3_loop.py",
        "kernel_src_path=data/2026_08_30/stencil_7x7_cuda/model_v1.py",
        # "kernel_src_path=../PolyLLMAgentTriton/triton_convstencil_1.py",
        # "ref_arch_src_path=data/conv_7x7.py",
        # "kernel_src_path=data/conv_7x7_loop_2026_08_31_14_10_18/map2mm_idx21_sample0_no_inter_params_perf_candidate_fix.py",
        # "ref_arch_src_path=data/conv_7x7.py",
        # "kernel_src_path=data/conv_7x7_codex_v3.py",
        "backend=triton",
        "precision=fp16",
    ]
    overrides = {
        "ref_arch_src_path": ref_arch_src_path,
        "kernel_src_path": kernel_src_path,
        "backend": backend,
    }
    for index, arg in enumerate(args):
        key = arg.split("=", 1)[0]
        if overrides.get(key) is not None:
            args[index] = f"{key}={overrides[key]}"

    # run_and_check resolves source paths relative to the working directory.
    previous_cwd = Path.cwd()
    previous_model_dir = os.environ.get("KERNELBENCH_MODEL_DIR")
    try:
        os.chdir(PROJECT_ROOT)
        kernel_src_arg = next(arg for arg in args if arg.startswith("kernel_src_path="))
        kernel_src_path = Path(kernel_src_arg.split("=", 1)[1]).resolve()
        if not kernel_src_path.is_file():
            raise FileNotFoundError(f"generated kernel does not exist: {kernel_src_path}")
        if kernel_src_path.stem.endswith("_metaschedule"):
            module_path = kernel_src_path.with_suffix(".so")
            if not module_path.is_file():
                raise FileNotFoundError(
                    f"generated TVM module does not exist: {module_path}")
        os.environ["KERNELBENCH_MODEL_DIR"] = str(kernel_src_path.parent)
        return run_and_check.main(args)
    finally:
        if previous_model_dir is None:
            os.environ.pop("KERNELBENCH_MODEL_DIR", None)
        else:
            os.environ["KERNELBENCH_MODEL_DIR"] = previous_model_dir
        os.chdir(previous_cwd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref-arch-src-path", "--ref_arch_src_path",
        help="Reference source path (relative to the project root or absolute); defaults to the path in check_triton.",
    )
    parser.add_argument(
        "--kernel-src-path", "--kernel_src_path",
        help="Kernel source path (relative to the project root or absolute); defaults to the path in check_triton.",
    )
    parser.add_argument("--backend", choices=("triton", "cuda"), default="triton")
    parser.add_argument("--result-json", type=Path, help="Write correctness and timing results as JSON.")
    args = parser.parse_args(argv)
    (
        kernel_eval_result,
        ref_exec_eager_time,
        ref_exec_compile_time,
        kernel_exec_time,
    ) = check_triton(args.ref_arch_src_path, args.kernel_src_path, args.backend)

    print(f"kernel_eval_result: {kernel_eval_result}")
    print(f"ref_exec_eager_time: {ref_exec_eager_time}")
    print(f"ref_exec_compile_time: {ref_exec_compile_time}")
    print(f"kernel_exec_time: {kernel_exec_time}")
    if args.result_json:
        args.result_json.write_text(json.dumps({
            "compiled": bool(getattr(kernel_eval_result, "compiled", False)),
            "correctness": bool(getattr(kernel_eval_result, "correctness", False)),
            "runtime_ms": kernel_exec_time,
            "ref_eager_ms": ref_exec_eager_time,
            "ref_compile_ms": ref_exec_compile_time,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
