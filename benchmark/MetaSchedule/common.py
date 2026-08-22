"""Runtime shared by the independent TVM operator benchmark scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--implementation", choices=("raw", "dlight", "metaschedule", "vendor"),
                   default="metaschedule")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    p.add_argument("--target", default="cuda", help="TVM target kind or a JSON target")
    p.add_argument("--arch", default=None, help="CUDA architecture, e.g. sm_80 or sm_89")
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--trials", type=int, default=2000)
    p.add_argument("--builder-workers", type=int, default=8,
                   help="Concurrent MetaSchedule builds; lower this if CUDA compilation is slow")
    p.add_argument("--builder-timeout", type=float, default=300.0,
                   help="Timeout in seconds for each MetaSchedule build")
    p.add_argument("--trials-per-iter", type=int, default=64,
                   help="MetaSchedule candidates generated/built in each iteration")
    p.add_argument("--tune-number", type=int, default=3,
                   help="Runs averaged for each MetaSchedule candidate")
    p.add_argument("--tune-repeat", type=int, default=1,
                   help="Measurement repeats for each MetaSchedule candidate")
    p.add_argument("--tune-min-repeat-ms", type=int, default=100,
                   help="Minimum measurement time per MetaSchedule repeat; use 0-10 for large operators")
    p.add_argument("--runner-timeout", type=float, default=30.0,
                   help="Timeout in seconds for measuring one MetaSchedule candidate")
    p.add_argument("--work-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--number", type=int, default=5)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--min-repeat-ms", type=int, default=100)
    p.add_argument("--benchmark", choices=("kernelbench", "tvm"), default="kernelbench",
                   help="Timing protocol; kernelbench uses CUDA events")
    p.add_argument("--warmup", type=int, default=3, help="KernelBench warmup calls")
    p.add_argument("--perf-trials", type=int, default=100, help="KernelBench timed calls")
    p.add_argument("--correctness-trials", type=int, default=5)
    p.add_argument("--tolerance", type=float, default=None,
                   help="Override KernelBench dtype-specific atol and rtol")
    p.add_argument("--skip-check", action="store_true")
    p.add_argument("--export-tir", type=Path, default=None,
                   help="Write the selected/scheduled TensorIR to this file")
    return p


def _scheduled_module(mod, target, implementation: str, work_dir: Path, args):
    import tvm
    from tvm.s_tir import dlight as dl
    from tvm.s_tir import meta_schedule as ms

    if implementation == "raw":
        # CUDA cannot compile a serial host-style PrimFunc. Fallback supplies only
        # the mandatory grid/block mapping and is the closest executable raw baseline.
        with target:
            return dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
    if implementation == "dlight":
        with target:
            return dl.ApplyDefaultSchedule(
                dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(), dl.gpu.Transpose(), dl.gpu.Fallback(),
            )(mod)
    if implementation == "metaschedule":
        if args.trials <= 0:
            raise SystemExit("--trials must be positive for metaschedule")
        if args.builder_workers <= 0:
            raise SystemExit("--builder-workers must be positive")
        if args.builder_timeout <= 0:
            raise SystemExit("--builder-timeout must be positive")
        if args.trials_per_iter <= 0:
            raise SystemExit("--trials-per-iter must be positive")
        if args.tune_number <= 0 or args.tune_repeat <= 0:
            raise SystemExit("--tune-number and --tune-repeat must be positive")
        if args.tune_min_repeat_ms < 0:
            raise SystemExit("--tune-min-repeat-ms must be non-negative")
        if args.runner_timeout <= 0:
            raise SystemExit("--runner-timeout must be positive")
        database = ms.tune_tir(
            mod=mod, target=target, work_dir=str(work_dir),
            max_trials_global=args.trials,
            num_trials_per_iter=min(args.trials_per_iter, args.trials), cost_model="xgb",
            builder=ms.builder.LocalBuilder(
                max_workers=args.builder_workers,
                timeout_sec=args.builder_timeout,
            ),
            runner=ms.runner.LocalRunner(
                timeout_sec=args.runner_timeout,
                evaluator_config=ms.runner.EvaluatorConfig(
                    number=args.tune_number,
                    repeat=args.tune_repeat,
                    min_repeat_ms=args.tune_min_repeat_ms,
                    enable_cpu_cache_flush=False,
                ),
            ),
            # TVM 0.26's Droplet post-optimizer cannot reliably parse all CUDA
            # SamplePerfectTile decisions (it may treat a categorical integer
            # as a tile list).  Keep the standard MetaSchedule result instead.
            strategy="evolutionary", seed=args.seed, post_optimization=False,
        )
        sch = ms.tir_integration.compile_tir(database, mod, target)
        if sch is None:
            raise RuntimeError("MetaSchedule produced no valid schedule")
        return sch.mod
    raise AssertionError(implementation)


def run(*, name: str, mod, shapes: Sequence[Sequence[int]],
        reference: Callable[..., np.ndarray], args: argparse.Namespace,
        vendor_factory: Callable[[], object] | None = None) -> None:
    try:
        import tvm
    except ImportError as exc:
        raise SystemExit("Apache TVM >= 0.26 is required") from exc
    version = tuple(int(x) for x in tvm.__version__.split(".")[:2])
    if version < (0, 26):
        raise SystemExit(f"Apache TVM >= 0.26 is required; found {tvm.__version__}")

    if not isinstance(mod, tvm.ir.IRModule):
        mod = tvm.IRModule({"main": mod})
    dev = tvm.device(args.device, args.device_id)
    if not dev.exist:
        raise SystemExit(f"Runtime device {args.device}:{args.device_id} is unavailable")
    if args.target == "cuda":
        # A bare CUDA target (even with ``arch``) does not contain the hardware
        # limits required by MetaSchedule's GPU rules.  Detect them from the
        # selected runtime device, then honor an explicitly requested codegen
        # architecture.
        target_config = dict(tvm.target.Target.from_device(dev).export())
        if args.arch:
            target_config["arch"] = args.arch
        target = tvm.target.Target(target_config)
    else:
        target = tvm.target.Target(args.target)
    work_dir = args.work_dir or Path(__file__).parent / "logs" / name
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.implementation == "vendor":
        if vendor_factory is None:
            raise SystemExit(f"vendor implementation is unavailable for {name}")
        selected_mod = vendor_factory()
        if not isinstance(selected_mod, tvm.ir.IRModule):
            selected_mod = tvm.IRModule({"main": selected_mod})
    else:
        selected_mod = _scheduled_module(mod, target, args.implementation, work_dir, args)

    if args.export_tir:
        args.export_tir.parent.mkdir(parents=True, exist_ok=True)
        args.export_tir.write_text(selected_mod.script(show_meta=True), encoding="utf-8")

    # The 0.26 wheel's default callback may lose Target.current() during codegen
    # when cross-compiling without a visible GPU. Pin the requested architecture.
    if args.arch and target.kind.name == "cuda":
        from tvm.support import nvcc
        def compile_cuda(code):
            with target:
                return nvcc.compile_cuda(code, target_format="cubin", arch=args.arch, compiler="nvcc")
        tvm.register_global_func(
            "tvm_callback_cuda_compile", compile_cuda, override=True,
        )
    rt_mod = tvm.build(selected_mod, target=target)
    numpy_dtype = args.dtype
    if args.dtype == "bfloat16":
        try:
            import ml_dtypes
            numpy_dtype = ml_dtypes.bfloat16
        except ImportError as exc:
            raise SystemExit("bfloat16 requires the ml_dtypes package") from exc
    tolerance = args.tolerance if args.tolerance is not None else (1e-4 if args.dtype == "float32" else 1e-2)
    rng = np.random.default_rng(args.seed)
    host_inputs = [rng.uniform(-1, 1, shape).astype(numpy_dtype) for shape in shapes[:-1]]
    rt_inputs = [tvm.runtime.tensor(x, dev) for x in host_inputs]
    rt_output = tvm.runtime.empty(shapes[-1], args.dtype, dev)
    rt_mod(*rt_inputs, rt_output)
    if not args.skip_check:
        for correctness_trial in range(args.correctness_trials):
            trial_rng = np.random.default_rng(args.seed + correctness_trial)
            check_inputs = [trial_rng.uniform(-1, 1, shape).astype(numpy_dtype) for shape in shapes[:-1]]
            check_rt_inputs = [tvm.runtime.tensor(x, dev) for x in check_inputs]
            rt_mod(*check_rt_inputs, rt_output)
            np.testing.assert_allclose(
                rt_output.numpy(), reference(*check_inputs),
                rtol=tolerance, atol=tolerance,
                err_msg=f"correctness trial {correctness_trial + 1} failed",
            )

    if args.benchmark == "kernelbench":
        try:
            import torch
        except ImportError as exc:
            raise SystemExit("--benchmark kernelbench requires PyTorch") from exc
        torch.cuda.set_device(args.device_id)
        for _ in range(args.warmup):
            rt_mod(*rt_inputs, rt_output)
            torch.cuda.synchronize(args.device_id)
        times_ms = []
        for _ in range(args.perf_trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            rt_mod(*rt_inputs, rt_output)
            end.record()
            torch.cuda.synchronize(args.device_id)
            times_ms.append(start.elapsed_time(end))
        times_ms = np.asarray(times_ms)
        timing_hardware = torch.cuda.get_device_name(args.device_id)
    else:
        evaluator = rt_mod.time_evaluator(rt_mod.entry_name, dev, number=args.number,
                                          repeat=args.repeat, min_repeat_ms=args.min_repeat_ms)
        times_ms = np.asarray(evaluator(*rt_inputs, rt_output).results) * 1e3
    timing_stats = {
        "mean": float(f"{times_ms.mean():.3g}"),
        "std": float(f"{times_ms.std():.3g}"),
        "min": float(f"{times_ms.min():.3g}"),
        "max": float(f"{times_ms.max():.3g}"),
        "num_trials": int(times_ms.size),
    }
    if args.benchmark == "kernelbench":
        timing_stats.update({"hardware": timing_hardware, "device": f"cuda:{args.device_id}"})
    print(json.dumps({
        "operator": name, "implementation": args.implementation,
        "dtype": args.dtype, "tolerance": tolerance,
        "tvm_version": tvm.__version__, "target": str(target),
        "shapes": [list(x) for x in shapes], "trials": args.trials,
        "benchmark": args.benchmark, "runtime": timing_stats["mean"],
        "runtime_stats": timing_stats, "work_dir": str(work_dir),
    }, indent=2))
