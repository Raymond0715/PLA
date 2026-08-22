"""Two-stage MetaSchedule tuning for NCHW depthwise convolution.

Stage 1 searches on a fast screening GPU (for example RTX 4070).  Stage 2
loads the screening database, takes its top-K traces, and rebuilds and measures
only those traces on the final GPU (for example A100).

The database must be copied as a directory, because it consists of both
``database_workload.json`` and ``database_tuning_record.json``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np


def make_workload(n: int, c: int, h: int, w: int, k: int, dtype: str):
    import tvm
    from tvm import te, topi

    x = te.placeholder((n, c, h, w), dtype, "X")
    weight = te.placeholder((c, 1, k, k), dtype, "W")
    y = topi.nn.depthwise_conv2d_nchw(x, weight, 1, k // 2, 1, dtype)
    prim = te.create_prim_func([x, weight, y]).with_attr("global_symbol", "main")
    return tvm.IRModule({"main": prim})


def cuda_target(tvm, device_id: int, arch: str | None):
    dev = tvm.cuda(device_id)
    if not dev.exist:
        raise SystemExit(f"CUDA device {device_id} is unavailable")
    config = dict(tvm.target.Target.from_device(dev).export())
    if arch:
        config["arch"] = arch
    return dev, tvm.target.Target(config)


def register_cuda_compiler(tvm, target, arch: str | None) -> None:
    if not arch:
        return
    from tvm.support import nvcc

    def compile_cuda(code):
        with target:
            return nvcc.compile_cuda(
                code, target_format="cubin", arch=arch, compiler="nvcc"
            )

    tvm.register_global_func("tvm_callback_cuda_compile", compile_cuda, override=True)


def search(args, mod, work_dir: Path) -> None:
    import tvm
    from tvm.s_tir import meta_schedule as ms

    dev, target = cuda_target(tvm, args.device_id, args.arch)
    del dev
    work_dir.mkdir(parents=True, exist_ok=True)
    database = ms.tune_tir(
        mod=mod,
        target=target,
        work_dir=str(work_dir),
        max_trials_global=args.trials,
        num_trials_per_iter=min(args.trials_per_iter, args.trials),
        builder=ms.builder.LocalBuilder(
            max_workers=args.builder_workers, timeout_sec=args.builder_timeout
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
        cost_model="xgb",
        strategy="evolutionary",
        seed=args.seed,
        post_optimization=False,
    )
    records = database.get_all_tuning_records()
    valid = [r for r in records if record_cost(r) < 1e9]
    print(json.dumps({
        "stage": "search",
        "hardware": str(target),
        "trials_requested": args.trials,
        "records": len(records),
        "valid_records": len(valid),
        "database": str(work_dir),
    }, indent=2))


def record_cost(record) -> float:
    """Mean screening-GPU latency in seconds; failed records sort last."""
    values = [float(x) for x in record.run_secs]
    if not values or any(not math.isfinite(x) for x in values):
        return math.inf
    return float(np.mean(values))


def load_database(ms, work_dir: Path):
    workload = work_dir / "database_workload.json"
    records = work_dir / "database_tuning_record.json"
    if not workload.is_file() or not records.is_file():
        raise SystemExit(
            f"{work_dir} is not a MetaSchedule JSON database directory "
            "(both database_workload.json and database_tuning_record.json are required)"
        )
    return ms.database.JSONDatabase(
        path_workload=str(workload), path_tuning_record=str(records)
    )


def replay(args, mod, source_dir: Path, output_dir: Path) -> None:
    import tvm
    from tvm.s_tir import meta_schedule as ms

    dev, target = cuda_target(tvm, args.device_id, args.arch)
    register_cuda_compiler(tvm, target, args.arch)
    database = load_database(ms, source_dir)
    records = sorted(database.get_all_tuning_records(), key=record_cost)
    records = [r for r in records if record_cost(r) < 1e9][: args.top_k]
    if not records:
        raise SystemExit("The source database contains no valid tuning records")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    shapes = ((args.batch, args.channels, args.height, args.width),
              (args.channels, 1, args.kernel, args.kernel))
    inputs = [tvm.runtime.tensor(rng.uniform(-1, 1, s).astype(args.dtype), dev)
              for s in shapes]
    output = tvm.runtime.empty(shapes[0], args.dtype, dev)

    results = []
    best = None
    for rank, record in enumerate(records, 1):
        row = {
            "screen_rank": rank,
            "screen_ms": record_cost(record) * 1e3,
        }
        started = time.monotonic()
        try:
            # Start from the original workload, apply only the saved scheduling
            # decisions, then code-generate for the *current* A100 target.
            sch = tvm.tir.Schedule(mod)
            record.trace.apply_to_schedule(sch, remove_postproc=False)
            rt_mod = tvm.build(sch.mod, target=target)
            rt_mod(*inputs, output)  # compilation/load warm-up
            evaluator = rt_mod.time_evaluator(
                rt_mod.entry_name,
                dev,
                number=args.number,
                repeat=args.repeat,
                min_repeat_ms=args.min_repeat_ms,
            )
            times = np.asarray(evaluator(*inputs, output).results) * 1e3
            row.update({
                "status": "ok",
                "a100_ms_mean": float(times.mean()),
                "a100_ms_median": float(np.median(times)),
                "a100_ms_min": float(times.min()),
            })
            if best is None or row["a100_ms_median"] < best[0]:
                best = (row["a100_ms_median"], rank, sch.mod)
        except Exception as exc:  # one illegal/failed trace must not abort top-K
            row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        row["wall_time_s"] = time.monotonic() - started
        results.append(row)
        print(json.dumps(row), flush=True)

    result_path = output_dir / "a100_replay_results.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if best is None:
        raise SystemExit(f"All {len(records)} traces failed; details: {result_path}")
    best_ms, best_rank, best_mod = best
    tir_path = output_dir / "best_a100_tir.py"
    tir_path.write_text(best_mod.script(show_meta=True), encoding="utf-8")
    print(json.dumps({
        "stage": "replay",
        "target": str(target),
        "source_database": str(source_dir),
        "requested_top_k": args.top_k,
        "measured": len(records),
        "best_screen_rank": best_rank,
        "best_a100_ms_median": best_ms,
        "results": str(result_path),
        "best_tir": str(tir_path),
    }, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("search", "replay"))
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--kernel", type=int, choices=(3, 7), default=7)
    p.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--arch", default=None, help="search: sm_89 for 4070; replay: sm_80 for A100")
    p.add_argument("--work-dir", type=Path, required=True,
                   help="search output DB, or replay source DB copied from the screening GPU")
    p.add_argument("--output-dir", type=Path, default=Path("a100_replay"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trials", type=int, default=2000)
    p.add_argument("--trials-per-iter", type=int, default=64)
    p.add_argument("--builder-workers", type=int, default=8)
    p.add_argument("--builder-timeout", type=float, default=300.0)
    p.add_argument("--runner-timeout", type=float, default=30.0)
    p.add_argument("--tune-number", type=int, default=3)
    p.add_argument("--tune-repeat", type=int, default=1)
    p.add_argument("--tune-min-repeat-ms", type=int, default=10)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--number", type=int, default=5)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--min-repeat-ms", type=int, default=100)
    args = p.parse_args()
    if args.trials <= 0 or args.top_k <= 0:
        p.error("--trials and --top-k must be positive")
    return args


def main():
    args = parse_args()
    mod = make_workload(
        args.batch, args.channels, args.height, args.width, args.kernel, args.dtype
    )
    if args.stage == "search":
        search(args, mod, args.work_dir)
    else:
        replay(args, mod, args.work_dir, args.output_dir)


if __name__ == "__main__":
    main()
