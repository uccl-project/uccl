import argparse
import importlib
import math
import os
import sys
from pathlib import Path

import torch


EP_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EP_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))


STAGE_ORDER = (
    "dispatch",
    "expanded_dispatch",
    "cached_dispatch",
    "combine",
    "reduced_combine",
)
FULL_PATH_MODE = (1, 128, 0, 0, 0, 0, 0)


def load_test_ep():
    return importlib.import_module("test_ep")


def parse_stages(value: str) -> tuple[str, ...]:
    if value == "all":
        return STAGE_ORDER
    stages = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = [stage for stage in stages if stage not in STAGE_ORDER]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown stage(s): {', '.join(unknown)}; valid values are: all,{','.join(STAGE_ORDER)}"
        )
    if not stages:
        raise argparse.ArgumentTypeError("at least one stage must be selected")
    return stages


def apply_transport_env(transport: str) -> None:
    os.environ.setdefault("EP_USE_UCCL_PROXY", "1")
    os.environ.setdefault("EP_FORCE_NO_NVLINK", "1")
    os.environ.setdefault("EP_SUPPRESS_NCCL_CHECK", "1")

    if transport == "nogdr":
        os.environ["UCCL_FORCE_NO_GDR"] = "1"
        os.environ["EP_FORCE_HOST_WINDOW"] = "1"
        os.environ["NCCL_NET_GDR_LEVEL"] = "0"
    elif transport == "gdr":
        os.environ["UCCL_FORCE_NO_GDR"] = "0"
        os.environ["EP_FORCE_HOST_WINDOW"] = "0"
        os.environ["NCCL_NET_GDR_LEVEL"] = "5"


def make_test_args(args: argparse.Namespace) -> argparse.Namespace:
    test_args = argparse.Namespace(
        num_processes=args.num_processes,
        num_sms=args.num_sms,
        num_qps=args.num_qps,
        num_allocated_qps=args.num_allocated_qps,
        num_gpu_timeout_secs=args.num_gpu_timeout_secs,
        num_cpu_timeout_secs=args.num_cpu_timeout_secs,
        sl_idx=args.sl_idx,
        num_tokens=args.num_tokens,
        hidden=args.hidden,
        num_topk=args.num_topk,
        num_experts=args.num_experts,
        do_cpu_sync=args.do_cpu_sync,
        allow_hybrid_mode=args.allow_hybrid_mode,
        allow_multiple_reduction=args.allow_multiple_reduction,
        prefer_overlap_with_compute=args.prefer_overlap_with_compute,
        deterministic=args.deterministic,
        seed=args.seed,
        trace_steps=args.trace_steps,
        skip_check=not args.check,
        skip_perf_test=args.validate_only,
        num_bench_tests=args.num_bench_tests,
        do_pressure_test=False,
        reuse_elastic_buffer=False,
        test_first_only=True,
        do_handle_copy_modes="1",
        expert_alignment_modes="128",
        fp8_dispatch_modes="0",
        unbalanced_ratio=args.unbalanced_ratio,
        precise_unbalanced_ratio=args.precise_unbalanced_ratio,
        masked_ratio=args.masked_ratio,
        dump_profile_traces=args.dump_profile_traces,
        ignore_local_traffic=args.ignore_local_traffic,
    )

    if int(os.environ.get("EP_FORCE_NO_NVLINK", "0")):
        test_args.allow_hybrid_mode = 0
        test_args.allow_multiple_reduction = 0
        test_args.num_qps = 1
        test_args.num_allocated_qps = max(test_args.num_allocated_qps, 1)
    return test_args


def install_mode_filter(test_ep) -> None:
    def enumerate_full_path_mode(_args=None):
        yield FULL_PATH_MODE

    test_ep.enumerate_ep_modes = enumerate_full_path_mode


def install_stage_filter(test_ep, measured_stages: tuple[str, ...], num_bench_tests: int) -> None:
    original_bench_kineto = test_ep.bench_kineto
    call_index = {"value": 0}
    measured_stage_set = set(measured_stages)

    def bench_kineto_full_path(fn, kernel_names, **kwargs):
        index = call_index["value"]
        call_index["value"] += 1
        stage = STAGE_ORDER[index] if index < len(STAGE_ORDER) else f"extra_{index}"

        if stage not in measured_stage_set:
            fn()
            torch.cuda.synchronize()
            if isinstance(kernel_names, tuple):
                return tuple(math.nan for _ in kernel_names)
            return math.nan

        kwargs["num_tests"] = num_bench_tests
        return original_bench_kineto(fn, kernel_names, **kwargs)

    test_ep.bench_kineto = bench_kineto_full_path


def run_worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    test_ep = load_test_ep()
    install_mode_filter(test_ep)
    install_stage_filter(test_ep, args.measure_stages, args.num_bench_tests)
    test_ep.test_loop(local_rank, num_local_ranks, args.test_args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeepEPv2 full-path UCCL benchmark stages")
    parser.add_argument("--transport", choices=("env", "nogdr", "gdr"), default="env",
                        help="Set common UCCL transport envs, or leave them unchanged")
    parser.add_argument("--measure-stages", type=parse_stages, default=STAGE_ORDER,
                        help="Comma-separated stages to profile, or 'all'")
    parser.add_argument("--num-bench-tests", type=int, default=1,
                        help="Kineto measurement iterations for selected stages")
    parser.add_argument("--validate-only", action="store_true",
                        help="Run full path without perf profiling")
    parser.add_argument("--check", action="store_true",
                        help="Enable correctness checks instead of benchmark-only validation")
    parser.add_argument("--trace-steps", action="store_true",
                        help="Print per-rank full-path trace steps")

    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--num-sms", type=int, default=0)
    parser.add_argument("--num-qps", type=int, default=0)
    parser.add_argument("--num-allocated-qps", type=int, default=0)
    parser.add_argument("--num-gpu-timeout-secs", type=int, default=100)
    parser.add_argument("--num-cpu-timeout-secs", type=int, default=100)
    parser.add_argument("--sl-idx", type=int, default=0)

    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=64)

    parser.add_argument("--do-cpu-sync", type=int, default=1)
    parser.add_argument("--allow-hybrid-mode", type=int, default=1)
    parser.add_argument("--allow-multiple-reduction", type=int, default=1)
    parser.add_argument("--prefer-overlap-with-compute", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unbalanced-ratio", type=float, default=1.0)
    parser.add_argument("--precise-unbalanced-ratio", action="store_true")
    parser.add_argument("--masked-ratio", type=float, default=0.0)
    parser.add_argument("--dump-profile-traces", type=str, default="")
    parser.add_argument("--ignore-local-traffic", action="store_true")
    args = parser.parse_args()

    if args.num_bench_tests < 1:
        raise ValueError("--num-bench-tests must be positive")
    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)
    apply_transport_env(args.transport)
    args.test_args = make_test_args(args)

    torch.multiprocessing.spawn(
        run_worker, args=(args.num_processes, args), nprocs=args.num_processes
    )
    if int(os.environ.get("EP_FORCE_PROCESS_EXIT", "0")):
        os._exit(0)


if __name__ == "__main__":
    main()
