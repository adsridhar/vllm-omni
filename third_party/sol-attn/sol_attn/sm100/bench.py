"""Minimal SM100 Sol-Attn forward benchmark.

Defaults match Wan2.1-T2V-14B at 720p, 81 frames: BTHD [1, 75600, 40, 128]
(21 latent frames x 45 x 80 spatial patches).

The editable sol-attn install does not include the optional SM100 runtime.
In a CUDA 13 environment, install it with::

    uv pip install --python .venv/bin/python "nvidia-cutlass-dsl[cu13]==4.8.0" apache-tvm-ffi numpy

Then run from any directory::

    python techniques/sparse_backends/sol_attn/sm100/bench.py
    python techniques/sparse_backends/sol_attn/sm100/bench.py --check
    python techniques/sparse_backends/sol_attn/sm100/bench.py --ncu
    python techniques/sparse_backends/sol_attn/sm100/bench.py --iket

Set IKET_CMD or NCU_CMD to the corresponding executable path if it is not on PATH.
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=75600)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--check", action="store_true", help="compare output with dense torch SDPA")
    parser.add_argument("--max-rel-rms", type=float, help="fail the check above this relative RMS error")
    profilers = parser.add_mutually_exclusive_group()
    profilers.add_argument("--ncu", action="store_true", help="profile one forward launch with NCU")
    profilers.add_argument("--iket", action="store_true", help="profile one forward launch with run-iket")
    parser.add_argument("--iket-cluster", default="0,0,0", help="CTA/cluster to dump with --iket (default: 0,0,0)")
    parser.add_argument("--iket-output", type=Path, help="output directory for --iket (default: timestamped directory in cwd)")
    parser.add_argument("--profile-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cuda-profiler-api", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if min(args.batch, args.tokens, args.heads, args.reps) <= 0 or args.warmup < 0:
        parser.error("batch, tokens, heads, and reps must be positive; warmup must be nonnegative")
    if args.max_rel_rms is not None:
        if not args.check or args.max_rel_rms < 0:
            parser.error("--max-rel-rms requires --check and a nonnegative value")
    if args.iket:
        try:
            cluster = [int(x) for x in args.iket_cluster.split(",")]
        except ValueError:
            parser.error("--iket-cluster must be X,Y,Z")
        if len(cluster) != 3 or any(x < 0 for x in cluster):
            parser.error("--iket-cluster must be three nonnegative integers: X,Y,Z")

    import torch
    try:
        import cutlass.cute as cute
        import tvm_ffi  # noqa: F401 - required by from_dlpack below
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing SM100 runtime dependency {exc.name!r}; see `python {__file__} --help` for the install command"
        ) from None
    from cuda.bindings import driver as cuda
    from sol_attn.common import to_cute_tensor
    from sol_attn.preprocess import prepare
    from sol_attn.sm100 import forward

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        parser.error("an SM100 CUDA GPU is required")

    torch.manual_seed(0)
    shape = (args.batch, args.tokens, args.heads, 128)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    scale = 128 ** -0.5
    kc, vc, threshold = prepare(q, k, v, tau=args.tau, scale=scale)
    out = torch.empty_like(v)
    lse = torch.empty(shape[:3], device="cuda", dtype=torch.float32)
    tensors = [to_cute_tensor(x) for x in (q, k, v, out, kc, vc, threshold, lse)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    # No sink: use the same empty block range as the public SM100 dispatch.
    sink_block = (args.tokens + 63) // 64
    kernel = cute.compile(
        forward, *tensors, scale, sink_block, sink_block,
        stream=stream, options="--enable-tvm-ffi",
    )

    def run():
        kernel(*tensors, scale, sink_block, sink_block, stream=stream)

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    if args.profile_only:
        if args.cuda_profiler_api:
            torch.cuda.cudart().cudaProfilerStart()
        run()
        torch.cuda.synchronize()
        if args.cuda_profiler_api:
            torch.cuda.cudart().cudaProfilerStop()
        return

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.reps):
        run()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / args.reps
    # 2 GEMMs (QK^T and PV), each with 2 FLOPs per multiply-add.
    dense_flops = 4 * args.batch * args.heads * args.tokens**2 * 128
    print(f"SM100 BF16 B={args.batch} T={args.tokens} H={args.heads} D=128 tau={args.tau}")
    print(f"forward: {ms:.2f} ms | dense-equivalent: {dense_flops / (ms * 1e-3) / 1e12:.2f} TFLOP/s")
    print("Timing excludes preprocessing, compilation, and allocation; actual routed FLOPs vary.")

    if args.check:
        import torch.nn.functional as F

        reference = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
        error = out.float() - reference.float()
        max_abs = error.abs().max().item()
        rms = error.square().mean().sqrt().item()
        ref_rms = reference.float().square().mean().sqrt().item()
        rel_rms = rms / max(ref_rms, 1e-12)
        print(f"vs torch SDPA: max_abs={max_abs:.6g} rms={rms:.6g} rel_rms={rel_rms:.6g}")
        if args.max_rel_rms is not None and rel_rms > args.max_rel_rms:
            raise SystemExit(f"relative RMS error {rel_rms:.6g} exceeds {args.max_rel_rms}")

    if args.ncu or args.iket:
        child = [
            sys.executable, str(Path(__file__).resolve()),
            "--batch", str(args.batch), "--tokens", str(args.tokens),
            "--heads", str(args.heads), "--tau", str(args.tau),
            "--warmup", "0", "--profile-only",
        ]
        if args.ncu:
            command = shlex.split(os.environ.get("NCU_CMD", "ncu"))
            if not command or shutil.which(command[0]) is None:
                raise SystemExit("ncu not found; install Nsight Compute or set NCU_CMD to its path")
            command += [
                "--profile-from-start", "off", "--force-overwrite",
                "--metrics",
                "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,"
                "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,"
                "sm__ops_path_tensor_op_utchmma_src_bf16_dst_fp32.sum,"
                "sm__inst_executed_pipe_tensor_subpipe_hmma_op_utchmma_utcqmma_utcomma_scope_1cta.sum,"
                "gpu__time_duration.sum",
                "-o", "sol_attn_sm100",
            ]
            child.append("--cuda-profiler-api")
        else:
            command = shlex.split(os.environ.get("IKET_CMD", "run-iket"))
            if not command:
                parser.error("IKET_CMD must contain a run-iket executable")
            if shutil.which(command[0]) is None:
                raise SystemExit(
                    "run-iket not found; install nvidia-cutlass-dsl[cu13]==4.8.0 "
                    "or set IKET_CMD to its path"
                )
            output_dir = args.iket_output or Path.cwd() / (
                "iket_sm100_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            )
            command += [
                "--output-dir", str(output_dir), "profile",
                "--enabled-cluster", args.iket_cluster,
                "--postprocess", "all", "--",
            ]
        print("Profiling:", shlex.join(command + child), flush=True)
        subprocess.run(command + child, check=True)
        if args.iket:
            renderer = Path(__file__).with_name("render_iket.py")
            for trace in output_dir.glob("*.trace.json"):
                subprocess.run([sys.executable, str(renderer), str(trace)], check=True)


if __name__ == "__main__":
    main()
