"""Compare SM100 Sol-Attn tau=1 with vLLM Omni's cuDNN SDPA path.

Cosmos3 uses 32 Q heads, 8 KV heads, D=128. Sol-Attn currently needs equal
head counts, so its K/V inputs are materialized with 4 repeated heads each.
cuDNN uses native GQA. Both see the same logical Q/K/V values; Sol is sparse
and approximate, whereas cuDNN computes dense attention.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def time_cuda(fn, warmup, reps):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / reps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-tokens", type=int, default=44000)
    parser.add_argument("--kv-tokens", type=int, default=46000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--check", action="store_true", help="report output error against dense cuDNN")
    args = parser.parse_args()
    if min(args.q_tokens, args.kv_tokens, args.batch, args.q_heads, args.kv_heads, args.reps) <= 0:
        parser.error("shape and reps must be positive")
    if args.warmup < 0 or args.q_heads % args.kv_heads:
        parser.error("warmup must be nonnegative and q-heads must be divisible by kv-heads")

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from sol_attn.common import to_cute_tensor
    from sol_attn.preprocess import prepare
    from sol_attn.sm100 import forward

    if torch.cuda.get_device_capability() != (10, 0):
        parser.error("SM100 GPU required")
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    d = 128
    scale = d ** -0.5
    q = torch.randn(args.batch, args.q_tokens, args.q_heads, d, device=device, dtype=dtype)
    k_gqa = torch.randn(args.batch, args.kv_tokens, args.kv_heads, d, device=device, dtype=dtype)
    v_gqa = torch.randn_like(k_gqa)
    group = args.q_heads // args.kv_heads
    k = k_gqa.repeat_interleave(group, dim=2).contiguous()
    v = v_gqa.repeat_interleave(group, dim=2).contiguous()

    def preprocess():
        return prepare(q, k, v, tau=args.tau, scale=scale)

    kc, vc, threshold = preprocess()
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:3], device=device, dtype=torch.float32)
    tensors = [to_cute_tensor(x) for x in (q, k, v, out, kc, vc, threshold, lse)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    sink_block = (args.kv_tokens + 63) // 64
    kernel = cute.compile(forward, *tensors, scale, sink_block, sink_block,
                          stream=stream, options="--enable-tvm-ffi")

    def sol():
        kernel(*tensors, scale, sink_block, sink_block, stream=stream)

    q_bhtd = q.permute(0, 2, 1, 3)
    k_bhtd = k_gqa.permute(0, 2, 1, 3)
    v_bhtd = v_gqa.permute(0, 2, 1, 3)

    def cudnn():
        # Same implementation as CuDNNAttentionImpl._attention in vLLM Omni:
        # a single pinned cuDNN SDPA backend, with native GQA enabled.
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            return F.scaled_dot_product_attention(
                q_bhtd, k_bhtd, v_bhtd, dropout_p=0.0,
                is_causal=False, scale=scale, enable_gqa=group != 1,
            )

    with torch.inference_mode():
        # Compile and build the cuDNN plan before either timer starts.
        sol()
        cudnn()
        prep_ms = time_cuda(preprocess, args.warmup, args.reps)
        sol_ms = time_cuda(sol, args.warmup, args.reps)
        cudnn_ms = time_cuda(cudnn, args.warmup, args.reps)

    dense_flops = 4 * args.batch * args.q_heads * args.q_tokens * args.kv_tokens * d
    print(f"GPU: {torch.cuda.get_device_name()} | cuDNN: {torch.backends.cudnn.version()}")
    print(f"BF16 Q=[{args.batch},{args.q_tokens},{args.q_heads},{d}] "
          f"KV=[{args.batch},{args.kv_tokens},{args.kv_heads},{d}] tau={args.tau}")
    print(f"SM100 preprocessing: {prep_ms:.3f} ms")
    print(f"SM100 attention:     {sol_ms:.3f} ms")
    print(f"SM100 total:         {prep_ms + sol_ms:.3f} ms (sum of separate timings)")
    print(f"cuDNN dense GQA:     {cudnn_ms:.3f} ms")
    print(f"Speedup, attention only: {cudnn_ms / sol_ms:.3f}x")
    print(f"Speedup, with SM100 preprocessing: {cudnn_ms / (prep_ms + sol_ms):.3f}x")
    print(f"Dense-equivalent TFLOP/s: SM100 {dense_flops / (sol_ms * 1e9):.2f}, "
          f"cuDNN {dense_flops / (cudnn_ms * 1e9):.2f}")
    if args.check:
        with torch.inference_mode():
            sol()
            reference = cudnn().permute(0, 2, 1, 3)
            torch.cuda.synchronize()
            error = out.float() - reference.float()
            rel_rms = (error.square().mean().sqrt() /
                       reference.float().square().mean().sqrt()).item()
            max_abs = error.abs().max().item()
        print(f"vs dense cuDNN: relative RMS={rel_rms:.6g}, max abs={max_abs:.6g}")


if __name__ == "__main__":
    main()
