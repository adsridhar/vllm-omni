"""L2 gather probe for the Wan-size Sol-Attn K/V access pattern.

This deliberately does no MMA or softmax. It reads the same 64-token BF16
stripes from BTHD K/V and the same 128-summary route stripes from Kc/Vc as
the SM100 kernel. Indices are synthetic and are prepared outside timing.
"""

import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def gather_probe(k, v, kc, vc, indices, result, tokens: tl.constexpr,
                 blocks: tl.constexpr, heads: tl.constexpr, pairs: tl.constexpr):
    q_block = tl.program_id(0)
    head = tl.program_id(1)
    row = tl.arange(0, 64)
    dim = tl.arange(0, 128)
    acc = tl.full((64, 128), 0, tl.float32)

    # One N128 Kc and Vc transaction per route tile (two N64 halves).
    for route in range(tl.cdiv(blocks, 128)):
        for half in range(2):
            summary = route * 128 + half * 64 + row
            offsets = (summary[:, None] * heads + head) * 128 + dim[None, :]
            valid = summary[:, None] < blocks
            acc += tl.load(kc + offsets, valid, other=0, cache_modifier=".cg").to(tl.float32)
            acc += tl.load(vc + offsets, valid, other=0, cache_modifier=".cg").to(tl.float32)

    # Each pair is two independently selected N64 token blocks, as in the
    # exact path. The source layout has a heads*128*2-byte row stride.
    base = (head * tl.cdiv(tokens, 64) + q_block) * pairs * 2
    for pair in range(pairs):
        for half in range(2):
            block = tl.load(indices + base + pair * 2 + half)
            token = block * 64 + row
            offsets = (token[:, None] * heads + head) * 128 + dim[None, :]
            valid = token[:, None] < tokens
            acc += tl.load(k + offsets, valid, other=0, cache_modifier=".cg").to(tl.float32)
            acc += tl.load(v + offsets, valid, other=0, cache_modifier=".cg").to(tl.float32)

    tl.store(result + head * tl.cdiv(tokens, 64) + q_block, tl.sum(acc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=75600)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--exact-pairs", type=int, default=96)
    parser.add_argument("--warps", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    blocks = triton.cdiv(args.tokens, 64)
    shape = (args.tokens, args.heads, 128)
    summary_shape = (blocks, args.heads, 128)
    k, v = (torch.empty(shape, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    kc, vc = (torch.empty(summary_shape, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    for tensor in (k, v, kc, vc):
        tensor.fill_(1)
    indices = torch.randint(
        blocks, (args.heads, blocks, args.exact_pairs, 2),
        device="cuda", dtype=torch.int32,
    )
    result = torch.empty((args.heads, blocks), device="cuda", dtype=torch.float32)

    def run():
        gather_probe[(blocks, args.heads)](
            k, v, kc, vc, indices, result,
            args.tokens, blocks, args.heads, args.exact_pairs,
            num_warps=args.warps,
        )

    run()  # Compile before timing and before CUDA profiler activation.
    torch.cuda.synchronize()
    if args.profile_only:
        torch.cuda.cudart().cudaProfilerStart()
        run()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return

    ms = triton.testing.do_bench(run, warmup=200, rep=args.reps * 100)
    payload = blocks * args.heads * (
        triton.cdiv(blocks, 128) * 2 * 128 * 128 * 2
        + args.exact_pairs * 2 * 128 * 128 * 2
    )
    print(f"T={args.tokens} H={args.heads} exact_pairs={args.exact_pairs} "
          f"warps={args.warps} time={ms:.3f} ms "
          f"payload={payload / 1e9:.2f} GB "
          f"payload_rate={payload / (ms * 1e-3) / 1e12:.2f} TB/s")
    print(f"checksum={result.sum().item():.0f}")


if __name__ == "__main__":
    main()
