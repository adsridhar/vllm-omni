"""Dense FlashAttention-4 forward benchmark at the Sol-Attn Wan shape.

Install the official FA4 package with ``uv pip install --prerelease allow
'flash-attn-4[cu13]'``. The inputs, seed, event timing, and default shape match
bench.py; this times forward attention only, without input preparation.
"""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=75600)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()

    from flash_attn.cute import flash_attn_func

    torch.manual_seed(0)
    shape = (args.batch, args.tokens, args.heads, 128)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    def run():
        return flash_attn_func(q, k, v, causal=False)[0]

    with torch.inference_mode():
        run()  # Compile before warmup and timing.
        for _ in range(args.warmup):
            run()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.reps):
            out = run()
        end.record()
        end.synchronize()

    ms = start.elapsed_time(end) / args.reps
    flops = 4 * args.batch * args.heads * args.tokens**2 * 128
    print(f"FA4 dense BF16 B={args.batch} T={args.tokens} H={args.heads} D=128")
    print(f"forward: {ms:.2f} ms | {flops / (ms * 1e-3) / 1e12:.2f} TFLOP/s")
    print(f"output shape: {tuple(out.shape)}")


if __name__ == "__main__":
    main()
