# SOL Attention (CuTe DSL)

`SOL_ATTN` is an opt-in adapter for the reference
[Sol-Attn](https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends)
CuTe DSL kernel. It computes selected blocks exactly and includes proxy
contributions from the other blocks. No retraining or offline calibration is
required; tune output quality against a dense baseline for your checkpoint.

## Installation

Use PyTorch 2.10 or newer and CUDA 12.8 or newer. Install the optional kernel
package and CuTe runtime in the worker environment:

```bash
python -m pip install 'nvidia-cutlass-dsl>=4.5' cuda-python apache-tvm-ffi
python -m pip install 'sol-attn @ git+https://github.com/NVlabs/Sana.git@92f2759f4b7754673a6edf2fab71a4f51568b664#subdirectory=techniques/sparse_backends'
```

The adapter uses the reference kernel directly; the TensorRT-LLM Python
package is not required. This adapter supports SM90 (H100/H200) and SM100 (B200/GB200),
BF16, head dimension 128, and noncausal self-attention. GQA is adapted by
expanding KV heads. Cosmos3 GEN attention supplies explicit text-prefix
metadata: the adapter pads unused text query rows, keeps text KV blocks exact,
and returns only video-query outputs. This preserves the conditioning context
but uses extra temporary memory and compute compared with a native GQA kernel.
Unequal Q/KV lengths require that explicit prefix contract.
Sequence parallelism, packed sequences, masks, quantized attention, and
general cross-attention are outside its supported path. Select a dense backend for
cross-attention. The adapter rejects the reference package's Triton fallback.

## Configuration

Select SOL only for the model's self-attention role:

```bash
vllm-omni serve <model> --diffusion-attention-config '{
  "default": {"backend": "TORCH_SDPA"},
  "per_role": {
    "self": {
      "backend": "SOL_ATTN",
      "sol": {"tau": 1.0, "thresh_type": "diag"}
    }
  }
}'
```

`tau` defaults to `1.0`; larger values route fewer blocks to exact attention.
`thresh_type` is `diag` (default) or `exact`, which uses full covariance for
the routing threshold. Neither option guarantees a fixed sparsity ratio.
Skip-Softmax thresholds and calibration coefficients do not apply to SOL.

By default, SOL applies at every denoising step on the selected roles. Set
`sol.start_step` to keep the first N steps dense, and optionally set
`sol.stop_step` to return to dense attention at that zero-based step. SOL is
active on `[start_step, stop_step)`; an omitted `stop_step` means through the
last step. For example, `"sol": {"tau": 0.85, "start_step": 5}` uses cuDNN
on steps 0–4 and SOL from step 5 onward. Step-gated SOL requires the pipeline
to publish its current denoising step; Cosmos3 does this in its denoising loop.
Exact per-role entries still select the eligible layers. The adapter wraps the
SOL kernel in a custom operator for `torch.compile`; warm up each input shape
before CUDA graph capture so JIT compilation finishes outside capture.

## Cosmos3 PAI runner

From the repository root, run one cached prompt/image pair with seed zero:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_sol.sh --smoke --sol-tau 1.0 --sol-thresh-type diag
```

Omit `--smoke` for the full 50-pair, three-seed benchmark. Add `--dry-run`
to validate cached inputs and write the manifest without loading the model.
The launcher also accepts `SOL_TAU` and `SOL_THRESH_TYPE` environment defaults;
explicit CLI values take precedence.

To use SOL only on the first 70% of GEN layers:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_sol.sh --smoke --sol-first-fraction 0.7
```

The fraction rounds up: 36 layers selects layers 0–25 (26 layers); layers
26–35 use the dense backend. It defaults to `1.0` and can also be set through
`SOL_FIRST_FRACTION`. The reference kernel tracks route counts internally but
does not expose exact/approximate block counters through its public API.

The runner resolves the checkpoint's generation layer count and selects SOL
for the selected `gen_layers.<index>.cross_attention.attn` roles. Other roles use
`CUDNN_ATTN`, including the causal UND path. Workers read the SOL controls from
the saved manifest. No Skip-Softmax calibration file is needed.
