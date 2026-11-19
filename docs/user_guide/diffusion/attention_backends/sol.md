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
python -m pip install ./third_party/sol-attn
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

For Cosmos3, select SOL through a generation-only strategy preset. Understanding
attention is causal and must remain on a dense backend.

`tau` defaults to `1.0`; larger values route fewer blocks to exact attention.
`thresh_type` is `diag` (default) or `exact`, which uses full covariance for
the routing threshold. Neither option guarantees a fixed sparsity ratio.
Skip-Softmax thresholds and calibration coefficients do not apply to SOL.

Use attention strategies for step and layer selection. For example:

```json
{
  "presets": {
    "dense": {"backend": "FLASH_ATTN"},
    "sol": {"backend": "SOL_ATTN", "sol": {"tau": 1.0}}
  },
  "layouts": {
    "dense": {"default": "dense"},
    "mixed": {
      "default": "dense",
      "overrides": [
        {"attention_role": "cosmos3.gen", "layers": [0], "use": "sol"}
      ]
    }
  },
  "schedule": {
    "coordinate": "step_index",
    "phases": [
      {"until": 2, "layout": "dense"},
      {"until": null, "layout": "mixed"}
    ]
  }
}
```

This uses FlashAttention for the first two iterations, then SOL on generation
layer zero. Every other attention operation stays on FlashAttention. The strategy
runner selects the layout outside compiled execution; SOL has no independent
step gate or dense fallback. See the attention strategy design for its topology
and compilation constraints. SOL currently requires sequence parallelism disabled.

The adapter consumes Cosmos3's `protected_kv_prefix` metadata directly, preserving
text conditioning without a model-code patch. Its custom operator keeps kernel
compilation outside Dynamo. Warm up each input shape before CUDA graph capture.

FP8 model weights are independent of attention arithmetic. The standard SOL path
requires BF16 Q/K/V, including when the checkpoint selects FP8 linear operations.
It preserves the checkpoint's diffusion-step precision policy.

The bootstrap under `scripts/physicsiq/` installs the bundled public SOL sources
and the worker dependencies before running a command. Benchmark inputs,
checkpoint locations, and scientific recipes belong to the caller.

When using SageAttention 2.2 for dense strategy presets on Hopper, apply
`scripts/physicsiq/sage-current-stream.patch` to the public SageAttention
`v2.2.0` source (`eb615cf6cf4d221338033340ee2de1c37fbdba4a`) before building it.
The public release launches its fused preprocessing and SM90 kernels on the
default stream. The patch uses PyTorch's current CUDA stream so those launches
respect the caller's tensor dependencies. It preserves the kernel arithmetic
and also renames a fake implementation that otherwise shadows the real SM90
operation in Python. The wheel is labeled `2.2.0+torchstream1`.
The non-default-stream regression in
`test_sol_attn_cuda.py` covers this integration boundary.
