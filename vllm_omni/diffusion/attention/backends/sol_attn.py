# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SM90/SM100 adapter for the reference Sol-Attn CuTe DSL kernel."""

import math
from functools import cache

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionBackend, AttentionImpl, AttentionMetadata
from vllm_omni.diffusion.attention.backends.cudnn_attn import CuDNNAttentionImpl
from vllm_omni.diffusion.attention.capabilities import AttentionComputation
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available


@cache
def _load_sol_attn():
    try:
        import cuda.bindings.driver  # noqa: F401
        import cutlass.cute  # noqa: F401
        from sol_attn import get_sol_attn_backend, sol_attn
    except ImportError as exc:
        raise ImportError(
            "SOL_ATTN requires sol-attn from NVlabs/Sana (techniques/sparse_backends), "
            "nvidia-cutlass-dsl, cuda-python and apache-tvm-ffi. See the SOL attention guide."
        ) from exc
    return get_sol_attn_backend, sol_attn


# Keep the kernel's Python compilation cache and launch code opaque to Dynamo.
if not hasattr(torch.ops.vllm_omni, "sol_attention"):

    @torch.library.custom_op("vllm_omni::sol_attention", mutates_args=())
    def _sol_attention_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        tau: float,
        thresh_type: str,
        kv_prefix_length: int = 0,
    ) -> torch.Tensor:
        get_backend, kernel = _load_sol_attn()
        # The public reference API otherwise silently selects Triton when CuTe
        # is unavailable or the GPU is unsupported. This adapter promises CuTe.
        if query.device.type != "cuda" or get_backend(query.device) not in ("cute_sm90", "cute_sm100"):
            raise ValueError("SOL_ATTN requires the SM90 or SM100 CuTe DSL backend (H100/H200/B200/GB200).")
        # The reference kernel serves equal-length MHA. Expand GQA heads and
        # supply unused text-query rows for Cosmos3's [text, video] KV layout.
        # Queries are independent in noncausal attention, so these extra rows
        # do not change the set of keys seen by any real video query.
        repeats = query.shape[2] // key.shape[2]
        if repeats != 1:
            key = key.repeat_interleave(repeats, dim=2)
            value = value.repeat_interleave(repeats, dim=2)
        if kv_prefix_length:
            padding = query.new_zeros(query.shape[0], kv_prefix_length, query.shape[2], query.shape[3])
            query = torch.cat((padding, query), dim=1)
        output = kernel(
            query,
            key,
            value,
            scale=scale,
            tau=tau,
            thresh_type=thresh_type,
            sink_start=0,
            sink_tokens=kv_prefix_length,
        )
        return output[:, kv_prefix_length:].contiguous()

    @_sol_attention_op.register_fake
    def _sol_attention_fake(query, key, value, scale, tau, thresh_type, kv_prefix_length=0):
        return torch.empty(query.shape, dtype=query.dtype, device=query.device)


_sol_attention_op = torch.ops.vllm_omni.sol_attention


class SolAttentionBackend(AttentionBackend):
    supported_platforms = ("cuda",)

    @classmethod
    def validate_available(cls) -> None:
        get_backend, _ = _load_sol_attn()
        if get_backend() not in ("cute_sm90", "cute_sm100"):
            raise ValueError("SOL_ATTN requires the SM90 or SM100 CuTe DSL backend (H100/H200/B200/GB200).")

    @staticmethod
    def get_name() -> str:
        return "SOL_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SolAttentionImpl"]:
        return SolAttentionImpl

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [128]

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError("SOL_ATTN uses caller-provided AttentionMetadata")


class SolAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if causal or head_size != 128 or num_heads <= 0 or kv_heads <= 0 or num_heads % kv_heads:
            raise ValueError(
                "SOL_ATTN requires noncausal attention with head_size=128 and Q heads divisible by KV heads."
            )
        if qkv_layout not in (None, "BSND"):
            raise ValueError("SOL_ATTN requires BSND Q/K/V layout.")
        if extra_impl_args.get("computation") == AttentionComputation.CROSS:
            raise ValueError("SOL_ATTN supports self-attention only; select a dense backend for cross-attention.")
        if extra_impl_args.get("dropout_p", 0.0) != 0.0:
            raise ValueError("SOL_ATTN does not support dropout.")
        self.num_heads = num_heads
        self.num_kv_heads = kv_heads
        self.softmax_scale = float(softmax_scale)
        if not math.isfinite(self.softmax_scale) or self.softmax_scale <= 0:
            raise ValueError("SOL_ATTN requires a finite positive softmax_scale.")
        kwargs = backend_kwargs or {}
        self.tau = float(kwargs.get("tau", 1.0))
        self.thresh_type = kwargs.get("thresh_type", "diag")
        self.start_step = kwargs.get("start_step", 0)
        self.stop_step = kwargs.get("stop_step")
        if not math.isfinite(self.tau):
            raise ValueError("SOL_ATTN tau must be finite.")
        if self.thresh_type not in ("diag", "exact"):
            raise ValueError("SOL_ATTN thresh_type must be 'diag' or 'exact'.")
        if type(self.start_step) is not int or self.start_step < 0:
            raise ValueError("SOL_ATTN start_step must be a nonnegative integer.")
        if self.stop_step is not None and (type(self.stop_step) is not int or self.stop_step <= self.start_step):
            raise ValueError("SOL_ATTN stop_step must be an integer greater than start_step.")
        self._dense_attention = (
            CuDNNAttentionImpl(
                num_heads=num_heads,
                head_size=head_size,
                softmax_scale=softmax_scale,
                causal=causal,
                num_kv_heads=kv_heads,
                prefix=prefix,
                backend_explicit=True,
            )
            if self.start_step or self.stop_step is not None
            else None
        )

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if self._dense_attention is not None:
            step = get_forward_context().denoise_step_idx if is_forward_context_available() else None
            if step is None:
                raise RuntimeError("Step-gated SOL_ATTN requires a current denoising step in ForwardContext.")
            if step < self.start_step or (self.stop_step is not None and step >= self.stop_step):
                return self._dense_attention.forward_cuda(query, key, value, attn_metadata)
        kv_prefix_length = 0
        if attn_metadata is not None:
            extra = attn_metadata.extra
            kv_prefix_length = extra.get("sol_kv_prefix_length", 0)
            if (
                attn_metadata.attn_mask is not None
                or attn_metadata.joint_attn_mask is not None
                or attn_metadata.full_attn_spans is not None
                or attn_metadata.packed_padding is not None
                or any(name in extra for name in ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k"))
                or extra.get("valid_kv_length") is not None
            ):
                raise ValueError("SOL_ATTN requires unmasked, unpacked self-attention.")
        if any(t.ndim != 4 for t in (query, key, value)) or key.shape != value.shape:
            raise ValueError("SOL_ATTN requires rank-4 Q/K/V and matching K/V shapes.")
        if (
            query.shape[0] != key.shape[0]
            or query.shape[2] != self.num_heads
            or key.shape[2] != self.num_kv_heads
            or query.shape[3] != 128
            or key.shape[3] != 128
            or any(s == 0 for t in (query, key, value) for s in t.shape)
        ):
            raise ValueError(
                "SOL_ATTN requires nonempty Q/K/V with matching batches, configured heads and head_size=128."
            )
        if (
            not isinstance(kv_prefix_length, int)
            or kv_prefix_length < 0
            or query.shape[1] + kv_prefix_length != key.shape[1]
        ):
            raise ValueError("SOL_ATTN requires equal Q/KV lengths or explicit sol_kv_prefix_length metadata.")
        if any(t.dtype != torch.bfloat16 for t in (query, key, value)):
            raise ValueError("SOL_ATTN requires BF16 Q/K/V.")
        if query.device != key.device or query.device != value.device:
            raise ValueError("SOL_ATTN Q/K/V devices must match.")
        return _sol_attention_op(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            self.softmax_scale,
            self.tau,
            self.thresh_type,
            kv_prefix_length,
        )
