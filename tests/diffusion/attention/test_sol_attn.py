# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.backends import sol_attn as sol
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.capabilities import AttentionComputation
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_sol_config_round_trip():
    config = AttentionConfig(per_role={"self": {"backend": "SOL_ATTN", "sol": {"tau": 0.5, "thresh_type": "exact"}}})
    assert config.per_role["self"].backend_kwargs() == {"tau": 0.5, "thresh_type": "exact"}
    assert AttentionSpec(backend="SOL_ATTN").backend_kwargs() is None
    for kwargs in ({"tau": float("nan")}, {"tau": float("inf")}, {"thresh_type": "other"}):
        with pytest.raises(ValueError):
            AttentionSpec(backend="SOL_ATTN", sol=kwargs)
    with pytest.raises(ValueError, match="requires the SOL_ATTN"):
        AttentionSpec(backend="TORCH_SDPA", sol={})
    for kwargs in ({"quant": {"dtype_qk": "int8"}}, {"skip_softmax": {"threshold": 0.1}}):
        with pytest.raises(ValueError):
            AttentionSpec(backend="SOL_ATTN", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"causal": True},
        {"head_size": 64},
        {"num_kv_heads": 3},
        {"computation": AttentionComputation.CROSS},
        {"qkv_layout": "BNSD"},
    ],
)
def test_sol_rejects_unsupported_geometry(kwargs):
    args = dict(num_heads=2, head_size=128, softmax_scale=128**-0.5)
    args.update(kwargs)
    with pytest.raises(ValueError):
        sol.SolAttentionImpl(**args)


def test_sol_preserves_layout_scale_and_inputs(monkeypatch):
    calls = []

    def kernel(q, k, v, scale, tau, thresh_type, kv_prefix_length):
        assert all(t.is_contiguous() for t in (q, k, v))
        assert kv_prefix_length == 0
        calls.append((scale, tau, thresh_type))
        return q + k + v

    monkeypatch.setattr(sol, "_sol_attention_op", kernel)
    impl = sol.SolAttentionImpl(2, 128, 0.2, backend_kwargs={"tau": 0.5, "thresh_type": "exact"})
    q, k, v = (torch.randn(1, 2, 7, 128, dtype=torch.bfloat16).transpose(1, 2) for _ in range(3))
    before = [t.clone() for t in (q, k, v)]
    out = impl.forward_cuda(q, k, v)
    torch.testing.assert_close(out, q + k + v)
    assert calls == [(0.2, 0.5, "exact")]
    for actual, original in zip((q, k, v), before):
        torch.testing.assert_close(actual, original)


@pytest.mark.parametrize(
    "metadata",
    [
        AttentionMetadata(attn_mask=torch.ones(1)),
        AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, 8])}),
        AttentionMetadata(full_attn_spans=[[(0, 8)]]),
    ],
)
def test_sol_rejects_masks_and_packing(metadata):
    impl = sol.SolAttentionImpl(2, 128, 128**-0.5)
    q = torch.empty(1, 8, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="unmasked, unpacked"):
        impl.forward_cuda(q, q, q, metadata)


def test_sol_requires_explicit_kv_prefix(monkeypatch):
    impl = sol.SolAttentionImpl(32, 128, 0.1, num_kv_heads=8)
    q = torch.empty(1, 129, 32, 128, dtype=torch.bfloat16)
    kv = torch.empty(1, 146, 8, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="sol_kv_prefix_length"):
        impl.forward_cuda(q, kv, kv)
    with pytest.raises(ValueError, match="sol_kv_prefix_length"):
        impl.forward_cuda(q, kv, kv, AttentionMetadata(extra={"sol_kv_prefix_length": 16}))
    seen = []

    def kernel(query, key, value, scale, tau, thresh_type, kv_prefix_length):
        seen.append(kv_prefix_length)
        return torch.empty_like(query)

    monkeypatch.setattr(sol, "_sol_attention_op", kernel)
    out = impl.forward_cuda(q, kv, kv, AttentionMetadata(extra={"sol_kv_prefix_length": 17}))
    assert out.shape == q.shape and seen == [17]


def test_sol_refuses_triton_fallback(monkeypatch):
    def unexpected_kernel(*args, **kwargs):
        pytest.fail("The Triton kernel must not run")

    monkeypatch.setattr(sol, "_load_sol_attn", lambda: (lambda *args: "triton", unexpected_kernel))
    with pytest.raises(ValueError, match="SM90 or SM100 CuTe"):
        sol.SolAttentionBackend.validate_available()


@pytest.mark.parametrize("backend", ["cute_sm90", "cute_sm100"])
def test_sol_accepts_supported_cute_backends(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    monkeypatch.setattr(sol, "_load_sol_attn", lambda: (lambda *args: backend, None))
    sol.SolAttentionBackend.validate_available()


@pytest.mark.parametrize("degree", ["ulysses_degree", "ring_degree", "allgather_degree"])
def test_sol_rejects_sequence_parallelism_before_dispatch(monkeypatch, degree):
    from vllm_omni.diffusion.attention import layer
    from vllm_omni.diffusion.config import set_current_diffusion_config

    monkeypatch.setattr(layer, "get_attn_backend_for_role", lambda **kwargs: (sol.SolAttentionBackend, None))
    config = SimpleNamespace(diffusion_attention_config=None, parallel_config=SimpleNamespace(**{degree: 2}))
    with set_current_diffusion_config(config), pytest.raises(ValueError, match="sequence parallelism"):
        layer.Attention(num_heads=2, head_size=128, softmax_scale=128**-0.5, causal=False)
