# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm_omni.diffusion.attention.backends import sol_attn as sol
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.capabilities import AttentionComputation
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_sol_config_round_trip() -> None:
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


def test_sol_sm120_kernel_selection() -> None:
    standard = AttentionSpec(backend="SOL_ATTN", sol={"tau": 1.0})
    assert standard.backend_kwargs() == {"tau": 1.0, "thresh_type": "diag"}
    quantized = AttentionSpec(backend="SOL_ATTN", sol={"tau": 1.0, "sm120_kernel": "int8_qk_fp8_pv"})
    assert quantized.backend_kwargs() == {
        "tau": 1.0,
        "thresh_type": "diag",
        "sm120_kernel": "int8_qk_fp8_pv",
    }
    with pytest.raises(ValueError, match="sm120_kernel"):
        AttentionSpec(backend="SOL_ATTN", sol={"sm120_kernel": "unknown"})


def test_sol_quantized_variant_reaches_custom_op(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[object, ...]] = []

    def capture(*args: object) -> torch.Tensor:
        seen.append(args)
        assert isinstance(args[0], torch.Tensor)
        return args[0]

    monkeypatch.setattr(sol, "_sol_attention_op", capture)
    impl = sol.SolAttentionImpl(2, 128, 0.2, backend_kwargs={"sm120_kernel": "int8_qk_fp8_pv"})
    q = torch.zeros(1, 2, 2, 128, dtype=torch.bfloat16)
    impl.forward_cuda(q, q, q)
    assert seen[0][-1] == "int8_qk_fp8_pv"


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
def test_sol_rejects_unsupported_geometry(kwargs: dict[str, Any]) -> None:
    args = dict(num_heads=2, head_size=128, softmax_scale=128**-0.5)
    args.update(kwargs)
    with pytest.raises(ValueError):
        sol.SolAttentionImpl(**args)


def test_sol_preserves_layout_scale_and_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        tau: float,
        thresh_type: str,
        kv_prefix_length: int,
    ) -> torch.Tensor:
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
def test_sol_rejects_masks_and_packing(metadata: AttentionMetadata) -> None:
    impl = sol.SolAttentionImpl(2, 128, 128**-0.5)
    q = torch.empty(1, 8, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="unmasked, unpacked"):
        impl.forward_cuda(q, q, q, metadata)


def test_sol_requires_explicit_kv_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    impl = sol.SolAttentionImpl(32, 128, 0.1, num_kv_heads=8)
    q = torch.empty(1, 129, 32, 128, dtype=torch.bfloat16)
    kv = torch.empty(1, 146, 8, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="protected_kv_prefix"):
        impl.forward_cuda(q, kv, kv)
    with pytest.raises(ValueError, match="protected_kv_prefix"):
        impl.forward_cuda(q, kv, kv, AttentionMetadata(extra={"protected_kv_prefix": 16}))
    seen = []

    def kernel(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        tau: float,
        thresh_type: str,
        kv_prefix_length: int,
    ) -> torch.Tensor:
        seen.append(kv_prefix_length)
        return torch.empty_like(query)

    monkeypatch.setattr(sol, "_sol_attention_op", kernel)
    out = impl.forward_cuda(q, kv, kv, AttentionMetadata(extra={"protected_kv_prefix": 17}))
    assert out.shape == q.shape and seen == [17]


def test_sol_refuses_triton_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_kernel(*args: object, **kwargs: object) -> None:
        pytest.fail("The Triton kernel must not run")

    monkeypatch.setattr(sol, "_load_sol_attn", lambda: (lambda *args: "triton", unexpected_kernel))
    with pytest.raises(ValueError, match="SM90, SM100, or SM120 CuTe"):
        sol.SolAttentionBackend.validate_available()


@pytest.mark.parametrize("backend", ["cute_sm90", "cute_sm100", "cute_sm120"])
def test_sol_accepts_supported_cute_backends(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    monkeypatch.setattr(sol, "_load_sol_attn", lambda: (lambda *args: backend, None))
    sol.SolAttentionBackend.validate_available()


@pytest.mark.parametrize("degree", ["ulysses_degree", "ring_degree", "allgather_degree"])
def test_sol_rejects_sequence_parallelism_before_dispatch(monkeypatch: pytest.MonkeyPatch, degree: str) -> None:
    from vllm_omni.diffusion.attention import layer
    from vllm_omni.diffusion.config import set_current_diffusion_config

    monkeypatch.setattr(layer, "get_attn_backend_for_role", lambda **kwargs: (sol.SolAttentionBackend, None))
    config = SimpleNamespace(diffusion_attention_config=None, parallel_config=SimpleNamespace(**{degree: 2}))
    with set_current_diffusion_config(config), pytest.raises(ValueError, match="sequence parallelism"):
        layer.Attention(num_heads=2, head_size=128, softmax_scale=128**-0.5, causal=False)


def test_sol_strategy_contract_has_no_independent_step_gate() -> None:
    from vllm_omni.diffusion.attention.backends.sage_attn import SageAttentionBackend
    from vllm_omni.diffusion.attention.contracts import validate_strategy_attention
    from vllm_omni.diffusion.attention.strategy import AttentionOperation, ForwardStrategyPlan

    config = AttentionConfig(
        presets={"dense": {"backend": "FLASH_ATTN"}, "sparse": {"backend": "SOL_ATTN", "sol": {"tau": 0.85}}},
        layouts={
            "dense": {"default": "dense"},
            "mixed": {
                "default": "dense",
                "overrides": [{"attention_role": "cosmos3.gen", "layers": [0], "use": "sparse"}],
            },
        },
        schedule={
            "coordinate": "step_index",
            "phases": [{"until": 2, "layout": "dense"}, {"until": None, "layout": "mixed"}],
        },
    )
    operations = [AttentionOperation(f"gen.{i}", i, "cosmos3.gen", "self", component="transformer") for i in range(2)]
    config.strategy.validate_inventory(operations)
    plan = ForwardStrategyPlan.from_strategy(config.strategy)
    for step in (0, 1, 2, 4):
        layout = plan.layout_for_step(step, 5)
        for operation in operations:
            spec = config.strategy.assignments(operation)[layout]
            assert spec.backend == ("SOL_ATTN" if step >= 2 and operation.layer == 0 else "FLASH_ATTN")
    validate_strategy_attention(config.presets["sparse"], sol.SolAttentionBackend)
    validate_strategy_attention(AttentionSpec(backend="SAGE_ATTN"), SageAttentionBackend)
    with pytest.raises(TypeError, match="start_step"):
        AttentionSpec(backend="SOL_ATTN", sol={"start_step": 2})
