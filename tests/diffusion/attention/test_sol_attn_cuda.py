# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.backends.sol_attn import SolAttentionImpl, _sol_attention_op

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": ["H100", "H200", "B200"]}, num_cards=1),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((9, 0), (10, 0))
        or importlib.util.find_spec("sol_attn") is None,
        reason="requires SM90/SM100 and the optional sol-attn CuTe DSL package",
    ),
]


@pytest.mark.parametrize("thresh_type", ["diag", "exact"])
def test_sol_real_kernel_compile_and_graph(thresh_type: str) -> None:
    from sol_attn import sol_attn

    torch.manual_seed(42)
    impl = SolAttentionImpl(2, 128, 0.1, backend_kwargs={"tau": 1.0, "thresh_type": thresh_type})
    compiled = torch.compile(impl.forward_cuda, fullgraph=True)
    for tokens in (129, 256):
        # Model projection split views need compaction before the kernel.
        q, k, v = (
            torch.randn(1, 2, tokens, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2) for _ in range(3)
        )
        before = [t.clone() for t in (q, k, v)]
        expected = sol_attn(*(t.contiguous() for t in (q, k, v)), scale=0.1, tau=1.0, thresh_type=thresh_type)
        for run in (impl.forward_cuda, compiled):
            out = run(q, k, v)
            assert out.is_contiguous() and torch.isfinite(out).all()
            torch.testing.assert_close(out, expected, atol=0, rtol=0)
        for actual, original in zip((q, k, v), before):
            torch.testing.assert_close(actual, original, atol=0, rtol=0)

        # Warm compilation outside capture, then verify replay reads new inputs.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                impl.forward_cuda(q, k, v)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = impl.forward_cuda(q, k, v)
        for tensor in (q, k, v):
            tensor.copy_(torch.randn_like(tensor))
        graph.replay()
        torch.testing.assert_close(captured, impl.forward_cuda(q, k, v), atol=0, rtol=0)

    checks = torch.library.opcheck(
        _sol_attention_op.default,
        (*(t.contiguous() for t in (q, k, v)), 0.1, 1.0, thresh_type),
        test_utils=("test_schema", "test_faketensor"),
    )
    assert all(result == "SUCCESS" for result in checks.values())


def test_strategy_layout_keeps_dense_executor_and_compiles_sol() -> None:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.layer import build_attention
    from vllm_omni.diffusion.attention.strategy import AttentionOperation, ForwardStrategyPlan
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig, DiffusionParallelConfig

    config = AttentionConfig(
        presets={"dense": {"backend": "TORCH_SDPA"}, "sol": {"backend": "SOL_ATTN"}},
        layouts={
            "dense": {"default": "dense"},
            "mixed": {
                "default": "dense",
                "overrides": [{"attention_role": "cosmos3.gen", "layers": [0], "use": "sol"}],
            },
        },
        schedule={
            "coordinate": "step_index",
            "phases": [{"until": 2, "layout": "dense"}, {"until": None, "layout": "mixed"}],
        },
    )
    with set_current_diffusion_config(
        SimpleNamespace(diffusion_attention_config=config, parallel_config=DiffusionParallelConfig())
    ):
        layers = [
            build_attention(
                operation=AttentionOperation(f"gen.{i}", i, "cosmos3.gen", "self", component="transformer"),
                num_heads=4,
                num_kv_heads=2,
                head_size=128,
                softmax_scale=0.1,
                causal=False,
                role="cosmos3.gen",
                role_category="self",
            )
            for i in range(2)
        ]
    q = torch.randn(1, 129, 4, 128, device="cuda", dtype=torch.bfloat16)  # [B,Sq,Hq,D]
    k, v = (torch.randn(1, 146, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2))  # [B,Sk,Hkv,D]
    metadata = AttentionMetadata(extra={"protected_kv_prefix": 17})
    plan = ForwardStrategyPlan.from_strategy(config.strategy)
    for step in (0, 1, 2, 4):
        layout = plan.layout_for_step(step, 5)
        for i, layer in enumerate(layers):
            executor = layer.for_layout(layout)
            assert executor.attn_backend.get_name() == ("SOL_ATTN" if step >= 2 and i == 0 else "SDPA")
    executor = layers[0].for_layout(1)
    eager = executor(q, k, v, metadata)  # [B,Sq,Hq,D]
    compiled = torch.compile(executor, fullgraph=True)
    torch.testing.assert_close(compiled(q, k, v, metadata), eager, atol=0, rtol=0)
    assert layers[1].for_layout(0) is layers[1].for_layout(1)


def test_sol_dense_routing_matches_sdpa() -> None:
    q, k, v = (torch.randn(2, 193, 4, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    impl = SolAttentionImpl(4, 128, 0.1, backend_kwargs={"tau": -100.0})
    expected = F.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v)), scale=0.1).transpose(1, 2)
    torch.testing.assert_close(impl.forward_cuda(q, k, v), expected, atol=3e-3, rtol=3e-3)


def test_sage_dense_respects_current_stream() -> None:
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("The Sage integration patch targets Hopper")
    pytest.importorskip("sageattention")
    from vllm_omni.diffusion.attention.backends.sage_attn import SageAttentionImpl

    torch.manual_seed(17)
    q = torch.randn(1, 129, 8, 128, device="cuda", dtype=torch.bfloat16)  # [B,Sq,Hq,D]
    k, v = (torch.randn(1, 257, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2))  # [B,Sk,Hkv,D]
    pending = [torch.full_like(t, float("nan")) for t in (q, k, v)]  # [B,S,H,D] each
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=0.1, enable_gqa=True
    ).transpose(1, 2)  # [B,Sq,Hq,D]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    impl = SageAttentionImpl(num_heads=8, head_size=128, softmax_scale=0.1, num_kv_heads=2)
    with torch.cuda.stream(stream):
        # Leave default-stream launches time to read poisoned buffers. A correct
        # kernel follows the queued copies on the caller's stream.
        torch.cuda._sleep(50_000_000)
        for target, source in zip(pending, (q, k, v)):
            target.copy_(source)  # [B,S,H,D]
        actual = impl.forward_cuda(*pending)  # [B,Sq,Hq,D]
    torch.accelerator.synchronize()
    assert torch.isfinite(actual).all()
    # Sage quantizes Q/K/V. Aggregate relative error avoids unstable relative
    # comparisons near zero while rejecting uncomputed or stale output.
    relative_error = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(
        expected.float()
    )  # []
    assert relative_error.item() < 0.05
    compiled = torch.compile(impl.forward_cuda, fullgraph=True)
    torch.testing.assert_close(compiled(q, k, v), actual, atol=0, rtol=0)
    with torch.cuda.stream(stream):
        impl.forward_cuda(q, k, v)  # [B,Sq,Hq,D]
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = impl.forward_cuda(q, k, v)  # [B,Sq,Hq,D]
    q.copy_(torch.randn_like(q))  # [B,Sq,Hq,D]
    graph.replay()
    torch.testing.assert_close(captured, impl.forward_cuda(q, k, v), atol=0, rtol=0)


@pytest.mark.parametrize("tau,thresh_type", [(1.0, "diag"), (1.0, "exact"), (-100.0, "diag")])
def test_sol_cosmos3_gqa_prefix(monkeypatch: pytest.MonkeyPatch, tau: float, thresh_type: str) -> None:
    from sol_attn import sol_attn
    from vllm_omni.diffusion.attention import layer
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.backends.sol_attn import SolAttentionBackend
    from vllm_omni.diffusion.data import AttentionSpec
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3CrossAttention

    torch.manual_seed(43)
    spec = AttentionSpec(backend="SOL_ATTN", sol={"tau": tau, "thresh_type": thresh_type})
    monkeypatch.setattr(layer, "get_attn_backend_for_role", lambda **kwargs: (SolAttentionBackend, spec))
    attention = layer.Attention(num_heads=32, num_kv_heads=8, head_size=128, softmax_scale=0.1, causal=False)
    q = torch.randn(2, 129, 32, 128, device="cuda", dtype=torch.bfloat16)
    k, v = (torch.randn(2, 146, 8, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    prefix = 17
    metadata = AttentionMetadata(extra={"protected_kv_prefix": prefix})
    padded_q = torch.cat((q.new_zeros(2, prefix, 32, 128), q), dim=1)
    expanded_k, expanded_v = (t.repeat_interleave(4, dim=2) for t in (k, v))
    expected = sol_attn(
        padded_q, expanded_k, expanded_v, scale=0.1, tau=tau, thresh_type=thresh_type, sink_start=0, sink_tokens=prefix
    )[:, prefix:].contiguous()

    # Exercise the exact Cosmos3 helper and common Attention layer that failed
    # at startup, including its GQA geometry and text-prefix metadata producer.
    holder = SimpleNamespace(attn=attention)
    actual = Cosmos3CrossAttention._forward_local(holder, q, k[:, prefix:], v[:, prefix:], k[:, :prefix], v[:, :prefix])
    torch.testing.assert_close(actual.reshape_as(q), expected, atol=0, rtol=0)
    compiled = torch.compile(attention.attention.forward_cuda, fullgraph=True)
    torch.testing.assert_close(compiled(q, k, v, metadata), expected, atol=0, rtol=0)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            attention.attention.forward_cuda(q, k, v, metadata)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = attention.attention.forward_cuda(q, k, v, metadata)
    for tensor in (q, k, v):
        tensor.copy_(torch.randn_like(tensor))
    graph.replay()
    torch.testing.assert_close(captured, attention.attention.forward_cuda(q, k, v, metadata), atol=0, rtol=0)
    if tau == -100.0:
        # Use an FP32 reference to avoid comparing two differently rounded
        # BF16 outputs. The reference CuTe kernel itself remains BF16.
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            dense = F.scaled_dot_product_attention(
                q.float().transpose(1, 2),
                k.float().transpose(1, 2),
                v.float().transpose(1, 2),
                scale=0.1,
                enable_gqa=True,
            ).transpose(1, 2)
        torch.testing.assert_close(captured.float(), dense, atol=3e-3, rtol=3e-3)
    checks = torch.library.opcheck(
        _sol_attention_op.default,
        (q, k, v, 0.1, tau, thresh_type, prefix),
        test_utils=("test_schema", "test_faketensor"),
    )
    assert all(result == "SUCCESS" for result in checks.values())
