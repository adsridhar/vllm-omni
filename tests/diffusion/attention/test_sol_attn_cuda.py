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
def test_sol_real_kernel_compile_and_graph(thresh_type):
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


def test_sol_dense_routing_matches_sdpa():
    q, k, v = (torch.randn(2, 193, 4, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    impl = SolAttentionImpl(4, 128, 0.1, backend_kwargs={"tau": -100.0})
    expected = F.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v)), scale=0.1).transpose(1, 2)
    torch.testing.assert_close(impl.forward_cuda(q, k, v), expected, atol=3e-3, rtol=3e-3)


@pytest.mark.parametrize("tau,thresh_type", [(1.0, "diag"), (1.0, "exact"), (-100.0, "diag")])
def test_sol_cosmos3_gqa_prefix(monkeypatch, tau, thresh_type):
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
    metadata = AttentionMetadata(extra={"sol_kv_prefix_length": prefix})
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
