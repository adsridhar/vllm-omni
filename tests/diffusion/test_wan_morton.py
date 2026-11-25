# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Wan Morton geometry and position-preserving attention checks."""

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.data import AttentionConfig, parse_attention_config
from vllm_omni.diffusion.models.wan2_2.morton import morton3d_permutation

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_morton_blocks_are_spatiotemporal_and_invertible() -> None:
    perm, inverse = morton3d_permutation((2, 4, 4), torch.device("cpu"))  # [32], [32]
    assert perm[:8].tolist() == [0, 1, 4, 5, 16, 17, 20, 21]
    torch.testing.assert_close(perm[inverse], torch.arange(32))
    # Exercise the non-power-of-two grid used by 720p, 81-frame Wan generation.
    perm, inverse = morton3d_permutation((21, 45, 80), torch.device("cpu"))  # [75600], [75600]
    torch.testing.assert_close(perm[inverse], torch.arange(75600))


def test_morton_preserves_dense_attention_with_position_rows() -> None:
    generator = torch.Generator().manual_seed(42)
    perm, inverse = morton3d_permutation((3, 5, 7), torch.device("cpu"))  # [105], [105]
    q = torch.randn(1, 2, 105, 16, generator=generator, dtype=torch.float64)  # [B,H,S,D]
    k = torch.randn(1, 2, 105, 16, generator=generator, dtype=torch.float64)  # [B,H,S,D]
    v = torch.randn(1, 2, 105, 16, generator=generator, dtype=torch.float64)  # [B,H,S,D]
    phase = torch.randn(1, 1, 105, 8, generator=generator, dtype=torch.float64)  # [1,1,S,D/2]

    def rotate(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:  # [B,H,S,D], [1,1,S,D/2] -> [B,H,S,D]
        even, odd = x[..., ::2], x[..., 1::2]  # two [B,H,S,D/2]
        return torch.stack(
            (even * positions.cos() - odd * positions.sin(), even * positions.sin() + odd * positions.cos()), dim=-1
        ).flatten(-2)  # [B,H,S,D]

    native = F.scaled_dot_product_attention(rotate(q, phase), rotate(k, phase), v)  # [B,H,S,D]
    reordered = F.scaled_dot_product_attention(
        rotate(q[:, :, perm], phase[:, :, perm]),
        rotate(k[:, :, perm], phase[:, :, perm]),
        v[:, :, perm],
    )  # [B,H,S,D]
    torch.testing.assert_close(reordered[:, :, inverse], native, rtol=1e-12, atol=1e-12)


def test_wan_token_order_config_round_trip_and_default() -> None:
    assert AttentionConfig().wan_token_order == "native"
    config = AttentionConfig(wan_token_order="morton3d")
    assert parse_attention_config(asdict(config)).wan_token_order == "morton3d"
    with pytest.raises(ValueError, match="wan_token_order"):
        AttentionConfig(wan_token_order="unknown")


def test_wan_forward_reorders_tokens_and_rope_and_restores_output() -> None:
    from vllm_omni.diffusion.models.wan2_2 import wan2_2_transformer as wan

    grid = (3, 5, 7)
    tokens = 105
    seen: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    class Block(torch.nn.Module):
        def forward(
            self,
            hidden: torch.Tensor,
            text: torch.Tensor,
            timestep: torch.Tensor,
            rotary: tuple[torch.Tensor, torch.Tensor],
            mask: torch.Tensor | None,
            shape: tuple[int, int, int],
            preserve: bool,
        ) -> torch.Tensor:  # hidden [B,S,C], text [B,L,C], rotary two [1,S,1,C] -> [B,S,C]
            seen.append((hidden.clone(), rotary[0].clone(), text.clone()))
            return hidden + rotary[0][:, :, 0, :]  # [B,S,C]

    hidden = torch.arange(4 * tokens, dtype=torch.float32).reshape(1, 4, *grid)  # [B,C,T,H,W]
    text = torch.randn(1, 7, 4)  # [B,L,C]
    positions = torch.arange(tokens).float().view(1, tokens, 1, 1).expand(1, tokens, 1, 8)  # [1,S,1,2*C]
    conditioning = torch.zeros(1, 4)  # [B,C]
    model = SimpleNamespace(
        config=SimpleNamespace(patch_size=(1, 1, 1)),
        _cached_rope_resolution=None,
        _cached_rope_emb=None,
        rope=lambda _: (positions, positions),
        patch_embedding=torch.nn.Identity(),
        _sp_shard_point=torch.nn.Identity(),
        condition_embedder=lambda *args, **kwargs: (conditioning, conditioning, text, None),
        timestep_proj_prepare=lambda value, _: value,
        blocks=[Block(), Block()],
        start_layer=0,
        end_layer=2,
        preserve_vsa_all_blocks=False,
        output_scale_shift_prepare=lambda _: (conditioning, conditioning),
        norm_out=lambda value, scale, shift: value,
        proj_out=torch.nn.Identity(),
    )
    config = SimpleNamespace(parallel_config=None, diffusion_attention_config=AttentionConfig())
    context = SimpleNamespace(omni_diffusion_config=config)
    with (
        patch.object(wan, "get_forward_context", return_value=context),
        patch.object(wan, "is_pipeline_first_stage", return_value=True),
        patch.object(wan, "is_pipeline_last_stage", return_value=True),
        patch.object(wan, "get_pipeline_parallel_world_size", return_value=1),
    ):
        native = wan.WanTransformer3DModel.forward(model, hidden, torch.tensor([1]), text).sample  # [B,C,T,H,W]
        native_hidden, native_rope, _ = seen[0]  # [B,S,C], [1,S,1,C], [B,L,C]
        seen.clear()
        config.diffusion_attention_config = AttentionConfig(wan_token_order="morton3d")
        output = wan.WanTransformer3DModel.forward(model, hidden, torch.tensor([1]), text).sample  # [B,C,T,H,W]
    perm, _ = morton3d_permutation(grid, hidden.device)  # [S], [S]
    torch.testing.assert_close(seen[0][0], native_hidden[:, perm])
    torch.testing.assert_close(seen[0][1], native_rope[:, perm])
    torch.testing.assert_close(seen[0][2], text)
    torch.testing.assert_close(output, native)
