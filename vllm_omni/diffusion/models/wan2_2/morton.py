# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Canonical Morton3D ordering for Wan video patch tokens."""

from functools import lru_cache

import torch


@lru_cache(maxsize=16)
def morton3d_permutation(grid: tuple[int, int, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return native-to-Morton indices and their inverse for a (T,H,W) grid."""
    frames, height, width = grid
    if min(grid) <= 0:
        raise ValueError("Morton grid dimensions must be positive.")
    # Interleave x, y, then time, matching the NVlabs/Sana sol-engine Wan adapter.
    linear = torch.arange(frames * height * width, dtype=torch.long)  # [T*H*W]
    x = linear % width  # [T*H*W]
    y = linear // width % height  # [T*H*W]
    t = linear // (height * width)  # [T*H*W]
    codes = torch.zeros_like(linear)  # [T*H*W]
    for bit in range(max(grid).bit_length()):
        codes |= ((x >> bit) & 1) << (3 * bit)  # [T*H*W]
        codes |= ((y >> bit) & 1) << (3 * bit + 1)  # [T*H*W]
        codes |= ((t >> bit) & 1) << (3 * bit + 2)  # [T*H*W]
    permutation = torch.argsort(codes)  # [T*H*W]
    inverse = torch.argsort(permutation)  # [T*H*W]
    return permutation.to(device), inverse.to(device)  # [T*H*W], [T*H*W]
