"""Check SM90/SM100 dispatch and dense-routing accuracy without model weights."""

import json

import torch
import torch.nn.functional as F
from sol_attn import get_sol_attn_backend, sol_attn


def main() -> None:
    capability = torch.cuda.get_device_capability()
    assert capability in ((9, 0), (10, 0))
    assert get_sol_attn_backend() == {(9, 0): "cute_sm90", (10, 0): "cute_sm100"}[capability]
    torch.manual_seed(42)
    query = torch.randn(1, 193, 4, 128, device="cuda", dtype=torch.bfloat16)  # [1,193,4,128]
    key = torch.randn_like(query)  # [1,193,4,128]
    value = torch.randn_like(query)  # [1,193,4,128]
    for threshold in ("diag", "exact"):
        actual = sol_attn(query, key, value, tau=-100.0, thresh_type=threshold)  # [1,193,4,128]
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            expected = F.scaled_dot_product_attention(
                query.float().transpose(1, 2),  # [1,4,193,128]
                key.float().transpose(1, 2),  # [1,4,193,128]
                value.float().transpose(1, 2),  # [1,4,193,128]
            ).transpose(1, 2)  # [1,193,4,128]
        torch.testing.assert_close(actual.float(), expected, atol=3e-3, rtol=3e-3)
    torch.accelerator.synchronize()
    print(
        json.dumps({"gpu": torch.cuda.get_device_name(), "backend": get_sol_attn_backend(), "dense_routing": "passed"})
    )


if __name__ == "__main__":
    main()
