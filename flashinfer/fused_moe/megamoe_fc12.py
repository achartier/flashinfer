"""Local CuTe-DSL MegaMOE FC12 launch adapters.

The adapters deliberately own only FC1/SwiGLU/FC2.  Routing and token movement
remain in the unified runner so they can be shared by direct and EP callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch

from .api import QuantConfig, QuantFormat, SwiGLU


def prepare_megamoe_fc12_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    quant: QuantConfig,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation=None,
    device=None,
) -> dict[str, torch.Tensor]:
    """Prepare canonical weights for a local MegaMOE FC12 kernel."""
    del device
    if not isinstance(activation or SwiGLU(), SwiGLU):
        raise NotImplementedError("MegaMOE FC12 currently implements SwiGLU only.")
    expected_w1 = (num_local_experts, 2 * intermediate_size, hidden_size)
    expected_w2 = (num_local_experts, hidden_size, intermediate_size)
    if tuple(w1_bf16.shape) != expected_w1 or tuple(w2_bf16.shape) != expected_w2:
        raise ValueError(
            "MegaMOE FC12 canonical weights must have shapes "
            f"{expected_w1} and {expected_w2}; got {tuple(w1_bf16.shape)} and "
            f"{tuple(w2_bf16.shape)}."
        )
    pair = quant.pair
    if pair == (QuantFormat.MXFP4, QuantFormat.MXFP8):
        from ..moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl.weights import (  # noqa: E501
            preprocess_mega_weights,
        )

        fc1, fc2 = preprocess_mega_weights(
            SimpleNamespace(w13=w1_bf16, w2=w2_bf16),
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
        )
        return {
            "fc1_weight": fc1[0],
            "fc1_weight_sf": fc1[1],
            "fc2_weight": fc2[0],
            "fc2_weight_sf": fc2[1],
        }
    raise NotImplementedError(
        "MegaMOE FC12 requires MXFP4 weights and MXFP8 activations."
    )


@dataclass
class Mxfp8Mxfp4Fc12Inputs:
    activation: torch.Tensor
    fc1_weight: torch.Tensor
    fc2_weight: torch.Tensor
    output: torch.Tensor
    expert_token_sizes: torch.Tensor
    fc1_weight_sf: Optional[torch.Tensor] = None
    fc2_weight_sf: Optional[torch.Tensor] = None

    activation_scales: Optional[torch.Tensor] = None


class Mxfp8Mxfp4Fc12Launcher:
    """Host adapter for the independent communication-free mixed-format core."""

    def __init__(self, num_experts, max_rows, hidden, intermediate):
        from ..moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.lean_abi import LeanFc12Config

        self.config = LeanFc12Config(
            local_num_experts=num_experts,
            hidden=hidden,
            intermediate=intermediate,
            data_row_capacity=max_rows,
            scale_row_capacity=max_rows,
            data_row_alignment=64,
            scale_row_alignment=64,
        )
        self._launcher = None

    def run(self, inputs: Mxfp8Mxfp4Fc12Inputs) -> None:
        from ..moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.lean_abi import LeanFc12Inputs
        from ..moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.lean_kernel import LeanFc12Launcher

        if self._launcher is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "MXFP8 x MXFP4 FC12 must be warmed up before capture"
                )
            self._launcher = LeanFc12Launcher(self.config)
        # moe_sort pads each expert independently to 64 rows. Keep the prefix
        # calculation on device: host count reads would break graph capture.
        counts = inputs.expert_token_sizes
        offsets = torch.cat(
            (
                counts.new_zeros(1),
                torch.cumsum(((counts + 63) // 64) * 64, 0, dtype=torch.int32),
            )
        )
        self._launcher.run(
            LeanFc12Inputs(
                activation=inputs.activation,
                activation_scales=inputs.activation_scales,
                fc1_weight=inputs.fc1_weight,
                fc1_weight_scales=inputs.fc1_weight_sf,
                fc2_weight=inputs.fc2_weight,
                fc2_weight_scales=inputs.fc2_weight_sf,
                output=inputs.output,
                expert_token_sizes=counts,
                expert_data_row_offsets=offsets,
                expert_scale_row_offsets=offsets,
            )
        )


def unpermute_mxfp8_mxfp4_routes(terms, output, expanded_to_permuted, weights):
    """Contract-exact eager/capture-safe finalization, with no fused multiply-add.

    Separate PyTorch operations deliberately preserve a FP32 product rounding
    and increasing top-k slot sum. Each term has already rounded to BF16.
    """
    num_tokens, top_k = weights.shape
    mapping = expanded_to_permuted.reshape(-1)[: num_tokens * top_k].reshape(
        num_tokens, top_k
    )
    result = torch.zeros_like(output, dtype=torch.float32)
    for slot in range(top_k):
        rows = mapping[:, slot]
        term = terms.index_select(0, rows.clamp(min=0).long()).float()
        term = torch.where((rows >= 0).unsqueeze(1), term, 0.0)
        score = torch.where(rows >= 0, weights[:, slot].float(), 0.0)
        product = term * score.unsqueeze(1)
        result.add_(product)
    output.copy_(result)


def mxfp8_mxfp4_expert_counts(topk_ids, local_expert_offset, num_local_experts):
    """Fixed-shape device histogram for the lean scheduler.

    Some native moe_sort routing variants do not populate out_expert_counts.
    Never feed that optional output to the persistent scheduler: stale counts
    can skip valid rows or schedule beyond the sorted buffers. scatter_add
    preserves graph capture and avoids a data-dependent bincount allocation.
    """
    local_ids = topk_ids.reshape(-1) - local_expert_offset
    valid = (local_ids >= 0) & (local_ids < num_local_experts)
    counts = torch.zeros(num_local_experts, dtype=torch.int32, device=topk_ids.device)
    counts.scatter_add_(
        0,
        local_ids.clamp(min=0, max=num_local_experts - 1).long(),
        valid.to(torch.int32),
    )
    return counts
