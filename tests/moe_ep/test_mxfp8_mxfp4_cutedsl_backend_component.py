"""Focused host checks for MXFP8 x MXFP4 backend/config/weight plumbing."""

from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from flashinfer.fused_moe.api import QuantConfig, QuantFormat
from flashinfer.moe_ep.core.kernel.registry import is_mega_kernel_config
from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl.config import (
    KERNEL_NAME,
    Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig,
    candidate_tactics,
    default_tactic,
    resolve_tactic,
    validate_tactic,
)
from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl import (
    weights as weight_impl,
)
from flashinfer.moe_ep.weights import MoEWeightPack


@pytest.mark.parametrize("owned_output", [False, True])
def test_compute_uses_stable_reducer_output(monkeypatch, owned_output):
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl.backend import (
        Mxfp8Mxfp4CutedslMegaKernelBackend,
    )
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4 import integration
    from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe

    backend = object.__new__(Mxfp8Mxfp4CutedslMegaKernelBackend)
    backend._autotune_pending = False
    backend._kernel_config = SimpleNamespace(gate_up_clamp=None, fast_math=False)
    view = torch.full((2, 128), 3.0, dtype=torch.bfloat16)
    output = torch.empty_like(view) if owned_output else None
    workspace = SimpleNamespace(topk_idx=object())

    def compute(destination, fc1, fc2, actual_workspace, **kwargs):
        assert destination is None
        assert actual_workspace is workspace
        assert kwargs["num_tokens"] == 2
        return view

    monkeypatch.setattr(integration, "mxfp8_mxfp4_mega_moe", compute)
    monkeypatch.setattr(cutedsl_megamoe, "staged_tokens", lambda _: 2)
    result = backend.compute(workspace, (object(), object()), output=output)
    assert result is (output if owned_output else view)
    torch.testing.assert_close(result, view)


def test_backend_registers_after_persistent_assembly_is_ready() -> None:
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.persistent_kernel import (
        ASSEMBLY_READY,
    )

    assert ASSEMBLY_READY is True
    config = Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=128, top_k=4
    )
    assert is_mega_kernel_config(config)


def test_public_name_and_quant_taxonomy_are_activation_first() -> None:
    config = Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=128, top_k=4
    )
    assert config.kernel_name == KERNEL_NAME == "sm100_mxfp8_mxfp4_bf16_cutedsl"
    assert config.quant.pair == (QuantFormat.MXFP4, QuantFormat.MXFP8)
    assert config.quant.output is QuantFormat.BF16
    assert config.quant.swizzled_scale_factors is True
    assert config.quant.per_token_scale is False
    assert Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=128,
        top_k=4,
        quant=QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
    )

    reversed_axes = QuantConfig(
        weight=QuantFormat.MXFP8,
        activation=QuantFormat.MXFP4,
        output=QuantFormat.BF16,
        swizzled_scale_factors=True,
        per_token_scale=False,
    )
    with pytest.raises(ValueError, match="weight=MXFP4, activation=MXFP8"):
        Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=128, top_k=4, quant=reversed_axes
        )


def test_tactic_defaults_candidates_and_partial_overrides_are_valid() -> None:
    small = default_tactic(8)
    large = default_tactic(2048)
    assert small["mma_tiler_mnk"] == (128, 64, 128)
    # The persistent kernel runs N64 and N128 only.
    assert large["mma_tiler_mnk"] == (128, 128, 128)
    assert small["group_hint"] == 4096
    assert large["group_hint"] == 512
    assert small["load_balance_mode"] == large["load_balance_mode"] == "static"
    assert small["cluster_shape_mnk"] == (1, 1, 1)
    assert large["cluster_shape_mnk"] == (1, 1, 1)
    # The operand-pipeline depth resolves at compile time unless pinned.
    assert small["num_stages"] == "auto"
    assert large["num_stages"] == "auto"
    assert resolve_tactic(8, {"num_stages": 4})["num_stages"] == 4
    validate_tactic({"num_stages": "auto"})
    with pytest.raises(ValueError, match="num_stages"):
        validate_tactic({"num_stages": "deep"})
    with pytest.raises(ValueError, match="num_stages"):
        validate_tactic({"num_stages": 9})
    candidates = candidate_tactics()
    assert len(candidates) == 16
    assert {candidate["num_stages"] for candidate in candidates} == {"auto", 3, 4, 5}
    assert small in candidates
    assert large in candidates
    assert {candidate["mma_tiler_mnk"][1] for candidate in candidates} == {64, 128}
    assert {candidate["group_hint"] for candidate in candidates} == {512, 4096}
    assert {candidate["load_balance_mode"] for candidate in candidates} == {"static"}
    assert {candidate["token_back_schedule_mode"] for candidate in candidates} == {
        "static"
    }
    assert {candidate["token_back_mode"] for candidate in candidates} == {"epi_warps"}
    assert {candidate["mma_tiler_mnk"][0] for candidate in candidates} == {128}
    assert {candidate["cluster_shape_mnk"] for candidate in candidates} == {(1, 1, 1)}
    for candidate in candidates:
        validate_tactic(candidate, partial=False)

    for unsupported_n in (32, 192, 256):
        with pytest.raises(ValueError, match="N must be 64 or 128"):
            validate_tactic({"mma_tiler_mnk": (128, unsupported_n, 128)})

    with pytest.raises(ValueError, match="unknown"):
        validate_tactic({"decode_warps": 8})
    with pytest.raises(ValueError, match="M=128"):
        validate_tactic(
            {
                "mma_tiler_mnk": (256, 128, 128),
                "cluster_shape_mnk": (1, 1, 1),
            }
        )


def test_prequantized_weight_layout_is_k_major_and_interleaves_gate_up(
    monkeypatch,
) -> None:
    # Keep this a CPU test: scale-layout math is covered by the kernel component,
    # while this component owns logical validation and K-major view construction.
    monkeypatch.setattr(weight_impl, "_swizzle_expert_scales", lambda x: x.flatten())
    experts, hidden, intermediate = 1, 128, 128
    w13 = (
        torch.arange(experts * 2 * intermediate * (hidden // 2), dtype=torch.int64)
        .to(torch.uint8)
        .reshape(experts, 2 * intermediate, hidden // 2)
    )
    w2 = torch.zeros(experts, hidden, intermediate // 2, dtype=torch.uint8)
    s13 = (
        torch.arange(experts * 2 * intermediate * (hidden // 32), dtype=torch.int64)
        .to(torch.uint8)
        .reshape(experts, 2 * intermediate, hidden // 32)
    )
    s2 = torch.zeros(experts, hidden, intermediate // 32, dtype=torch.uint8)

    transformed = weight_impl.preprocess_mega_weights(
        MoEWeightPack(w13, w2, s13, s2),
        intermediate_size=intermediate,
        hidden_size=hidden,
    )
    fc1, fc1_sf = transformed[0]
    fc2, fc2_sf = transformed[1]
    assert fc1.shape == (experts, hidden // 2, 2 * intermediate)
    assert fc2.shape == (experts, intermediate // 2, hidden)
    assert fc1.stride(1) == 1 and fc2.stride(1) == 1
    assert fc1_sf.dtype == torch.uint8 and fc2_sf.dtype == torch.uint8
    assert fc1_sf.shape == (experts, s13[0].numel())

    # First 32 rows are gate, next 32 are up; then the second gate/up pair.
    interleaved_sf = fc1_sf.reshape(experts, 2 * intermediate, hidden // 32)
    torch.testing.assert_close(interleaved_sf[:, :32], s13[:, :32])
    torch.testing.assert_close(
        interleaved_sf[:, 32:64], s13[:, intermediate : intermediate + 32]
    )


def test_weight_validation_rejects_non_bf16_and_wrong_e8m0_contract() -> None:
    with pytest.raises(TypeError, match="must be BF16"):
        weight_impl.preprocess_mega_weights(
            MoEWeightPack(
                torch.zeros(1, 256, 128),
                torch.zeros(1, 128, 128),
            ),
            intermediate_size=128,
            hidden_size=128,
        )

    with pytest.raises(TypeError, match="scale tensors must have dtype torch.uint8"):
        weight_impl.preprocess_mega_weights(
            MoEWeightPack(
                torch.zeros(1, 256, 64, dtype=torch.uint8),
                torch.zeros(1, 128, 64, dtype=torch.uint8),
                torch.zeros(1, 256, 4, dtype=torch.float32),
                torch.zeros(1, 128, 4, dtype=torch.float32),
            ),
            intermediate_size=128,
            hidden_size=128,
        )
