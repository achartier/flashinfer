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


def test_epilogue_warps_default_band_env_and_overrides(monkeypatch) -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl.config import (
        EIGHT_WARP_EPILOGUE_MIN_TOKENS,
        EPILOGUE_WARPS_ENV,
    )

    monkeypatch.delenv(EPILOGUE_WARPS_ENV, raising=False)
    shape = dict(hidden=4096, intermediate=6400)
    # The pair tile uses eight epilogue warps from 256 tokens per rank.
    for tokens, warps in ((1, 4), (255, 4), (256, 8), (1024, 8), (4096, 8)):
        assert default_tactic(tokens, **shape)["epilogue_warps"] == warps
    assert EIGHT_WARP_EPILOGUE_MIN_TOKENS == 256
    # 1-CTA tiles (no shape, or a shape the pair tile cannot divide) keep four.
    assert default_tactic(2048)["epilogue_warps"] == 4
    assert default_tactic(2048, hidden=6528, intermediate=4096)["epilogue_warps"] == 4
    # An explicit override wins; the environment replaces only the default.
    assert resolve_tactic(2048, {"epilogue_warps": 4}, **shape)["epilogue_warps"] == 4
    assert resolve_tactic(8, {"epilogue_warps": 8}, **shape)["epilogue_warps"] == 8
    monkeypatch.setenv(EPILOGUE_WARPS_ENV, "4")
    assert default_tactic(2048, **shape)["epilogue_warps"] == 4
    assert resolve_tactic(2048, {"epilogue_warps": 8}, **shape)["epilogue_warps"] == 8
    monkeypatch.setenv(EPILOGUE_WARPS_ENV, "8")
    assert default_tactic(8)["epilogue_warps"] == 8
    monkeypatch.setenv(EPILOGUE_WARPS_ENV, "6")
    with pytest.raises(ValueError, match=EPILOGUE_WARPS_ENV):
        default_tactic(8)
    monkeypatch.delenv(EPILOGUE_WARPS_ENV)
    with pytest.raises(ValueError, match="epilogue_warps must be 4 or 8"):
        validate_tactic({"epilogue_warps": 6})


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_in_kernel_reduce_rejects_non_boolean_tactic(value) -> None:
    with pytest.raises(ValueError, match="in_kernel_reduce must be a bool"):
        validate_tactic({"in_kernel_reduce": value})


def test_pair_tile_candidates_cover_every_default() -> None:
    shape = dict(hidden=4096, intermediate=6400)
    candidates = candidate_tactics(**shape)
    assert len(candidates) == 28
    alternating = [c for c in candidates if c["load_balance_mode"] == "alternating"]
    assert len(alternating) == 4
    assert {c["mma_tiler_mnk"] for c in alternating} == {(256, 192, 128)}
    assert {c["mma_tiler_mnk"] for c in candidates} == {
        (256, 64, 128),
        (256, 128, 128),
        (256, 192, 128),
    }
    assert {c["cluster_shape_mnk"] for c in candidates} == {(2, 1, 1)}
    assert {c["epilogue_warps"] for c in candidates} == {4, 8}
    assert {c["in_kernel_reduce"] for c in candidates} == {False, True}
    assert {c["group_hint"] for c in candidates} == {512, 4096}
    # Tuning can never lose the shipped band on the timed problem.
    for tokens in (1, 64, 255, 256, 511, 512, 1023, 1024, 4096):
        assert default_tactic(tokens, **shape) in candidates
    # Shapes the pair tile cannot divide keep the 1-CTA grid.
    assert candidate_tactics(hidden=6528, intermediate=4096) == candidate_tactics()


def test_tune_cli_dispatches_mxfp8_mxfp4_and_schedule_candidates_validate() -> None:
    from flashinfer.moe_ep.tune import _parse_args
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl import (
        tuner,
    )

    argv = ["--dtype", "mxfp8_mxfp4", "--hidden", "4096", "--intermediate", "6400"]
    argv += ["--num-experts", "512", "--topk", "8", "--max-tokens", "512"]
    args = _parse_args(argv)
    assert args.dtype == "mxfp8_mxfp4"
    assert tuner._candidates(args, 512, rank=1) == candidate_tactics(
        hidden=4096, intermediate=6400
    )
    args = _parse_args(
        [*argv, "--sweep", "schedule", "--base-knobs", '{"epilogue_warps": 4}']
    )
    schedule = tuner._candidates(args, 512, rank=1)
    assert schedule and all(c["epilogue_warps"] == 4 for c in schedule)
    assert {c["load_balance_mode"] for c in schedule} == {"static", "atomic_counter"}
    for candidate in schedule:
        validate_tactic(candidate, partial=False)


def test_alternating_claims_default_band_env_and_overrides(monkeypatch) -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl.config import (
        ALTERNATING_CLAIM_TOKENS,
        LOAD_BALANCE_MODE_ENV,
    )

    monkeypatch.delenv(LOAD_BALANCE_MODE_ENV, raising=False)
    shape = dict(hidden=4096, intermediate=6400)
    assert ALTERNATING_CLAIM_TOKENS == (512, 1024)
    # The pair tile alternates claims in the N192 band only.
    for tokens, mode in (
        (64, "static"),
        (511, "static"),
        (512, "alternating"),
        (1023, "alternating"),
        (1024, "static"),
        (4096, "static"),
    ):
        tactic = default_tactic(tokens, **shape)
        assert tactic["load_balance_mode"] == mode
        assert (tactic["mma_tiler_mnk"][1] == 192) == (mode == "alternating")
    # 1-CTA tiles keep static claims.
    assert default_tactic(512)["load_balance_mode"] == "static"
    assert default_tactic(512, hidden=6528, intermediate=4096)["load_balance_mode"] == (
        "static"
    )
    # An explicit override wins; the environment replaces only the default.
    pinned = resolve_tactic(512, {"load_balance_mode": "static"}, **shape)
    assert pinned["load_balance_mode"] == "static"
    monkeypatch.setenv(LOAD_BALANCE_MODE_ENV, "static")
    assert default_tactic(512, **shape)["load_balance_mode"] == "static"
    monkeypatch.setenv(LOAD_BALANCE_MODE_ENV, "alternating")
    assert default_tactic(64)["load_balance_mode"] == "alternating"
    monkeypatch.setenv(LOAD_BALANCE_MODE_ENV, "dynamic")
    with pytest.raises(ValueError, match=LOAD_BALANCE_MODE_ENV):
        default_tactic(64)


def test_in_kernel_reduce_default_band_env_and_overrides(monkeypatch) -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl.config import (
        IN_KERNEL_REDUCE_ENV,
        IN_KERNEL_REDUCE_MAX_TOKENS,
    )

    monkeypatch.delenv(IN_KERNEL_REDUCE_ENV, raising=False)
    shape = dict(hidden=4096, intermediate=6400)
    # The tail reduction is the default below 256 tokens per rank, any tile.
    for tokens, enabled in (
        (1, True),
        (64, True),
        (255, True),
        (256, False),
        (1024, False),
    ):
        assert default_tactic(tokens, **shape)["in_kernel_reduce"] is enabled
        assert default_tactic(tokens)["in_kernel_reduce"] is enabled
    assert IN_KERNEL_REDUCE_MAX_TOKENS == 256
    # An explicit override wins; the environment replaces only the default.
    assert resolve_tactic(64, {"in_kernel_reduce": False})["in_kernel_reduce"] is False
    assert resolve_tactic(1024, {"in_kernel_reduce": True})["in_kernel_reduce"] is True
    monkeypatch.setenv(IN_KERNEL_REDUCE_ENV, "0")
    assert default_tactic(64)["in_kernel_reduce"] is False
    assert resolve_tactic(64, {"in_kernel_reduce": True})["in_kernel_reduce"] is True
    monkeypatch.setenv(IN_KERNEL_REDUCE_ENV, "1")
    assert default_tactic(1024)["in_kernel_reduce"] is True
    monkeypatch.setenv(IN_KERNEL_REDUCE_ENV, "true")
    with pytest.raises(ValueError, match=IN_KERNEL_REDUCE_ENV):
        default_tactic(64)


def test_tactic_defaults_candidates_and_partial_overrides_are_valid() -> None:
    small = default_tactic(8)
    large = default_tactic(2048)
    assert small["mma_tiler_mnk"] == (128, 64, 128)
    # The persistent kernel runs N64 and N128 only.
    assert large["mma_tiler_mnk"] == (128, 128, 128)
    assert small["group_hint"] == 4096
    assert large["group_hint"] == 512
    # Thresholds (tokens per rank): N128 from 256, group_hint=512 from 512.
    for tokens, n, group_hint in (
        (255, 64, 4096),
        (256, 128, 4096),
        (511, 128, 4096),
        (512, 128, 512),
    ):
        tactic = default_tactic(tokens)
        assert tactic["mma_tiler_mnk"] == (128, n, 128)
        assert tactic["group_hint"] == group_hint
    # With a 256-aligned shape the pair tile (2-CTA) is always used, with N192
    # from 512 to 1023 tokens per rank; otherwise the 1-CTA tiles above apply.
    shape = dict(hidden=6400, intermediate=4096)
    for tokens, m, n, group_hint in (
        (1, 256, 64, 4096),
        (255, 256, 64, 4096),
        (256, 256, 128, 4096),
        (511, 256, 128, 4096),
        (512, 256, 192, 512),
        (1023, 256, 192, 512),
        (1024, 256, 128, 512),
    ):
        tactic = default_tactic(tokens, **shape)
        assert tactic["mma_tiler_mnk"] == (m, n, 128)
        assert tactic["cluster_shape_mnk"] == ((2, 1, 1) if m == 256 else (1, 1, 1))
        assert tactic["group_hint"] == group_hint
    assert (
        default_tactic(512, hidden=6400, intermediate=4032)["mma_tiler_mnk"][0] == 128
    )
    assert (
        default_tactic(512, hidden=6528, intermediate=4096)["mma_tiler_mnk"][0] == 128
    )
    # Pinning a 1-CTA tile over a 2-CTA default also pins the 1-CTA cluster.
    pinned = resolve_tactic(512, {"mma_tiler_mnk": (128, 64, 128)}, **shape)
    assert pinned["cluster_shape_mnk"] == (1, 1, 1)
    paired = resolve_tactic(8, {"mma_tiler_mnk": (256, 64, 128)})
    assert paired["cluster_shape_mnk"] == (2, 1, 1)
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
    assert len(candidates) == 32
    assert {candidate["in_kernel_reduce"] for candidate in candidates} == {False, True}
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

    validate_tactic({"load_balance_mode": "alternating"})
    with pytest.raises(ValueError, match="load_balance_mode"):
        validate_tactic({"load_balance_mode": "unknown"})
    # FC1/FC2 alternation does not change the token-back scheduler's modes.
    with pytest.raises(ValueError, match="token_back_schedule_mode"):
        validate_tactic({"token_back_schedule_mode": "alternating"})

    for unsupported_n in (32, 256):
        with pytest.raises(ValueError, match="N must be 64, 128, or 192"):
            validate_tactic({"mma_tiler_mnk": (128, unsupported_n, 128)})
    validate_tactic({"mma_tiler_mnk": (256, 192, 128), "cluster_shape_mnk": (2, 1, 1)})

    with pytest.raises(ValueError, match="unknown"):
        validate_tactic({"decode_warps": 8})
    with pytest.raises(ValueError, match="requires cluster_shape_mnk"):
        validate_tactic(
            {
                "mma_tiler_mnk": (256, 128, 128),
                "cluster_shape_mnk": (1, 1, 1),
            }
        )
    # The 2-CTA pair tile is valid with its (2, 1, 1) cluster.
    validate_tactic({"mma_tiler_mnk": (256, 64, 128), "cluster_shape_mnk": (2, 1, 1)})
    with pytest.raises(ValueError, match="M must be 128"):
        validate_tactic({"mma_tiler_mnk": (64, 128, 128)})


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

    # Verify every row of both packed data and scales, including later pairs.
    order = torch.stack(
        (
            torch.arange(intermediate).reshape(-1, 8),
            torch.arange(intermediate, 2 * intermediate).reshape(-1, 8),
        ),
        dim=1,
    ).flatten()
    interleaved_sf = fc1_sf.reshape(experts, 2 * intermediate, hidden // 32)
    torch.testing.assert_close(interleaved_sf, s13[:, order])
    torch.testing.assert_close(fc1.view(torch.uint8).transpose(1, 2), w13[:, order])


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
