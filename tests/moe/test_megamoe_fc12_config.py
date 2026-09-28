"""Host-side contracts for the unified MegaMOE FC12 backend."""

from __future__ import annotations

import torch
import pytest

from flashinfer.fused_moe import (
    BackendOptions,
    ExpertConfig,
    MegaMoeFc12Config,
    MoEConfig,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
)
from flashinfer.fused_moe.megamoe_fc12 import prepare_megamoe_fc12_weights


def _config() -> MoEConfig:
    return MoEConfig(
        routing=RoutingConfig(num_experts=2, top_k=1),
        quant=QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
        experts=ExpertConfig(intermediate_size=128, local_num_experts=2),
        backend=BackendOptions(candidates=(MegaMoeFc12Config(),)),
    )


def test_config_is_registered_and_exported():
    cfg = _config()
    assert cfg.backend.valid_for(100) == [MegaMoeFc12Config()]
    assert cfg.backend.valid_for(89) == []


def test_unsupported_family_is_not_silently_prepared():
    w13 = torch.empty(1, 128, 32, dtype=torch.bfloat16)
    w2 = torch.empty(1, 32, 64, dtype=torch.bfloat16)
    try:
        prepare_megamoe_fc12_weights(
            w13,
            w2,
            quant=QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8),
            num_local_experts=1,
            hidden_size=32,
            intermediate_size=64,
        )
    except NotImplementedError as error:
        assert "MXFP8" in str(error)
    else:
        raise AssertionError("unsupported FC12 family must not get a BF16 view")


def test_mxfp8_mxfp4_fc12_is_registered():
    from flashinfer.fused_moe.runners import MegaMoeFc12Runner

    assert (
        (
            QuantFormat.MXFP4,
            QuantFormat.MXFP8,
        ),
    ) == MegaMoeFc12Runner.supported_quant_variants


def test_mxfp8_mxfp4_preparer_selects_independent_weight_path(monkeypatch):
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl import (
        weights,
    )

    sentinel = [torch.empty(0) for _ in range(4)]
    calls = []

    def prepare(pack, **kwargs):
        calls.append((pack, kwargs))
        return (sentinel[0], sentinel[1]), (sentinel[2], sentinel[3])

    monkeypatch.setattr(weights, "preprocess_mega_weights", prepare)
    view = prepare_megamoe_fc12_weights(
        torch.empty(2, 256, 128, dtype=torch.bfloat16),
        torch.empty(2, 128, 128, dtype=torch.bfloat16),
        quant=QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
        num_local_experts=2,
        hidden_size=128,
        intermediate_size=128,
    )
    assert view["fc1_weight"] is sentinel[0]
    assert view["fc2_weight_sf"] is sentinel[3]
    assert calls[0][1] == {"intermediate_size": 128, "hidden_size": 128}


def test_mxfp8_mxfp4_finalize_uses_slot_order_and_ignores_masked_weights():
    from flashinfer.fused_moe.megamoe_fc12 import unpermute_mxfp8_mxfp4_routes

    # Cancellation distinguishes ascending slots from a reassociated sum.
    terms = torch.tensor([[2**24], [-(2**24)], [1]], dtype=torch.bfloat16)
    mapping = torch.tensor([[0, 2, 1], [0, 1, 2], [-1, -1, -1]], dtype=torch.int32)
    scores = torch.ones(3, 3, dtype=torch.float32)
    scores[2] = float("nan")
    output = torch.empty(3, 1, dtype=torch.bfloat16)
    unpermute_mxfp8_mxfp4_routes(terms, output, mapping, scores)
    assert output[:, 0].tolist() == [0, 1, 0]


@pytest.mark.parametrize(
    "counts, expected", [([1, 0, 65], [0, 64, 64, 192]), ([0, 0, 0], [0, 0, 0, 0])]
)
def test_mxfp8_mxfp4_fc12_launcher_passes_local_offsets(monkeypatch, counts, expected):
    import sys
    import types
    from flashinfer.fused_moe.megamoe_fc12 import (
        Mxfp8Mxfp4Fc12Inputs,
        Mxfp8Mxfp4Fc12Launcher,
    )

    seen = []

    class Launcher:
        def __init__(self, config):
            assert config.local_num_experts == 3

        def run(self, inputs):
            seen.append(inputs)

    module = types.ModuleType("lean_kernel")
    module.LeanFc12Launcher = Launcher
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.lean_kernel",
        module,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    dummy = torch.empty(0)
    launcher = Mxfp8Mxfp4Fc12Launcher(3, 192, 128, 128)
    launcher.run(
        Mxfp8Mxfp4Fc12Inputs(
            dummy,
            dummy,
            dummy,
            dummy,
            torch.tensor(counts, dtype=torch.int32),
            dummy,
            dummy,
            dummy,
        )
    )
    assert seen[0].expert_data_row_offsets.tolist() == expected
    assert seen[0].expert_scale_row_offsets is seen[0].expert_data_row_offsets


@pytest.mark.parametrize(
    "pair",
    [
        (QuantFormat.BF16, QuantFormat.BF16),
        (QuantFormat.NVFP4, QuantFormat.BF16),
    ],
)
def test_fc12_rejects_non_mxfp8_mxfp4_weights(pair):
    with pytest.raises(NotImplementedError, match="MXFP4 weights and MXFP8"):
        prepare_megamoe_fc12_weights(
            torch.empty(1, 256, 128, dtype=torch.bfloat16),
            torch.empty(1, 128, 128, dtype=torch.bfloat16),
            quant=QuantConfig(weight=pair[0], activation=pair[1]),
            num_local_experts=1,
            hidden_size=128,
            intermediate_size=128,
        )


def test_ep_materializes_mxfp8_mxfp4_view(monkeypatch):
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.split.kernel.fused_moe.weights import (
        materialize_fused_moe_weights,
    )

    sentinel = {"fc1_weight": torch.empty(0), "fc2_weight": torch.empty(0)}
    calls = []

    def prepare(w13, w2, **kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(MegaMoeFc12Config, "prepare_weights", staticmethod(prepare))
    native = materialize_fused_moe_weights(
        MoEWeightPack(
            w13=torch.empty(2, 256, 128, dtype=torch.bfloat16),
            w2=torch.empty(2, 128, 128, dtype=torch.bfloat16),
        ),
        _config(),
    )
    assert native.get_view("megamoe_fc12")["fc1_weight"] is sentinel["fc1_weight"]
    assert calls[0]["quant"].pair == (QuantFormat.MXFP4, QuantFormat.MXFP8)
