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
        quant=QuantConfig(),
        experts=ExpertConfig(intermediate_size=64, local_num_experts=2),
        backend=BackendOptions(candidates=(MegaMoeFc12Config(),)),
    )


def test_config_is_registered_and_exported():
    cfg = _config()
    assert cfg.backend.valid_for(100) == [MegaMoeFc12Config()]
    assert cfg.backend.valid_for(89) == []


def test_bf16_weight_view_uses_fc12_layout():
    w13 = torch.arange(2 * 128 * 32, dtype=torch.float32).reshape(2, 128, 32).bfloat16()
    w2 = torch.arange(2 * 32 * 64, dtype=torch.float32).reshape(2, 32, 64).bfloat16()
    view = prepare_megamoe_fc12_weights(
        w13,
        w2,
        quant=QuantConfig(),
        num_local_experts=2,
        hidden_size=32,
        intermediate_size=64,
    )
    assert view["fc1_weight"].shape == (2, 32, 128)
    assert view["fc2_weight"].shape == (2, 64, 32)
    # FC12 uses gate/up groups of 32: gate group 0, then up group 0.
    assert torch.equal(view["fc1_weight"][0, :, :32], w13[0, :32].transpose(0, 1))
    assert torch.equal(view["fc1_weight"][0, :, 32:64], w13[0, 64:96].transpose(0, 1))


def test_non_bf16_family_is_not_silently_prepared():
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


def test_ep_materializes_megamoe_view_alongside_other_candidates():
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.split.kernel.fused_moe.weights import (
        materialize_fused_moe_weights,
    )

    w13 = torch.zeros(1, 128, 32, dtype=torch.bfloat16)
    w2 = torch.zeros(1, 32, 64, dtype=torch.bfloat16)
    cfg = MoEConfig(
        routing=RoutingConfig(num_experts=1, top_k=1),
        quant=QuantConfig(),
        experts=ExpertConfig(intermediate_size=64, local_num_experts=1),
        backend=BackendOptions(candidates=(MegaMoeFc12Config(),)),
    )
    native = materialize_fused_moe_weights(MoEWeightPack(w13=w13, w2=w2), cfg)
    assert set(native.get_view("megamoe_fc12")) == {"fc1_weight", "fc2_weight"}


def test_rank_major_bridge_preserves_int32_ids_and_bf16_weights():
    from flashinfer.moe_ep.backends.split.kernel.fused_moe.bridge import (
        build_activation_pack_rank_major,
    )

    received = torch.zeros(1, 2, 32, dtype=torch.bfloat16)
    ids = torch.tensor([[0, -1], [0, 0]], dtype=torch.int32)
    weights = torch.tensor([[0.5, 1.0], [0.25, 0.75]], dtype=torch.bfloat16)
    pack = build_activation_pack_rank_major(
        received,
        ids,
        weights,
        num_local_experts=1,
        quant=QuantConfig(),
    )
    assert pack.topk_ids.dtype is torch.int32
    assert pack.topk_weights.dtype is torch.bfloat16
    assert pack.topk_ids.tolist() == [[0, 0], [0, 0]]
    assert pack.topk_weights.tolist() == [[0.5, 0.0], [0.25, 0.75]]


def test_mxfp8_mxfp4_preparer_selects_native_weight_path(monkeypatch):
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl import (
        weights,
    )

    sentinel = [torch.empty(0) for _ in range(4)]

    def prepare(pack, **kwargs):
        assert kwargs == {"intermediate_size": 128, "hidden_size": 128}
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
    assert view["fc1_weight_sf"] is sentinel[1]
    assert view["fc2_weight"] is sentinel[2]
    assert view["fc2_weight_sf"] is sentinel[3]


def test_mxfp8_mxfp4_finalize_uses_slot_order_and_masks_nonfinite_weights():
    from flashinfer.fused_moe.megamoe_fc12 import unpermute_mxfp8_mxfp4_routes

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
def test_mxfp8_mxfp4_launcher_keeps_offsets_on_device(monkeypatch, counts, expected):
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
    assert seen[0].expert_data_row_offsets.dtype is torch.int32
    assert seen[0].expert_scale_row_offsets is seen[0].expert_data_row_offsets


def test_ep_materializes_mxfp8_mxfp4_view(monkeypatch):
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.split.kernel.fused_moe.weights import (
        materialize_fused_moe_weights,
    )

    sentinel = {"fc1_weight": torch.empty(0), "fc2_weight": torch.empty(0)}

    def prepare(w13, w2, **kwargs):
        assert kwargs["quant"].pair == (QuantFormat.MXFP4, QuantFormat.MXFP8)
        return sentinel

    monkeypatch.setattr(MegaMoeFc12Config, "prepare_weights", staticmethod(prepare))
    cfg = MoEConfig(
        routing=RoutingConfig(num_experts=2, top_k=1),
        quant=QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
        experts=ExpertConfig(intermediate_size=128, local_num_experts=2),
        backend=BackendOptions(candidates=(MegaMoeFc12Config(),)),
    )
    native = materialize_fused_moe_weights(
        MoEWeightPack(
            w13=torch.empty(2, 256, 128, dtype=torch.bfloat16),
            w2=torch.empty(2, 128, 128, dtype=torch.bfloat16),
        ),
        cfg,
    )
    assert native.get_view("megamoe_fc12") is sentinel


def test_mxfp8_mxfp4_counts_skip_masked_and_nonlocal_routes():
    from flashinfer.fused_moe.megamoe_fc12 import mxfp8_mxfp4_expert_counts

    ids = torch.tensor([[3, -1, 7], [4, 3, 8]], dtype=torch.int32)
    counts = mxfp8_mxfp4_expert_counts(ids, local_expert_offset=3, num_local_experts=2)
    assert counts.dtype is torch.int32
    assert counts.tolist() == [2, 1]
