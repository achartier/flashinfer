"""Host tests for explicit wire padding; logical MoE shapes never change."""

import pytest
import torch

from flashinfer.moe_ep.config import EpAlgorithm, EpLayout, FleetParams
from flashinfer.moe_ep.core.comm.padding import (
    finish_ll_combine,
    pad_ll_rows,
    prepare_ll_combine,
)
from flashinfer.moe_ep.core.validation.common import validate_ll_hidden_size


def test_mai_ll_output_padding_is_byte_exact():
    params = FleetParams(512, 128, 6656, transport_hidden_size=7168)
    validate_ll_hidden_size(params, "nccl_ep")
    bits = torch.arange(2 * 3 * 6656, dtype=torch.int32).to(torch.int16)
    x = bits.view(torch.bfloat16).reshape(2, 3, 6656)
    out = torch.empty(3, 6656, dtype=torch.bfloat16)
    wire_x, wire_out, logical_out = prepare_ll_combine(x, out, params, 3)
    assert logical_out is out
    assert wire_x.shape == (2, 3, 7168)
    assert torch.equal(wire_x[..., :6656].view(torch.int16), x.view(torch.int16))
    assert torch.count_nonzero(wire_x[..., 6656:].view(torch.int16)) == 0
    wire_out.copy_(wire_x[0])
    assert finish_ll_combine(wire_out, out) is out
    assert torch.equal(out.view(torch.int16), x[0].view(torch.int16))


def test_unpadded_combine_reuses_buffers():
    params = FleetParams(8, 16, 2048)
    x = torch.empty(2, 16, 2048, dtype=torch.bfloat16)
    out = torch.empty(4, 2048, dtype=torch.bfloat16)
    wire_x, wire_out, logical_out = prepare_ll_combine(x, out, params, 4)
    assert wire_x is x and wire_out is out and logical_out is out


@pytest.mark.parametrize(
    "kwargs",
    [
        {"transport_hidden_size": 6144},
        {"transport_hidden_size": 7168, "algorithm": EpAlgorithm.HIGH_THROUGHPUT},
        {"transport_hidden_size": 7168, "layout": EpLayout.RANK_MAJOR},
        {"transport_hidden_size": 7168, "dtype_bytes": 1},
    ],
)
def test_invalid_padding_config(kwargs):
    with pytest.raises(ValueError):
        FleetParams(8, 16, 6656, **kwargs)


def test_ll_padding_still_requires_instantiated_wire_width():
    with pytest.raises(ValueError, match="does not support"):
        validate_ll_hidden_size(
            FleetParams(8, 16, 6656, transport_hidden_size=6784), "nccl_ep"
        )


def test_padding_rejects_wrong_width_dtype_and_output():
    with pytest.raises(ValueError, match="smaller"):
        pad_ll_rows(torch.empty(2, 32), 16)
    with pytest.raises(ValueError, match="BF16"):
        pad_ll_rows(torch.empty(2, 16), 32)
    params = FleetParams(8, 16, 2048)
    with pytest.raises(ValueError, match="logical width"):
        prepare_ll_combine(torch.empty(2, 1024), None, params, 2)
    with pytest.raises(ValueError, match="combine out"):
        prepare_ll_combine(torch.empty(2, 2048), torch.empty(3, 2048), params, 2)


@pytest.mark.parametrize("staged", [False, True])
def test_nixl_padding_copy_waits_for_recv_hook(staged):
    from types import SimpleNamespace

    from flashinfer.moe_ep.algo_knobs import HandleAlgoKnobTopKWeights
    from flashinfer.moe_ep.backends.split.comm.nixl_ep.handle import NixlEpHandle
    from flashinfer.moe_ep.config import CombineInputParams

    calls = []

    def combine(x, *args, **kwargs):
        assert x.shape[-1] == 7168
        assert not torch.count_nonzero(x[..., 6656:])
        wire_out = kwargs["out"]

        def hook():
            calls.append("receive")
            wire_out.fill_(7)

        return wire_out, None, hook

    handle = object.__new__(NixlEpHandle)
    handle._fleet = SimpleNamespace(
        params=FleetParams(8, 16, 6656, transport_hidden_size=7168),
        buffer=SimpleNamespace(low_latency_combine=combine),
    )
    handle._topk_ids = torch.zeros(3, 2, dtype=torch.int64)
    handle._handle_knobs = {
        HandleAlgoKnobTopKWeights: HandleAlgoKnobTopKWeights(torch.ones(3, 2))
    }
    handle._nixl_handle = None
    handle._staged = staged
    out = torch.full((3, 6656), -1, dtype=torch.bfloat16)
    result = handle.combine(
        CombineInputParams(x=[torch.zeros(2, 16, 6656, dtype=torch.bfloat16)], out=out)
    )
    assert result.x is out
    if staged:
        assert calls == [] and torch.all(out == -1)
        handle.complete()
    assert calls == ["receive"] and torch.all(out == 7)
