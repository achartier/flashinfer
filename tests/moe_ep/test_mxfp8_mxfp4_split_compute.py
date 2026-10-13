"""Single-GPU split FC12 compute validation without a transport dependency."""

import pytest
import torch


@pytest.mark.arch_blackwell
def test_mxfp8_split_dispatch_payload_roundtrips_data_and_scales():
    from flashinfer.fused_moe.api import QuantConfig, QuantFormat
    from flashinfer.moe_ep.backends.split.kernel.fused_moe.bridge import (
        build_activation_pack,
        pack_mxfp8_dispatch_payload,
    )
    from flashinfer.quantization.fp8_quantization import mxfp8_quantize

    x = torch.randn(2, 2048, device="cuda", dtype=torch.bfloat16) / 10
    packed = pack_mxfp8_dispatch_payload(x)
    act = build_activation_pack(
        packed.unsqueeze(0),
        quant=QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
        mxfp8_dispatch=True,
        hidden_size=2048,
    )
    expected_data, expected_scales = mxfp8_quantize(
        x.contiguous(), is_sf_swizzled_layout=False
    )
    assert torch.equal(
        act.hidden_states_q.view(torch.uint8), expected_data.view(torch.uint8)
    )
    assert torch.equal(
        act.hidden_states_scale, expected_scales.view(torch.uint8).reshape(2, 64)
    )


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("prequantized", [False, True])
def test_mxfp8_mxfp4_split_compute_masks_padding_and_matches_oracle(prequantized):
    from flashinfer import fused_moe as fm
    from flashinfer.moe_ep import (
        EpLayout,
        FleetParams,
        MoEWeightPack,
        SplitKernelContext,
    )
    from flashinfer.moe_ep.backends.split.kernel.fused_moe.backend import (
        FusedMoeSplitKernelBackend,
    )
    from flashinfer.moe_ep.backends.split.kernel.fused_moe.config import (
        FusedMoeKernelConfig,
    )
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4 import reference

    device = torch.device("cuda")
    torch.manual_seed(5309)
    experts, cap, hidden, intermediate = 2, 4, 128, 128
    counts = torch.tensor([1, 3], dtype=torch.int32, device=device)
    x = torch.randn(experts, cap, hidden, device=device, dtype=torch.bfloat16) / 10
    w13 = (
        torch.randn(
            experts, 2 * intermediate, hidden, device=device, dtype=torch.bfloat16
        )
        / 10
    )
    w2 = (
        torch.randn(experts, hidden, intermediate, device=device, dtype=torch.bfloat16)
        / 10
    )
    config = fm.MoEConfig(
        routing=fm.RoutingConfig(num_experts=experts, top_k=1),
        quant=fm.QuantConfig(
            weight=fm.QuantFormat.MXFP4, activation=fm.QuantFormat.MXFP8
        ),
        experts=fm.ExpertConfig(
            intermediate_size=intermediate, local_num_experts=experts
        ),
        backend=fm.BackendOptions(candidates=(fm.MegaMoeFc12Config(),)),
        execution=fm.ExecutionConfig(
            enable_pdl=False, tune_max_num_tokens=experts * cap
        ),
    )
    fleet = FleetParams(
        num_experts=experts,
        max_tokens_per_rank=cap,
        token_hidden_size=hidden,
        layout=EpLayout.EXPERT_MAJOR,
    )
    backend = FusedMoeSplitKernelBackend(FusedMoeKernelConfig(moe_config=config))
    if prequantized:
        q13, sf13 = zip(
            *(reference.quantize_mxfp4(w13[e]) for e in range(experts)), strict=True
        )
        q2, sf2 = zip(
            *(reference.quantize_mxfp4(w2[e]) for e in range(experts)), strict=True
        )
        weights = MoEWeightPack(
            w13=torch.stack(q13),
            w2=torch.stack(q2),
            w13_scale=torch.stack(sf13),
            w2_scale=torch.stack(sf2),
        )
    else:
        weights = MoEWeightPack(w13=w13, w2=w2)
    backend.preprocess_weights(weights, fleet)
    ctx = SplitKernelContext(
        expert_tensors=x,
        num_tokens=cap,
        fleet_params=fleet,
        recv_count=counts,
    )
    output = backend.compute(ctx)
    torch.cuda.synchronize()

    expected = torch.zeros_like(output)
    for expert, n in enumerate(counts.cpu().tolist()):
        if n == 0:
            continue
        x_q, x_sf = reference.quantize_mxfp8(x[expert, :n].float())
        x_fp32 = reference.dequantize_mxfp8(x_q, x_sf)
        interleaved = reference.interleave_gate_up_8(w13[expert])
        q13, sf13 = reference.quantize_mxfp4(interleaved)
        fc1 = x_fp32 @ reference.dequantize_mxfp4(q13, sf13).T
        paired = fc1.reshape(n, -1, 2, 8)
        gate, up = paired[:, :, 0, :], paired[:, :, 1, :]
        activated = (up * (gate * torch.sigmoid(gate))).reshape(n, -1)
        a2_q, a2_sf = reference.quantize_mxfp8(activated)
        q2, sf2 = reference.quantize_mxfp4(w2[expert])
        fc2 = (
            reference.dequantize_mxfp8(a2_q, a2_sf)
            @ reference.dequantize_mxfp4(q2, sf2).T
        )
        expected[expert, :n] = fc2.to(torch.bfloat16)

    torch.testing.assert_close(output, expected, atol=0.02, rtol=0.02)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = backend.compute(ctx)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, output, atol=0, rtol=0)

    x.neg_()
    counts.copy_(torch.tensor([0, 2], dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()
    eager = backend.compute(ctx)
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, atol=0, rtol=0)
