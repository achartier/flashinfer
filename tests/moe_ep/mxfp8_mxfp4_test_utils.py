# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent distributed oracle and split-backend setup for correctness gates.

Kept with tests so multi-rank validation does not import performance scripts.
The oracle checks the first token on each selected rank; graph/eager checks
in the calling harness cover all output elements.
"""

from typing import Any


def _broadcast_rank0_first(torch: Any, dist: Any, value: Any) -> Any:
    sample = value[:1].clone() if dist.get_rank() == 0 else torch.empty_like(value[:1])
    dist.broadcast(sample, src=0)
    return sample


def _quantize_mxfp4_rows(torch: Any, reference: Any, value: Any, chunk_rows: int = 32):
    """Bound the oracle's E2M1 nearest-value temporary for model-size rows."""

    packed, scales = [], []
    for begin in range(0, value.shape[0], chunk_rows):
        q, sf = reference.quantize_mxfp4(value[begin : begin + chunk_rows])
        packed.append(q)
        scales.append(sf)
    return torch.cat(packed), torch.cat(scales)


def _independent_refcheck(
    *,
    torch: Any,
    dist: Any,
    rank: int,
    world_size: int,
    hidden_states: Any,
    topk_ids: Any,
    topk_weights: Any,
    w13: Any,
    w2: Any,
    output: Any,
    gate_up_clamp: float | None,
    atol: float,
    rtol: float,
    rel_l2_limit: float,
    all_ranks: bool = False,
) -> float:
    """Check the first token on rank zero or every rank independently.

    Each rank evaluates the sampled routes for experts it owns.  Route terms
    are all-reduced before the specified fixed-slot routing reduction, avoiding
    an impractical all-gather of model-size expert weights.
    """

    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4 import reference

    def sample(value):
        if not all_ranks:
            return _broadcast_rank0_first(torch, dist, value)
        first = value[:1].contiguous()
        gathered = [torch.empty_like(first) for _ in range(world_size)]
        dist.all_gather(gathered, first)
        return torch.cat(gathered, dim=0)

    x = sample(hidden_states)
    ids = sample(topk_ids)
    scores = sample(topk_weights)
    x_q, x_sf = reference.quantize_mxfp8(x.float())
    x_fp32 = reference.dequantize_mxfp8(x_q, x_sf)
    route_terms = torch.zeros(
        (x.shape[0], ids.shape[-1], x.shape[-1]),
        dtype=torch.bfloat16,
        device=x.device,
    )
    local_experts = w13.shape[0]
    first_expert = rank * local_experts
    last_expert = first_expert + local_experts
    selected = torch.unique(ids[(ids >= first_expert) & (ids < last_expert)]).tolist()

    for global_expert in selected:
        local_expert = int(global_expert) - first_expert
        coordinates = (ids == int(global_expert)).nonzero(as_tuple=False)
        owner_rank, slot = coordinates.unbind(dim=1)

        interleaved = reference.interleave_gate_up_32(w13[local_expert])
        q13, sf13 = _quantize_mxfp4_rows(torch, reference, interleaved)
        w13_fp32 = reference.dequantize_mxfp4(q13, sf13)
        fc1 = x_fp32[owner_rank] @ w13_fp32.transpose(0, 1)
        del interleaved, q13, sf13, w13_fp32

        paired = fc1.reshape(fc1.shape[0], -1, 2, 32)
        gate, up = paired[:, :, 0, :], paired[:, :, 1, :]
        if gate_up_clamp is not None:
            limit = abs(float(gate_up_clamp))
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        swiglu = (up * (gate * torch.sigmoid(gate))).reshape(fc1.shape[0], -1)
        fc1_q, fc1_sf = reference.quantize_mxfp8(swiglu)
        fc1_roundtrip = reference.dequantize_mxfp8(fc1_q, fc1_sf)
        del fc1, paired, gate, up, swiglu, fc1_q, fc1_sf

        q2, sf2 = _quantize_mxfp4_rows(torch, reference, w2[local_expert])
        w2_fp32 = reference.dequantize_mxfp4(q2, sf2)
        fc2 = fc1_roundtrip @ w2_fp32.transpose(0, 1)
        route_terms[owner_rank, slot] = fc2.to(torch.bfloat16)
        del fc1_roundtrip, q2, sf2, w2_fp32, fc2

    dist.all_reduce(route_terms, op=dist.ReduceOp.SUM)
    expected = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for slot in range(ids.shape[-1]):
        expected += route_terms[:, slot].float() * scores[:, slot, None].float()
    expected = expected.to(torch.bfloat16)
    actual = sample(output).float()
    wanted = expected.float()
    # Gate each source rank separately: one bad rank must not be diluted by
    # other samples with larger output norms. Report the worst sampled rank.
    relative_errors = (actual - wanted).norm(dim=1) / wanted.norm(dim=1).clamp_min(1e-6)
    rel_l2 = float(relative_errors.max())
    torch.testing.assert_close(actual, wanted, atol=atol, rtol=rtol)
    if rel_l2 >= rel_l2_limit:
        raise AssertionError(
            f"independent oracle rel-L2 {rel_l2:.6g} >= {rel_l2_limit:.6g}"
        )
    return rel_l2


def _make_split_backend(args, rank: int, world_size: int, tokens: int) -> Any:
    from flashinfer import fused_moe as fm
    from flashinfer.moe_ep import (
        SplitConfig,
        NcclEpConfig,
        NvepConfig,
        FusedMoeKernelConfig,
    )

    local_experts = args.experts // world_size
    candidate = fm.MegaMoeFc12Config()
    moe = fm.MoEConfig(
        routing=fm.RoutingConfig(num_experts=args.experts, top_k=args.top_k),
        quant=fm.QuantConfig(
            weight=fm.QuantFormat.MXFP4, activation=fm.QuantFormat.MXFP8
        ),
        experts=fm.ExpertConfig(
            intermediate_size=args.intermediate,
            local_expert_offset=rank * local_experts,
            local_num_experts=local_experts,
        ),
        backend=fm.BackendOptions(candidates=(candidate,)),
        execution=fm.ExecutionConfig(
            enable_pdl=False, tune_max_num_tokens=tokens * world_size * local_experts
        ),
    )
    return SplitConfig(
        comm=NcclEpConfig() if args.transport == "nccl_ep" else NvepConfig(),
        kernel=FusedMoeKernelConfig(moe_config=moe, mxfp8_dispatch=True),
    )
