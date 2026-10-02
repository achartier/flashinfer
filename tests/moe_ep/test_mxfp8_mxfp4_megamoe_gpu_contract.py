# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-B200 staging, graph, and gated full-kernel contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch


def _require_b200() -> None:
    if not torch.cuda.is_available():
        pytest.skip("MXFP8 x MXFP4 GPU contracts need CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (10, 0):
        pytest.skip(
            f"MXFP8 x MXFP4 GPU contracts need B200 (sm_100), got sm_{major}{minor}"
        )


def _buffers(capacity: int, hidden: int, topk: int):
    return (
        torch.empty((capacity, hidden), dtype=torch.float8_e4m3fn, device="cuda"),
        torch.empty(
            (capacity, ((hidden // 32 + 3) // 4) * 4),
            dtype=torch.float8_e8m0fnu,
            device="cuda",
        ),
        torch.empty((capacity, topk), dtype=torch.int64, device="cuda"),
        torch.empty((capacity, topk), dtype=torch.float32, device="cuda"),
    )


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("tokens", [0, 1, 31, 32, 33])
def test_mxfp8_mxfp4_megamoe_prequantized_staging_is_byte_exact(tokens: int) -> None:
    _require_b200()
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.staging import stage_inputs

    capacity, hidden, topk = 64, 160, 4
    data = torch.randn((tokens, hidden), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.randint(
        1, 254, (tokens, hidden // 32), dtype=torch.uint8, device="cuda"
    ).view(torch.float8_e8m0fnu)
    ids = (
        torch.arange(tokens * topk, device="cuda", dtype=torch.int32).reshape(
            tokens, topk
        )
        % 8
    )
    scores = torch.randn((tokens, topk), device="cuda", dtype=torch.float32)
    out = _buffers(capacity, hidden, topk)
    staged = stage_inputs(data, scales, ids, scores, *out, quantize_input=False)
    torch.cuda.synchronize()
    assert staged.num_tokens == tokens
    assert torch.equal(staged.data.view(torch.uint8), data.view(torch.uint8))
    assert torch.equal(staged.scales.view(torch.uint8), scales.view(torch.uint8))
    assert torch.equal(staged.route_ids, ids.to(torch.int64))
    assert torch.equal(staged.route_scores, scores)
    if tokens < capacity:
        assert bool((out[2][tokens:] == -1).all())


@pytest.mark.arch_blackwell
def test_mxfp8_mxfp4_megamoe_frontend_requires_warmup_before_graph_capture() -> None:
    _require_b200()
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.frontend import (
        FrontendConfig,
        KernelLaunchSpec,
        Mxfp8Mxfp4Frontend,
    )

    class Adapter:
        def __init__(self) -> None:
            self.compiles = 0
            self.binds = 0

        def compile(self, config, spec):
            self.compiles += 1
            return object()

        def bind(self, compiled, config, spec):
            self.binds += 1

            def launch():
                spec.output[: spec.num_tokens].copy_(
                    spec.activation[: spec.num_tokens].to(torch.bfloat16)
                )

            return launch

    hidden, intermediate, topk, capacity, tokens = 128, 128, 2, 8, 3
    activation, scales, ids, scores = _buffers(capacity, hidden, topk)
    activation.copy_(torch.randn((capacity, hidden), device="cuda"))
    tactic = (256, 128, 128)
    config = FrontendConfig(hidden, intermediate, topk, 2, capacity, tactic)
    spec = KernelLaunchSpec(
        activation=activation,
        activation_scales=scales,
        fc1_weight=torch.zeros(
            (2, hidden // 2, 2 * intermediate), dtype=torch.uint8, device="cuda"
        ),
        fc1_weight_scales=torch.zeros(
            (2, 1), dtype=torch.float8_e8m0fnu, device="cuda"
        ),
        fc2_weight=torch.zeros(
            (2, intermediate // 2, hidden), dtype=torch.uint8, device="cuda"
        ),
        fc2_weight_scales=torch.zeros(
            (2, 1), dtype=torch.float8_e8m0fnu, device="cuda"
        ),
        route_ids=ids,
        route_scores=scores,
        output=torch.empty((capacity, hidden), dtype=torch.bfloat16, device="cuda"),
        workspace_regions={
            "scratch": torch.zeros(16, dtype=torch.uint8, device="cuda")
        },
        num_tokens=tokens,
        stream=torch.cuda.current_stream().cuda_stream,
        static_tactic=tactic,
    )
    adapter = Adapter()
    frontend = Mxfp8Mxfp4Frontend(config, adapter)
    frontend.warmup(spec)
    assert (adapter.compiles, adapter.binds) == (1, 1)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = frontend.run(spec)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, activation[:tokens].to(torch.bfloat16))
    assert (adapter.compiles, adapter.binds) == (1, 1)

    missed = replace(spec, output=torch.empty_like(spec.output))
    with (
        pytest.raises(RuntimeError, match="warmup"),
        torch.cuda.graph(torch.cuda.CUDAGraph()),
    ):
        frontend.run(missed)


def _full_kernel_case(tokens, hidden, intermediate, experts, topk, *, knobs=None):
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.integration import (
        get_symm_buffer_for_mxfp8_mxfp4_mega_moe,
        mxfp8_mxfp4_mega_moe,
    )
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.reference import (
        compute_reference,
        interleave_gate_up_32,
        quantize_mxfp4,
        quantize_mxfp8,
    )
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.weight_scale_views import (
        to_atom_swizzled,
    )

    generator = torch.Generator(device="cuda").manual_seed(20260922)

    def random(shape, scale):
        return torch.randn(shape, generator=generator, device="cuda") * scale

    xq, xs = quantize_mxfp8(random((tokens, hidden), 0.5))
    w13q, w13s = quantize_mxfp4(
        interleave_gate_up_32(random((experts, 2 * intermediate, hidden), 0.125))
    )
    w2q, w2s = quantize_mxfp4(random((experts, hidden, intermediate), 0.125))
    ids = torch.arange(tokens * topk, device="cuda").reshape(tokens, topk) % experts
    # Empty experts, masked routes, nonuniform scores, and partial token tiles.
    if experts > topk:
        ids %= experts - 1
    if tokens:
        ids[-1, -1] = -1
    scores = random((tokens, topk), 0.3)
    expected = compute_reference(
        input_q=xq.unsqueeze(0),
        input_scale=xs.unsqueeze(0),
        topk_ids=ids.unsqueeze(0),
        topk_weights=scores.unsqueeze(0),
        w13_packed=w13q,
        w13_scale=w13s,
        w2_packed=w2q,
        w2_scale=w2s,
    ).output[0]
    workspace = get_symm_buffer_for_mxfp8_mxfp4_mega_moe(
        experts,
        max(tokens, 1),
        topk,
        hidden,
        intermediate,
        0,
        1,
        knobs=knobs,
    )
    workspace.x[:tokens].copy_(xq)
    workspace.x_sf[:tokens].view(torch.uint8).copy_(xs)
    workspace.topk_idx[:tokens].copy_(ids)
    workspace.topk_weights[:tokens].copy_(scores)
    w13 = (w13q.transpose(1, 2), torch.stack([to_atom_swizzled(s) for s in w13s]))
    w2 = (w2q.transpose(1, 2), torch.stack([to_atom_swizzled(s) for s in w2s]))

    def launch():
        return mxfp8_mxfp4_mega_moe(None, w13, w2, workspace, num_tokens=tokens)

    return workspace, launch, expected


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("load_balance_mode", ["static", "atomic_counter"])
@pytest.mark.parametrize(
    "token_back_mode,token_back_schedule_mode",
    [
        ("epi_warps", "static"),
        ("reuse_dispatch_warps", "static"),
        ("reuse_dispatch_warps", "atomic_counter"),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (4, 128, 128, 1, 1),
        (33, 256, 256, 4, 2),
        (129, 256, 128, 4, 2),
        (0, 128, 128, 2, 1),
    ],
)
def test_mxfp8_mxfp4_megamoe_full_kernel_boundary_shapes(
    shape, load_balance_mode, token_back_mode, token_back_schedule_mode
) -> None:
    _require_b200()
    workspace, launch, expected = _full_kernel_case(
        *shape,
        knobs={
            "load_balance_mode": load_balance_mode,
            "token_back_mode": token_back_mode,
            "token_back_schedule_mode": token_back_schedule_mode,
        },
    )
    try:
        assert workspace.plan.config.token_back_by_dispatch == (
            token_back_mode == "reuse_dispatch_warps"
        )
        for _ in range(3):
            actual = launch()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.05)
    finally:
        workspace.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("load_balance_mode", ["static", "atomic_counter"])
@pytest.mark.parametrize(
    "token_back_mode,token_back_schedule_mode",
    [
        ("epi_warps", "static"),
        ("reuse_dispatch_warps", "static"),
        ("reuse_dispatch_warps", "atomic_counter"),
    ],
)
@pytest.mark.parametrize(
    "tokens,tile_tokens,cluster_shape",
    [
        (33, 64, (1, 1, 1)),
        # More tiles than resident clusters exercises repeated atomic claims.
        (4097, 128, (1, 1, 1)),
        (129, 128, (2, 1, 1)),
    ],
)
def test_mxfp8_mxfp4_megamoe_full_kernel_cuda_graph_replay(
    load_balance_mode,
    tokens,
    tile_tokens,
    cluster_shape,
    token_back_mode,
    token_back_schedule_mode,
) -> None:
    _require_b200()
    workspace, launch, expected = _full_kernel_case(
        tokens,
        256,
        256,
        4,
        2,
        knobs={
            "load_balance_mode": load_balance_mode,
            "token_back_mode": token_back_mode,
            "token_back_schedule_mode": token_back_schedule_mode,
            "mma_tiler_mnk": (128, tile_tokens, 128),
            "cluster_shape_mnk": cluster_shape,
            "group_hint": 1,
        },
    )
    try:
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            baseline = launch().clone()
            for _ in range(2):
                torch.testing.assert_close(launch(), expected, atol=0.015, rtol=0.05)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            actual = launch()
        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
            torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.05)
    finally:
        workspace.destroy()


@pytest.mark.arch_blackwell
def test_mxfp8_mxfp4_token_back_modes_are_bit_exact() -> None:
    _require_b200()
    baseline = None
    for mode, schedule in (
        ("epi_warps", "static"),
        ("reuse_dispatch_warps", "static"),
        ("reuse_dispatch_warps", "atomic_counter"),
    ):
        workspace, launch, expected = _full_kernel_case(
            129,
            384,
            256,
            4,
            2,
            knobs={
                "token_back_mode": mode,
                "token_back_schedule_mode": schedule,
                "load_balance_mode": "atomic_counter",
                "group_hint": 1,
            },
        )
        try:
            actual = launch().clone()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.05)
            if baseline is None:
                baseline = actual
            else:
                torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
            # Empty and restore routes without resetting the workspace. Stale
            # pool rows must never leak into the deterministic home reduction.
            ids = workspace.topk_idx.clone()
            workspace.topk_idx.fill_(-1)
            torch.testing.assert_close(
                launch(), torch.zeros_like(actual), atol=0, rtol=0
            )
            workspace.topk_idx.copy_(ids)
            torch.testing.assert_close(launch(), actual, atol=0, rtol=0)
        finally:
            workspace.destroy()
