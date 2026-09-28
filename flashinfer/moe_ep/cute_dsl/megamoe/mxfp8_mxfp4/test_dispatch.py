# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused one-B200 test for native MXFP8 single-rank dispatch."""

from __future__ import annotations

import pytest
import torch

from .device_workspace import DeviceWorkspacePartition
from .dispatch import SingleRankMxfp8Dispatch
from .workspace import WorkspaceConfig, make_workspace_plan


def _config() -> WorkspaceConfig:
    return WorkspaceConfig(
        world_size=1,
        num_topk=2,
        num_experts_per_rank=4,
        max_tokens_per_rank=8,
        hidden=128,
        intermediate=128,
        token_padding_block=64,
        sf_padding_block=128,
        cluster_tile_tokens=64,
    )


def _metadata(token: int, topk: int) -> int:
    return token | (topk << 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_sms", [1, 4])
def test_single_rank_dispatch_compacts_mxfp8_routes_on_device(num_sms) -> None:
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        pytest.skip("requires an SM100-class B200")

    config = _config()
    plan = make_workspace_plan(config)
    device = torch.device("cuda")
    local_bytes = torch.zeros(plan.local.total_bytes, dtype=torch.uint8, device=device)
    shared_bytes = torch.zeros(
        plan.shared.total_bytes, dtype=torch.uint8, device=device
    )
    route_terms = torch.zeros(
        config.max_tokens_per_rank,
        config.num_topk,
        config.hidden,
        dtype=torch.bfloat16,
        device=device,
    )
    partition = DeviceWorkspacePartition(
        plan,
        local_bytes,
        route_terms,
        shared_workspace=shared_bytes,
    )

    token_bytes = (
        torch.arange(
            config.max_tokens_per_rank * config.hidden,
            dtype=torch.int32,
            device=device,
        )
        .to(torch.uint8)
        .reshape(config.max_tokens_per_rank, config.hidden)
    )
    activation = token_bytes.view(torch.float8_e4m3fn)
    scale_bytes = (
        torch.arange(
            config.max_tokens_per_rank * config.sf_uint32_per_token * 4,
            dtype=torch.int32,
            device=device,
        )
        .add_(1)
        .to(torch.uint8)
        .reshape(config.max_tokens_per_rank, config.sf_uint32_per_token * 4)
    )
    scales = scale_bytes.view(torch.float8_e8m0fnu)
    route_ids = torch.full(
        (config.max_tokens_per_rank, config.num_topk),
        -1,
        dtype=torch.int64,
        device=device,
    )
    route_ids[:4].copy_(torch.tensor([[2, 0], [-1, 2], [1, 0], [2, 1]], device=device))
    route_scores = torch.arange(
        config.max_tokens_per_rank * config.num_topk,
        dtype=torch.float32,
        device=device,
    ).reshape(config.max_tokens_per_rank, config.num_topk)

    # launch() issues on the stream bound at compile time, so the graph below
    # must capture that same stream.
    stream = torch.cuda.Stream()
    products = SingleRankMxfp8Dispatch(config, num_sms=num_sms).compile(
        partition,
        activation,
        scales,
        route_ids,
        route_scores,
        stream=stream.cuda_stream,
    )
    products()
    torch.cuda.synchronize()

    # Pool rows are expert-major; within an expert the stable order is the
    # original (token, top-k slot) scan order.  Expert 3 is empty.
    expected_routes = (((0, 1), (2, 1)), ((2, 0), (3, 1)), ((0, 0), (1, 1), (3, 0)))
    starts = (0, 64, 128)
    counts = products.expert_pool_offsets.expert_counts.cpu()
    torch.testing.assert_close(counts, torch.tensor([2, 2, 3, 0], dtype=torch.int32))

    metadata = products.token_src_metadata.view(torch.int64).reshape(-1)
    scale_pool = products.scale_pool
    for start, routes in zip(starts, expected_routes, strict=True):
        for position, (token, topk) in enumerate(routes):
            pool_row = start + position
            torch.testing.assert_close(
                products.token_pool[pool_row].cpu(), token_bytes[token].cpu()
            )
            assert int(metadata[pool_row].item()) == _metadata(token, topk)
            actual_score = products.compatibility_route_scores[pool_row].item()
            assert actual_score == pytest.approx(route_scores[token, topk].item())
            # hidden=128 has one uint32 SF atom per token.  At the start of
            # each expert's 128-row SF block, token positions map to 0,4,8...
            sf_row = (start // 64) * 128
            sf_index = (sf_row // 128) * 128 + position * 4
            actual_sf = scale_pool[sf_index : sf_index + 1].view(torch.uint8).cpu()
            torch.testing.assert_close(actual_sf, scale_bytes[token].cpu())

    torch.testing.assert_close(
        products.fc1_ready_counters[:3].cpu(),
        torch.tensor([2, 2, 3], dtype=torch.int32),
    )

    # A bound launcher must be reusable: counts/readiness are per launch while
    # the NVLink barrier phase state intentionally persists.
    products()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        products.expert_pool_offsets.expert_counts.cpu(),
        torch.tensor([2, 2, 3, 0], dtype=torch.int32),
    )

    # The packed completion word counts every CTA, including empty ones.
    torch.testing.assert_close(
        (products.expert_recv_count_sum >> 32).cpu(),
        torch.full((4,), num_sms, dtype=torch.int64),
    )
    saved_routes = route_ids.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        products()
    for _ in range(3):
        route_ids.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        assert not products.expert_pool_offsets.expert_counts.any().item()
        torch.testing.assert_close(
            (products.expert_recv_count_sum >> 32).cpu(),
            torch.full((4,), num_sms, dtype=torch.int64),
        )
        route_ids.copy_(saved_routes)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            products.expert_pool_offsets.expert_counts.cpu(),
            torch.tensor([2, 2, 3, 0], dtype=torch.int32),
        )


def test_dispatch_rejects_distributed_configuration() -> None:
    config = _config()
    distributed = WorkspaceConfig(**{**config.__dict__, "world_size": 2})
    with pytest.raises(NotImplementedError, match="world_size=1"):
        SingleRankMxfp8Dispatch(distributed)
