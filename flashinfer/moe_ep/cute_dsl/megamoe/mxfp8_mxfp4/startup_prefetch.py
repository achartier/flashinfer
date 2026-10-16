# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host policy for the speculative FC1 weight prefix (no CUDA imports)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StartupPrefetchPlan:
    waves: int
    bytes_per_wave: int
    budget_bytes: int
    l2_bytes: int


def plan_startup_prefetch(
    requested_waves: int,
    *,
    hidden: int,
    intermediate: int,
    experts: int,
    mma_m: int,
    clusters: int,
    l2_bytes: int,
) -> StartupPrefetchPlan:
    """Cap whole waves to 80% of L2 and to one pass over unique FC1 weights.

    Each feature row contains H/2 packed E2M1 bytes and H/32 E8M0 bytes.
    Reserve 20% of L2 for dispatch and activations. This is a capacity bound,
    not a residency guarantee; actual routing is unknown at this point.
    """
    if type(requested_waves) is not int or not 0 <= requested_waves <= 3:
        raise ValueError("requested_waves must be an integer in [0, 3]")
    if min(hidden, intermediate, experts, clusters) <= 0 or l2_bytes < 0:
        raise ValueError("geometry must be positive and L2 size nonnegative")
    if mma_m not in (128, 256) or hidden % 128 or 2 * intermediate % mma_m:
        raise ValueError("prefetch geometry must contain whole weight tiles")
    tiles = experts * (2 * intermediate // mma_m)
    bytes_per_tile = mma_m * (hidden // 2 + hidden // 32)
    bytes_per_wave = min(tiles, clusters) * bytes_per_tile
    budget = l2_bytes * 4 // 5
    waves = min(
        requested_waves, budget // bytes_per_wave, (tiles + clusters - 1) // clusters
    )
    return StartupPrefetchPlan(waves, bytes_per_wave, budget, l2_bytes)
