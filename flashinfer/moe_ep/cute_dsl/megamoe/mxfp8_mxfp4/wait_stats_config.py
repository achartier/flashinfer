# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side layout of the round-4 wait accounting (no CuTe DSL import).

See :mod:`.wait_stats` for the device helpers.  The workspace planner uses
this module to size the mega ``wait_stats`` region, so it must stay free of
CUTLASS imports.
"""

from __future__ import annotations

import os
from enum import IntEnum
from typing import Optional

WAIT_STATS_ENV = "FLASHINFER_MXFP8_MXFP4_WAIT_STATS"
# Upper bound on persistent CTAs covered by the stats area (GB200 has 152).
MAX_CTAS = 256


class Slot(IntEnum):
    """Per-CTA int32 accumulators, in nanoseconds unless named ``*_COUNT``."""

    TMA_A_EMPTY_NS = 0
    TMA_B_EMPTY_NS = 1
    TMA_B_FC2_READY_NS = 2
    TMA_B_FC2_READY_COUNT = 3
    MMA_OPERAND_FULL_NS = 4
    MMA_ACC_EMPTY_NS = 5
    EPI_ACC_FULL_NS = 6
    EPI_FC1_TILE_NS = 7
    EPI_FC1_TILE_COUNT = 8
    EPI_FC2_TILE_NS = 9
    EPI_FC2_TILE_COUNT = 10
    MMA_LOOP_NS = 11
    EPI_LOOP_NS = 12
    TMA_B_LOOP_NS = 13
    TMA_B_FIRST_FC2_READY_NS = 14
    TMA_B_FIRST_FC2_START_NS = 15
    # Mega only: FC1 waits on dispatched-token arrival, the scheduler's wait
    # for dispatch before scheduling, and the all-warp tail rendezvous.
    TMA_B_FC1_READY_NS = 16
    TMA_B_FC1_READY_COUNT = 17
    SCHED_DISPATCH_WAIT_NS = 18
    KERNEL_TAIL_NS = 19
    # Scheduler hand-off and clock: time consumers block in consume_work(),
    # the scheduler warp's publish (blocked on consumers) vs generation time,
    # and SM clock cycles over the MMA loop (cycles / MMA_LOOP_NS = GHz).
    MMA_CONSUME_NS = 20
    TMA_A_CONSUME_NS = 21
    TMA_B_CONSUME_NS = 22
    SCHED_PUBLISH_NS = 23
    SCHED_GENERATE_NS = 24
    MMA_LOOP_CYCLES = 25
    # Start-up: the scheduler's first work generation (mega: waits for the
    # dispatched expert counts) and the MMA warp's first hand-off.
    SCHED_FIRST_GENERATE_NS = 26
    MMA_FIRST_CONSUME_NS = 27
    # Start-up timeline, each relative to that warp's own read right after
    # the kernel's pipeline-init barrier (all warps pass it together).
    SCHED_INIT_DONE_AT_NS = 28
    SCHED_FIRST_PUBLISH_AT_NS = 29
    MMA_FIRST_WORK_AT_NS = 30
    SCHED_WAIT_DONE_AT_NS = 31
    # internal_init split: a timed pass loading every local expert count,
    # then the first claim's decode (group construction and readiness peek).
    SCHED_INIT_COUNT_LOADS_NS = 32
    SCHED_INIT_DECODE_NS = 33
    # Raw %globaltimer_lo values (host takes the difference of per-CTA means):
    # just after the dispatch barrier, and at the start of the count loads.
    SCHED_WAIT_DONE_RAW = 34
    SCHED_LOADS_START_RAW = 35
    # Round-5 start-up split on the dispatch warps (mega only, recorded by
    # local dispatch warp 0 of each CTA): count/slot prep, the first grid
    # sync plus CTA 0's per-expert count sends to the owner ranks, and the
    # cross-rank NVLink barrier including the closing grid sync.
    DISPATCH_PREP_NS = 36
    DISPATCH_SYNC_SEND_NS = 37
    DISPATCH_NVLINK_BARRIER_NS = 38


SLOTS = 40
STATS_WORDS = MAX_CTAS * SLOTS


def wait_stats_enabled() -> bool:
    return os.environ.get(WAIT_STATS_ENV, "0").strip() == "1"


def mega_stats_word_offset(plan) -> Optional[int]:
    """Word offset of the ``wait_stats`` region from ``fc1_done_counter``."""

    try:
        stats = plan.local.region("wait_stats")
    except KeyError:
        return None
    done = plan.local.region("fc1_done_counter")
    return (stats.offset - done.offset) // 4


def read_mega_wait_stats(workspace):
    """The mega ``wait_stats`` region of a workspace as an int32 tensor view."""

    region = workspace.plan.local.region("wait_stats")
    import torch

    return workspace.workspace_bytes[region.offset : region.end].view(torch.int32)


def summarize(stats, num_ctas: int) -> dict:
    """Host-side mean/max per slot (µs for times) over the first CTAs."""

    rows = stats.reshape(MAX_CTAS, SLOTS)[:num_ctas].double()
    summary = {}
    for slot in Slot:
        column = rows[:, slot.value]
        if slot.name.endswith(("_COUNT", "_CYCLES", "_RAW")):
            summary[slot.name.lower()] = {
                "mean": round(float(column.mean()), 2),
                "max": int(column.max()),
            }
        else:
            summary[slot.name.lower().removesuffix("_ns") + "_us"] = {
                "mean": round(float(column.mean()) / 1000, 2),
                "max": round(float(column.max()) / 1000, 2),
            }
    return summary


__all__ = [
    "MAX_CTAS",
    "SLOTS",
    "STATS_WORDS",
    "Slot",
    "WAIT_STATS_ENV",
    "mega_stats_word_offset",
    "read_mega_wait_stats",
    "summarize",
    "wait_stats_enabled",
]
