# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Overwrite-only count exchange and a single drain barrier for MXFP8 MegaMoE.

Kept outside the vendored helper so other backends retain their reset protocol.
"""

import os
from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import Int32, Int64, dsl_user_op
from cutlass._mlir.dialects import llvm

from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.src.src.grid_sync import (
    software_grid_sync,
)
from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.src.src.ptx_helpers import (
    stg_b64_raw,
    red_add_release_sys_s32_raw,
)

from .wait_stats import Slot, SLOTS, globaltimer_lo


@dsl_user_op
def _startup_phase_assert(actual, expected, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [actual.ir_value(), expected.ir_value()],
        "{ .reg .pred p; setp.ne.s32 p, $0, $1; @p trap; }",
        "r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


def overwrite_count_token_comm_class(base: Any) -> type:
    class OverwriteCountTokenComm(base):  # type: ignore[misc, valid-type]
        overwrite_counts = True
        cta_local_counts = False

        @cute.jit
        def dispatch_barrier(
            self,
            expert_send_count,
            expert_recv_count,
            expert_recv_count_sum,
            nvlink_barrier_signal,
            grid_sync_counter,
            peer_rank_ptr_mapper,
            sm_idx,
            warp_idx,
            lane_idx,
            *,
            local_rank,
            num_sms,
            nvlink_barrier_counter,
        ):
            # software_grid_sync expects a dispatch-group-relative thread id.
            tid_in_group = warp_idx * Int32(self.warp_threads) + lane_idx

            # Snapshot before the grid rendezvous: CTA 0 cannot advance the
            # phase until every CTA has read it. Tail/NCU phase cycling is intact.
            if cutlass.const_expr(self.cta_local_counts):
                startup_status = cute.arch.load(
                    nvlink_barrier_counter.iterator, Int32, sem="relaxed", scope="gpu"
                ) & Int32(3)
                if cutlass.const_expr(hasattr(self, "_stamp")):
                    if warp_idx == Int32(0) and lane_idx == Int32(0):
                        phase_ptr = (
                            grid_sync_counter.iterator
                            + Int32(self.stats_word_offsets["grid_sync_counter"])
                            + sm_idx * Int32(SLOTS)
                            + Int32(Slot.STARTUP_PHASE_RAW)
                        )
                        cute.make_tensor(phase_ptr, cute.make_layout(1))[0] = (
                            startup_status
                        )

            if cutlass.const_expr(hasattr(self, "_stamp")):
                grid_start = globaltimer_lo()
            software_grid_sync(
                grid_sync_counter,
                sm_idx,
                num_sms,
                tid_in_group,
                num_threads=self.num_dispatch_threads,
            )

            if cutlass.const_expr(self.cta_local_counts and hasattr(self, "_stamp")):
                if warp_idx == Int32(0) and lane_idx == Int32(0):
                    # The first grid sync orders CTA 0's debug snapshot. Unlike
                    # the live barrier counter, it cannot advance during acquire.
                    phase_ptr = (
                        grid_sync_counter.iterator
                        + Int32(self.stats_word_offsets["grid_sync_counter"])
                        + Int32(Slot.STARTUP_PHASE_RAW)
                    )
                    expected_phase = cute.arch.load(
                        phase_ptr, Int32, sem="relaxed", scope="gpu"
                    )
                    _startup_phase_assert(startup_status, expected_phase)

            if cutlass.const_expr(hasattr(self, "_stamp")):
                send_start = globaltimer_lo()
                self._stamp(
                    grid_sync_counter.iterator,
                    "grid_sync_counter",
                    sm_idx,
                    warp_idx,
                    lane_idx,
                    Slot.DISPATCH_GRID_SYNC_NS,
                    send_start - grid_start,
                )
            if sm_idx == 0:
                for offset in cutlass.range_constexpr(
                    0,
                    self.num_total_experts,
                    self.experts_per_dispatch_pass,
                ):
                    expert_id = Int32(offset + warp_idx * self.warp_threads + lane_idx)
                    if expert_id < Int32(self.num_total_experts):
                        dst_rank = expert_id // Int32(self.num_experts_per_rank)
                        dst_local_expert = expert_id % Int32(self.num_experts_per_rank)
                        status_u64 = cute.arch.load(
                            expert_send_count.iterator + expert_id,
                            Int64,
                            sem="relaxed",
                            scope="gpu",
                        )
                        token_count_u32 = Int32(status_u64 & Int64(0xFFFFFFFF))
                        erc_local_base = expert_recv_count.iterator.toint()
                        erc_elem_off = (
                            Int32(local_rank) * Int32(self.num_experts_per_rank)
                            + dst_local_expert
                        ) * Int32(8)
                        erc_peer_addr = peer_rank_ptr_mapper.map(
                            erc_local_base,
                            dst_rank,
                            Int64(erc_elem_off),
                        )
                        stg_b64_raw(erc_peer_addr, Int64(token_count_u32))
                cute.arch.fence_acq_rel_sys()
            cute.arch.barrier(
                barrier_id=self.dispatch_intra_cta_bar_id,
                number_of_threads=self.num_dispatch_threads,
            )

            if cutlass.const_expr(hasattr(self, "_stamp")):
                rendezvous_start = globaltimer_lo()
                self._stamp(
                    grid_sync_counter.iterator,
                    "grid_sync_counter",
                    sm_idx,
                    warp_idx,
                    lane_idx,
                    Slot.DISPATCH_COUNT_SEND_NS,
                    rendezvous_start - send_start,
                )
            if cutlass.const_expr(self.cta_local_counts):
                self.startup_acquire(
                    nvlink_barrier_signal,
                    nvlink_barrier_counter,
                    peer_rank_ptr_mapper,
                    sm_idx,
                    warp_idx,
                    lane_idx,
                    startup_status,
                    grid_sync_counter,
                )
            else:
                self.nvlink_barrier(
                    nvlink_barrier_signal,
                    nvlink_barrier_counter,
                    grid_sync_counter,
                    peer_rank_ptr_mapper,
                    sm_idx,
                    warp_idx,
                    lane_idx,
                    num_sms=num_sms,
                    prologue_grid_sync=False,
                    epilogue_grid_sync=False,
                )
            if cutlass.const_expr(hasattr(self, "_stamp")):
                sum_start = globaltimer_lo()
                self._stamp(
                    grid_sync_counter.iterator,
                    "grid_sync_counter",
                    sm_idx,
                    warp_idx,
                    lane_idx,
                    Slot.DISPATCH_RENDEZVOUS_NS,
                    sum_start - rendezvous_start,
                )

            # Each CTA acquired peer count stores through startup_acquire.
            # Private sums reuse prep's now-dead scratch. The dispatch barrier
            # orders shared stores before pull; the dispatch-to-scheduler named
            # barrier carries the same acquire and sums to the scheduler warp.
            # The global packed diagnostic copy has exactly one writer per word.
            if cutlass.const_expr(self.cta_local_counts):
                global_count_sum = self.global_count_sum
                for offset in cutlass.range_constexpr(
                    0, self.num_experts_per_rank, self.num_dispatch_threads
                ):
                    expert = Int32(offset) + tid_in_group
                    if expert < Int32(self.num_experts_per_rank):
                        total = Int64(0)
                        for rank in cutlass.range_constexpr(self.world_size):
                            total += expert_recv_count[rank, expert]
                        expert_recv_count_sum[expert] = Int32(total)
                        if sm_idx == Int32(0):
                            global_count_sum[expert] = (
                                Int64(self.world_size * num_sms) << Int64(32)
                            ) | total
                cute.arch.barrier(
                    barrier_id=self.dispatch_intra_cta_bar_id,
                    number_of_threads=self.num_dispatch_threads,
                )
            else:
                if sm_idx == Int32(0):
                    cute.arch.barrier(
                        barrier_id=self.dispatch_intra_cta_bar_id,
                        number_of_threads=self.num_dispatch_threads,
                    )
                    for offset in cutlass.range_constexpr(
                        0, self.num_experts_per_rank, self.num_dispatch_threads
                    ):
                        expert = Int32(offset) + tid_in_group
                        if expert < Int32(self.num_experts_per_rank):
                            total = Int64(0)
                            for rank in cutlass.range_constexpr(self.world_size):
                                total += expert_recv_count[rank, expert]
                            expert_recv_count_sum[expert] = (
                                Int64(self.world_size * num_sms) << Int64(32)
                            ) | total
            if cutlass.const_expr(hasattr(self, "_stamp")):
                grid_start = globaltimer_lo()
                self._stamp(
                    grid_sync_counter.iterator,
                    "grid_sync_counter",
                    sm_idx,
                    warp_idx,
                    lane_idx,
                    Slot.DISPATCH_COUNT_SUM_NS,
                    grid_start - sum_start,
                )
            if cutlass.const_expr(not self.cta_local_counts):
                software_grid_sync(
                    grid_sync_counter,
                    sm_idx,
                    num_sms,
                    tid_in_group,
                    num_threads=self.num_dispatch_threads,
                )

                if cutlass.const_expr(hasattr(self, "_stamp")):
                    self._stamp(
                        grid_sync_counter.iterator,
                        "grid_sync_counter",
                        sm_idx,
                        warp_idx,
                        lane_idx,
                        Slot.DISPATCH_GRID_SYNC_NS,
                        globaltimer_lo() - grid_start,
                    )

        @cute.jit
        def startup_acquire(
            self,
            signal,
            counter,
            mapper,
            sm_idx,
            warp_idx,
            lane_idx,
            status,
            grid_sync_counter,
        ):
            if cutlass.const_expr(hasattr(self, "_stamp")):
                self._nvlink_probed = True
                self._nvlink_start = globaltimer_lo()
                self._stamp(
                    grid_sync_counter.iterator,
                    "grid_sync_counter",
                    sm_idx,
                    warp_idx,
                    lane_idx,
                    Slot.DISPATCH_SYNC_SEND_NS,
                    self._nvlink_start - self._barrier_start,
                )
            phase = status & Int32(1)
            delta = Int32(1)
            target = Int32(self.world_size)
            if (status >> Int32(1)) != Int32(0):
                delta = Int32(-1)
                target = Int32(0)
            if cutlass.const_expr(hasattr(self, "_stamp")):
                send_at = globaltimer_lo()
            if warp_idx == Int32(0):
                if sm_idx == Int32(0):
                    if lane_idx < Int32(self.world_size):
                        addr = mapper.map(
                            signal.iterator.toint(), lane_idx, Int64(phase * Int32(4))
                        )
                        red_add_release_sys_s32_raw(addr, delta)
                    cute.arch.sync_warp()
                    if lane_idx == Int32(0):
                        cute.arch.atomic_add(
                            counter.iterator, Int32(1), sem="relaxed", scope="gpu"
                        )
                if lane_idx == Int32(0):
                    # Memory-model chain: peer count stores -> peer system fence
                    # -> peer release-add -> this CTA's system acquire -> CTA
                    # barrier -> count reads. Every CTA executes its own acquire;
                    # no CTA relies on another CTA's polling load for visibility.
                    while (
                        cute.arch.load(
                            signal.iterator + phase, Int32, sem="acquire", scope="sys"
                        )
                        != target
                    ):
                        pass
                    if cutlass.const_expr(hasattr(self, "_stamp")):
                        if sm_idx == Int32(0):
                            self._record_startup_signal(
                                grid_sync_counter, send_at, globaltimer_lo() - send_at
                            )
            cute.arch.barrier(
                barrier_id=self.dispatch_intra_cta_bar_id,
                number_of_threads=self.num_dispatch_threads,
            )

            if cutlass.const_expr(hasattr(self, "_stamp")):
                self._signal_done = globaltimer_lo()

        @cute.jit
        def kernel_tail(
            self,
            token_comm_args,
            *,
            warp_idx,
            lane_idx,
            tidx,
            tail_work: cutlass.Constexpr = None,
        ):
            nb = pipeline.NamedBarrier(
                barrier_id=self.kernel_tail_named_barrier_id,
                num_threads=self.kernel_tail_threads,
            )
            # Mxfp8DispatchPhase fences every producer's peer stores before this
            # rendezvous. All count/slot readers and local counter users are done.
            nb.arrive_and_wait()
            if (warp_idx >= self.dispatch_warp_start) and (
                warp_idx < self.dispatch_warp_start + self.num_dispatch_warps
            ):
                cta = self._cta_linear_id()
                warp = Int32(warp_idx) - Int32(self.dispatch_warp_start)
                # Normal launches have two NVLink barriers (startup and drain).
                # Keep the phase counter: the two signal slots self-cancel across
                # TWO launches, not one. NCU restores local counters but cannot
                # restore peer signal memory, so kernel replay still needs four
                # calls per launch. Add two profiling-only barriers in that case.
                for _ in cutlass.range_constexpr(
                    3 if os.environ.get("MEGA_USE_NCU", "0") == "1" else 1
                ):
                    self.nvlink_barrier(
                        token_comm_args.nvlink_barrier_signal,
                        token_comm_args.nvlink_barrier_counter,
                        token_comm_args.grid_sync_counter,
                        token_comm_args.peer_rank_ptr_mapper,
                        cta,
                        warp,
                        lane_idx,
                        num_sms=token_comm_args.sm_count,
                        prologue_grid_sync=True,
                        epilogue_grid_sync=True,
                    )

            if cutlass.const_expr(tail_work is not None):
                # Publish the drain acquire before any warp reads peer route terms.
                nb.arrive_and_wait()

            if (warp_idx >= self.dispatch_warp_start) and (
                warp_idx < self.dispatch_warp_start + self.num_dispatch_warps
            ):
                # Rank-local reset overlaps other warps' reduction. Stream order
                # protects these counters on relaunch; persistent phase counters
                # live outside this prefix and are never reset here.
                self.tail_reset_counters(
                    token_comm_args,
                    token_comm_args.local_zero_prefix,
                    cta_linear_id=self._cta_linear_id(),
                    local_warp_idx=Int32(warp_idx) - Int32(self.dispatch_warp_start),
                    lane_idx=lane_idx,
                )

            if cutlass.const_expr(tail_work is not None):
                tail_work(
                    token_comm_args,
                    self._cta_linear_id(),
                    tidx,
                    self.kernel_tail_threads,
                )
            # No final rendezvous is needed: a peer may overwrite counts and slot
            # metadata now (their readers finished before the drain), but cannot
            # overwrite route_terms until its next startup barrier, which waits
            # for this rank to finish reducing and enter its next launch.

    return OverwriteCountTokenComm
