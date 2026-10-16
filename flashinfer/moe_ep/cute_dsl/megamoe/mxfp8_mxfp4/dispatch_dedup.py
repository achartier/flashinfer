# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashInfer-owned deduplicating overrides of the vendored pull protocol.

Each source token has one primary per owner (lowest expert, then top-k slot).
The primary fetches its E4M3 row and E8M0 scales once and fills every linked
route. A negative source word marks a secondary; its low 31 bits retain the
original token/top-k index. Links encode ``expert * max_slot + slot``; -1 ends
the chain. Every live link is rewritten during prep on every invocation.

No additional shared storage is required. The symmetric sibling region is
addressed relative to src_token_topk_idx, using the host workspace layout,
so the vendored argument ABI and source remain unchanged.
"""

from typing import Any


def round_robin_slot(counts: list[int], rank: int, slot: int) -> int:
    """Host model of the inverse rank/slot mapping used by fan-out."""
    if not 0 <= rank < len(counts) or not 0 <= slot < counts[rank]:
        raise ValueError("rank/slot must identify a live route")
    return sum(min(count, slot) for count in counts) + sum(
        count > slot for count in counts[:rank]
    )


def sibling_chains(experts: list[int], experts_per_rank: int) -> list[list[int]]:
    """Host model: top-k positions grouped by owner in primary-first order."""
    owners: dict[int, list[int]] = {}
    for k, expert in enumerate(experts):
        if expert >= 0:
            owners.setdefault(expert // experts_per_rank, []).append(k)
    return [sorted(slots, key=lambda k: (experts[k], k)) for slots in owners.values()]


def deduplicating_token_comm_class(
    base: Any, sibling_word_offset: int, fetch_stats_offset: int | None = None
) -> type:
    """Specialize the two device overrides without importing DSL on the host."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cutlass_dsl import Int32, Int64
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.src.src.token_comm import (
        GpuReleaseFlagBatchTracker,
        TokenSrcMetadata,
        _iket,
        fns_b32,
        ldg_b32_raw,
        ldg_f32_raw,
        sf_atom_int32_offset,
        stg_b32_raw,
        tma_load_1d_raw,
        tma_store_1d,
    )

    from .wait_stats import Slot, stat_add
    from .wait_stats_config import SLOTS

    class DeduplicatingTokenComm(base):
        @cute.jit
        def dispatch_prep(
            self,
            token_comm_storage,
            topk_idx,
            expert_send_count,
            src_token_topk_idx,
            peer_rank_ptr_mapper,
            sm_idx,
            warp_idx,
            lane_idx,
            *,
            local_rank,
            num_tokens,
            num_sms,
        ):
            thread_idx_in_dispatch = Int32(warp_idx * self.warp_threads + lane_idx)
            smem_count_ptr = token_comm_storage.smem_expert_count.data_ptr()
            i = thread_idx_in_dispatch
            while i < Int32(self.num_total_experts):
                (smem_count_ptr + i).store(Int32(0))
                i = i + Int32(self.num_dispatch_threads)
            cute.arch.barrier(
                barrier_id=self.dispatch_intra_cta_bar_id,
                number_of_threads=self.num_dispatch_threads,
            )

            tokens_per_warp: cutlass.Constexpr[int] = 32 // self.num_topk
            active_lanes: cutlass.Constexpr[int] = tokens_per_warp * self.num_topk
            num_dispatch_warps_per_grid: cutlass.Constexpr[int] = (
                num_sms * self.num_dispatch_warps
            )

            base_token_for_warp = (
                sm_idx * self.num_dispatch_warps + warp_idx
            ) * tokens_per_warp
            grid_token_stride = num_dispatch_warps_per_grid * tokens_per_warp

            t = base_token_for_warp
            while t < num_tokens:
                token_offset_in_warp = lane_idx // self.num_topk
                token_global = t + token_offset_in_warp
                if lane_idx < active_lanes and token_global < num_tokens:
                    topk_slot = lane_idx % self.num_topk
                    expert_id = Int32(topk_idx[token_global, topk_slot])
                    if expert_id >= Int32(0):
                        cute.arch.atomic_add(
                            smem_count_ptr + expert_id,
                            Int32(1),
                            sem="relaxed",
                            scope="cta",
                        )
                cute.arch.sync_warp()
                t += grid_token_stride

            cute.arch.barrier(
                barrier_id=self.dispatch_intra_cta_bar_id,
                number_of_threads=self.num_dispatch_threads,
            )

            for offset in cutlass.range_constexpr(
                0,
                self.num_total_experts,
                self.experts_per_dispatch_pass,
            ):
                expert_id = Int32(offset + warp_idx * self.warp_threads + lane_idx)
                if expert_id < Int32(self.num_total_experts):
                    slot_ptr = smem_count_ptr + expert_id
                    local_count = (slot_ptr).load()
                    delta = (Int64(1) << Int64(32)) | (
                        Int64(local_count) & Int64(0xFFFFFFFF)
                    )
                    old_packed = cute.arch.atomic_add(
                        expert_send_count.iterator + expert_id,
                        delta,
                        sem="relaxed",
                        scope="gpu",
                    )
                    base_slot = Int32(old_packed & Int64(0xFFFFFFFF))
                    (slot_ptr).store(base_slot)
            cute.arch.barrier(
                barrier_id=self.dispatch_intra_cta_bar_id,
                number_of_threads=self.num_dispatch_threads,
            )

            # All lanes participate in sibling shuffles, including masked routes
            # and the unused tail lanes when top-k does not divide 32.
            sibling_ptr = src_token_topk_idx.iterator + sibling_word_offset
            max_slot: cutlass.Constexpr[int] = num_tokens * self.num_topk
            t = base_token_for_warp
            while t < num_tokens:
                token_global = t + lane_idx // self.num_topk
                topk_slot = lane_idx % self.num_topk
                expert_id = Int32(-1)
                dst_rank = Int32(-1)
                local_expert = Int32(-1)
                slot = Int32(-1)
                if lane_idx < active_lanes and token_global < num_tokens:
                    expert_id = Int32(topk_idx[token_global, topk_slot])
                    if expert_id >= Int32(0):
                        dst_rank = expert_id // Int32(self.num_experts_per_rank)
                        local_expert = expert_id % Int32(self.num_experts_per_rank)
                        slot = cute.arch.atomic_add(
                            smem_count_ptr + expert_id,
                            Int32(1),
                            sem="relaxed",
                            scope="cta",
                        )
                primary = Int32(1)
                next_key = Int32(0x7FFFFFFF)
                next_link = Int32(-1)
                own_key = local_expert * Int32(self.num_topk) + topk_slot
                token_lane = (lane_idx // self.num_topk) * self.num_topk
                for k in cutlass.range_constexpr(self.num_topk):
                    # Clamp only the unused lanes' shuffle source to a valid lane.
                    other_lane = min(token_lane + Int32(k), Int32(31))
                    other_rank = cute.arch.shuffle_sync(dst_rank, other_lane)
                    other_expert = cute.arch.shuffle_sync(local_expert, other_lane)
                    other_slot = cute.arch.shuffle_sync(slot, other_lane)
                    other_key = other_expert * Int32(self.num_topk) + Int32(k)
                    if other_rank == dst_rank and other_expert >= Int32(0):
                        if other_key < own_key:
                            primary = Int32(0)
                        if other_key > own_key and other_key < next_key:
                            next_key = other_key
                            next_link = other_expert * Int32(max_slot) + other_slot
                if expert_id >= Int32(0):
                    word = Int32(token_global * self.num_topk + topk_slot)
                    if primary == Int32(0):
                        word = word | Int32(-2147483648)
                    elem_off = Int64(
                        (local_expert * Int32(self.world_size) + Int32(local_rank))
                        * Int32(max_slot)
                        + slot
                    ) * Int64(4)
                    stg_b32_raw(
                        peer_rank_ptr_mapper.map(
                            src_token_topk_idx.iterator.toint(), dst_rank, elem_off
                        ),
                        word,
                    )
                    stg_b32_raw(
                        peer_rank_ptr_mapper.map(
                            sibling_ptr.toint(), dst_rank, elem_off
                        ),
                        next_link,
                    )
                cute.arch.sync_warp()
                t += grid_token_stride

        @cute.jit
        def dispatch_pull(
            self,
            token_comm_storage,
            input_token_buffer,
            input_sf_buffer,
            input_topk_weights_buffer,
            src_token_topk_idx,
            expert_recv_count,
            expert_recv_count_sum,
            fc1_input_token_buffer,
            fc1_input_sf_buffer,
            fc1_input_topk_weights_buffer,
            fc1_ready_counter,
            token_src_metadata,
            peer_rank_ptr_mapper,
            sm_idx,
            warp_idx,
            lane_idx,
            *,
            num_sms,
        ):
            # MemRange does not support dynamic indexing here; use raw pointers.
            pull_mbar_ptr = token_comm_storage.pull_mbar.data_ptr()
            pull_buffer_ptr = token_comm_storage.pull_buffer.data_ptr()
            if lane_idx == Int32(0):
                cute.arch.mbarrier_init(pull_mbar_ptr + warp_idx, 1)
            cute.arch.sync_warp()

            phase_bit = Int32(0)
            fetched_rows = Int32(0)

            current_expert_idx = Int32(-1)
            expert_start_idx = Int32(0)
            expert_end_idx = Int32(0)
            expert_pool_block_offset = Int32(0)
            expert_task_tile_offset = Int32(0)
            # SF rows use their own padding; token and SF pool offsets can diverge.
            expert_sf_pool_block_offset = Int32(0)

            # ── Release-flag batching ────────────────────────────────────────
            # Delay fc1-ready counter publication with the same rotating-lane
            # tracker used by the epilogue.  Each token's TMA store to the FC1 pool
            # is drained CTA-locally by ``cp_async_bulk_wait_group(0)`` before its
            # release target is accumulated; the eventual red.release.gpu add
            # publishes the corresponding pool data to GPU scope.
            flag_tracker = GpuReleaseFlagBatchTracker(
                flag_addr=Int64(0),
                cumulated_flags=Int32(0),
                phase=Int32(0),
                tid=lane_idx,
            )

            stored_rank_count_lane = Int32(0)

            NUM_EXPERTS_PER_LANE: cutlass.Constexpr[int] = (
                self.num_experts_per_rank + 31
            ) // 32
            stored_num_tokens_per_expert = []
            for _ in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
                stored_num_tokens_per_expert.append(Int32(0))
            for i in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
                e_idx_for_lane = Int32(i * self.warp_threads) + lane_idx
                if e_idx_for_lane < Int32(self.num_experts_per_rank):
                    sum_packed_init = expert_recv_count_sum[e_idx_for_lane]
                    stored_num_tokens_per_expert[i] = Int32(
                        Int64(sum_packed_init) & Int64(0xFFFFFFFF)
                    )
            cute.arch.sync_warp()

            num_global_warps: cutlass.Constexpr[int] = num_sms * self.num_dispatch_warps
            token_idx = sm_idx * Int32(self.num_dispatch_warps) + warp_idx

            _iket_pull_emit = (
                (sm_idx == Int32(0))
                and (warp_idx == Int32(0))
                and (lane_idx == Int32(0))
            )

            while current_expert_idx < Int32(self.num_experts_per_rank):
                if _iket_pull_emit:
                    _iket.range_push("Pull.ChooseToken")
                old_expert_idx = current_expert_idx
                while (token_idx >= expert_end_idx) and (
                    current_expert_idx < Int32(self.num_experts_per_rank)
                ):
                    prev_valid_count = expert_end_idx - expert_start_idx
                    prev_block_count = (
                        prev_valid_count + Int32(self.token_padding_block) - Int32(1)
                    ) // Int32(self.token_padding_block)
                    expert_pool_block_offset = (
                        expert_pool_block_offset + prev_block_count
                    )
                    # Mirror cumul for the release-counter granularity (self.cluster_tile_tokens).
                    prev_task_tile_count = (
                        prev_valid_count + Int32(self.cluster_tile_tokens) - Int32(1)
                    ) // Int32(self.cluster_tile_tokens)
                    expert_task_tile_offset = (
                        expert_task_tile_offset + prev_task_tile_count
                    )
                    # Mirror cumul for the SF axis granularity (self.sf_padding_block).
                    prev_sf_block_count = (
                        prev_valid_count + Int32(self.sf_padding_block) - Int32(1)
                    ) // Int32(self.sf_padding_block)
                    expert_sf_pool_block_offset = (
                        expert_sf_pool_block_offset + prev_sf_block_count
                    )
                    current_expert_idx = current_expert_idx + Int32(1)
                    if current_expert_idx < Int32(self.num_experts_per_rank):
                        expert_start_idx = expert_end_idx
                        valid_value = Int32(0)
                        for i in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
                            if (
                                current_expert_idx
                                == Int32(i * self.warp_threads) + lane_idx
                            ):
                                valid_value = stored_num_tokens_per_expert[i]
                        total_for_expert = cute.arch.shuffle_sync(
                            valid_value, current_expert_idx % Int32(self.warp_threads)
                        )
                        expert_end_idx = expert_end_idx + total_for_expert

                if current_expert_idx < Int32(self.num_experts_per_rank):
                    if old_expert_idx != current_expert_idx:
                        if lane_idx < Int32(self.world_size):
                            stored_rank_count_lane = Int32(
                                expert_recv_count[lane_idx, current_expert_idx]
                            )
                        else:
                            stored_rank_count_lane = Int32(0)

                    token_idx_in_expert = token_idx - expert_start_idx
                    slot_idx = token_idx_in_expert
                    offset = Int32(0)
                    remaining_lane = stored_rank_count_lane

                    current_rank_in_expert_idx = Int32(0)
                    token_idx_in_rank = Int32(0)

                    decided = Int32(0)
                    for _round in cutlass.range_constexpr(0, self.world_size + 1, 1):
                        if decided == Int32(0):
                            active = remaining_lane > Int32(0)
                            mask = cute.arch.vote_ballot_sync(active)
                            num_active_ranks = Int32(cute.arch.popc(Int32(mask)))
                            v_for_min = Int32(0x7FFFFFFF)
                            if active:
                                v_for_min = remaining_lane
                            length = Int32(cute.arch.warp_redux_sync(v_for_min, "min"))

                            if num_active_ranks > Int32(0):
                                num_round_tokens = length * num_active_ranks
                                if slot_idx < num_round_tokens:
                                    slot_idx_in_round = slot_idx % num_active_ranks
                                    current_rank_in_expert_idx = fns_b32(
                                        Int32(mask),
                                        Int32(0),
                                        slot_idx_in_round + Int32(1),
                                    )
                                    token_idx_in_rank = offset + (
                                        slot_idx // num_active_ranks
                                    )
                                    decided = Int32(1)
                                else:
                                    slot_idx = slot_idx - num_round_tokens
                                    offset = offset + length
                                    if remaining_lane > length:
                                        remaining_lane = remaining_lane - length
                                    else:
                                        remaining_lane = Int32(0)
                            else:
                                decided = Int32(1)

                    if _iket_pull_emit:
                        _iket.range_pop()

                    src_word = Int32(
                        src_token_topk_idx[
                            current_expert_idx,
                            current_rank_in_expert_idx,
                            token_idx_in_rank,
                        ]
                    )
                    # A primary fills every sibling, even if another warp visits
                    # that secondary first. Only the primary publishes its arrival.
                    if src_word >= Int32(0):
                        if cutlass.const_expr(fetch_stats_offset is not None):
                            fetched_rows += Int32(1)
                        src_token = src_word // Int32(self.num_topk)
                        cur_peer_offset = peer_rank_ptr_mapper.map(
                            Int64(0), current_rank_in_expert_idx, Int64(0)
                        )
                        pull_buffer_warp_ptr = pull_buffer_ptr + warp_idx * Int32(
                            self.hidden_bytes
                        )
                        if _iket_pull_emit:
                            _iket.range_push("Pull.TMA_NVLink_Roundtrip")
                        with cute.arch.elect_one():
                            tma_load_1d_raw(
                                pull_buffer_warp_ptr,
                                input_token_buffer.iterator.toint()
                                + cur_peer_offset
                                + Int64(src_token) * Int64(self.hidden_bytes),
                                pull_mbar_ptr + warp_idx,
                                Int32(self.hidden_bytes),
                            )
                        sf_passes: cutlass.Constexpr[int] = (
                            self.sf_uint32_per_token + 31
                        ) // 32
                        sf_vals = []
                        for i in cutlass.range_constexpr(sf_passes):
                            sf_vals.append(Int32(0))
                            j = Int32(i * 32) + lane_idx
                            if j < Int32(self.sf_uint32_per_token):
                                sf_vals[i] = ldg_b32_raw(
                                    input_sf_buffer.iterator.toint()
                                    + cur_peer_offset
                                    + (
                                        Int64(src_token)
                                        * Int64(self.sf_uint32_per_token)
                                        + Int64(j)
                                    )
                                    * Int64(4)
                                )
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                pull_mbar_ptr + warp_idx, Int32(self.hidden_bytes)
                            )
                            cute.arch.mbarrier_wait(pull_mbar_ptr + warp_idx, phase_bit)
                        cute.arch.sync_warp()
                        if _iket_pull_emit:
                            _iket.range_pop()
                            _iket.range_push("Pull.Fanout")

                        dest_expert = current_expert_idx
                        dest_slot = token_idx_in_rank
                        dest_token = token_idx_in_expert
                        pool_offset = expert_pool_block_offset
                        sf_offset = expert_sf_pool_block_offset
                        task_offset = expert_task_tile_offset
                        more = Int32(1)
                        # At most top-k <= 32 arrivals: one pending tile per
                        # lane avoids shared memory and per-route store waits.
                        pending_task = Int32(0)
                        fanout_count = Int32(0)
                        while more != Int32(0):
                            pool_row = (
                                pool_offset * Int32(self.token_padding_block)
                                + dest_token
                            )
                            sf_row = (
                                sf_offset * Int32(self.sf_padding_block) + dest_token
                            )
                            dest_word = Int32(
                                src_token_topk_idx[
                                    dest_expert, current_rank_in_expert_idx, dest_slot
                                ]
                            ) & Int32(0x7FFFFFFF)
                            dest_topk = dest_word % Int32(self.num_topk)
                            for i in cutlass.range_constexpr(sf_passes):
                                j = Int32(i * 32) + lane_idx
                                if j < Int32(self.sf_uint32_per_token):
                                    sf_pos = sf_atom_int32_offset(
                                        sf_row, j, num_k_atoms=self.sf_uint32_per_token
                                    )
                                    fc1_input_sf_buffer[sf_pos] = sf_vals[i]
                            if lane_idx == Int32(0):
                                fc1_input_topk_weights_buffer[pool_row] = ldg_f32_raw(
                                    input_topk_weights_buffer.iterator.toint()
                                    + cur_peer_offset
                                    + Int64(dest_word) * Int64(4)
                                )
                            with cute.arch.elect_one():
                                tma_store_1d(
                                    fc1_input_token_buffer.iterator
                                    + Int64(pool_row) * Int64(self.hidden_bytes),
                                    pull_buffer_warp_ptr,
                                    Int32(self.hidden_bytes),
                                )
                                TokenSrcMetadata(
                                    src_rank=current_rank_in_expert_idx,
                                    src_token=src_token,
                                    src_topk=dest_topk,
                                ).store(
                                    token_src_metadata.iterator
                                    + Int64(pool_row) * Int64(TokenSrcMetadata.nbytes)
                                )
                            cute.arch.sync_warp()
                            task_tile = task_offset + dest_token // Int32(
                                self.cluster_tile_tokens
                            )
                            if lane_idx == fanout_count:
                                pending_task = task_tile
                            fanout_count += Int32(1)
                            max_slot: cutlass.Constexpr[int] = cute.size(
                                src_token_topk_idx.shape[2]
                            )
                            link = (
                                src_token_topk_idx.iterator
                                + sibling_word_offset
                                + (
                                    (
                                        dest_expert * Int32(self.world_size)
                                        + current_rank_in_expert_idx
                                    )
                                    * Int32(max_slot)
                                    + dest_slot
                                )
                            ).load()
                            if link < Int32(0):
                                more = Int32(0)
                            else:
                                dest_expert = link // Int32(max_slot)
                                dest_slot = link % Int32(max_slot)
                                # Round-robin inverse: all earlier rounds, then
                                # the active lower ranks in this round. Zero ranks
                                # contribute nothing; no peeling loop is necessary.
                                count = Int32(0)
                                if lane_idx < Int32(self.world_size):
                                    count = Int32(
                                        expert_recv_count[lane_idx, dest_expert]
                                    )
                                before = min(count, dest_slot)
                                if (
                                    lane_idx < current_rank_in_expert_idx
                                    and count > dest_slot
                                ):
                                    before += Int32(1)
                                dest_token = Int32(
                                    cute.arch.warp_redux_sync(before, "add")
                                )
                                # Counts already live in registers, distributed by
                                # expert over the warp. Prefix sums need no shared
                                # memory and support more than 32 local experts.
                                pool_prefix = Int32(0)
                                sf_prefix = Int32(0)
                                task_prefix = Int32(0)
                                for i in cutlass.range_constexpr(NUM_EXPERTS_PER_LANE):
                                    if Int32(i * 32) + lane_idx < dest_expert:
                                        count = stored_num_tokens_per_expert[i]
                                        pool_prefix += (
                                            count + Int32(self.token_padding_block - 1)
                                        ) // Int32(self.token_padding_block)
                                        sf_prefix += (
                                            count + Int32(self.sf_padding_block - 1)
                                        ) // Int32(self.sf_padding_block)
                                        task_prefix += (
                                            count + Int32(self.cluster_tile_tokens - 1)
                                        ) // Int32(self.cluster_tile_tokens)
                                pool_offset = Int32(
                                    cute.arch.warp_redux_sync(pool_prefix, "add")
                                )
                                sf_offset = Int32(
                                    cute.arch.warp_redux_sync(sf_prefix, "add")
                                )
                                task_offset = Int32(
                                    cute.arch.warp_redux_sync(task_prefix, "add")
                                )
                        # Every store reads the same unchanged pull buffer.
                        # Drain the complete group before the tracker can fire
                        # (including flag_batch=1) or the next fetch reuses it.
                        with cute.arch.elect_one():
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(0)
                        cute.arch.sync_warp()
                        published = Int32(0)
                        while published < fanout_count:
                            task_tile = cute.arch.shuffle_sync(pending_task, published)
                            flag_tracker = flag_tracker.accumulate(
                                Int32(0),
                                self._flag_batch,
                                (fc1_ready_counter.iterator + task_tile).toint(),
                            )
                            cute.arch.sync_warp()
                            published += Int32(1)
                        if _iket_pull_emit:
                            _iket.range_pop()
                        phase_bit ^= Int32(1)
                    token_idx += Int32(num_global_warps)

            if cutlass.const_expr(fetch_stats_offset is not None):
                if lane_idx == Int32(0):
                    stat_add(
                        fc1_ready_counter.iterator
                        + Int32(fetch_stats_offset)
                        + sm_idx * Int32(SLOTS)
                        + Int32(Slot.DISPATCH_FETCH_COUNT),
                        fetched_rows,
                    )
            flag_tracker.fire()
            cute.arch.sync_warp()
            return phase_bit, stored_num_tokens_per_expert

    return DeduplicatingTokenComm
