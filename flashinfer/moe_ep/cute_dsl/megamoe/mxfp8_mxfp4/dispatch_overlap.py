# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-buffer pipelined pull for MXFP8 x MXFP4 only.

Preserves the vendored round-robin slot mapping and flag batching. The next
route is selected during the current load; store read completion frees the
buffer and full completion precedes deferred arrival publication.
"""

import os
from typing import Any


def overlapping_token_comm_class(base: Any) -> type:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cutlass_dsl import Float32, Int32, Int64, Uint32
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.src.src.token_comm import (
        GpuReleaseFlagBatchTracker,
        TokenSrcMetadata,
        _iket,
        fns_b32,
        ldg_b32_raw,
        ldg_f32_raw,
        sf_atom_int32_offset,
        tma_load_1d_raw,
        tma_store_1d,
    )

    fine_ctas = max(
        1, int(os.environ.get("FLASHINFER_MXFP8_MXFP4_IKET_FINE_CTAS", "1"))
    )

    class OverlappingTokenComm(base):
        @cute.jit
        def _choose_pull_token(
            self,
            state,
            token_idx,
            stored_num_tokens_per_expert,
            expert_recv_count,
            src_token_topk_idx,
            lane_idx,
        ):
            NUM_EXPERTS_PER_LANE: cutlass.Constexpr = (
                self.num_experts_per_rank + 31
            ) // 32
            current_expert_idx = state[0]
            expert_start_idx = state[1]
            expert_end_idx = state[2]
            expert_pool_block_offset = state[3]
            expert_task_tile_offset = state[4]
            expert_sf_pool_block_offset = state[5]
            stored_rank_count_lane = state[6]
            current_rank_in_expert_idx = Int32(0)
            src_token = Int32(0)
            src_topk = Int32(0)
            old_expert_idx = current_expert_idx
            while (token_idx >= expert_end_idx) and (
                current_expert_idx < Int32(self.num_experts_per_rank)
            ):
                prev_valid_count = expert_end_idx - expert_start_idx
                prev_block_count = (
                    prev_valid_count + Int32(self.token_padding_block) - Int32(1)
                ) // Int32(self.token_padding_block)
                expert_pool_block_offset = expert_pool_block_offset + prev_block_count
                # Mirror cumul for the release-counter granularity (self.cluster_tile_tokens).
                prev_task_tile_count = (
                    prev_valid_count + Int32(self.cluster_tile_tokens) - Int32(1)
                ) // Int32(self.cluster_tile_tokens)
                expert_task_tile_offset = expert_task_tile_offset + prev_task_tile_count
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

                src_token_topk = Uint32(
                    src_token_topk_idx[
                        current_expert_idx,
                        current_rank_in_expert_idx,
                        token_idx_in_rank,
                    ]
                )
                src_token = Int32(src_token_topk // Uint32(self.num_topk))
                src_topk = Int32(src_token_topk % Uint32(self.num_topk))

            return (
                current_expert_idx,
                expert_start_idx,
                expert_end_idx,
                expert_pool_block_offset,
                expert_task_tile_offset,
                expert_sf_pool_block_offset,
                stored_rank_count_lane,
                current_rank_in_expert_idx,
                src_token,
                src_topk,
            )

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

            current_expert_idx = Int32(-1)
            expert_start_idx = Int32(0)
            expert_end_idx = Int32(0)
            expert_pool_block_offset = Int32(0)
            expert_task_tile_offset = Int32(0)
            # SF rows use their own padding; token and SF pool offsets can diverge.
            expert_sf_pool_block_offset = Int32(0)

            # One deferred store per warp; the tracker only sees completed writes.
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
                (sm_idx < Int32(fine_ctas))
                and (warp_idx == Int32(0))
                and (lane_idx == Int32(0))
            )

            state = (
                current_expert_idx,
                expert_start_idx,
                expert_end_idx,
                expert_pool_block_offset,
                expert_task_tile_offset,
                expert_sf_pool_block_offset,
                stored_rank_count_lane,
            )
            if _iket_pull_emit:
                _iket.range_push("Pull.ChooseToken")
            state = self._choose_pull_token(
                state,
                token_idx,
                stored_num_tokens_per_expert,
                expert_recv_count,
                src_token_topk_idx,
                lane_idx,
            )
            if _iket_pull_emit:
                _iket.range_pop()
            pending = Int32(0)
            pending_addr = Int64(0)
            while state[0] < Int32(self.num_experts_per_rank):
                current_expert_idx = state[0]
                expert_start_idx = state[1]
                expert_end_idx = state[2]
                expert_pool_block_offset = state[3]
                expert_task_tile_offset = state[4]
                expert_sf_pool_block_offset = state[5]
                stored_rank_count_lane = state[6]
                current_rank_in_expert_idx = state[7]
                src_token = state[8]
                src_topk = state[9]
                token_idx_in_expert = token_idx - expert_start_idx
                if _iket_pull_emit:
                    _iket.range_push("Pull.TMA_NVLink_Roundtrip")
                cur_peer_offset = peer_rank_ptr_mapper.map(
                    Int64(0), current_rank_in_expert_idx, Int64(0)
                )
                inp_tok_local_base = input_token_buffer.iterator.toint()
                # SF-free callers pass input_sf_buffer=None (their SF loops are
                # compiled out by sf_uint32_per_token=0, so this base address
                # is never consumed).
                if cutlass.const_expr(input_sf_buffer is not None):
                    inp_sf_local_base = input_sf_buffer.iterator.toint()
                inp_w_local_base = input_topk_weights_buffer.iterator.toint()

                with cute.arch.elect_one():
                    pull_buffer_warp_ptr = pull_buffer_ptr + (
                        warp_idx * Int32(self.hidden_bytes)
                    )
                    tma_src_addr = (
                        inp_tok_local_base
                        + cur_peer_offset
                        + Int64(src_token * Int32(self.hidden_bytes))
                    )
                    tma_load_1d_raw(
                        pull_buffer_warp_ptr,
                        tma_src_addr,
                        pull_mbar_ptr + warp_idx,
                        Int32(self.hidden_bytes),
                    )
                cute.arch.sync_warp()

                # The preceding store has released this warp's shared buffer,
                # but its global write may still be in flight. Overlap completion
                # with the row load just issued. Only full completion permits the
                # release flag: FC1 reads the pool through the async proxy.
                if pending != Int32(0):
                    if _iket_pull_emit:
                        _iket.range_push("Pull.StoreCompletion")
                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_wait_group(0, read=False)
                    cute.arch.sync_warp()
                    if _iket_pull_emit:
                        _iket.range_pop()
                        _iket.range_push("Pull.Arrival_Atomic")
                    flag_tracker = flag_tracker.accumulate(
                        Int32(0), self._flag_batch, pending_addr
                    )
                    if _iket_pull_emit:
                        _iket.range_pop()
                # Select the next route while this row's asynchronous load is in
                # flight. Keep current offsets/source metadata until its store.
                if _iket_pull_emit:
                    _iket.range_push("Pull.ChooseToken")
                next_state = self._choose_pull_token(
                    state,
                    token_idx + Int32(num_global_warps),
                    stored_num_tokens_per_expert,
                    expert_recv_count,
                    src_token_topk_idx,
                    lane_idx,
                )
                if _iket_pull_emit:
                    _iket.range_pop()

                if _iket_pull_emit:
                    _iket.range_push("Pull.SF_LDG_STG")

                sf_token_in_pool_axis = (
                    expert_sf_pool_block_offset * Int32(self.sf_padding_block)
                    + token_idx_in_expert
                )
                pool_token_idx = (
                    expert_pool_block_offset * Int32(self.token_padding_block)
                    + token_idx_in_expert
                )
                sf_passes: cutlass.Constexpr[int] = (
                    self.sf_uint32_per_token + 31
                ) // 32

                sf_vals = []
                for _ in cutlass.range_constexpr(0, sf_passes, 1):
                    sf_vals.append(Int32(0))

                for i in cutlass.range_constexpr(0, sf_passes, 1):
                    j = Int32(i * self.warp_threads) + lane_idx
                    if j < Int32(self.sf_uint32_per_token):
                        sf_addr = (
                            inp_sf_local_base
                            + cur_peer_offset
                            + Int64(
                                (src_token * Int32(self.sf_uint32_per_token) + j)
                                * Int32(4)
                            )
                        )
                        sf_vals[i] = ldg_b32_raw(sf_addr)

                weight = Float32(0.0)
                if lane_idx == Int32(0):
                    weight_addr = (
                        inp_w_local_base
                        + cur_peer_offset
                        + Int64(
                            (src_token * Int32(self.num_topk) + src_topk) * Int32(4)
                        )
                    )
                    weight = ldg_f32_raw(weight_addr)

                if _iket_pull_emit:
                    _iket.range_pop()  # Pull.SF_LDG_STG  (= LD phase)
                    _iket.range_push("Pull.Weight_LDG")  # (= ST phase)

                for i in cutlass.range_constexpr(0, sf_passes, 1):
                    j = Int32(i * self.warp_threads) + lane_idx
                    if j < Int32(self.sf_uint32_per_token):
                        if cutlass.const_expr(self.sf_atom_swizzled):
                            sf_int32_pos = sf_atom_int32_offset(
                                sf_token_in_pool_axis,
                                j,
                                num_k_atoms=self.sf_uint32_per_token,
                            )
                        else:
                            sf_int32_pos = (
                                sf_token_in_pool_axis * Int32(self.sf_uint32_per_token)
                                + j
                            )
                        fc1_input_sf_buffer[sf_int32_pos] = sf_vals[i]
                cute.arch.sync_warp()

                if lane_idx == Int32(0):
                    fc1_input_topk_weights_buffer[pool_token_idx] = weight

                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        pull_mbar_ptr + warp_idx, Int32(self.hidden_bytes)
                    )
                    cute.arch.mbarrier_wait(
                        pull_mbar_ptr + warp_idx,
                        phase_bit,
                    )

                if _iket_pull_emit:
                    _iket.range_pop()  # Pull.Weight_LDG (ST phase)
                    _iket.range_pop()  # Pull.TMA_NVLink_Roundtrip (outer)
                    _iket.range_push("Pull.TMA_Store")

                with cute.arch.elect_one():
                    pull_buffer_warp_ptr = pull_buffer_ptr + (
                        warp_idx * Int32(self.hidden_bytes)
                    )
                    tma_store_1d(
                        fc1_input_token_buffer.iterator
                        # T=128k) × self.hidden_bytes overflows int32 (max 2.1 G).
                        # 64-bit address math is required for large token pools.
                        + (Int64(pool_token_idx) * Int64(self.hidden_bytes)),
                        pull_buffer_warp_ptr,
                        Int32(self.hidden_bytes),
                    )

                with cute.arch.elect_one():
                    TokenSrcMetadata(
                        src_rank=current_rank_in_expert_idx,
                        src_token=src_token,
                        src_topk=src_topk,
                    ).store(
                        token_src_metadata.iterator
                        + Int64(pool_token_idx) * Int64(TokenSrcMetadata.nbytes)
                    )

                with cute.arch.elect_one():
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)

                if _iket_pull_emit:
                    _iket.range_pop()  # Pull.TMA_Store

                # Read completion permits reusing the one shared buffer. The
                # next iteration drains the global write before accumulating this
                # tile's arrival; the final token is drained explicitly below.
                task_tile_idx = expert_task_tile_offset + (
                    token_idx_in_expert // Int32(self.cluster_tile_tokens)
                )
                pending_addr = (fc1_ready_counter.iterator + task_tile_idx).toint()
                pending = Int32(1)
                cute.arch.sync_warp()
                state = next_state

                phase_bit = phase_bit ^ Int32(1)

                token_idx = token_idx + Int32(num_global_warps)

            # No next row exists to trigger deferred completion for the last token.
            if pending != Int32(0):
                if _iket_pull_emit:
                    _iket.range_push("Pull.StoreCompletion")
                with cute.arch.elect_one():
                    cute.arch.cp_async_bulk_wait_group(0, read=False)
                cute.arch.sync_warp()
                if _iket_pull_emit:
                    _iket.range_pop()
                    _iket.range_push("Pull.Arrival_Atomic")
                flag_tracker = flag_tracker.accumulate(
                    Int32(0), self._flag_batch, pending_addr
                )
                if _iket_pull_emit:
                    _iket.range_pop()
            flag_tracker.fire()
            cute.arch.sync_warp()
            return phase_bit, stored_num_tokens_per_expert

    return OverlappingTokenComm
