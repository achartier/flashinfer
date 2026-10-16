# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warp-owned, double-buffered BF16 combine in dead operand storage."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Float32, Int32, dsl_user_op

from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.src.src.ptx_helpers import (
    fns_b32,
    tma_load_1d,
)

from .fc2_epilogue import (
    DeterministicTopKReducer,
    _add_rn_f32,
    _mul_rn_f32,
)


@dsl_user_op
def _invalidate_barrier(ptr, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [ptr.toint(loc=loc, ip=ip).ir_value()],
        "mbarrier.inval.shared::cta.b64 [$0];",
        "r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


def combine_chunk_elements(hidden: int, warps: int, scratch_bytes: int) -> int:
    """Fit two stages and two barriers per warp without changing stage depth.

    Zero requests the scalar fallback for unusually small operand rings.
    A 256-element minimum preserves 16-byte vector accesses per lane.
    """
    chunk = 256
    while chunk < min(hidden, 1024):
        chunk *= 2
    while chunk >= 256:
        if warps * (4 * chunk + 16) <= scratch_bytes:
            return chunk
        chunk //= 2
    return 0


class PipelinedTopKReducer:
    """One warp per token chunk, two stages, ascending live-slot order.

    Each lane owns interleaved vectors of eight BF16 elements: vector j
    starts at j * 256 + lane * 8. This keeps shared/global vector accesses
    contiguous across the warp without the bank conflicts of 64-byte lane
    strides. Only the final conversion rounds the FP32 accumulator to BF16.
    """

    def __init__(self, *, hidden: int, num_topk: int, chunk_elements: int = 1024):
        self.reference = DeterministicTopKReducer(hidden=hidden, num_topk=num_topk)
        if not 1 <= num_topk <= 32:
            raise ValueError("pipelined combine requires top-k in [1, 32]")
        if chunk_elements < 256 or chunk_elements % 256:
            raise ValueError("combine chunk must be a positive multiple of 256")
        self.hidden = hidden
        self.num_topk = num_topk
        self.chunk = chunk_elements
        self.chunks = (hidden + self.chunk - 1) // self.chunk
        self.vectors = self.chunk // 256
        self.warp_bytes = 4 * self.chunk + 16

    @cute.jit
    def reduce(
        self,
        terms,
        scores,
        ids,
        output,
        scratch,
        cta,
        tidx,
        sm_count: cutlass.Constexpr,
        threads: cutlass.Constexpr,
    ):
        warp = tidx // Int32(32)
        lane = tidx % Int32(32)
        # Stripe independent token chunks across CTAs before using another
        # warp in a CTA. Small batches then use many SMs, and wide rows do not
        # serialize every chunk on the same warp. Each element still visits
        # live slots in exactly the same ascending order.
        work = cta + Int32(sm_count) * warp
        # Idle warps need no asynchronous-copy state or proxy fences.
        if work < output.shape[0] * self.chunks:
            # The caller has drained all operand users and acquired peer stores.
            # Strip the operand pointer's swizzle: these are plain linear stages.
            base = cute.make_ptr(
                cutlass.BFloat16,
                scratch.toint() + warp * Int32(self.warp_bytes),
                cute.AddressSpace.smem,
                assumed_align=16,
            )
            barriers = cute.recast_ptr(base + 2 * self.chunk, dtype=cutlass.Int64)
            if lane == 0:
                cute.arch.mbarrier_init(barriers, 1)
                cute.arch.mbarrier_init(barriers + 1, 1)
                cute.arch.fence_proxy("async.shared", space="cta")
                # The drain acquired FC2's generic peer stores; publish that
                # ordering to the proxy which performs the route-term bulk loads.
                cute.arch.fence_proxy("async.global")
            cute.arch.sync_warp()
            phases = Int32(0)
            while work < output.shape[0] * self.chunks:
                token = work // Int32(self.chunks)
                chunk_idx = work % Int32(self.chunks)
                live = False
                score_lane = Float32(0.0)
                if lane < self.num_topk:
                    live = ids[token, lane] != -1
                    if live:
                        score_lane = Float32(scores[token, lane])
                live_mask = Int32(cute.arch.vote_ballot_sync(live))
                start = chunk_idx * Int32(self.chunk)
                valid = cutlass.min(Int32(self.chunk), Int32(self.hidden) - start)
                acc = cute.make_rmem_tensor((8, self.vectors), cutlass.Float32)
                acc.fill(Float32(0.0))
                mask = live_mask
                stage = Int32(0)
                if mask != 0:
                    slot = fns_b32(mask, Int32(0), Int32(1))
                    if lane == 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(barriers, valid * 2)
                        tma_load_1d(
                            base,
                            terms[token, slot, None].iterator + start,
                            barriers,
                            valid * 2,
                        )
                while mask != 0:
                    slot = fns_b32(mask, Int32(0), Int32(1))
                    # Clearing the lowest set bit preserves exact slot order,
                    # including slot 31 (the signed mask's high bit).
                    next_mask = mask & (mask - Int32(1))
                    other = stage ^ Int32(1)
                    if next_mask != 0:
                        next_slot = fns_b32(next_mask, Int32(0), Int32(1))
                        if lane == 0:
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                barriers + other, valid * 2
                            )
                            tma_load_1d(
                                base + other * self.chunk,
                                terms[token, next_slot, None].iterator + start,
                                barriers + other,
                                valid * 2,
                            )
                    cute.arch.mbarrier_wait(barriers + stage, (phases >> stage) & 1)
                    phases = phases ^ (Int32(1) << stage)
                    score = cute.arch.shuffle_sync(score_lane, slot)
                    for vector in cutlass.range_constexpr(self.vectors):
                        offset = lane * Int32(8) + Int32(vector * 256)
                        if offset < valid:
                            values = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                            src = cute.make_tensor(
                                cute.make_ptr(
                                    cutlass.BFloat16,
                                    (base + stage * self.chunk + offset).toint(),
                                    cute.AddressSpace.smem,
                                    assumed_align=16,
                                ),
                                cute.make_layout(8),
                            )
                            cute.copy(
                                cute.make_copy_atom(
                                    cute.nvgpu.CopyUniversalOp(),
                                    cutlass.BFloat16,
                                    num_bits_per_copy=128,
                                ),
                                src,
                                values,
                            )
                            for i in cutlass.range_constexpr(8):
                                acc[i, vector] = _add_rn_f32(
                                    acc[i, vector],
                                    _mul_rn_f32(Float32(values[i]), score),
                                )
                    # No lane may refill this stage until every lane has read it.
                    cute.arch.sync_warp()
                    mask = next_mask
                    stage = other
                for vector in cutlass.range_constexpr(self.vectors):
                    offset = lane * Int32(8) + Int32(vector * 256)
                    if offset < valid:
                        result = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                        result.store(acc[None, vector].load().to(cutlass.BFloat16))
                        dst = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.BFloat16,
                                (output[token, None].iterator + start + offset).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(8),
                        )
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                cutlass.BFloat16,
                                num_bits_per_copy=128,
                            ),
                            result,
                            dst,
                        )
                work += Int32(sm_count * (threads // 32))
            # Every issued copy has been waited on, including the last live slot.
            cute.arch.sync_warp()
            if lane == 0:
                _invalidate_barrier(barriers)
                _invalidate_barrier(barriers + 1)


class PipelinedInKernelTopKReduction:
    """Tail callback; scratch aliases the now-dead mainloop operand ring."""

    def __init__(self, *, hidden, num_topk, route_terms, output, scratch, chunk):
        self.reducer = PipelinedTopKReducer(
            hidden=hidden, num_topk=num_topk, chunk_elements=chunk
        )
        self.route_terms = route_terms
        self.output = output
        self.scratch = scratch

    @cute.jit
    def __call__(self, args, cta_linear_id, tidx, threads_per_cta: cutlass.Constexpr):
        terms, scores, ids, output = self.reducer.reference.views(
            self.route_terms, args.input_topk_weights_buffer, args.topk_idx, self.output
        )
        self.reducer.reduce(
            terms,
            scores,
            ids,
            output,
            self.scratch,
            cta_linear_id,
            tidx,
            args.sm_count,
            threads_per_cta,
        )
