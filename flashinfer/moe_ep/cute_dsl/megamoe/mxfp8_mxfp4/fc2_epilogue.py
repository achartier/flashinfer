# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FC2 route stores and deterministic combine for MXFP8 x MXFP4 MegaMoE.

The two operations in this module deliberately have separate interfaces:

* :class:`Fc2Bf16Epilogue` rounds each unweighted FP32 FC2 accumulator to
  BF16 exactly once and stores one row per ``(src_token_idx, src_topk_idx)``.
* :class:`DeterministicTopKReducer` reads those BF16 route terms, converts
  them back to FP32, applies the FP32 routing score, accumulates top-k slots in
  increasing slot order, and rounds the final output to BF16 exactly once.

In particular, the route-store path has no routing-score operand.  This keeps
routing after FC2 and prevents it from changing the FC1-to-FC2 MXFP8
quantization point.  The default implementation intentionally does not use an
in-kernel reduction (IKR): local and peer stores share the same route-slot
layout, followed by the deterministic external reducer.

The integrating kernel owns TMEM lifetime, accumulator-pipeline barriers,
source metadata, and inter-rank completion signalling.  This component only
owns the final TMEM/RMEM conversion, route destination resolution, and combine
arithmetic.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import Float32, Int32, T, dsl_user_op


FC2_EPILOGUE_TILE = 32
REDUCE_HIDDEN_PER_THREAD = 8
REDUCE_THREADS = 128


def validate_fc2_contract(*, hidden: int, num_topk: int) -> None:
    """Validate the static shape assumptions of the FC2 output path."""

    if hidden <= 0 or hidden % FC2_EPILOGUE_TILE:
        raise ValueError(
            f"hidden must be a positive multiple of {FC2_EPILOGUE_TILE}, got {hidden}"
        )
    if num_topk <= 0:
        raise ValueError(f"num_topk must be positive, got {num_topk}")


@dataclasses.dataclass(frozen=True)
class Fc2RouteStore:
    """Resolve a route slot in local or peer symmetric memory.

    ``tensor`` is the local-rank view of a BF16
    ``(max_tokens_per_rank, num_topk, hidden)`` route-term allocation.  With no
    mapper, ``resolve_route_row`` returns its local
    ``[src_token_idx, src_topk_idx, :]`` row.  With a mapper, the same local
    row pointer is rebased to ``src_rank``; consequently the data layout is
    identical for local and peer stores.

    The scheduler/token-communication component supplies the three source
    coordinates.  It must use this component only after route ids have been
    validated against the numerical contract.  Expert id ``-1`` never creates
    an FC2 task, so its route slot remains the zero initialized term consumed
    by the reducer.
    """

    tensor: cute.Tensor
    peer_rank_ptr_mapper: Any = None

    @property
    def is_peer_store(self) -> bool:
        """Whether source-rank pointer rebasing is enabled."""

        return self.peer_rank_ptr_mapper is not None

    @cute.jit
    def resolve_route_row(
        self,
        src_rank,
        src_token_idx,
        src_topk_idx,
    ) -> cute.Tensor:
        local_row = cute.slice_(
            self.tensor,
            (src_token_idx, src_topk_idx, None),
        )
        if cutlass.const_expr(self.peer_rank_ptr_mapper is None):
            return local_row
        peer_iter = self.peer_rank_ptr_mapper.ptr_map_to_rank(
            local_row.iterator,
            src_rank,
        )
        return cute.make_tensor(peer_iter, local_row.layout)


class Fc2Bf16Epilogue:
    """Store unweighted FP32 FC2 accumulator subtiles as BF16 route terms."""

    def __init__(self, *, hidden: int) -> None:
        validate_fc2_contract(hidden=hidden, num_topk=1)
        self.hidden = int(hidden)

    @cute.jit
    def store_rmem_subtile(
        self,
        r_acc: cute.Tensor,
        route_store: Fc2RouteStore,
        src_rank,
        src_token_idx,
        src_topk_idx,
        hidden_col_start,
    ) -> None:
        """Round 32 FP32 accumulator cells once and store one route subtile.

        ``hidden_col_start`` must be 32-aligned.  The static hidden contract
        makes every subtile complete, so no partial vector store or second
        conversion path exists.
        """

        # This is the sole route-term rounding point.  Keep it before any
        # destination handling and do not add a routing-score argument here.
        r_bf16 = cute.make_rmem_tensor((FC2_EPILOGUE_TILE,), cutlass.BFloat16)
        r_bf16.store(r_acc.load().to(cutlass.BFloat16))

        dest_row = route_store.resolve_route_row(
            src_rank,
            src_token_idx,
            src_topk_idx,
        )
        dest_ptr = cute.make_ptr(
            cutlass.BFloat16,
            dest_row.iterator.toint() + cutlass.Int64(hidden_col_start) * 2,
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        stg_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.BFloat16,
            num_bits_per_copy=256,
        )
        for half in cutlass.range_constexpr(FC2_EPILOGUE_TILE // 16):
            src = cute.make_tensor(
                r_bf16.iterator + half * 16,
                cute.make_layout(16),
            )
            dst = cute.make_tensor(
                dest_ptr + half * 16,
                cute.make_layout(16),
            )
            cute.copy(stg_atom, src, dst)

    @cute.jit
    def load_tmem_and_store_subtile(
        self,
        tmem_subtile: cute.Tensor,
        route_store: Fc2RouteStore,
        src_rank,
        src_token_idx,
        src_topk_idx,
        hidden_col_start,
    ) -> None:
        """Load one 32-cell FP32 TMEM subtile and emit its BF16 route term."""

        r_acc_layout = cute.make_layout(
            (((FC2_EPILOGUE_TILE,), 1),),
            stride=(((1,), 0),),
        )
        r_acc = cute.make_rmem_tensor(r_acc_layout.shape, cutlass.Float32)
        ldtm_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x32),
            cutlass.Float32,
        )
        cute.copy(ldtm_atom, tmem_subtile, r_acc)
        self.store_rmem_subtile(
            r_acc,
            route_store,
            src_rank,
            src_token_idx,
            src_topk_idx,
            hidden_col_start,
        )


@dsl_user_op
def _mul_rn_f32(
    lhs,
    rhs,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Float32:
    """Separate FP32 multiply used by the contract's multiply-then-add step."""

    return Float32(
        llvm.inline_asm(
            T.f32(),
            [lhs.ir_value(), rhs.ir_value()],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _add_rn_f32(
    lhs,
    rhs,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Float32:
    """Separate FP32 add; intentionally not contracted into an FMA."""

    return Float32(
        llvm.inline_asm(
            T.f32(),
            [lhs.ir_value(), rhs.ir_value()],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


class DeterministicTopKReducer:
    """External BF16-route-term reducer with a fixed ascending slot order.

    ``route_ids`` is required so the ``-1`` sentinel ignores its score even if
    the caller filled that score with a nonzero value.  All other ids are
    treated as active here: invalid ids must be rejected by frontend route
    validation and are never silently dropped by this kernel.
    """

    def __init__(self, *, hidden: int, num_topk: int) -> None:
        validate_fc2_contract(hidden=hidden, num_topk=num_topk)
        self.hidden = int(hidden)
        self.num_topk = int(num_topk)
        self.hidden_tiles = self.hidden // REDUCE_HIDDEN_PER_THREAD
        self.requires_token_guard = self.hidden_tiles % REDUCE_THREADS != 0

    @cute.jit
    def __call__(
        self,
        route_terms: cute.Tensor,
        route_scores: cute.Tensor,
        route_ids: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Launch reduction for local ``(token, topk, hidden)`` route terms."""

        route_terms, route_scores, route_ids, output = self.views(
            route_terms, route_scores, route_ids, output
        )
        total_workers = output.shape[0] * self.hidden_tiles
        self._reduce(route_terms, route_scores, route_ids, output).launch(
            grid=((total_workers + REDUCE_THREADS - 1) // REDUCE_THREADS, 1, 1),
            block=(REDUCE_THREADS, 1, 1),
            stream=stream,
        )

    @cute.jit
    def views(self, route_terms, route_scores, route_ids, output):
        """``(token, topk, hidden)`` / ``(token, topk)`` / ``(token, hidden)``
        views over the first ``output.shape[0]`` tokens."""

        token_count = output.shape[0]
        route_terms = cute.make_tensor(
            route_terms.iterator,
            cute.make_layout(
                (token_count, self.num_topk, self.hidden),
                stride=route_terms.stride,
            ),
        )
        route_scores = cute.make_tensor(
            route_scores.iterator,
            cute.make_layout(
                (token_count, self.num_topk),
                stride=route_scores.stride,
            ),
        )
        route_ids = cute.make_tensor(
            route_ids.iterator,
            cute.make_layout(
                (token_count, self.num_topk),
                stride=route_ids.stride,
            ),
        )
        output = cute.make_tensor(
            output.iterator,
            cute.make_layout(
                (token_count, self.hidden),
                stride=output.stride,
            ),
        )
        return route_terms, route_scores, route_ids, output

    @cute.kernel
    def _reduce(
        self,
        route_terms: cute.Tensor,
        route_scores: cute.Tensor,
        route_ids: cute.Tensor,
        output: cute.Tensor,
    ):
        worker_idx = (
            cute.arch.block_idx()[0] * Int32(REDUCE_THREADS) + cute.arch.thread_idx()[0]
        )
        if (not self.requires_token_guard) or worker_idx < (
            output.shape[0] * self.hidden_tiles
        ):
            self.reduce_worker(route_terms, route_scores, route_ids, output, worker_idx)

    @cute.jit
    def reduce_worker(self, route_terms, route_scores, route_ids, output, worker_idx):
        """Reduce eight hidden values of one token (views from :meth:`views`).

        Shared by the external kernel and the persistent kernel's in-kernel
        reduction, so both round identically.
        """

        token_idx = worker_idx // self.hidden_tiles
        hidden_tile_idx = worker_idx % self.hidden_tiles
        terms = cute.zipped_divide(
            route_terms[token_idx, None, None],
            (self.num_topk, REDUCE_HIDDEN_PER_THREAD),
        )[(None, None), (0, hidden_tile_idx)]
        dst = cute.zipped_divide(
            output[token_idx, None],
            (REDUCE_HIDDEN_PER_THREAD,),
        )[(None,), (hidden_tile_idx,)]

        acc = cute.make_rmem_tensor(
            (REDUCE_HIDDEN_PER_THREAD,),
            cutlass.Float32,
        )
        for i in cutlass.range_constexpr(REDUCE_HIDDEN_PER_THREAD):
            acc[i] = Float32(0.0)

        load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.BFloat16,
            num_bits_per_copy=128,
        )
        # This constexpr loop is the deterministic reduction order.  Do
        # not tree-reduce, atomically reduce, or reorder the slot axis.
        for slot in cutlass.range_constexpr(0, self.num_topk, 1):
            if route_ids[token_idx, Int32(slot)] != -1:
                term_bf16 = cute.make_rmem_tensor(
                    (REDUCE_HIDDEN_PER_THREAD,),
                    cutlass.BFloat16,
                )
                term_slice = terms[slot, None]
                # Route rows are hidden-major and hidden is K32 aligned;
                # each worker advances by eight BF16 values.  Rebuild the
                # dynamically sliced iterator with its guaranteed 16-byte
                # alignment so the 128-bit load survives verification.
                aligned_term = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.BFloat16,
                        term_slice.iterator.toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout(REDUCE_HIDDEN_PER_THREAD),
                )
                cute.copy(load_atom, aligned_term, term_bf16)
                score = Float32(route_scores[token_idx, Int32(slot)])
                for i in cutlass.range_constexpr(REDUCE_HIDDEN_PER_THREAD):
                    product = _mul_rn_f32(Float32(term_bf16[i]), score)
                    acc[i] = _add_rn_f32(acc[i], product)

        # The only output rounding point follows the complete FP32 slot
        # reduction.
        result = cute.make_rmem_tensor(
            (REDUCE_HIDDEN_PER_THREAD,),
            cutlass.BFloat16,
        )
        result.store(acc.load().to(cutlass.BFloat16))
        aligned_dst = cute.make_tensor(
            cute.make_ptr(
                cutlass.BFloat16,
                dst.iterator.toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            dst.layout,
        )
        cute.copy(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.BFloat16,
                num_bits_per_copy=128,
            ),
            result,
            aligned_dst,
        )


__all__ = [
    "DeterministicTopKReducer",
    "FC2_EPILOGUE_TILE",
    "Fc2Bf16Epilogue",
    "Fc2RouteStore",
    "REDUCE_HIDDEN_PER_THREAD",
    "REDUCE_THREADS",
    "validate_fc2_contract",
]
