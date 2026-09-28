# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autonomous persistent epilogue for MXFP8 x MXFP4 MegaMoE.

The native swap-AB MMA leaves an accumulator tile in TMEM with feature rows
and token columns.  The numerical epilogues in :mod:`fc1_epilogue` and
:mod:`fc2_epilogue` intentionally operate on one token-major K32 register
block.  This module owns the missing device orchestration:

* wait for and release accumulator pipeline stages;
* traverse FC1 gate/up and FC2 hidden TMEM tiles;
* transpose each 32x32 tile from feature-major to token-major registers;
* invoke the format-specific FC1 and FC2 primitives;
* publish FC1 readiness only after data and scale stores are visible; and
* resolve FC2 route metadata for identical local and peer route-slot stores.

This is a correctness-first implementation.  It deliberately uses a regular
two-stage, non-overlapping accumulator pipeline.  Register shuffles replace
the older in-place TMEM transpose and keep the release boundary explicit.

Routing scores are absent by construction.  FC2 publishes an unweighted BF16
term for every ``(source token, source top-k slot)``; the deterministic reducer
in :mod:`fc2_epilogue` owns score multiplication and ordered accumulation.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import Int32, Int64, dsl_user_op

from .device_scheduler import DevicePhase
from .fc1_epilogue import Fc1Epilogue, Fc1EpilogueConfig, SF_VEC_SIZE
from .wait_stats import Slot, globaltimer_lo, stat_add, stats_slot_ptr
from .fc2_epilogue import FC2_EPILOGUE_TILE, Fc2Bf16Epilogue, Fc2RouteStore


WARP_SIZE = 32
EPILOGUE_WARPS = 4
TOKEN_METADATA_BYTES = 8
TMEM_ROW_STRIDE = 1 << 16
# Gate/up exchange tile: [warp pair][lane][token] FP32. Every lane writes the
# same token index per step, so a 32-word lane stride puts all 32 lanes in one
# shared-memory bank; one pad word per lane row makes the access conflict-free.
EXCHANGE_LANE_STRIDE = WARP_SIZE + 1
# Rows 0-31 carry lane rows; rows 32/33 carry the two half-block amax values
# of the split FC1 epilogue.
EXCHANGE_ROWS = WARP_SIZE + 2
EXCHANGE_WORDS = (EPILOGUE_WARPS // 2) * EXCHANGE_ROWS * EXCHANGE_LANE_STRIDE

# Skip 32-token epilogue groups whose first token lies beyond the tile's valid
# token count (performance handoff E1).  The default, "fc1", skips empty FC1
# groups: FC2 mainloops wait on FC1 completion, so this shortens the critical
# path (+11-13% end to end at EP16 on GB200).  Skipping FC2 groups measured no
# gain and stays opt-in.  Values: "fc1" (default), "none" (process every
# group, the original behavior), "fc2", or "both".
E1_SKIP_ENV = "FLASHINFER_MXFP8_MXFP4_E1_SKIP"
# The FC1 epilogue splits each 32-token group's SwiGLU/quantization across
# both warps of a gate/up pair by feature halves (round 4: about 6-11% faster
# at 64+ tokens/expert, neutral below). "1" restores the original path, where
# the gate warp transposes both matrices and does all the math.
FC1_EPI_LEGACY_ENV = "FLASHINFER_MXFP8_MXFP4_FC1_EPI_LEGACY"
_E1_SKIP_CHOICES = ("none", "fc1", "fc2", "both")


def _e1_skip_mode() -> str:
    mode = os.environ.get(E1_SKIP_ENV, "fc1").strip().lower() or "fc1"
    if mode not in _E1_SKIP_CHOICES:
        raise ValueError(
            f"{E1_SKIP_ENV} must be one of {_E1_SKIP_CHOICES}, got {mode!r}"
        )
    return mode


@dataclass(frozen=True)
class DeviceEpilogueConfig:
    """Compile-time geometry and policy for the persistent epilogue.

    ``cta_tile_features`` is the swap-AB MMA M extent (weight/output rows),
    while ``cta_tile_tokens`` is its N extent.  FC1 rows are interleaved as
    K32 gate followed by K32 up, so a 128-row feature tile produces 64
    post-SwiGLU values.  FC2 uses all four epilogue warps and produces 128
    hidden values per tile.
    """

    cta_tile_features: int = 128
    cta_tile_tokens: int = 128
    accumulator_stages: int = 2
    epilogue_warps: int = EPILOGUE_WARPS
    barrier_id: int = 1
    gate_up_clamp: Optional[float] = None
    fast_math: bool = True
    # E1 switches, defaulted from E1_SKIP_ENV.  A skipped group's first token is at or beyond
    # ``valid_tokens_in_cta_tile``, which is uniform across the epilogue warps,
    # so all four warps skip together and the FC1 barriers stay balanced.
    skip_empty_fc1_groups: bool = field(
        default_factory=lambda: _e1_skip_mode() in ("fc1", "both")
    )
    skip_empty_fc2_groups: bool = field(
        default_factory=lambda: _e1_skip_mode() in ("fc2", "both")
    )

    def __post_init__(self) -> None:
        if self.cta_tile_features != 128:
            raise ValueError(
                "the correctness epilogue currently requires a 128-row "
                "swap-AB feature tile"
            )
        if self.cta_tile_tokens not in (2 * WARP_SIZE, EPILOGUE_WARPS * WARP_SIZE):
            # Every warp covers each 32-token group of its feature rows; N64
            # and N128 are the token tiles validated end to end.
            raise ValueError("the four-warp epilogue requires N64 or N128")
        if self.accumulator_stages != 2:
            raise ValueError(
                "the correctness epilogue uses exactly two non-overlapping "
                "accumulator stages"
            )
        if self.epilogue_warps != EPILOGUE_WARPS:
            raise ValueError("the epilogue requires exactly four warps")
        if self.barrier_id < 0:
            raise ValueError("barrier_id must be non-negative")
        if self.gate_up_clamp is not None and self.gate_up_clamp < 0.0:
            raise ValueError("gate_up_clamp must be None or non-negative")

    @property
    def token_groups(self) -> int:
        return self.cta_tile_tokens // WARP_SIZE

    @property
    def fc1_output_features_per_tile(self) -> int:
        return self.cta_tile_features // 2

    @property
    def fc1_active_warps(self) -> int:
        return self.epilogue_warps

    @property
    def fc2_active_warps(self) -> int:
        return self.cta_tile_features // FC2_EPILOGUE_TILE

    @property
    def accumulator_tmem_columns(self) -> int:
        return self.cta_tile_tokens * self.accumulator_stages


def route_metadata_host(packed: int) -> tuple[int, int, int]:
    """Decode the stable eight-byte token-source record on the host.

    Low 32 bits are ``source token``.  The high word contains
    ``source rank`` in bits 16..31 and ``source top-k slot`` in bits 0..15.
    This helper is also the host-side test oracle for the device decoder.
    """

    value = packed & ((1 << 64) - 1)
    high = value >> 32
    return ((high >> 16) & 0xFFFF, value & 0xFFFFFFFF, high & 0xFFFF)


@dsl_user_op
def _red_add_release_gpu_s32(
    counter_ptr,
    value,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Publish a task-tile completion after the caller's device fence."""

    llvm.inline_asm(
        None,
        [
            counter_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _load_route_metadata(token_metadata: cute.Tensor, pool_token):
    """Load ``(source rank, source token, source top-k)`` from one i64."""

    byte_address = cute.recast_tensor(
        token_metadata, cutlass.Uint8
    ).iterator.toint() + Int64(pool_token) * Int64(TOKEN_METADATA_BYTES)
    pointer = cute.make_ptr(
        Int64,
        byte_address,
        cute.AddressSpace.gmem,
        assumed_align=TOKEN_METADATA_BYTES,
    )
    packed = Int64(cute.arch.load(pointer, Int64, scope="gpu"))
    high = packed >> Int64(32)
    source_rank = Int32((high >> Int64(16)) & Int64(0xFFFF))
    source_token = Int32(packed & Int64(0xFFFFFFFF))
    source_topk = Int32(high & Int64(0xFFFF))
    return source_rank, source_token, source_topk


class PersistentMxfp8Mxfp4Epilogue:
    """Own the complete epilogue side of the persistent FC1/FC2 loop.

    The integrating kernel supplies the accumulator TMEM allocation, pipeline,
    scheduler consumer, output tensors, and optional source metadata/peer
    mapper.  It does not traverse accumulator tiles or publish readiness
    counters itself.
    """

    def __init__(self, config: Optional[DeviceEpilogueConfig] = None) -> None:
        if config is None:
            config = DeviceEpilogueConfig()
        self.config = config
        self.fc1_split = os.environ.get(FC1_EPI_LEGACY_ENV, "0").strip() != "1"
        if FC1_EPI_LEGACY_ENV in os.environ:
            print(
                f"[mxfp8_mxfp4 epilogue] fc1_split={self.fc1_split}",
                file=sys.stderr,
                flush=True,
            )
        if E1_SKIP_ENV in os.environ:
            # Evidence that an A/B run compiled the intended variant.
            print(
                f"[mxfp8_mxfp4 epilogue] E1 skip_empty_fc1_groups="
                f"{config.skip_empty_fc1_groups} skip_empty_fc2_groups="
                f"{config.skip_empty_fc2_groups}",
                file=sys.stderr,
                flush=True,
            )
        self.fc1 = Fc1Epilogue(
            Fc1EpilogueConfig(
                gate_up_clamp=config.gate_up_clamp,
                fast_math=config.fast_math,
            )
        )
        # FC2 validates the full hidden extent at run time in the integrating
        # kernel; this primitive only needs a legal static K32 store width.
        self.fc2_subtile = Fc2Bf16Epilogue(hidden=FC2_EPILOGUE_TILE)

    @property
    def num_acc_pipeline_stages(self) -> int:
        return self.config.accumulator_stages

    @property
    def num_accumulator_tmem_cols(self) -> int:
        return self.config.accumulator_tmem_columns

    @staticmethod
    def _tmem_layout() -> cute.Layout:
        return cute.make_layout(
            (((WARP_SIZE, WARP_SIZE), 1),),
            stride=(((TMEM_ROW_STRIDE, 1), 0),),
        )

    @staticmethod
    def _rmem_layout() -> cute.Layout:
        return cute.make_layout((((WARP_SIZE,), 1),), stride=(((1,), 0),))

    @cute.jit
    def _load_native_feature_group(
        self,
        tmem_acc_tensor: cute.Tensor,
        warp_idx,
        token_group,
        acc_stage,
    ) -> cute.Tensor:
        """Load the M32 group natively owned by one epilogue warp."""

        registers = cute.make_rmem_tensor(self._rmem_layout().shape, cutlass.Float32)
        offset = (
            warp_idx * Int32(WARP_SIZE * TMEM_ROW_STRIDE)
            + token_group * Int32(WARP_SIZE)
            + acc_stage * Int32(self.config.cta_tile_tokens)
        )
        pointer = tmem_acc_tensor.iterator + cute.assume(offset, divby=WARP_SIZE)
        source = cute.make_tensor(pointer, self._tmem_layout())
        atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Float32
        )
        cute.copy(atom, source, registers)
        return registers

    @cute.jit
    def _transpose_to_token_major(
        self,
        feature_major: cute.Tensor,
        lane_idx,
    ) -> cute.Tensor:
        """Gather one value per feature so each lane owns one token row."""

        current = cute.make_rmem_tensor(self._rmem_layout().shape, cutlass.Float32)
        current.store(feature_major.load())
        for stage in cutlass.range_constexpr(5):
            mask = 1 << stage
            exchanged = cute.make_rmem_tensor(
                self._rmem_layout().shape, cutlass.Float32
            )
            for register in cutlass.range_constexpr(WARP_SIZE):
                shuffled = cutlass.Float32(
                    cute.arch.shuffle_sync_bfly(current[register ^ mask], offset=mask)
                )
                value = current[register]
                if (lane_idx & Int32(mask)) != Int32(register & mask):
                    value = shuffled
                exchanged[register] = value
            current = exchanged
        return current

    @cute.jit
    def _store_fc1_block(
        self,
        quantized: cute.Tensor,
        raw_scale,
        output_data: cute.Tensor,
        output_scales: cute.Tensor,
        data_token,
        scale_token,
        intermediate_block,
    ) -> None:
        """Store one E4M3 K32 block and its raw E8M0 scale byte."""

        destination_row = cute.slice_(output_data, (data_token, None))
        # Every row is 128-byte aligned and the block offset is K32, so this
        # vector destination is at least 32-byte aligned.  Preserve that fact
        # explicitly: slicing a dynamic row otherwise degrades the iterator's
        # inferred alignment to one byte and fails the 256-bit store verifier.
        destination = cute.make_tensor(
            cute.make_ptr(
                cutlass.Float8E4M3FN,
                destination_row.iterator.toint()
                + Int64(intermediate_block) * Int64(SF_VEC_SIZE),
                cute.AddressSpace.gmem,
                assumed_align=32,
            ),
            cute.make_layout(SF_VEC_SIZE),
        )
        cute.copy(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.Float8E4M3FN,
                num_bits_per_copy=256,
            ),
            cute.coalesce(quantized),
            destination,
        )
        scale_cell = cute.local_tile(
            output_scales,
            (1, 1, 1),
            (
                scale_token,
                intermediate_block * Int32(SF_VEC_SIZE),
                Int32(0),
            ),
        )
        scale_bytes = cute.recast_tensor(scale_cell, cutlass.Uint8)
        scale_bytes[0] = raw_scale.to(cutlass.Uint8)

    @cute.jit
    def _run_fc1_task_tile(
        self,
        *,
        work,
        tmem_acc_tensor: cute.Tensor,
        acc_stage,
        fc1_output: cute.Tensor,
        fc1_output_scales: cute.Tensor,
        epilogue_exchange: cute.Tensor,
        boundary,
        warp_idx,
        lane_idx,
    ) -> None:
        """Traverse gate/up TMEM and publish E4M3+E8M0/K32 handoff rows."""

        # Swap-AB TMEM rows are interleaved gate/up features and columns are
        # tokens.  Each epilogue warp owns one K32 token-column group, and LDTM
        # directly gives each lane one token's contiguous K32 feature block.
        pair_idx = warp_idx // Int32(2)
        is_gate_warp = (warp_idx & Int32(1)) == Int32(0)
        for token_group in cutlass.range_constexpr(self.config.token_groups):
            if cutlass.const_expr(self.config.skip_empty_fc1_groups):
                if Int32(token_group * WARP_SIZE) < work.valid_tokens_in_cta_tile:
                    self._run_fc1_token_group(
                        work=work,
                        token_group=token_group,
                        tmem_acc_tensor=tmem_acc_tensor,
                        acc_stage=acc_stage,
                        fc1_output=fc1_output,
                        fc1_output_scales=fc1_output_scales,
                        epilogue_exchange=epilogue_exchange,
                        boundary=boundary,
                        warp_idx=warp_idx,
                        lane_idx=lane_idx,
                        pair_idx=pair_idx,
                        is_gate_warp=is_gate_warp,
                    )
            else:
                self._run_fc1_token_group(
                    work=work,
                    token_group=token_group,
                    tmem_acc_tensor=tmem_acc_tensor,
                    acc_stage=acc_stage,
                    fc1_output=fc1_output,
                    fc1_output_scales=fc1_output_scales,
                    epilogue_exchange=epilogue_exchange,
                    boundary=boundary,
                    warp_idx=warp_idx,
                    lane_idx=lane_idx,
                    pair_idx=pair_idx,
                    is_gate_warp=is_gate_warp,
                )

    @cute.jit
    def _run_fc1_token_group(
        self,
        *,
        work,
        token_group: cutlass.Constexpr[int],
        tmem_acc_tensor: cute.Tensor,
        acc_stage,
        fc1_output: cute.Tensor,
        fc1_output_scales: cute.Tensor,
        epilogue_exchange: cute.Tensor,
        boundary,
        warp_idx,
        lane_idx,
        pair_idx,
        is_gate_warp,
    ) -> None:
        """One 32-token FC1 group: load, gate/up exchange, SwiGLU, store."""

        native = self._load_native_feature_group(
            tmem_acc_tensor, warp_idx, Int32(token_group), acc_stage
        )
        if cutlass.const_expr(self.fc1_split):
            self._run_fc1_token_group_split(
                work=work,
                token_group=token_group,
                native=native,
                fc1_output=fc1_output,
                fc1_output_scales=fc1_output_scales,
                epilogue_exchange=epilogue_exchange,
                boundary=boundary,
                warp_idx=warp_idx,
                lane_idx=lane_idx,
                pair_idx=pair_idx,
                is_gate_warp=is_gate_warp,
            )
        else:
            if not is_gate_warp:
                for token in cutlass.range_constexpr(WARP_SIZE):
                    epilogue_exchange[pair_idx, lane_idx, token] = native[token]
            boundary.arrive_and_wait()
            if is_gate_warp:
                up_raw = cute.make_rmem_tensor(
                    self._rmem_layout().shape, cutlass.Float32
                )
                for token in cutlass.range_constexpr(WARP_SIZE):
                    up_raw[token] = epilogue_exchange[pair_idx, lane_idx, token]
                gate_token_major = self._transpose_to_token_major(native, lane_idx)
                up_token_major = self._transpose_to_token_major(up_raw, lane_idx)
                quantized, raw_scale = self.fc1.transform(
                    gate_token_major, up_token_major
                )
                token_offset = Int32(token_group * WARP_SIZE) + lane_idx
                if token_offset < work.valid_tokens_in_cta_tile:
                    data_token = (
                        work.cumulative_data_physical_row
                        + work.tile_n_idx * Int32(self.config.cta_tile_tokens)
                        + token_offset
                    )
                    scale_token = (
                        work.cumulative_sf_physical_row
                        + work.tile_n_idx * Int32(self.config.cta_tile_tokens)
                        + token_offset
                    )
                    intermediate_block = work.tile_m_idx * Int32(2) + pair_idx
                    self._store_fc1_block(
                        quantized,
                        raw_scale,
                        fc1_output,
                        fc1_output_scales,
                        data_token,
                        scale_token,
                        intermediate_block,
                    )
            boundary.arrive_and_wait()

    @cute.jit
    def _run_fc1_token_group_split(
        self,
        *,
        work,
        token_group: cutlass.Constexpr[int],
        native: cute.Tensor,
        fc1_output: cute.Tensor,
        fc1_output_scales: cute.Tensor,
        epilogue_exchange: cute.Tensor,
        boundary,
        warp_idx,
        lane_idx,
        pair_idx,
        is_gate_warp,
    ) -> None:
        """Each warp of a pair computes one feature half of every token.

        After each warp transposes its own matrix (lane = token), the gate
        warp hands up features 16-31 of gate and the up warp features 0-15 of
        up through one 33-word padded buffer (disjoint halves). The gate warp
        then owns features 0-15 and the up warp features 16-31; the two
        half-block amax values meet in rows 32/33 so both halves share the
        block's E8M0 scale.
        """

        half = SF_VEC_SIZE // 2
        token_major = self._transpose_to_token_major(native, lane_idx)
        if is_gate_warp:
            for feature in cutlass.range_constexpr(half):
                epilogue_exchange[pair_idx, lane_idx, half + feature] = token_major[
                    half + feature
                ]
        else:
            for feature in cutlass.range_constexpr(half):
                epilogue_exchange[pair_idx, lane_idx, feature] = token_major[feature]
        boundary.arrive_and_wait()
        gate = cute.make_rmem_tensor(self.fc1.half_rmem_layout().shape, cutlass.Float32)
        up = cute.make_rmem_tensor(self.fc1.half_rmem_layout().shape, cutlass.Float32)
        if is_gate_warp:
            for feature in cutlass.range_constexpr(half):
                gate[feature] = token_major[feature]
                up[feature] = epilogue_exchange[pair_idx, lane_idx, feature]
        else:
            for feature in cutlass.range_constexpr(half):
                gate[feature] = epilogue_exchange[pair_idx, lane_idx, half + feature]
                up[feature] = token_major[half + feature]
        swiglu, half_amax = self.fc1.swiglu_half(gate, up)
        own_row = Int32(WARP_SIZE) + (warp_idx & Int32(1))
        other_row = Int32(WARP_SIZE + 1) - (warp_idx & Int32(1))
        epilogue_exchange[pair_idx, own_row, lane_idx] = half_amax
        boundary.arrive_and_wait()
        block_amax = cute.arch.fmax(
            half_amax, epilogue_exchange[pair_idx, other_row, lane_idx]
        )
        quantized, raw_scale = self.fc1.quantize_half(swiglu, block_amax)
        token_offset = Int32(token_group * WARP_SIZE) + lane_idx
        if token_offset < work.valid_tokens_in_cta_tile:
            data_token = (
                work.cumulative_data_physical_row
                + work.tile_n_idx * Int32(self.config.cta_tile_tokens)
                + token_offset
            )
            scale_token = (
                work.cumulative_sf_physical_row
                + work.tile_n_idx * Int32(self.config.cta_tile_tokens)
                + token_offset
            )
            intermediate_block = work.tile_m_idx * Int32(2) + pair_idx
            self._store_fc1_half(
                quantized,
                raw_scale,
                fc1_output,
                fc1_output_scales,
                data_token,
                scale_token,
                intermediate_block,
                warp_idx & Int32(1),
            )
        boundary.arrive_and_wait()

    @cute.jit
    def _store_fc1_half(
        self,
        quantized: cute.Tensor,
        raw_scale,
        output_data: cute.Tensor,
        output_scales: cute.Tensor,
        data_token,
        scale_token,
        intermediate_block,
        half_idx,
    ) -> None:
        """Store 16 E4M3 bytes of a K32 block; half 0 also stores the scale."""

        half = SF_VEC_SIZE // 2
        destination_row = cute.slice_(output_data, (data_token, None))
        destination = cute.make_tensor(
            cute.make_ptr(
                cutlass.Float8E4M3FN,
                destination_row.iterator.toint()
                + Int64(intermediate_block) * Int64(SF_VEC_SIZE)
                + Int64(half_idx) * Int64(half),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            cute.make_layout(half),
        )
        cute.copy(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.Float8E4M3FN,
                num_bits_per_copy=128,
            ),
            cute.coalesce(quantized),
            destination,
        )
        if half_idx == Int32(0):
            scale_cell = cute.local_tile(
                output_scales,
                (1, 1, 1),
                (
                    scale_token,
                    intermediate_block * Int32(SF_VEC_SIZE),
                    Int32(0),
                ),
            )
            scale_bytes = cute.recast_tensor(scale_cell, cutlass.Uint8)
            scale_bytes[0] = raw_scale.to(cutlass.Uint8)

    @cute.jit
    def _run_fc2_task_tile(
        self,
        *,
        work,
        tmem_acc_tensor: cute.Tensor,
        acc_stage,
        route_store: Fc2RouteStore,
        token_metadata,
        warp_idx,
        lane_idx,
    ) -> None:
        """Traverse FC2 TMEM and store unweighted BF16 route terms."""

        for token_group in cutlass.range_constexpr(self.config.token_groups):
            if cutlass.const_expr(self.config.skip_empty_fc2_groups):
                if Int32(token_group * WARP_SIZE) < work.valid_tokens_in_cta_tile:
                    self._run_fc2_token_group(
                        work=work,
                        token_group=token_group,
                        tmem_acc_tensor=tmem_acc_tensor,
                        acc_stage=acc_stage,
                        route_store=route_store,
                        token_metadata=token_metadata,
                        warp_idx=warp_idx,
                        lane_idx=lane_idx,
                    )
            else:
                self._run_fc2_token_group(
                    work=work,
                    token_group=token_group,
                    tmem_acc_tensor=tmem_acc_tensor,
                    acc_stage=acc_stage,
                    route_store=route_store,
                    token_metadata=token_metadata,
                    warp_idx=warp_idx,
                    lane_idx=lane_idx,
                )

    @cute.jit
    def _run_fc2_token_group(
        self,
        *,
        work,
        token_group: cutlass.Constexpr[int],
        tmem_acc_tensor: cute.Tensor,
        acc_stage,
        route_store: Fc2RouteStore,
        token_metadata,
        warp_idx,
        lane_idx,
    ) -> None:
        """One 32-token FC2 group: load, transpose, store route terms."""

        feature_major = self._load_native_feature_group(
            tmem_acc_tensor, warp_idx, Int32(token_group), acc_stage
        )
        token_major = self._transpose_to_token_major(feature_major, lane_idx)
        token_offset = Int32(token_group * WARP_SIZE) + lane_idx
        if token_offset < work.valid_tokens_in_cta_tile:
            pool_token = (
                work.cumulative_data_physical_row
                + work.tile_n_idx * Int32(self.config.cta_tile_tokens)
                + token_offset
            )
            source_rank = Int32(0)
            source_token = pool_token
            source_topk = Int32(0)
            if cutlass.const_expr(token_metadata is not None):
                source_rank, source_token, source_topk = _load_route_metadata(
                    token_metadata, pool_token
                )
            hidden_col_start = work.tile_m_idx * Int32(
                self.config.cta_tile_features
            ) + warp_idx * Int32(FC2_EPILOGUE_TILE)
            self.fc2_subtile.store_rmem_subtile(
                token_major,
                route_store,
                source_rank,
                source_token,
                source_topk,
                hidden_col_start,
            )

    @cute.jit
    def run(
        self,
        *,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        scheduler_consumer,
        fc1_output: cute.Tensor,
        fc1_output_scales: cute.Tensor,
        route_store: Fc2RouteStore,
        fc1_done_counter: cute.Tensor,
        epilogue_exchange: cute.Tensor,
        warp_idx,
        tidx,
        token_metadata=None,
        fc2_done_counter=None,
        stats=None,
        stats_word_offset=None,
    ) -> None:
        """Consume and retire all persistent FC1/FC2 work records.

        Publication happens after a four-warp CTA barrier and a device-scope
        fence.  The FC1 release-add unlocks FC2 mainloop work.  The optional
        FC2 release-add is for future dispatch-warp token-back; the current
        deterministic single-GPU reducer is launched after kernel completion
        and therefore needs no in-kernel completion counter.
        """

        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.config.accumulator_stages,
        )
        boundary = pipeline.NamedBarrier(
            barrier_id=self.config.barrier_id,
            num_threads=self.config.epilogue_warps * WARP_SIZE,
        )
        lane_idx = tidx % Int32(WARP_SIZE)
        if cutlass.const_expr(stats is not None):
            loop_start = globaltimer_lo()
        work = scheduler_consumer.consume_work()
        while work.is_valid_tile:
            # The scheduler's peek concerns input/readiness and never permits
            # the epilogue to consume an accumulator before its pipeline stage.
            if cutlass.const_expr(stats is not None):
                waited = globaltimer_lo()
            acc_pipeline.consumer_wait(consumer_state)
            if cutlass.const_expr(stats is not None):
                tile_start = globaltimer_lo()
                if tidx == Int32(0):
                    stat_add(
                        stats_slot_ptr(stats, Slot.EPI_ACC_FULL_NS, stats_word_offset),
                        tile_start - waited,
                    )
            acc_stage = Int32(consumer_state.index)

            if work.phase == Int32(DevicePhase.FC1):
                self._run_fc1_task_tile(
                    work=work,
                    tmem_acc_tensor=tmem_acc_tensor,
                    acc_stage=acc_stage,
                    fc1_output=fc1_output,
                    fc1_output_scales=fc1_output_scales,
                    epilogue_exchange=epilogue_exchange,
                    boundary=boundary,
                    warp_idx=warp_idx,
                    lane_idx=lane_idx,
                )
            else:
                self._run_fc2_task_tile(
                    work=work,
                    tmem_acc_tensor=tmem_acc_tensor,
                    acc_stage=acc_stage,
                    route_store=route_store,
                    token_metadata=token_metadata,
                    warp_idx=warp_idx,
                    lane_idx=lane_idx,
                )

            # Every LDTM has completed before the accumulator slot is handed
            # back.  Register math and global stores no longer depend on TMEM.
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(consumer_state)
            consumer_state.advance()

            boundary.arrive_and_wait()
            if tidx == Int32(0):
                cute.arch.fence_acq_rel_gpu()
                if work.phase == Int32(DevicePhase.FC1):
                    _red_add_release_gpu_s32(
                        fc1_done_counter.iterator + work.counter_slot,
                        Int32(1),
                    )
                elif cutlass.const_expr(fc2_done_counter is not None):
                    _red_add_release_gpu_s32(
                        fc2_done_counter.iterator + work.expert_idx,
                        Int32(1),
                    )
                if cutlass.const_expr(stats is not None):
                    # Accumulator ready -> publication: the FC2 readiness latency.
                    elapsed = globaltimer_lo() - tile_start
                    if work.phase == Int32(DevicePhase.FC1):
                        stat_add(
                            stats_slot_ptr(
                                stats, Slot.EPI_FC1_TILE_NS, stats_word_offset
                            ),
                            elapsed,
                        )
                        stat_add(
                            stats_slot_ptr(
                                stats, Slot.EPI_FC1_TILE_COUNT, stats_word_offset
                            ),
                            Int32(1),
                        )
                    else:
                        stat_add(
                            stats_slot_ptr(
                                stats, Slot.EPI_FC2_TILE_NS, stats_word_offset
                            ),
                            elapsed,
                        )
                        stat_add(
                            stats_slot_ptr(
                                stats, Slot.EPI_FC2_TILE_COUNT, stats_word_offset
                            ),
                            Int32(1),
                        )

            work = scheduler_consumer.consume_work()
        if cutlass.const_expr(stats is not None):
            if tidx == Int32(0):
                stat_add(
                    stats_slot_ptr(stats, Slot.EPI_LOOP_NS, stats_word_offset),
                    globaltimer_lo() - loop_start,
                )


__all__ = [
    "DeviceEpilogueConfig",
    "EPILOGUE_WARPS",
    "PersistentMxfp8Mxfp4Epilogue",
    "TOKEN_METADATA_BYTES",
    "TMEM_ROW_STRIDE",
    "EXCHANGE_LANE_STRIDE",
    "EXCHANGE_ROWS",
    "EXCHANGE_WORDS",
    "WARP_SIZE",
    "route_metadata_host",
]
