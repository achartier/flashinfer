# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autonomous persistent epilogue for MXFP8 x MXFP4 MegaMoE.

The swap-AB MMA leaves feature rows and token columns in TMEM. FC1's
16dp256b loads match the gate8/up8 weight interleave, so each lane computes
SwiGLU directly without transposing or exchanging gate/up values. Only K32
block maxima cross lanes and paired warps. FC2 retains the token-major
register transpose for contiguous BF16 route stores.

This module owns accumulator wait/release, tile traversal, output stores,
and FC1 readiness publication after the output data and scales are visible.

Routing scores are absent by construction.  FC2 publishes an unweighted BF16
term for every ``(source token, source top-k slot)``; the deterministic reducer
in :mod:`fc2_epilogue` owns score multiplication and ordered accumulation.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import Int32, Int64, dsl_user_op

from .device_scheduler import DevicePhase
from .fc1_epilogue import (
    Fc1Epilogue,
    Fc1EpilogueConfig,
    SF_VEC_SIZE,
    _clamp_and_swiglu,
    _quantize_with_amax,
)
from .fc1_store import fc1_tma_store_enabled, issue_fc1_store, store_native_e4m3
from .wait_stats import Slot, globaltimer_lo, stat_add, stats_slot_ptr
from .iket_markers import iket_enabled, iket_fine_ctas, is_fine_cta, load_iket
from .fc2_epilogue import FC2_EPILOGUE_TILE, Fc2Bf16Epilogue, Fc2RouteStore


WARP_SIZE = 32
EPILOGUE_WARPS = 4
EPILOGUE_WARPS_ENV = "FLASHINFER_MXFP8_MXFP4_EPILOGUE_WARPS"
# IDs 0/1/2 are CTA/tile/TMEM barriers; dispatch owns 8/9/10.
EXCHANGE_BARRIER_IDS = (3, 4)
TOKEN_METADATA_BYTES = 8
TMEM_ROW_STRIDE = 1 << 16
# Only the two warps' half-block maxima cross shared memory. Each row
# contains one amax per token; the final barrier protects reuse by the next group.
EXCHANGE_LANE_STRIDE = WARP_SIZE
EXCHANGE_ROWS = 2
EXCHANGE_WORDS = (EPILOGUE_WARPS // 2) * EXCHANGE_ROWS * EXCHANGE_LANE_STRIDE

# Skip 32-token epilogue groups whose first token lies beyond the tile's valid
# token count. The default, "fc1", skips empty FC1 groups because FC2
# mainloops wait on FC1 completion. Skipping FC2 groups is opt-in.
# Values: "fc1" (default), "none" (process every
# group, the original behavior), "fc2", or "both".
E1_SKIP_ENV = "FLASHINFER_MXFP8_MXFP4_E1_SKIP"
# Debug comparison path: use the register transpose with the same gate8/up8
# weights. The default reads gate/up directly into each lane's registers.
FC1_EPI_LEGACY_ENV = "FLASHINFER_MXFP8_MXFP4_FC1_EPI_LEGACY"
_E1_SKIP_CHOICES = ("none", "fc1", "fc2", "both")


def epilogue_warps_from_env() -> int:
    """Read the experimental width once when constructing a launcher."""
    value = os.environ.get(EPILOGUE_WARPS_ENV, "4").strip()
    if value not in ("4", "8"):
        raise ValueError(f"{EPILOGUE_WARPS_ENV} must be 4 or 8, got {value!r}")
    return int(value)


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
    eight gate rows followed by eight up rows, so a 128-row feature tile produces 64
    post-SwiGLU values. Each warpgroup covers all feature rows; with eight
    epilogue warps, the two warpgroups process alternating token groups.
    """

    cta_tile_features: int = 128
    cta_tile_tokens: int = 128
    accumulator_stages: int = 2
    epilogue_warps: int = EPILOGUE_WARPS
    barrier_id: int = 1
    gate_up_clamp: Optional[float] = None
    fast_math: bool = True
    # E1 switches, defaulted from E1_SKIP_ENV.  A skipped group's first token is at or beyond
    # ``valid_tokens_in_cta_tile``, which is uniform across each warpgroup,
    # so its four warps skip together and the FC1 barriers stay balanced.
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
        if self.cta_tile_tokens not in (2 * WARP_SIZE, 4 * WARP_SIZE, 6 * WARP_SIZE):
            # Every warp covers each 32-token group of its feature rows; N64,
            # N128, and N192 are the token tiles validated end to end.
            raise ValueError("the epilogue requires N64, N128, or N192")
        if self.accumulator_stages != 2:
            raise ValueError(
                "the correctness epilogue uses exactly two non-overlapping "
                "accumulator stages"
            )
        if self.epilogue_warps not in (4, 8):
            raise ValueError("the epilogue requires four or eight warps")
        if self.barrier_id < 0:
            raise ValueError("barrier_id must be non-negative")
        if self.epilogue_warps == 8 and self.barrier_id in EXCHANGE_BARRIER_IDS:
            raise ValueError("tile and exchange barriers must differ")
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
        return self.epilogue_warps

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
def _store_predicated_u8(
    ptr,
    value,
    valid,
    offset: cutlass.Constexpr[int] = 0,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Keep the byte offset immediate and predicate only the global store."""

    llvm.inline_asm(
        None,
        [
            ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(value).ir_value(loc=loc, ip=ip),
            Int32(valid).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; "
        f"@p st.global.u8 [$0 + {offset}], $1; }}",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


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
        if EPILOGUE_WARPS_ENV in os.environ:
            print(
                f"[mxfp8_mxfp4 epilogue] epilogue_warps={config.epilogue_warps}",
                file=sys.stderr,
                flush=True,
            )
        self.fc1_tma_store = fc1_tma_store_enabled()
        self.fc1_native = os.environ.get(FC1_EPI_LEGACY_ENV, "0").strip() != "1"
        # IKET ranges (CUTE_DSL_COMPILER_OPT=iket only; see iket_markers).
        self.iket = iket_enabled()
        self._iket: Any = load_iket() if self.iket else None
        # Per-step FC1 tile ranges on the first CTAs only (see iket_markers).
        self.iket_fine_ctas = iket_fine_ctas()
        if FC1_EPI_LEGACY_ENV in os.environ:
            print(
                f"[mxfp8_mxfp4 epilogue] fc1_native={self.fc1_native}",
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
        feature_warp_idx = warp_idx
        if cutlass.const_expr(self.config.epilogue_warps == 8):
            feature_warp_idx = warp_idx % Int32(4)
        offset = (
            feature_warp_idx * Int32(WARP_SIZE * TMEM_ROW_STRIDE)
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
    def _load_fc1_native_half(
        self,
        tmem_acc_tensor,
        warp_idx,
        token_group,
        acc_stage,
        half: cutlass.Constexpr[int],
    ):
        """Load 16 feature rows x 32 tokens without a register transpose.

        For register r and lane l, 16dp256b maps to row l//4 + 8*((r//2)%2)
        and column 2*(l%4) + r%2 + 8*(r//4). Gate8/up8 interleaving puts
        matching gate/up in registers (4*t,4*t+1)/(4*t+2,4*t+3).
        """
        offset = (
            (warp_idx % Int32(4)) * Int32(32 * TMEM_ROW_STRIDE)
            + Int32(half * 16 * TMEM_ROW_STRIDE)
            + token_group * Int32(32)
            + acc_stage * Int32(self.config.cta_tile_tokens)
        )
        source = cute.make_tensor(
            tmem_acc_tensor.iterator + cute.assume(offset, divby=32),
            cute.make_layout((((16, 32), 1),), stride=(((TMEM_ROW_STRIDE, 1), 0),)),
        )
        registers = cute.make_rmem_tensor(
            cute.make_layout((((16,), 1),), stride=(((1,), 0),)).shape,
            cutlass.Float32,
        )
        cute.copy(
            cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition.x4), cutlass.Float32
            ),
            source,
            registers,
        )
        return registers

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
        fc1_store,
        boundary,
        warp_idx,
        lane_idx,
        warpgroup_idx,
        fine_cta,
    ) -> None:
        """Traverse gate/up TMEM and publish E4M3+E8M0/K32 handoff rows."""

        # Swap-AB TMEM rows are interleaved gate/up features and columns are
        # tokens. Each warp owns 32 feature rows. Warpgroup-local TMEM rows
        # repeat in the second warpgroup, but exchange pairs must be disjoint.
        pair_idx = warp_idx // Int32(2)
        for token_group in cutlass.range_constexpr(self.config.token_groups):
            if cutlass.const_expr(self.config.epilogue_warps == 4) or (
                Int32(token_group % 2) == warpgroup_idx
            ):
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
                            fc1_store=fc1_store,
                            boundary=boundary,
                            warp_idx=warp_idx,
                            lane_idx=lane_idx,
                            pair_idx=pair_idx,
                            fine_cta=fine_cta,
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
                        fc1_store=fc1_store,
                        boundary=boundary,
                        warp_idx=warp_idx,
                        lane_idx=lane_idx,
                        pair_idx=pair_idx,
                        fine_cta=fine_cta,
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
        fc1_store,
        boundary,
        warp_idx,
        lane_idx,
        pair_idx,
        fine_cta,
    ) -> None:
        """One 32-token FC1 group: native load, SwiGLU, amax exchange, store."""

        if cutlass.const_expr(self.fc1_tma_store):
            store_stage = (token_group // (self.config.epilogue_warps // 4)) % 2
            full_group = Int32((token_group + 1) * 32) <= work.valid_tokens_in_cta_tile
            stage = fc1_store[2][None, None, store_stage, warp_idx // Int32(4)]
            staged_bytes = cute.make_rmem_tensor((16,), cutlass.Uint8)
            if cutlass.const_expr(token_group >= 2 * (self.config.epilogue_warps // 4)):
                if full_group:
                    if warp_idx % Int32(4) == Int32(0):
                        with cute.arch.elect_one():
                            cute.arch.cp_async_bulk_wait_group(1, read=True)
                    boundary.arrive_and_wait()

        if cutlass.const_expr(self.fc1_native):
            if cutlass.const_expr(self.iket_fine_ctas > 0):
                if fine_cta:
                    self._iket.range_push("fc1_ld_math")
            first = self._load_fc1_native_half(
                tmem_acc_tensor, warp_idx, Int32(token_group), acc_stage, 0
            )
            second = self._load_fc1_native_half(
                tmem_acc_tensor, warp_idx, Int32(token_group), acc_stage, 1
            )
            gate = cute.make_rmem_tensor((16,), cutlass.Float32)
            up = cute.make_rmem_tensor((16,), cutlass.Float32)
            for token in cutlass.range_constexpr(8):
                register = (token // 2) * 4 + token % 2
                gate[token * 2] = first[register]
                up[token * 2] = first[register + 2]
                gate[token * 2 + 1] = second[register]
                up[token * 2 + 1] = second[register + 2]
            values = cute.make_rmem_tensor((16,), cutlass.Float32)
            _clamp_and_swiglu(
                values, gate, up, self.fc1._clamp, self.config.fast_math, 16
            )
            maxima = cute.make_rmem_tensor((8,), cutlass.Float32)
            for token in cutlass.range_constexpr(8):
                v0, v1 = values[token * 2], values[token * 2 + 1]
                # Match _block_amax's zero seed, including all-NaN fragments.
                amax = cute.arch.fmax(
                    cutlass.Float32(0.0),
                    cute.arch.fmax(cute.arch.fmax(v0, -v0), cute.arch.fmax(v1, -v1)),
                )
                # Eight lanes, spaced four apart, cover 16 features for
                # this token. The partner warp covers the other 16.
                for shift in cutlass.range_constexpr(3):
                    amax = cute.arch.fmax(
                        amax, cute.arch.shuffle_sync_bfly(amax, offset=4 << shift)
                    )
                maxima[token] = amax
                token_in_group = (lane_idx % Int32(4)) * Int32(2) + Int32(
                    (token // 2) * 8 + token % 2
                )
                if lane_idx < Int32(4):
                    epilogue_exchange[pair_idx, warp_idx % Int32(2), token_in_group] = (
                        amax
                    )
            if cutlass.const_expr(self.iket_fine_ctas > 0):
                if fine_cta:
                    self._iket.range_pop()
                    self._iket.range_push("fc1_amax_bar")
            boundary.arrive_and_wait()
            if cutlass.const_expr(self.iket_fine_ctas > 0):
                if fine_cta:
                    self._iket.range_pop()
                    self._iket.range_push("fc1_quant_store")
            # One address per lane and group. The native fragment owns rows
            # {0, 1, 8, 9, 16, 17, 24, 25} and features {0, 8}. The launcher
            # supplies a static row stride, so all remaining offsets are
            # store immediates rather than repeated 64-bit row multiplies.
            lane_row = (lane_idx % Int32(4)) * Int32(2)
            tile_row = work.tile_n_idx * Int32(self.config.cta_tile_tokens)
            group_row = tile_row + Int32(token_group * 32)
            block = work.tile_m_idx * Int32(2) + pair_idx % Int32(2)
            feature = (
                block * Int32(32)
                + (warp_idx % Int32(2)) * Int32(16)
                + lane_idx // Int32(4)
            )
            data_row = work.cumulative_data_physical_row + group_row + lane_row
            data_ptr = fc1_output.iterator + (
                Int64(data_row) * Int64(fc1_output.stride[0]) + Int64(feature)
            )
            scales = cute.make_rmem_tensor((8,), cutlass.Uint8)
            for token in cutlass.range_constexpr(8):
                token_in_group = (lane_idx % Int32(4)) * Int32(2) + Int32(
                    (token // 2) * 8 + token % 2
                )
                amax = cute.arch.fmax(
                    maxima[token],
                    epilogue_exchange[
                        pair_idx, Int32(1) - warp_idx % Int32(2), token_in_group
                    ],
                )
                pair_values = cute.make_rmem_tensor((2,), cutlass.Float32)
                pair_values[0], pair_values[1] = (
                    values[token * 2],
                    values[token * 2 + 1],
                )
                quantized = cute.make_rmem_tensor((2,), cutlass.Float8E4M3FN)
                raw_scale = _quantize_with_amax(pair_values, quantized, amax, 2)
                scales[token] = raw_scale.to(cutlass.Uint8)
                token_offset = Int32(token_group * 32) + token_in_group
                valid = token_offset < work.valid_tokens_in_cta_tile
                raw = cute.recast_tensor(quantized, cutlass.Uint8)
                if cutlass.const_expr(self.fc1_tma_store):
                    staged_bytes[(token // 2) * 4 + token % 2] = raw[0]
                    staged_bytes[(token // 2) * 4 + token % 2 + 2] = raw[1]
                    # A full TMA box would clobber the next expert's rows.
                    valid = valid and not full_group
                # Use one predicated path for full and partial groups to avoid
                # duplicating the unrolled quantization and its register state.
                row_offset = (token // 2) * 8 + token % 2
                _store_predicated_u8(
                    data_ptr, raw[0], valid, row_offset * fc1_output.stride[0]
                )
                _store_predicated_u8(
                    data_ptr, raw[1], valid, row_offset * fc1_output.stride[0] + 8
                )
            if lane_idx < Int32(4) and warp_idx % Int32(2) == Int32(0):
                # Expert SF rows are padded to 128, and tile/group starts are
                # multiples of 32. Within a group only the atom's row%32 mode
                # varies (stride 16); derive the base from the canonical layout
                # once so K-block and outer-atom strides need no duplication.
                scale_row = work.cumulative_sf_physical_row + group_row + lane_row
                scale_ptr = cute.recast_ptr(
                    fc1_output_scales.iterator, dtype=cutlass.Uint8
                ) + cute.crd2idx(
                    (scale_row, block * Int32(32), Int32(0)), fc1_output_scales.layout
                )
                for token in cutlass.range_constexpr(8):
                    row_offset = (token // 2) * 8 + token % 2
                    valid = (
                        Int32(token_group * 32 + row_offset) + lane_row
                        < work.valid_tokens_in_cta_tile
                    )
                    _store_predicated_u8(
                        scale_ptr, scales[token], valid, row_offset * 16
                    )
            if cutlass.const_expr(self.fc1_tma_store):
                if full_group:
                    # STSM transposes the native register map into token-major
                    # rows, replacing sixteen shared byte stores per lane.
                    packed = cute.recast_tensor(staged_bytes, cutlass.Uint32)
                    store_native_e4m3(
                        stage.iterator,
                        cute.crd2idx(
                            (lane_idx, (warp_idx % Int32(4)) * Int32(16)), stage.layout
                        ),
                        packed[0],
                        packed[1],
                        packed[2],
                        packed[3],
                    )
                cute.arch.fence_proxy("async.shared", space="cta")
            if cutlass.const_expr(self.iket_fine_ctas > 0):
                if fine_cta:
                    self._iket.range_pop()
                    self._iket.range_push("fc1_group_bar")
            boundary.arrive_and_wait()
            if cutlass.const_expr(self.iket_fine_ctas > 0):
                if fine_cta:
                    self._iket.range_pop()
            if cutlass.const_expr(self.fc1_tma_store):
                if full_group and warp_idx % Int32(4) == Int32(0):
                    with cute.arch.elect_one():
                        issue_fc1_store(
                            fc1_store[0],
                            fc1_store[1],
                            stage,
                            work.cumulative_data_physical_row
                            + work.tile_n_idx * Int32(self.config.cta_tile_tokens)
                            + Int32(token_group * 32),
                            work.tile_m_idx * Int32(64),
                        )
        else:
            # Diagnostic transpose path for the same gate8/up8 physical rows.
            native = self._load_native_feature_group(
                tmem_acc_tensor, warp_idx, Int32(token_group), acc_stage
            )
            token_major = self._transpose_to_token_major(native, lane_idx)
            gate = cute.make_rmem_tensor(
                self.fc1.half_rmem_layout().shape, cutlass.Float32
            )
            up = cute.make_rmem_tensor(
                self.fc1.half_rmem_layout().shape, cutlass.Float32
            )
            for feature in cutlass.range_constexpr(16):
                row = (feature // 8) * 16 + feature % 8
                gate[feature], up[feature] = token_major[row], token_major[row + 8]
            values, half_amax = self.fc1.swiglu_half(gate, up)
            epilogue_exchange[pair_idx, warp_idx % Int32(2), lane_idx] = half_amax
            boundary.arrive_and_wait()
            block_amax = cute.arch.fmax(
                half_amax,
                epilogue_exchange[pair_idx, Int32(1) - warp_idx % Int32(2), lane_idx],
            )
            quantized, raw_scale = self.fc1.quantize_half(values, block_amax)
            token_offset = Int32(token_group * 32) + lane_idx
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
                block = work.tile_m_idx * Int32(2) + pair_idx % Int32(2)
                self._store_fc1_half(
                    quantized,
                    raw_scale,
                    fc1_output,
                    fc1_output_scales,
                    data_token,
                    scale_token,
                    block,
                    warp_idx % Int32(2),
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
        warpgroup_idx,
    ) -> None:
        """Traverse FC2 TMEM and store unweighted BF16 route terms."""

        for token_group in cutlass.range_constexpr(self.config.token_groups):
            if cutlass.const_expr(self.config.epilogue_warps == 4) or (
                Int32(token_group % 2) == warpgroup_idx
            ):
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
            feature_warp_idx = warp_idx
            if cutlass.const_expr(self.config.epilogue_warps == 8):
                feature_warp_idx = warp_idx % Int32(4)
            hidden_col_start = work.tile_m_idx * Int32(
                self.config.cta_tile_features
            ) + feature_warp_idx * Int32(FC2_EPILOGUE_TILE)
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
        fc1_store,
        warp_idx,
        tidx,
        token_metadata=None,
        fc2_done_counter=None,
        stats=None,
        stats_word_offset=None,
    ) -> None:
        """Consume and retire all persistent FC1/FC2 work records.

        Publication happens after an all-epilogue barrier and a device-scope
        fence.  The FC1 release-add unlocks FC2 mainloop work.  The optional
        FC2 release-add unlocks dispatch-warp token-back after all feature
        and token tiles for an expert have reached the local BF16 pool.
        Direct epilogue token-back needs no completion counter: the reducer
        runs after the collective kernel tail.
        """

        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.config.accumulator_stages,
        )
        boundary = pipeline.NamedBarrier(
            barrier_id=self.config.barrier_id,
            num_threads=self.config.epilogue_warps * WARP_SIZE,
        )
        # Only the tile boundary spans both warpgroups. E1 can leave one
        # warpgroup with fewer groups (or none), so exchange barriers cannot.
        warpgroup_idx = warp_idx // Int32(4)
        exchange_boundary = boundary
        if cutlass.const_expr(self.config.epilogue_warps == 8):
            exchange_boundary = pipeline.NamedBarrier(
                barrier_id=Int32(EXCHANGE_BARRIER_IDS[0]) + warpgroup_idx,
                num_threads=4 * WARP_SIZE,
            )
        lane_idx = tidx % Int32(WARP_SIZE)
        fine_cta = cutlass.Boolean(False)
        if cutlass.const_expr(self.iket_fine_ctas > 0):
            fine_cta = is_fine_cta(self.iket_fine_ctas)
        if cutlass.const_expr(stats is not None):
            loop_start = globaltimer_lo()
        work = scheduler_consumer.consume_work()
        while work.is_valid_tile:
            # The scheduler's peek concerns input/readiness and never permits
            # the epilogue to consume an accumulator before its pipeline stage.
            if cutlass.const_expr(stats is not None):
                waited = globaltimer_lo()
            if cutlass.const_expr(self.iket):
                self._iket.range_push("epi_wait_acc")
            acc_pipeline.consumer_wait(consumer_state)
            if cutlass.const_expr(self.iket):
                self._iket.range_pop()
                if work.phase == Int32(DevicePhase.FC1):
                    self._iket.range_push("epi_fc1")
                else:
                    self._iket.range_push("epi_fc2")
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
                    fc1_store=fc1_store,
                    boundary=exchange_boundary,
                    warp_idx=warp_idx,
                    lane_idx=lane_idx,
                    warpgroup_idx=warpgroup_idx,
                    fine_cta=fine_cta,
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
                    warpgroup_idx=warpgroup_idx,
                )

            # Every LDTM has completed before the accumulator slot is handed
            # back.  Register math and global stores no longer depend on TMEM.
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(consumer_state)
            consumer_state.advance()

            # Full completion, not just SMEM-read completion: FC2 consumes
            # these bytes through TMA as soon as the release counter is visible.
            if cutlass.const_expr(self.fc1_tma_store):
                if work.phase == Int32(DevicePhase.FC1) and warp_idx % Int32(
                    4
                ) == Int32(0):
                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_wait_group(0, read=False)
            # Tile end, then the FC1-done publication (warp 0 only: tid 0
            # issues the GPU-scope fence and release-add).
            if cutlass.const_expr(self.iket_fine_ctas > 0):
                if fine_cta and work.phase == Int32(DevicePhase.FC1):
                    self._iket.range_push("fc1_tile_bar")
            boundary.arrive_and_wait()
            if cutlass.const_expr(self.iket_fine_ctas > 0):
                if fine_cta and work.phase == Int32(DevicePhase.FC1):
                    self._iket.range_pop()
            if cutlass.const_expr(self.iket):
                self._iket.range_pop()
            if cutlass.const_expr(self.iket_fine_ctas > 0):
                if (
                    fine_cta
                    and work.phase == Int32(DevicePhase.FC1)
                    and warp_idx == Int32(0)
                ):
                    self._iket.range_push("fc1_publish")
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
            if cutlass.const_expr(self.iket_fine_ctas > 0):
                if (
                    fine_cta
                    and work.phase == Int32(DevicePhase.FC1)
                    and warp_idx == Int32(0)
                ):
                    self._iket.range_pop()

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
    "EPILOGUE_WARPS_ENV",
    "PersistentMxfp8Mxfp4Epilogue",
    "TOKEN_METADATA_BYTES",
    "TMEM_ROW_STRIDE",
    "EXCHANGE_LANE_STRIDE",
    "EXCHANGE_ROWS",
    "EXCHANGE_WORDS",
    "EXCHANGE_BARRIER_IDS",
    "WARP_SIZE",
    "epilogue_warps_from_env",
    "route_metadata_host",
]
