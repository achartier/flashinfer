# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Double-buffered FC1 data stores; scale bytes retain the FC2 SFB layout."""

import os

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op


FC1_TMA_STORE_ENV = "FLASHINFER_MXFP8_MXFP4_FC1_TMA_STORE"
STORE_TILE = (32, 64)
STORE_STAGES = 2


def fc1_tma_store_enabled() -> bool:
    """Resolve the experimental opt-in before compiling a launcher."""
    value = os.environ.get(FC1_TMA_STORE_ENV, "0").strip()
    if value not in ("0", "1"):
        raise ValueError(f"{FC1_TMA_STORE_ENV} must be 0 or 1, got {value!r}")
    return (
        value == "1"
        and os.environ.get("FLASHINFER_MXFP8_MXFP4_FC1_EPI_LEGACY", "0").strip() != "1"
    )


def fc1_store_bytes(epilogue_warps: int) -> int:
    return (
        STORE_TILE[0] * STORE_TILE[1] * STORE_STAGES * (epilogue_warps // 4)
        if fc1_tma_store_enabled()
        else 0
    )


@cute.jit
def fc1_store_layout(warpgroups: cutlass.Constexpr[int] = 1):
    # SW64: stage and warpgroup strides span whole swizzle periods.
    return cute.make_composed_layout(
        cute.make_swizzle(2, 4, 3),
        0,
        cute.make_layout((32, 64, 2, warpgroups), stride=(64, 1, 2048, 4096)),
    )


@cute.jit
def issue_fc1_store(atom, tensor, stage, row, feature):
    """Called by one elected thread after all writers fence and synchronize."""
    destination = cute.local_tile(
        cute.domain_offset((row, feature), tensor), STORE_TILE, (0, 0)
    )
    source, destination = cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(stage, 0, 2),
        cute.group_modes(destination, 0, 2),
    )
    cute.copy(atom, source, destination)
    cute.arch.cp_async_bulk_commit_group()


@dsl_user_op
def store_native_e4m3(smem_ptr, byte_offset, r0, r1, r2, r3, *, loc=None, ip=None):
    """Store a warp's 32 tokens x 16 features with one transposing STSM.

    Each register packs two adjacent tokens at feature f, followed by the
    same tokens at feature f+8. Successive registers advance eight tokens.
    Each lane supplies its token row's logical 16-byte-aligned byte offset.
    """
    # Pointer.toint() drops the CuTe swizzle. Apply SW64 explicitly to the
    # shared address, including the control allocation preceding the ring.
    # SW64 XORs address bits 7:8 into bits 4:5.
    address = smem_ptr.toint(loc=loc, ip=ip) + byte_offset
    address = address ^ ((address >> 3) & 0x30)
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip)]
        + [cutlass.Uint32(r).ir_value(loc=loc, ip=ip) for r in (r0, r1, r2, r3)],
        "stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
