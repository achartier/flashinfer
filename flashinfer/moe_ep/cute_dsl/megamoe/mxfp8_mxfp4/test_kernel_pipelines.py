# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused storage/layout trace for persistent kernel pipelines."""

from types import SimpleNamespace

import pytest
import torch

cutlass = pytest.importorskip("cutlass")
import cutlass.cute as cute  # noqa: E402
from cuda.bindings import driver as cuda  # noqa: E402

from .device_epilogue import (  # noqa: E402
    DeviceEpilogueConfig,
    PersistentMxfp8Mxfp4Epilogue,
)
from .dispatch import Mxfp8DispatchPhase, SingleRankMxfp8DispatchPhase  # noqa: E402
from .kernel_pipelines import (  # noqa: E402
    KernelPipelines,
    KernelWarpIds,
    fit_operand_stages,
    mma_operand_stage_bytes,
)
from .scale_tma_staging import Mxfp8Mxfp4TmaStaging  # noqa: E402
from .workspace import WorkspaceConfig  # noqa: E402


def _make_pipelines(*, mma_m: int = 128, dispatch_start: int = 8):
    cluster_shape_mn = (2, 1) if mma_m == 256 else (1, 1)
    staging = Mxfp8Mxfp4TmaStaging(
        mma_tiler_mn=(mma_m, 128),
        cluster_shape_mn=cluster_shape_mn,
        num_stages=2,
    )
    workspace = WorkspaceConfig(
        world_size=1,
        num_topk=2,
        num_experts_per_rank=4,
        max_tokens_per_rank=64,
        hidden=128,
        intermediate=256,
        token_padding_block=64,
        sf_padding_block=128,
        cluster_tile_tokens=128,
    )
    dispatch = SingleRankMxfp8DispatchPhase(
        workspace,
        cluster_shape_mn=cluster_shape_mn,
        dispatch_warp_start=dispatch_start,
        num_other_warps=8,
    )
    epilogue = PersistentMxfp8Mxfp4Epilogue(
        DeviceEpilogueConfig(cta_tile_features=128, cta_tile_tokens=128)
    )
    # Storage shape depends only on these two scheduler fields.  Device-valued
    # scheduler parameters are supplied by the final driver before it calls
    # create_scheduler.
    scheduler_shape = SimpleNamespace(num_stages=2, load_balance_mode="static")
    return KernelPipelines(
        staging=staging,
        scheduler_params=scheduler_shape,
        dispatch_phase=dispatch,
        epilogue=epilogue,
    )


def test_pipeline_contract_is_m128_one_cta_block32() -> None:
    KernelPipelines.validate_mma_tiler_mn((128, 128))
    assert KernelWarpIds().dispatch == (8, 9, 10, 11)
    assert KernelWarpIds().threads_per_cta == 12 * 32


def test_pipeline_rejects_two_cta_mma_and_shifted_dispatch_warps() -> None:
    with pytest.raises(ValueError, match="M128"):
        KernelPipelines.validate_mma_tiler_mn((256, 128))
    with pytest.raises(ValueError, match="distinct"):
        KernelWarpIds(dispatch=(7, 9, 10, 11))


class _PipelineTraceHarness:
    """Trace allocation of the combined struct, operands, and both pipes."""

    @cute.jit
    def __call__(self, stream):
        self._kernel().launch(
            grid=(1, 1, 1),
            block=(12 * 32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def _kernel(self):
        # Layouts contain MLIR values and therefore must be created and
        # consumed wholly within this kernel region.  Storing ``pipelines`` on
        # the harness would leak composed layouts across CuTe region isolation.
        pipelines = _make_pipelines()
        stage_bytes = pipelines.staging.operand_stage_bytes()
        assert stage_bytes.weight_a == 128 * 128
        # The host-side stage fit uses this formula; keep it tied to layouts.
        assert stage_bytes == mma_operand_stage_bytes((128, 128, 128))
        assert pipelines.staging.tma_transaction_bytes_a() == 128 * 128 // 2 + 512
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(pipelines.shared_storage_type())
        pipelines.allocate_operand_smem(smem)
        pipelines.create_mainloop_pipelines(storage)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_b200_traces_shared_storage_and_pipeline_layout() -> None:
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        pytest.skip("requires an SM100 B200")

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    # Compilation traces every allocation and both pipeline constructors.  No
    # launch is required: execution behavior belongs to the final driver test.
    cute.compile(_PipelineTraceHarness(), stream)


# Shared-memory stage fit.  GB200/B200 allow 232448 dynamic bytes per CTA.
SM100_SMEM_CAPACITY = 232448


def _ep16_dispatch(hidden: int) -> Mxfp8DispatchPhase:
    """Dispatch phase of the EP16 multirank gate (2 experts/rank, top-k 2)."""

    warps = KernelWarpIds()
    workspace = WorkspaceConfig(
        world_size=16,
        num_topk=2,
        num_experts_per_rank=2,
        max_tokens_per_rank=7,
        hidden=hidden,
        intermediate=4992,
        token_padding_block=64,
        sf_padding_block=128,
        cluster_tile_tokens=128,
        load_balance_mode="static",
        token_back_by_dispatch=False,
        token_back_schedule_mode="static",
    )
    return Mxfp8DispatchPhase(
        workspace,
        rank=0,
        cluster_shape_mn=(1, 1),
        dispatch_warp_start=warps.dispatch[0],
        num_other_warps=len(warps.epilogue) + 4,
        flag_batch=2,
    )


def _fit(hidden: int, capacity: int, maximum_stages: int = 8) -> int:
    return fit_operand_stages(
        mma_tiler_mnk=(128, 128, 128),
        scheduler_params=SimpleNamespace(num_stages=2, load_balance_mode="static"),
        dispatch_phase=_ep16_dispatch(hidden),
        smem_capacity_bytes=capacity,
        maximum_stages=maximum_stages,
    ).stages


def test_stage_fit_reproduces_measured_gb200_allocation() -> None:
    # The DSL rejected six stages at this geometry with exactly 238080 bytes
    # allocated (GB200, EP16, H6656/I4992); the model must match to the byte.
    # Round 3 padded the epilogue exchange by 256 bytes (33-word lane rows)
    # and round 4 added two amax rows per pair (528 bytes, 512 after the
    # 128-byte rounding), so the same allocation is now 238848 bytes.
    assert _fit(6656, 238848, maximum_stages=6) == 6
    assert _fit(6656, 238847, maximum_stages=6) == 5
    assert _fit(6656, SM100_SMEM_CAPACITY) == 5
    # Dispatch storage scales with hidden size: six stages launch at H256.
    assert _fit(256, SM100_SMEM_CAPACITY) == 6


class _StageFitLaunchHarness:
    """Allocate exactly the persistent kernel's shared memory, then exit."""

    def __init__(self, stages: int, hidden: int):
        self.stages = stages
        self.hidden = hidden

    @cute.jit
    def __call__(self, stream):
        self._kernel().launch(
            grid=(1, 1, 1),
            block=(KernelWarpIds().threads_per_cta, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def _kernel(self):
        pipelines = KernelPipelines(
            staging=Mxfp8Mxfp4TmaStaging(
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(1, 1),
                num_stages=self.stages,
            ),
            scheduler_params=SimpleNamespace(num_stages=2, load_balance_mode="static"),
            dispatch_phase=_ep16_dispatch(self.hidden),
            epilogue=PersistentMxfp8Mxfp4Epilogue(
                DeviceEpilogueConfig(cta_tile_features=128, cta_tile_tokens=128)
            ),
        )
        smem = cutlass.utils.SmemAllocator()
        smem.allocate(pipelines.shared_storage_type())
        pipelines.allocate_operand_smem(smem)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_b200_stage_fit_boundary_matches_launch_limit() -> None:
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        pytest.skip("requires an SM100 B200/GB200")
    fitted = _fit(6656, SM100_SMEM_CAPACITY)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    # The fitted depth launches; one more stage exceeds the per-CTA limit.
    cute.compile(_StageFitLaunchHarness(fitted, 6656), stream)(stream)
    torch.cuda.synchronize()
    with pytest.raises(Exception, match="(?i)shared memory"):
        cute.compile(_StageFitLaunchHarness(fitted + 1, 6656), stream)(stream)
        torch.cuda.synchronize()
