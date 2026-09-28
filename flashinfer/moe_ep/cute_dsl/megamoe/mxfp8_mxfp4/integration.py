# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host integration boundary for the MXFP8 x MXFP4 persistent MegaMoE.

This module deliberately lives outside ``kernel_src/sm100/cutedsl_megamoe``: the
mixed-format kernel is FlashInfer-owned and must not become part of the
vendored source drop.  It owns workspace lifetime and translates the backend
ABI into the component frontend's stable launch specification.

The actual device assembler is imported lazily from ``persistent_kernel`` so
that config discovery and CPU-only tests do not import CuTe DSL or initialize
CUDA.  Compilation and launch binding remain owned by that adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from typing import Any

import torch

from .frontend import FrontendConfig, KernelLaunchSpec, Mxfp8Mxfp4Frontend
from .workspace import WorkspaceConfig, WorkspacePlan, make_workspace_plan, round_up


def _tactic_key(tactic: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Return a deterministic, hashable representation of a tactic dict."""

    return tuple(sorted(tactic.items()))


@dataclass
class Mxfp8Mxfp4MegaMoeWorkspace:
    """NVLink workspace and lazy-compile state.

    ``num_experts`` is global; weights and compute use ``num_local_experts``.
    Distributed allocation and destruction are collective on the EP world.
    One symmetric root contains every remotely accessed tensor, so a single
    peer offset table maps inputs, control words, and output route slots.
    """

    num_experts: int
    max_tokens_per_rank: int
    num_topk: int
    hidden: int
    intermediate: int
    rank: int
    world_size: int
    tactic: dict[str, Any]
    plan: WorkspacePlan
    x: torch.Tensor
    x_sf: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    route_terms: torch.Tensor
    y: torch.Tensor
    workspace_bytes: torch.Tensor
    shared_workspace_bytes: torch.Tensor
    symmetric_base: int = 0
    peer_offsets_list: tuple[int, ...] = (0,)
    _symmetric_root: torch.Tensor | None = field(default=None, repr=False)
    _frontends: dict[tuple[Any, ...], Mxfp8Mxfp4Frontend] = field(
        default_factory=dict, init=False, repr=False
    )
    _destroyed: bool = field(default=False, init=False, repr=False)

    @property
    def num_local_experts(self) -> int:
        return self.num_experts // self.world_size

    @property
    def workspace_regions(self) -> dict[str, torch.Tensor]:
        return {
            "local": self.workspace_bytes,
            "shared": self.shared_workspace_bytes,
        }

    def frontend(
        self, *, gate_up_clamp: float | None, fast_math: bool
    ) -> Mxfp8Mxfp4Frontend:
        if self._destroyed:
            raise RuntimeError("MXFP8 x MXFP4 MegaMoE workspace is destroyed")
        key = (gate_up_clamp, bool(fast_math), _tactic_key(self.tactic))
        frontend = self._frontends.get(key)
        if frontend is None:
            from .persistent_kernel import PersistentMxfp8Mxfp4KernelAdapter

            static_tactic = (
                *key[2],
                ("gate_up_clamp", gate_up_clamp),
                ("fast_math", bool(fast_math)),
            )
            config = FrontendConfig(
                hidden=self.hidden,
                intermediate=self.intermediate,
                topk=self.num_topk,
                num_experts=self.num_local_experts,
                max_tokens=self.max_tokens_per_rank,
                tactic=static_tactic,
            )
            frontend = Mxfp8Mxfp4Frontend(
                config,
                PersistentMxfp8Mxfp4KernelAdapter(
                    workspace=self,
                    tactic=self.tactic,
                    gate_up_clamp=gate_up_clamp,
                    fast_math=fast_math,
                ),
            )
            self._frontends[key] = frontend
        return frontend

    def destroy(self) -> None:
        """Collectively release after all launches/graphs using this workspace.

        Call on every EP rank before NVSHMEM or torch.distributed teardown.
        Captured graphs must not be replayed after destruction.
        """
        if self._destroyed:
            return
        from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.comm import (
            ensure_not_capturing,
            free_sym_tensor,
        )

        ensure_not_capturing("MXFP8 x MXFP4 workspace destruction")
        if self._symmetric_root is not None:
            import torch.distributed as dist

            torch.cuda.synchronize(self.x.device)
            dist.barrier()
        for frontend in self._frontends.values():
            frontend.invalidate()
        self._frontends.clear()
        if self._symmetric_root is not None:
            free_sym_tensor(self._symmetric_root)
            self._symmetric_root = None
        self._destroyed = True


def _validate_ep_geometry(num_experts: int, rank: int, world_size: int) -> None:
    if not 1 <= world_size <= 16:
        raise ValueError("NVLink mega supports 1 <= world_size <= 16")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    if num_experts < 1 or num_experts % world_size:
        raise ValueError(
            "global num_experts must be positive and divisible by world_size"
        )


def _symmetric_regions(
    max_tokens: int, hidden: int, topk: int, shared_bytes: int
) -> tuple[dict[str, tuple[int, tuple[int, ...], torch.dtype]], int]:
    """Byte offsets for the one-allocation peer mapping ABI (128B aligned)."""
    specs = (
        ("x", (max_tokens, hidden), torch.float8_e4m3fn, 1),
        ("x_sf", (max_tokens, round_up(hidden // 32, 4)), torch.float8_e8m0fnu, 1),
        ("topk_idx", (max_tokens, topk), torch.int64, 8),
        ("topk_weights", (max_tokens, topk), torch.float32, 4),
        ("route_terms", (max_tokens, topk, hidden), torch.bfloat16, 2),
        ("shared_workspace_bytes", (shared_bytes,), torch.uint8, 1),
    )
    regions = {}
    cursor = 0
    for name, shape, dtype, itemsize in specs:
        cursor = round_up(cursor, 128)
        regions[name] = (cursor, shape, dtype)
        cursor += math.prod(shape) * itemsize
    return regions, round_up(cursor, 128)


def _allocate_peer_regions(
    *,
    rank: int,
    world_size: int,
    geometry: tuple[Any, ...],
    regions: dict[str, tuple[int, tuple[int, ...], torch.dtype]],
    total_bytes: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int, tuple[int, ...]]:
    """Collective allocation; reject missing direct peer mappings before JIT."""
    import torch.distributed as dist
    import nvshmem.core

    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.comm import (
        _compute_peer_offsets,
        free_sym_tensor,
        sym_zeros,
    )

    if int(os.environ.get("MEGA_NO_DIST", "0")):
        raise ValueError("MEGA_NO_DIST cannot be used for multi-rank mega")
    if not dist.is_initialized():
        raise RuntimeError(
            "initialize the EP distributed/NVSHMEM runtime before allocation"
        )
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise ValueError("workspace rank/world must match the bootstrapped EP world")
    if int(nvshmem.core.my_pe()) != rank or int(nvshmem.core.n_pes()) != world_size:
        raise ValueError("NVSHMEM rank/world must match the EP world")
    configs: list[Any] = [None] * world_size
    dist.all_gather_object(configs, geometry)
    if any(config != geometry for config in configs):
        raise ValueError(
            "all EP ranks must allocate the same workspace geometry/tactic"
        )
    root = sym_zeros((total_bytes,), torch.uint8)
    error = None
    base, offsets = 0, ()
    try:
        base, offsets = _compute_peer_offsets(root, world_size)
        if offsets[rank] != 0 or any(base + delta <= 0 for delta in offsets):
            raise RuntimeError("invalid direct peer pointer mapping")
    except (RuntimeError, ValueError, TypeError) as exc:
        error = str(exc)
    errors: list[Any] = [None] * world_size
    dist.all_gather_object(errors, error)
    if any(item is not None for item in errors):
        free_sym_tensor(root)
        raise RuntimeError(
            "mega requires direct GPU peer access across the entire EP world; "
            f"use split mode for IB-only peers. Peer mapping errors: {errors}"
        )
    views = {}
    for name, (offset, shape, dtype) in regions.items():
        size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        views[name] = root.narrow(0, offset, size).view(dtype).reshape(shape)
    # No rank may start publishing dispatch counters until every peer's initial
    # zero-fill has completed. This is allocation-time only, never graph capture.
    torch.cuda.synchronize(root.device)
    dist.barrier()
    return views, root, base, offsets


def get_symm_buffer_for_mxfp8_mxfp4_mega_moe(
    num_experts: int,
    max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate: int,
    rank: int,
    world_size: int,
    *,
    gate_up_clamp: float | None = None,
    knobs: dict[str, Any] | None = None,
) -> Mxfp8Mxfp4MegaMoeWorkspace:
    """Allocate the mixed-format workspace without importing the device kernel."""

    del gate_up_clamp
    _validate_ep_geometry(num_experts, rank, world_size)
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.comm import (
        ensure_not_capturing,
    )

    ensure_not_capturing("MXFP8 x MXFP4 workspace allocation")
    from ....backends.mega.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl.config import (
        resolve_tactic,
    )

    tactic = dict(resolve_tactic(max_tokens_per_rank, knobs))
    cluster_tokens = tactic["mma_tiler_mnk"][1] * tactic["cluster_shape_mnk"][1]
    workspace_config = WorkspaceConfig(
        world_size=world_size,
        num_topk=num_topk,
        num_experts_per_rank=num_experts // world_size,
        max_tokens_per_rank=max_tokens_per_rank,
        hidden=hidden,
        intermediate=intermediate,
        token_padding_block=64,
        sf_padding_block=128,
        cluster_tile_tokens=cluster_tokens,
        load_balance_mode=tactic["load_balance_mode"],
        token_back_by_dispatch=False,
        token_back_schedule_mode=tactic["token_back_schedule_mode"],
    )
    plan = make_workspace_plan(workspace_config)
    device = torch.device("cuda", torch.cuda.current_device())
    sf_cols = round_up(hidden // 32, 4)
    peer_views = {}
    symmetric_root = None
    symmetric_base, peer_offsets = 0, (0,)
    if world_size > 1:
        regions, total_bytes = _symmetric_regions(
            max_tokens_per_rank, hidden, num_topk, plan.shared.total_bytes
        )
        peer_views, symmetric_root, symmetric_base, peer_offsets = (
            _allocate_peer_regions(
                rank=rank,
                world_size=world_size,
                geometry=(
                    num_experts,
                    max_tokens_per_rank,
                    num_topk,
                    hidden,
                    intermediate,
                    _tactic_key(tactic),
                ),
                regions=regions,
                total_bytes=total_bytes,
            )
        )
        peer_views["topk_idx"].fill_(-1)

    def allocate(name, shape, dtype, *, fill=None):
        if name in peer_views:
            return peer_views[name]
        result = torch.empty(shape, dtype=dtype, device=device)
        if fill is not None:
            result.fill_(fill)
        return result

    return Mxfp8Mxfp4MegaMoeWorkspace(
        num_experts=num_experts,
        max_tokens_per_rank=max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        intermediate=intermediate,
        rank=rank,
        world_size=world_size,
        tactic=tactic,
        plan=plan,
        x=allocate("x", (max_tokens_per_rank, hidden), torch.float8_e4m3fn),
        x_sf=allocate("x_sf", (max_tokens_per_rank, sf_cols), torch.float8_e8m0fnu),
        topk_idx=allocate(
            "topk_idx", (max_tokens_per_rank, num_topk), torch.int64, fill=-1
        ),
        topk_weights=allocate(
            "topk_weights", (max_tokens_per_rank, num_topk), torch.float32
        ),
        route_terms=allocate(
            "route_terms",
            (max_tokens_per_rank, num_topk, hidden),
            dtype=torch.bfloat16,
            fill=0,
        ),
        y=torch.empty(
            (max_tokens_per_rank, hidden), dtype=torch.bfloat16, device=device
        ),
        # Token communication has persistent phase/signal words outside the
        # per-launch zero prefixes.  They must start from zero once when the
        # workspace is created; subsequent graph-safe launches reset only the
        # declared prefixes in the adapter.
        workspace_bytes=torch.zeros(
            plan.local.total_bytes, dtype=torch.uint8, device=device
        ),
        shared_workspace_bytes=allocate(
            "shared_workspace_bytes", (plan.shared.total_bytes,), torch.uint8, fill=0
        ),
        symmetric_base=symmetric_base,
        peer_offsets_list=peer_offsets,
        _symmetric_root=symmetric_root,
    )


def mxfp8_mxfp4_mega_moe(
    output: torch.Tensor | None,
    fc1: tuple[torch.Tensor, torch.Tensor],
    fc2: tuple[torch.Tensor, torch.Tensor],
    workspace: Mxfp8Mxfp4MegaMoeWorkspace,
    *,
    num_tokens: int,
    gate_up_clamp: float | None = None,
    fast_math: bool = True,
) -> torch.Tensor:
    """Compile/bind lazily and launch the persistent mixed-format kernel."""

    if not 0 <= num_tokens <= workspace.max_tokens_per_rank:
        raise ValueError("num_tokens exceeds the allocated workspace")
    destination = workspace.y if output is None else output
    frontend = workspace.frontend(
        gate_up_clamp=gate_up_clamp,
        fast_math=fast_math,
    )
    spec = KernelLaunchSpec(
        activation=workspace.x,
        activation_scales=workspace.x_sf,
        fc1_weight=fc1[0],
        fc1_weight_scales=fc1[1],
        fc2_weight=fc2[0],
        fc2_weight_scales=fc2[1],
        route_ids=workspace.topk_idx,
        route_scores=workspace.topk_weights,
        output=destination,
        workspace_regions={
            **workspace.workspace_regions,
            "route_terms": workspace.route_terms,
        },
        num_tokens=num_tokens,
        stream=torch.cuda.current_stream().cuda_stream,
        static_tactic=frontend.config.tactic,
    )
    return frontend.run(spec)


def autotune_mxfp8_mxfp4_mega_moe(*args, **kwargs) -> None:
    """Autotuning hook reserved for the post-correctness B200 tuning pass."""

    raise NotImplementedError(
        "MXFP8 x MXFP4 persistent-kernel autotuning is deferred until the "
        "single-B200 correctness gate passes; pin knobs for bring-up"
    )


__all__ = [
    "Mxfp8Mxfp4MegaMoeWorkspace",
    "autotune_mxfp8_mxfp4_mega_moe",
    "get_symm_buffer_for_mxfp8_mxfp4_mega_moe",
    "mxfp8_mxfp4_mega_moe",
]
