"""Configuration and tactic schema for native MXFP8 x MXFP4 MegaMoE."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, TypedDict

from flashinfer.fused_moe.api import QuantConfig, QuantFormat


KERNEL_NAME = "sm100_mxfp8_mxfp4_bf16_cutedsl"


class Mxfp8Mxfp4Tactic(TypedDict, total=False):
    """Compile-time and scheduling knobs understood by the native frontend."""

    mma_tiler_mnk: tuple[int, int, int]
    cluster_shape_mnk: tuple[int, int, int]
    num_stages: int | Literal["auto"]
    group_hint: int | None
    flag_batch: int
    load_balance_mode: Literal["static", "atomic_counter"]
    token_back_mode: Literal["epi_warps", "reuse_dispatch_warps"]
    token_back_schedule_mode: Literal["static", "atomic_counter"]


_TACTIC_KEYS = frozenset(Mxfp8Mxfp4Tactic.__annotations__)


def validate_tactic(tactic: dict, *, partial: bool = True) -> None:
    """Validate a pinned tactic without importing CuTe DSL or CUDA modules."""

    unknown = set(tactic) - _TACTIC_KEYS
    if unknown:
        raise ValueError(f"unknown MXFP8 x MXFP4 tactic keys: {sorted(unknown)}")
    if not partial:
        missing = _TACTIC_KEYS - set(tactic)
        if missing:
            raise ValueError(
                f"incomplete MXFP8 x MXFP4 tactic: missing {sorted(missing)}"
            )

    tile = tactic.get("mma_tiler_mnk")
    if tile is not None:
        if not isinstance(tile, tuple) or len(tile) != 3:
            raise ValueError("mma_tiler_mnk must be a three-element tuple")
        m, n, k = tile
        if m != 128:
            raise ValueError(
                "initial MXFP8 x MXFP4 bring-up requires MMA M=128 (1-CTA)"
            )
        if n not in (64, 128):
            raise ValueError("mma_tiler_mnk N must be 64 or 128")
        if k <= 0 or k % 32:
            raise ValueError("mma_tiler_mnk K must be a positive multiple of 32")

    cluster = tactic.get("cluster_shape_mnk")
    if cluster is not None:
        if not isinstance(cluster, tuple) or len(cluster) != 3:
            raise ValueError("cluster_shape_mnk must be a three-element tuple")
        if cluster[2] != 1 or any(v not in (1, 2, 4) for v in cluster[:2]):
            raise ValueError("cluster_shape_mnk must be (1|2|4, 1|2|4, 1)")
        if cluster[0] * cluster[1] > 16:
            raise ValueError("cluster_shape_mnk may contain at most 16 CTAs")

    stages = tactic.get("num_stages")
    if (
        stages is not None
        and stages != "auto"
        and (not isinstance(stages, int) or not 2 <= stages <= 8)
    ):
        raise ValueError("num_stages must be 'auto' or an integer in [2, 8]")
    group_hint = tactic.get("group_hint")
    if group_hint is not None and (not isinstance(group_hint, int) or group_hint <= 0):
        raise ValueError("group_hint must be None or a positive integer")
    flag_batch = tactic.get("flag_batch")
    if flag_batch is not None and (
        not isinstance(flag_batch, int) or not 1 <= flag_batch <= 32
    ):
        raise ValueError("flag_batch must be an integer in [1, 32]")
    if tactic.get("load_balance_mode", "static") not in (
        "static",
        "atomic_counter",
    ):
        raise ValueError("load_balance_mode must be static or atomic_counter")
    if tactic.get("token_back_mode", "epi_warps") not in (
        "epi_warps",
        "reuse_dispatch_warps",
    ):
        raise ValueError("unsupported token_back_mode")
    if tactic.get("token_back_schedule_mode", "static") not in (
        "static",
        "atomic_counter",
    ):
        raise ValueError("unsupported token_back_schedule_mode")


def default_tactic(max_tokens_per_rank: int) -> Mxfp8Mxfp4Tactic:
    """Return the default launch tactic.

    Below 1024 tokens per rank, routed experts hold few tokens, so the N64
    token tile halves padded MMA and epilogue work.  ``group_hint=4096`` lets
    one scheduler group span about all local experts, removing the barrier
    between expert groups.  At EP16 on GB200 (H6656, 512 experts, top-8,
    T256-4096 global) the pair measured 1.35-1.60x faster than N128 with the
    SM-count group budget.  At 1024+ tokens per rank, N128 with
    ``group_hint=512`` measured 2-5% faster than N128 with 4096 (and N64
    lost), since smaller groups let FC2 overlap the next group's FC1.  N256
    and ``atomic_counter`` scheduling do not run in the persistent kernel yet.

    ``num_stages="auto"`` resolves at compile time to the deepest operand
    pipeline that fits shared memory for the actual geometry; pin an integer
    to override.  Offline tuning is expected to supersede the remaining fields
    through the knob cache.
    """

    if max_tokens_per_rank <= 0:
        raise ValueError("max_tokens_per_rank must be positive")
    large = max_tokens_per_rank >= 1024
    tactic: Mxfp8Mxfp4Tactic = {
        "mma_tiler_mnk": (128, 128 if large else 64, 128),
        "cluster_shape_mnk": (1, 1, 1),
        "num_stages": "auto",
        "group_hint": 512 if large else 4096,
        "flag_batch": 2,
        "load_balance_mode": "static",
        "token_back_mode": "epi_warps",
        "token_back_schedule_mode": "static",
    }
    validate_tactic(tactic, partial=False)
    return tactic


def candidate_tactics() -> list[Mxfp8Mxfp4Tactic]:
    """Bounded search over supported geometry and measured scheduling settings.

    GB200 round-2 experiments established N64 and larger expert groups as
    useful axes. Do not reintroduce N256 or atomic-counter scheduling here:
    the former is rejected by the assembly and the latter fails DSL lowering.
    Include auto depth so tuning cannot omit the shipped default (which fits
    more than four operand stages for the MAI geometry).
    """

    candidates: list[Mxfp8Mxfp4Tactic] = []
    for n in (64, 128):
        for stages in ("auto", 3, 4, 5):
            for group_hint in (512, 4096):
                tactic = default_tactic(1)
                tactic.update(
                    mma_tiler_mnk=(128, n, 128),
                    num_stages=stages,
                    group_hint=group_hint,
                )
                validate_tactic(tactic, partial=False)
                candidates.append(tactic)
    return candidates


def resolve_tactic(
    max_tokens_per_rank: int,
    overrides: Mxfp8Mxfp4Tactic | None,
) -> Mxfp8Mxfp4Tactic:
    """Merge a partial pinned tactic over the deterministic host default."""

    tactic = default_tactic(max_tokens_per_rank)
    if overrides is not None:
        validate_tactic(overrides)
        tactic.update(overrides)
    validate_tactic(tactic, partial=False)
    return tactic


def _mxfp8_mxfp4_quant_config() -> QuantConfig:
    return QuantConfig(
        weight=QuantFormat.MXFP4,
        activation=QuantFormat.MXFP8,
        output=QuantFormat.BF16,
        swizzled_scale_factors=True,
        per_token_scale=False,
    )


@dataclass
class Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig:
    """Public config for E4M3/K32 activations and E2M1/K32 weights."""

    intermediate_size: int
    top_k: int
    kernel_name: str = KERNEL_NAME
    quant: QuantConfig = field(default_factory=_mxfp8_mxfp4_quant_config)
    gate_up_clamp: float | None = None
    fast_math: bool = True
    knobs: Mxfp8Mxfp4Tactic | Literal["auto"] | None = None

    def __post_init__(self) -> None:
        if self.kernel_name != KERNEL_NAME:
            raise ValueError(f"kernel_name is fixed to {KERNEL_NAME!r}")
        if self.quant.pair != (QuantFormat.MXFP4, QuantFormat.MXFP8) or (
            self.quant.output is not QuantFormat.BF16
        ):
            raise ValueError(
                "MXFP8 x MXFP4 MegaMoE requires "
                "QuantConfig(weight=MXFP4, activation=MXFP8, output=BF16)"
            )
        if self.quant.swizzled_scale_factors is False:
            raise ValueError("MXFP4 kernel weights require swizzled scale factors")
        if self.quant.per_token_scale is True:
            raise ValueError("MXFP8 activations require E8M0 scales per K32 block")
        if self.intermediate_size <= 0 or self.intermediate_size % 128:
            raise ValueError("intermediate_size must be a positive multiple of 128")
        if self.top_k <= 0 or self.top_k > 32:
            raise ValueError("top_k must be in [1, 32]")
        if self.gate_up_clamp is not None and (
            not math.isfinite(self.gate_up_clamp) or self.gate_up_clamp <= 0
        ):
            raise ValueError("gate_up_clamp must be finite and positive")
        if isinstance(self.knobs, dict):
            validate_tactic(self.knobs)
        elif self.knobs not in (None, "auto"):
            raise TypeError("knobs must be a tactic dict, 'auto', or None")


__all__ = [
    "KERNEL_NAME",
    "Mxfp8Mxfp4Tactic",
    "Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig",
    "candidate_tactics",
    "default_tactic",
    "resolve_tactic",
    "validate_tactic",
]
