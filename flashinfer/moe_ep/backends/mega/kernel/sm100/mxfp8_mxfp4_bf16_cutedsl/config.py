"""Configuration and tactic schema for native MXFP8 x MXFP4 MegaMoE."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, TypedDict, cast

from flashinfer.fused_moe.api import QuantConfig, QuantFormat


KERNEL_NAME = "sm100_mxfp8_mxfp4_bf16_cutedsl"
# Same name as ``device_epilogue.EPILOGUE_WARPS_ENV`` (that module imports
# CuTe DSL, so the host config keeps its own copy). When set, it replaces the
# default epilogue width; explicit tactic overrides still win.
EPILOGUE_WARPS_ENV = "FLASHINFER_MXFP8_MXFP4_EPILOGUE_WARPS"
# The pair tile uses eight epilogue warps from this many tokens per rank.
EIGHT_WARP_EPILOGUE_MIN_TOKENS = 256
# When set to 0 or 1, replaces the default ``in_kernel_reduce`` band; explicit
# tactic overrides still win.
IN_KERNEL_REDUCE_ENV = "FLASHINFER_MXFP8_MXFP4_IN_KERNEL_REDUCE"
# The top-k reduction runs in the kernel tail below this many tokens per rank.
IN_KERNEL_REDUCE_MAX_TOKENS = 256
# Experimental per-owner fetch dedup; explicit tactics override the environment.
DISPATCH_DEDUP_ENV = "FLASHINFER_MXFP8_MXFP4_DISPATCH_DEDUP"
STARTUP_PREFETCH_WAVES_ENV = "FLASHINFER_MXFP8_MXFP4_STARTUP_PREFETCH_WAVES"
FC2_STORE_MODE_ENV = "FLASHINFER_MXFP8_MXFP4_FC2_STORE_MODE"
FC2_STORE_MODES = ("direct", "staged64", "staged256")
Fc2StoreMode = Literal["direct", "staged64", "staged256"]
LoadBalanceMode = Literal["static", "atomic_counter", "alternating"]
# When set (static, atomic_counter or alternating), replaces the default
# ``load_balance_mode`` band; explicit tactic overrides still win.
LOAD_BALANCE_MODE_ENV = "FLASHINFER_MXFP8_MXFP4_LOAD_BALANCE_MODE"
# The pair tile alternates FC1/FC2 claims in this tokens-per-rank band
# [min, max), the same band that selects the N192 token tile.
ALTERNATING_CLAIM_TOKENS = (512, 1024)
# The pair tile claims tiles atomically below this many tokens per rank.
ATOMIC_CLAIM_MAX_TOKENS = 256


class Mxfp8Mxfp4Tactic(TypedDict, total=False):
    """Compile-time and scheduling knobs understood by the native frontend."""

    mma_tiler_mnk: tuple[int, int, int]
    cluster_shape_mnk: tuple[int, int, int]
    num_stages: int | Literal["auto"]
    group_hint: int | None
    flag_batch: int
    load_balance_mode: LoadBalanceMode
    token_back_mode: Literal["epi_warps", "reuse_dispatch_warps"]
    token_back_schedule_mode: Literal["static", "atomic_counter"]
    epilogue_warps: Literal[4, 8]
    in_kernel_reduce: bool
    dispatch_dedup: bool
    fc2_store_mode: Fc2StoreMode
    startup_prefetch_waves: int
    vector_count_publish: bool
    cta_local_counts: bool
    overlap_dispatch_pull: bool


_TACTIC_KEYS = frozenset(Mxfp8Mxfp4Tactic.__annotations__)


def validate_tactic(tactic: Mapping[str, Any], *, partial: bool = True) -> None:
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
        if m not in (128, 256):
            raise ValueError(
                "MXFP8 x MXFP4 MMA M must be 128 (1-CTA) or 256 (2-CTA pair)"
            )
        if n not in (64, 128, 192):
            raise ValueError("mma_tiler_mnk N must be 64, 128, or 192")
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

    # A partial override may pin M256 alone; resolve_tactic supplies the
    # matching cluster before validating the complete tactic.
    if (
        tile is not None
        and tile[0] == 256
        and (cluster is not None or not partial)
        and cluster != (2, 1, 1)
    ):
        raise ValueError("MMA M=256 (2-CTA) requires cluster_shape_mnk=(2, 1, 1)")

    waves = tactic.get("startup_prefetch_waves", 0)
    if type(waves) is not int or not 0 <= waves <= 3:
        raise ValueError("startup_prefetch_waves must be an integer in [0, 3]")
    if waves and cluster is not None and cluster not in ((1, 1, 1), (2, 1, 1)):
        raise ValueError(
            "startup prefetch requires a single CTA or feature pair cluster"
        )
    if waves and tile is not None and cluster == (2, 1, 1) and tile[0] != 256:
        raise ValueError("startup prefetch with a pair cluster requires MMA M=256")

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
        "alternating",
        "static",
        "atomic_counter",
    ):
        raise ValueError(
            "load_balance_mode must be static, atomic_counter, or alternating"
        )
    if tactic.get("token_back_mode", "epi_warps") not in (
        "epi_warps",
        "reuse_dispatch_warps",
    ):
        raise ValueError(
            "token_back_mode must be epi_warps or reuse_dispatch_warps; "
            "standalone_warps is unsupported"
        )
    if tactic.get("token_back_schedule_mode", "static") not in (
        "static",
        "atomic_counter",
    ):
        raise ValueError("unsupported token_back_schedule_mode")
    if tactic.get("epilogue_warps", 4) not in (4, 8):
        raise ValueError("epilogue_warps must be 4 or 8")
    if tactic.get("fc2_store_mode", "direct") not in FC2_STORE_MODES:
        raise ValueError(f"fc2_store_mode must be one of {FC2_STORE_MODES}")
    for key in ("cta_local_counts", "overlap_dispatch_pull"):
        if not isinstance(tactic.get(key, False), bool):
            raise ValueError(f"{key} must be a bool")
    if tactic.get("overlap_dispatch_pull", False) and tactic.get(
        "dispatch_dedup", False
    ):
        raise ValueError("overlap_dispatch_pull cannot be combined with dispatch_dedup")
    if not isinstance(tactic.get("vector_count_publish", False), bool):
        raise ValueError("vector_count_publish must be a bool")
    if not isinstance(tactic.get("dispatch_dedup", False), bool):
        raise ValueError("dispatch_dedup must be a bool")
    if not isinstance(tactic.get("in_kernel_reduce", False), bool):
        raise ValueError("in_kernel_reduce must be a bool")


def _default_epilogue_warps(max_tokens_per_rank: int, pair: bool) -> Literal[4, 8]:
    value = os.environ.get(EPILOGUE_WARPS_ENV, "").strip()
    if value:
        if value not in ("4", "8"):
            raise ValueError(f"{EPILOGUE_WARPS_ENV} must be 4 or 8, got {value!r}")
        return 4 if value == "4" else 8
    return 8 if pair and max_tokens_per_rank >= EIGHT_WARP_EPILOGUE_MIN_TOKENS else 4


def supports_two_cta(hidden: int | None, intermediate: int | None) -> bool:
    """Whether the 2-CTA (M256) pair tile divides both GEMM feature extents."""

    return (
        hidden is not None
        and intermediate is not None
        and hidden % 256 == 0
        and (2 * intermediate) % 256 == 0
    )


def default_startup_prefetch_waves() -> int:
    value = os.environ.get(STARTUP_PREFETCH_WAVES_ENV, "0").strip() or "0"
    if value not in ("0", "1", "2", "3"):
        raise ValueError(f"{STARTUP_PREFETCH_WAVES_ENV} must be 0, 1, 2, or 3")
    return int(value)


def default_fc2_store_mode() -> Fc2StoreMode:
    value = os.environ.get(FC2_STORE_MODE_ENV, "staged256").strip() or "staged256"
    if value not in FC2_STORE_MODES:
        raise ValueError(f"{FC2_STORE_MODE_ENV} must be one of {FC2_STORE_MODES}")
    return cast(Fc2StoreMode, value)


def _default_dispatch_dedup() -> bool:
    value = os.environ.get(DISPATCH_DEDUP_ENV, "0").strip() or "0"
    if value not in ("0", "1"):
        raise ValueError(f"{DISPATCH_DEDUP_ENV} must be 0 or 1, got {value!r}")
    return value == "1"


def _default_in_kernel_reduce(max_tokens_per_rank: int) -> bool:
    value = os.environ.get(IN_KERNEL_REDUCE_ENV, "").strip()
    if value:
        if value not in ("0", "1"):
            raise ValueError(f"{IN_KERNEL_REDUCE_ENV} must be 0 or 1, got {value!r}")
        return value == "1"
    return max_tokens_per_rank < IN_KERNEL_REDUCE_MAX_TOKENS


def _default_load_balance_mode(max_tokens_per_rank: int, pair: bool) -> LoadBalanceMode:
    value = os.environ.get(LOAD_BALANCE_MODE_ENV, "").strip()
    if value:
        if value not in ("static", "atomic_counter", "alternating"):
            raise ValueError(
                f"{LOAD_BALANCE_MODE_ENV} must be static, atomic_counter or "
                f"alternating, got {value!r}"
            )
        return cast(LoadBalanceMode, value)
    if not pair:
        return "static"
    if max_tokens_per_rank < ATOMIC_CLAIM_MAX_TOKENS:
        return "atomic_counter"
    low, high = ALTERNATING_CLAIM_TOKENS
    return "alternating" if low <= max_tokens_per_rank < high else "static"


def default_tactic(
    max_tokens_per_rank: int,
    *,
    hidden: int | None = None,
    intermediate: int | None = None,
) -> Mxfp8Mxfp4Tactic:
    """Return a deterministic launch tactic for the supplied workload.

    Token tiles and scheduler groups trade padding and repeated weight loads
    against shared-memory use. Automatic stage depth selects the deepest
    operand pipeline that fits. Offline tuning may override these defaults."""

    if max_tokens_per_rank <= 0:
        raise ValueError("max_tokens_per_rank must be positive")
    small_groups = max_tokens_per_rank >= 512
    pair = supports_two_cta(hidden, intermediate)
    token_tile = 64
    if max_tokens_per_rank >= 256:
        token_tile = 128
    if pair and 512 <= max_tokens_per_rank < 1024:
        token_tile = 192
    tactic: Mxfp8Mxfp4Tactic = {
        "mma_tiler_mnk": (256 if pair else 128, token_tile, 128),
        "cluster_shape_mnk": (2, 1, 1) if pair else (1, 1, 1),
        "num_stages": "auto",
        "group_hint": 512 if small_groups else 4096,
        "flag_batch": 2,
        "load_balance_mode": _default_load_balance_mode(max_tokens_per_rank, pair),
        "token_back_mode": "epi_warps",
        "token_back_schedule_mode": "static",
        "epilogue_warps": _default_epilogue_warps(max_tokens_per_rank, pair),
        "in_kernel_reduce": _default_in_kernel_reduce(max_tokens_per_rank),
        "dispatch_dedup": _default_dispatch_dedup(),
        "fc2_store_mode": default_fc2_store_mode(),
        "startup_prefetch_waves": default_startup_prefetch_waves(),
        "vector_count_publish": False,
        "cta_local_counts": False,
        "overlap_dispatch_pull": False,
    }
    validate_tactic(tactic, partial=False)
    return tactic


def candidate_tactics(
    *, hidden: int | None = None, intermediate: int | None = None
) -> list[Mxfp8Mxfp4Tactic]:
    """Enumerate supported launch tactics for offline tuning.

    Search tile widths, stage depth, and scheduling choices while including
    the default tactic. Shape-dependent candidates must satisfy the kernel
    alignment and shared-memory constraints."""

    epilogue_warps: Literal[4, 8]
    candidates: list[Mxfp8Mxfp4Tactic] = []
    if supports_two_cta(hidden, intermediate):
        for n in (64, 128, 192):
            for group_hint in (512, 4096):
                for epilogue_warps in (4, 8):
                    for in_kernel_reduce in (False, True):
                        tactic = default_tactic(
                            1, hidden=hidden, intermediate=intermediate
                        )
                        tactic["mma_tiler_mnk"] = (256, n, 128)
                        tactic["group_hint"] = group_hint
                        tactic["epilogue_warps"] = cast(Literal[4, 8], epilogue_warps)
                        tactic["in_kernel_reduce"] = in_kernel_reduce
                        tactic["load_balance_mode"] = "static"
                        validate_tactic(tactic, partial=False)
                        candidates.append(tactic)
        for epilogue_warps in (4, 8):
            for in_kernel_reduce in (False, True):
                tactic = default_tactic(1, hidden=hidden, intermediate=intermediate)
                tactic["mma_tiler_mnk"] = (256, 192, 128)
                tactic["group_hint"] = 512
                tactic["epilogue_warps"] = cast(Literal[4, 8], epilogue_warps)
                tactic["in_kernel_reduce"] = in_kernel_reduce
                tactic["load_balance_mode"] = "alternating"
                validate_tactic(tactic, partial=False)
                candidates.append(tactic)
                # The atomic-claim default below 256 tokens per rank.
                tactic = tactic.copy()
                tactic["mma_tiler_mnk"] = (256, 64, 128)
                tactic["group_hint"] = 4096
                tactic["load_balance_mode"] = "atomic_counter"
                validate_tactic(tactic, partial=False)
                candidates.append(tactic)
        return candidates
    for n in (64, 128):
        for stages in ("auto", 3, 4, 5):
            for group_hint in (512, 4096):
                for in_kernel_reduce in (False, True):
                    tactic = default_tactic(1)
                    tactic["mma_tiler_mnk"] = (128, n, 128)
                    tactic["num_stages"] = stages
                    tactic["group_hint"] = group_hint
                    tactic["in_kernel_reduce"] = in_kernel_reduce
                    validate_tactic(tactic, partial=False)
                    candidates.append(tactic)
    return candidates


def resolve_tactic(
    max_tokens_per_rank: int,
    overrides: Mxfp8Mxfp4Tactic | None,
    *,
    hidden: int | None = None,
    intermediate: int | None = None,
) -> Mxfp8Mxfp4Tactic:
    """Merge a partial pinned tactic over the deterministic host default.

    An override that pins the MMA M without a cluster shape gets the cluster
    that M implies ((1, 1, 1) for M128, (2, 1, 1) for M256), so pinning a
    1-CTA tile never inherits the default's 2-CTA cluster.
    """

    tactic = default_tactic(
        max_tokens_per_rank, hidden=hidden, intermediate=intermediate
    )
    if overrides is not None:
        validate_tactic(overrides)
        tactic.update(overrides)
        if "mma_tiler_mnk" in overrides and "cluster_shape_mnk" not in overrides:
            tactic["cluster_shape_mnk"] = (
                (2, 1, 1) if overrides["mma_tiler_mnk"][0] == 256 else (1, 1, 1)
            )
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
    "DISPATCH_DEDUP_ENV",
    "EIGHT_WARP_EPILOGUE_MIN_TOKENS",
    "EPILOGUE_WARPS_ENV",
    "ALTERNATING_CLAIM_TOKENS",
    "ATOMIC_CLAIM_MAX_TOKENS",
    "IN_KERNEL_REDUCE_ENV",
    "IN_KERNEL_REDUCE_MAX_TOKENS",
    "LOAD_BALANCE_MODE_ENV",
    "KERNEL_NAME",
    "Mxfp8Mxfp4Tactic",
    "Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig",
    "candidate_tactics",
    "default_tactic",
    "supports_two_cta",
    "resolve_tactic",
    "validate_tactic",
]
