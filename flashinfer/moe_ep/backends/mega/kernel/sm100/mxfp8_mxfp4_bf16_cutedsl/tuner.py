"""Offline tactic sweep for native MXFP8 x MXFP4 MegaMoE."""

from __future__ import annotations

from typing import Any

from ...tuning import finish_sweep, run_tuning as _run_tuning, schedule_candidates
from .config import candidate_tactics, default_tactic, validate_tactic


def tune_one(args, rank: int, world_size: int, max_tokens: int) -> dict:
    """Run one max-token bucket; invoked by the shared future CLI wiring."""

    from ......cute_dsl.megamoe.mxfp8_mxfp4.integration import (
        autotune_mxfp8_mxfp4_mega_moe,
        create_dummy_mxfp8_mxfp4_inputs,
    )

    live_tokens = args.live_tokens if args.live_tokens is not None else max_tokens
    if live_tokens > max_tokens:
        raise SystemExit("--live-tokens must be <= --max-tokens")
    symm_buffer: Any = None
    try:
        y, fc1, fc2, symm_buffer = create_dummy_mxfp8_mxfp4_inputs(
            rank,
            world_size,
            args.num_experts,
            max_tokens,
            live_tokens,
            args.topk,
            args.hidden,
            args.intermediate,
            gate_up_clamp=args.gate_up_clamp,
            seed=args.seed,
        )
        candidates = candidate_tactics()
        if args.sweep == "schedule":
            import json

            if args.base_knobs:
                base = json.loads(args.base_knobs)
                base = {
                    key: tuple(value) if isinstance(value, list) else value
                    for key, value in base.items()
                }
                validate_tactic(base)
                base = {**default_tactic(max_tokens), **base}
            else:
                base = default_tactic(max_tokens)
            candidates = schedule_candidates(base)

        return finish_sweep(
            args,
            rank,
            max_tokens,
            live_tokens,
            symm_buffer,
            y,
            fc1,
            fc2,
            candidates,
            autotune_mxfp8_mxfp4_mega_moe,
        )
    finally:
        if symm_buffer is not None:
            symm_buffer.destroy()


def run_tuning(args) -> int:
    return _run_tuning(args, tune_one)


__all__ = ["run_tuning", "tune_one"]
