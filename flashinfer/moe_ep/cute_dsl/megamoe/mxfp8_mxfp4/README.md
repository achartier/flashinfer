# MXFP8 × MXFP4 MegaMoE

This Blackwell CuTe DSL kernel combines expert dispatch, FC1, SwiGLU,
FC2, token return, and deterministic top-k reduction. Activations use E4M3
with E8M0 scales per 32 values; weights use packed E2M1 with E8M0 scales
per 32 values. The output is BF16. The native kernel supports both
single-CTA and paired-CTA MMA, and the lean FC12 path shares its compute
implementation with MegaMoE.

The MoE EP backend is `sm100_mxfp8_mxfp4_bf16_cutedsl`. Its configuration,
weight preprocessing, and offline tuner live in
[`backends/mega/kernel/sm100/mxfp8_mxfp4_bf16_cutedsl`](../../../backends/mega/kernel/sm100/mxfp8_mxfp4_bf16_cutedsl).
Use a supported Blackwell CUDA/CuTe DSL installation. Multi-rank MegaMoE
requires the MoE EP symmetric-memory setup and peer-accessible NVLink GPUs;
all participating ranks must use the same configuration and collective order.

## Input and lifecycle contract

`hidden` and the post-SwiGLU `intermediate_size` must be positive multiples
of 128. `top_k` is in `[1, 32]` and cannot exceed the number of experts.
Experts divide evenly among the EP ranks; native dispatch supports 1–16
ranks. Symbolically, BF16 source weights have shapes
`w13=[local_experts, 2 * intermediate, hidden]` and
`w2=[local_experts, hidden, intermediate]`. Use the backend's
`preprocess_mega_weights` to obtain its packed and swizzled representation.
FC1 gate/up rows are interleaved in groups of eight. Passing an arbitrary
packed FP4 tensor without the corresponding scale layout is unsupported.

The staging path accepts BF16 inputs for quantization, or prequantized E4M3
inputs with E8M0/K32 scales. Routing IDs are int64 and routing weights are
FP32. Each FC2 route is rounded to BF16, weighted in FP32, and reduced in
fixed top-k order before the final BF16 conversion.

Allocate the workspace, preprocess weights, stage inputs, tune, and run an
eager warmup before CUDA graph capture. Keep workspace and tensor addresses
alive through replay. A tactic change can alter workspace offsets and the
compiled kernel, so apply it outside capture and warm up again. Destroy the
workspace only after outstanding GPU work completes, collectively for EP.

## Selecting a tactic

Set `knobs` on `Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig`:

| Value | Behavior |
|---|---|
| `None` | Look up the persistent MoE EP knob cache, then fall back to the default tactic. |
| `{}` | Bypass the knob cache and use the default tactic. |
| A partial tactic dictionary | Bypass the knob cache and merge the supplied fields over defaults. |
| `"auto"` | Run the collective candidate sweep on the first eager compute and retain the winner. |

Explicit tactic fields override environment-derived defaults. Environment
values must still be valid because defaults are resolved before overrides.
Treat environment switches as compile-time settings: use a fresh process and
workspace when comparing them. `gate_up_clamp` optionally supplies a finite,
positive SwiGLU clamp; `fast_math` defaults to `True`. Keep these numerical
settings fixed while comparing tactics.

### Default policy

MMA M is 256 with cluster `(2, 1, 1)` when `hidden % 256 == 0` and
`(2 * intermediate) % 256 == 0`; otherwise M is 128 with cluster `(1, 1, 1)`.
The following table uses **workspace capacity in tokens per rank**, rather
than the live token count or the global batch size.

| Capacity | Token tile N | Group hint | Pair-tile claim mode | Pair-tile epilogue warps | In-kernel reduction |
|---|---|---|---|---|---|
| Below 256 | 64 | 4096 | `atomic_counter` | 4 | On |
| 256–511 | 128 | 4096 | `static` | 8 | Off |
| 512–1023 | 192 for pairs, 128 otherwise | 512 | `alternating` | 8 | Off |
| 1024 and above | 128 | 512 | `static` | 8 | Off |

Single-CTA defaults always use static claims and four epilogue warps.
All defaults use K128, automatic operand stages, `flag_batch=2`,
`token_back_mode="epi_warps"`, static token-back claims, and
`fc2_store_mode="staged256"`. Startup prefetch is disabled, and the four
boolean dispatch/startup switches below are off. Tune against representative
routing and live token counts to replace these heuristics.

### Tactic dictionary reference

| Knob | Values and effect |
|---|---|
| `mma_tiler_mnk` | `(M, N, K)`; M128 or M256, N64/N128/N192. M is output features, N is routed tokens. The host schema accepts positive K multiples of 32; use K128 for the current device implementation. N256 is unsupported. |
| `cluster_shape_mnk` | M256 requires `(2, 1, 1)`. The host schema allows cluster axes 1/2/4, a final axis of 1, and at most 16 CTAs; device assembly enforces additional layout constraints. Start with the shipped single-CTA or paired-CTA configurations. Pinning M without a cluster automatically selects the matching cluster. |
| `num_stages` | `"auto"` or 2–8. Auto fits the deepest operand ring into shared memory, accounting for dispatch, epilogues, and optional stores. Auto can select up to 12 stages for pairs or 8 for single-CTA tiles; a pinned value still must fit. |
| `group_hint` | Positive integer or `None`. Controls expert grouping in the static work order; smaller groups can expose FC2 earlier. Alternating claims ignore it. |
| `flag_batch` | 1–32. Number of dispatched arrivals batched before publishing readiness. |
| `load_balance_mode` | `static`: fixed tile assignment; `atomic_counter`: dynamic claims over the same linear order; `alternating`: independent FC1/FC2 counters with FC1 warmup and dependency-aware FC2 claims. |
| `token_back_mode` | `epi_warps` sends FC2 rows from epilogue warps; `reuse_dispatch_warps` stages local rows and reuses dispatch warps for return. Both retain deterministic reduction. `standalone_warps` is unsupported. |
| `token_back_schedule_mode` | `static` or `atomic_counter`; affects return claims only with `reuse_dispatch_warps`. |
| `epilogue_warps` | 4 or 8. Eight splits token groups across two warpgroups, increasing threads and shared storage and potentially reducing operand stages. |
| `in_kernel_reduce` | Boolean. Reduce top-k in the persistent kernel tail instead of launching the standalone reducer. Mega only; reduction order is unchanged. |
| `fc2_store_mode` | `direct`: register transpose; `staged64`: warp-local shared transpose and 64-byte row segments; `staged256`: warpgroup shared transpose and 256-byte segments. Staged modes alias FC1 exchange storage and participate in stage fitting. |
| `dispatch_dedup` | Boolean, default `False`. Fetch each source token once per owner rank and fan out unchanged activation/scale bytes to local routes. Route metadata and arrivals remain independent. Mega only. |
| `startup_prefetch_waves` | Integer 0–3, default 0. Prefetch an FC1 weight/scale prefix during startup. Whole waves are capped at 80% of L2 and one pass over unique tiles. Requires single-CTA or M256 pair geometry; Mega only. |
| `vector_count_publish` | Boolean, default `False`. Publish adjacent expert counts with 16-byte stores when experts per rank is even; retain scalar stores otherwise. |
| `cta_local_counts` | Boolean, default `False`. Each CTA acquires startup signals and computes owner counts into reusable shared scratch, removing the closing startup grid rendezvous. CTA 0 retains a global diagnostic copy. |
| `overlap_dispatch_pull` | Boolean, default `False`. Select the next route during a row load and overlap completion of the preceding pool store with the next load. Arrival publication still follows store completion. Cannot be combined with `dispatch_dedup`. |

Dispatch and startup options affect Mega execution, not lean/split FC12.
Use `validate_tactic` for host validation and `resolve_tactic` to inspect the
complete tactic. Passing host validation alone does not guarantee that a
configuration fits shared memory or is accepted by the device assembler.

### Environment switches

In the next table, append each suffix to `FLASHINFER_MXFP8_MXFP4_`.

| Suffix | Default | Purpose |
|---|---|---|
| `EPILOGUE_WARPS` | Tactic policy | `4` or `8`; replaces the default width. |
| `IN_KERNEL_REDUCE` | Tactic policy | `0` or `1`; replaces the default reduction placement. |
| `LOAD_BALANCE_MODE` | Tactic policy | `static`, `atomic_counter`, or `alternating`. |
| `FC2_STORE_MODE` | `staged256` | `direct`, `staged64`, or `staged256`. |
| `DISPATCH_DEDUP` | `0` | `0` or `1`; replaces the dedup tactic default. |
| `STARTUP_PREFETCH_WAVES` | `0` | `0`, `1`, `2`, or `3`; replaces the prefetch tactic default. |
| `DYNAMIC_N` | `1` | `0` issues static full-N MMA; otherwise issue valid token columns rounded up to 16. |
| `E1_SKIP` | `fc1` | Skip empty 32-token epilogue groups in `none`, `fc1`, `fc2`, or `both` phases. |
| `FC1_EPI_LEGACY` | `0` | `1` selects the diagnostic register-transpose FC1 path; otherwise use native TMEM loads. |
| `FC1_TMA_STORE` | `0` | `1` enables shared staging/TMA stores for full FC1 groups. Partial groups use direct stores; the legacy FC1 path disables it. Adds shared memory accounted for by stage fitting. |
| `PIPELINED_COMBINE` | `0` | `1` double-buffers bulk loads for in-kernel reduction using the dead operand ring. Falls back to scalar loads if scratch is insufficient. Does not itself enable in-kernel reduction. |
| `WAIT_STATS` | `0` | `1` allocates and instruments per-CTA wait statistics and the startup ring. Set before workspace allocation. Read via `read_mega_wait_stats`, `read_mega_startup_ring`, `summarize`, and `decode_startup_ring` in `wait_stats_config.py`. |
| `IKET_FINE_CTAS` | `0` | Number of initial CTAs with detailed IKET ranges; requires `CUTE_DSL_COMPILER_OPT=iket`. Overlapped-pull tracing instruments at least one CTA, including when this is unset. |

Additional controls: `FLASHINFER_MEGA_FUSED_STAGE=0` disables fused input
quantization/staging; `FLASHINFER_MOE_EP_KNOB_CACHE` selects the tuning cache
file (`0` or `off` disables it); `MEGA_NO_DIST=1` selects single-process
setup. `MEGA_USE_NCU=1` selects the profiling tail signal path. Instrumentation
changes timing, so disable it when selecting performance winners.

## Autotuning

This backend currently uses the **collective MoE EP tuner**, rather than
[`flashinfer/autotuner/autotuner.py`](../../../../autotuner/autotuner.py).
Wrapping a call in the generic `flashinfer.autotune()` context does not tune
this kernel. Use the CLI, `knobs="auto"`, or the explicit function below.
The collective tuner compiles and times the same candidates on every rank,
then minimizes the maximum rank's median synchronized launch latency. It
persists the winner on rank 0 in the MoE EP knob cache.

### Offline CLI

Run outside the serving process, with the deployment's GPU type, EP world
size, capacities, and representative routing. Supply your own dimensions;
the following commands intentionally leave them as shell variables.

```bash
export FLASHINFER_MOE_EP_KNOB_CACHE=/tmp/mxfp8_mxfp4_knobs.json
torchrun --nproc_per_node="$EP_SIZE" -m flashinfer.moe_ep.tune \
    --dtype mxfp8_mxfp4 \
    --hidden "$HIDDEN" --intermediate "$INTERMEDIATE" \
    --num-experts "$EXPERTS" --topk "$TOP_K" \
    --max-tokens "$CAPACITY" --live-tokens "$LIVE_TOKENS" \
    --warmup-iters 3 --timed-iters 10
```

For one GPU, replace the `torchrun` prefix with
`MEGA_NO_DIST=1 python -m flashinfer.moe_ep.tune` and keep the arguments.
`--max-tokens` accepts several capacities, with a separate sweep for each.
Omit `--live-tokens` to use each capacity as its live count.

The default pair-tile search has 32 candidates: 24 static candidates over
N64/N128/N192, group hints 512/4096, four/eight epilogue warps, and both
reduction placements; four alternating N192 candidates; and four atomic N64
candidates. The single-CTA search has 32 candidates over N64/N128,
automatic or pinned 3–5 stages, group hints 512/4096, and both reduction
placements. Experimental dispatch/startup switches are not separate search
axes; pass explicit candidates to compare them.

`--max-candidates` truncates the list for smoke testing. `--skew` changes
synthetic routing to a target maximum-to-mean expert load ratio. The
`--sweep schedule` option pins the default geometry (or `--base-knobs` JSON)
and sweeps static/atomic claims with group hints `None`, 128, 256, and 512;
it does not include alternating claims. `--base-knobs` applies only to that
schedule sweep. `--seed` controls synthetic inputs and routing;
`--gate-up-clamp` sets the activation clamp. This backend uses BF16 combine
and deterministic reduction, so `--allow-nondeterministic` adds no candidates.

### Explicit candidate sweep

This single-GPU example uses deliberately small synthetic fixtures. Run it
with `MEGA_NO_DIST=1`. The helper stages random inputs, preprocesses weights,
and reserves enough capacity for every supplied candidate.

```python
import torch
from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.integration import (
    autotune_mxfp8_mxfp4_mega_moe,
    create_dummy_mxfp8_mxfp4_inputs,
    mxfp8_mxfp4_mega_moe,
)

candidates = [
    {"fc2_store_mode": "direct"},
    {"fc2_store_mode": "staged256"},
]
y, fc1, fc2, workspace = create_dummy_mxfp8_mxfp4_inputs(
    rank=0, world_size=1, num_experts=4, max_tokens=32,
    live_tokens=16, topk=2, hidden=256, intermediate=256,
    capacity_tactics=candidates,
)
try:
    winner = autotune_mxfp8_mxfp4_mega_moe(
        y, fc1, fc2, workspace, num_tokens=16, candidates=candidates,
    )
    result = mxfp8_mxfp4_mega_moe(None, fc1, fc2, workspace, num_tokens=16)
    torch.cuda.synchronize()
    print(winner)
finally:
    torch.cuda.synchronize()
    workspace.destroy()
```

For live inputs, allocate with
`get_symm_buffer_for_mxfp8_mxfp4_mega_moe(..., capacity_tactics=candidates)`,
stage the inputs, and call the same tuner. All EP ranks must participate
with identical candidate order. Candidates that do not fit the allocation
are dropped with a warning. The sweep overwrites workspace outputs and is
not allowed during graph capture.

### Reusing results and measuring performance

Start the application with the same `FLASHINFER_MOE_EP_KNOB_CACHE` path and
`knobs=None`. Cache keys include GPU name, dtype, world size, hidden and
intermediate dimensions, expert count, top-k, combine format, and capacity.
Capacity lookup prefers exact matches, then the smallest larger bucket,
then the largest smaller bucket. Keep separate cache files for distinct
live-token/routing workloads sharing a capacity; live token count and routing
distribution are not cache-key fields. Retune when code, compiler settings,
or hardware changes. The default path is
`~/.cache/flashinfer/moe_ep_knob_cache.json`.

For measurements, finish compilation and tuning first, preallocate outputs,
and keep the measured scope consistent: compute-only calls exclude input
quantization/staging, while a full layer call includes them. Validate against
the reference before timing; include sparse/empty experts and skewed routing.
For graph timing, warm up the exact buffers before capture and replay on all
ranks in the same order. Report the slowest rank's latency for collective
execution. Debug wait counters and IKET traces are diagnostic runs, separate
from uninstrumented performance measurements.
