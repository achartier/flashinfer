# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bitwise combine contracts, including partial chunks and stage reuse."""

import pytest
import torch

cutlass = pytest.importorskip("cutlass")
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402
from cuda.bindings import driver as cuda  # noqa: E402

from .combine import PipelinedTopKReducer, combine_chunk_elements  # noqa: E402
from .fc2_epilogue import DeterministicTopKReducer  # noqa: E402


class _CombineKernel:
    def __init__(self, hidden, topk, chunk):
        self.reducer = PipelinedTopKReducer(
            hidden=hidden, num_topk=topk, chunk_elements=chunk
        )

    @cute.jit
    def __call__(self, terms, scores, ids, output, stream):
        self.kernel(terms, scores, ids, output).launch(
            grid=(2, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, terms, scores, ids, output):
        scratch = cutlass.utils.SmemAllocator().allocate_tensor(
            cutlass.Uint8,
            cute.make_layout(4 * self.reducer.warp_bytes),
            byte_alignment=128,
        )
        self.reducer.reduce(
            terms,
            scores,
            ids,
            output,
            scratch.iterator,
            cute.arch.block_idx()[0],
            cute.arch.thread_idx()[0],
            2,
            128,
        )


def test_combine_scratch_fit():
    assert combine_chunk_elements(4096, 16, 65536 + 256) == 1024
    assert combine_chunk_elements(4096, 16, 65536) == 512
    assert combine_chunk_elements(384, 12, 65536) == 512
    assert combine_chunk_elements(128, 12, 65536) == 256
    assert combine_chunk_elements(4096, 16, 16384) == 0


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("tokens", [3, 23])
@pytest.mark.parametrize("topk", [1, 8, 32])
@pytest.mark.parametrize(
    "hidden,chunk",
    [(32, 256), (384, 512), (1152, 1024), (4096, 1024), (4096, 4096), (6400, 1024)],
)
def test_pipelined_combine_bit_exact(hidden, chunk, topk, tokens):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires Blackwell")
    torch.manual_seed(713)
    # Small batches leave idle warps; larger ones reuse both stage phases.
    terms = torch.randn(tokens, topk, hidden, device="cuda", dtype=torch.bfloat16)
    scores = torch.randn(tokens, topk, device="cuda", dtype=torch.float32)
    ids = torch.arange(topk, device="cuda").expand(tokens, -1).clone()
    ids[1::3, ::2] = -1
    ids[0] = -1
    scores.masked_fill_(ids == -1, float("nan"))
    terms.masked_fill_((ids == -1).unsqueeze(-1), float("nan"))
    output = torch.empty(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    expected = torch.empty_like(output)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    inputs = [from_dlpack(x, assumed_align=16) for x in (terms, scores, ids)]
    pipelined = cute.compile(
        _CombineKernel(hidden, topk, chunk),
        *inputs,
        from_dlpack(output, assumed_align=16),
        stream,
    )
    reference = cute.compile(
        DeterministicTopKReducer(hidden=hidden, num_topk=topk),
        *inputs,
        from_dlpack(expected, assumed_align=16),
        stream,
    )

    def run_reference():
        reference(
            *inputs,
            from_dlpack(expected, assumed_align=16),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )

    def run_combine():
        pipelined(
            *inputs,
            from_dlpack(output, assumed_align=16),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )

    run_reference()
    run_combine()
    torch.cuda.synchronize()
    assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_combine()
    saved_ids, saved_scores = ids.clone(), scores.clone()
    for live in (True, False, True, False, True):
        if live:
            ids.copy_(saved_ids)
            scores.copy_(saved_scores)
        else:
            ids.fill_(-1)
            scores.fill_(float("nan"))
        run_reference()
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
