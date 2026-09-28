"""Standalone multi-rank split correctness gate (not a performance benchmark).

Example: torchrun --standalone --nproc_per_node=2 \
    tests/moe_ep/run_mxfp8_mxfp4_split.py --hidden 6656 --intermediate 4992

Uses an independent quantized oracle, then compares every output
element across eager execution and repeated CUDA graph replay, including changed
activation and routing values at fixed addresses. Use --no-cuda-graph for eager
transport bring-up (in particular, NIXL's graph lifecycle is not yet validated).
"""

import argparse
import os
import sys
import traceback

from mxfp8_mxfp4_test_utils import _independent_refcheck, _make_split_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=128)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--tokens-per-rank", type=int, default=4)
    parser.add_argument(
        "--transport", choices=("nccl_ep", "nixl_ep"), default="nccl_ep"
    )
    parser.add_argument(
        "--cuda-graph", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        EpLayout,
        FleetParams,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert args.experts % world == 0
    torch.manual_seed(90210 + rank)
    device = torch.device("cuda", local_rank)
    local_experts = args.experts // world
    w13 = torch.randn(
        local_experts,
        2 * args.intermediate,
        args.hidden,
        device=device,
        dtype=torch.bfloat16,
    ).div_(10)
    w2 = torch.randn(
        local_experts,
        args.hidden,
        args.intermediate,
        device=device,
        dtype=torch.bfloat16,
    ).div_(10)
    hidden = torch.randn(
        args.tokens_per_rank, args.hidden, device=device, dtype=torch.bfloat16
    ).div_(10)
    ids = (
        torch.rand(args.tokens_per_rank, args.experts, device=device)
        .topk(args.top_k, dim=-1)
        .indices.to(torch.int32)
    )
    scores = torch.rand(
        args.tokens_per_rank, args.top_k, device=device, dtype=torch.float32
    )
    scores.div_(scores.sum(dim=-1, keepdim=True))
    tensors = MoEEpTensors(hidden_states=hidden, topk_ids=ids, topk_weights=scores)
    transport_width = next(
        x for x in (2048, 2560, 4096, 5120, 6144, 7168, 8192) if x >= args.hidden
    )
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=world,
            rank=rank,
            process_group=dist.group.WORLD,
            device=local_rank,
        ),
        fleet_params=FleetParams(
            num_experts=args.experts,
            max_tokens_per_rank=args.tokens_per_rank,
            token_hidden_size=args.hidden,
            transport_hidden_size=transport_width,
            algorithm=EpAlgorithm.LOW_LATENCY,
            layout=EpLayout.EXPERT_MAJOR,
        ),
        weights=MoEWeightPack(w13=w13, w2=w2),
        backend=_make_split_backend(args, rank, world, args.tokens_per_rank),
    )

    def oracle(output):
        return _independent_refcheck(
            torch=torch,
            dist=dist,
            rank=rank,
            world_size=world,
            hidden_states=hidden,
            topk_ids=ids,
            topk_weights=scores,
            w13=w13,
            w2=w2,
            output=output,
            gate_up_clamp=None,
            atol=0.01,
            rtol=0.01,
            rel_l2_limit=0.01,
            all_ranks=True,
        )

    layer.forward(tensors)
    torch.cuda.synchronize()
    eager = layer.forward(tensors).clone()
    initial_error = oracle(eager)
    if args.cuda_graph:
        graph_state = layer.create_graph_state(tensors)
        layer.forward(tensors, graph_state=graph_state)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output = layer.forward(tensors, graph_state=graph_state)
        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured_output, eager, atol=0, rtol=0)
    for mutation in range(2):
        hidden.normal_(mean=0, std=0.1)
        ids.copy_((ids + 1) % args.experts)
        scores.uniform_(0.1, 1.0)
        scores.div_(scores.sum(dim=-1, keepdim=True))
        if args.cuda_graph:
            graph.replay()
            replayed = captured_output.clone()
        eager = layer.forward(tensors).clone()
        error = oracle(eager)
        if args.cuda_graph:
            torch.testing.assert_close(replayed, eager, atol=0, rtol=0)
            for _ in range(3):
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(captured_output, eager, atol=0, rtol=0)
        if rank == 0:
            print(
                f"mutation={mutation} all_rank_sample_rel_l2={error} all_output_graph_match={args.cuda_graph}",
                flush=True,
            )
    torch.cuda.synchronize()
    if args.cuda_graph:
        del graph, captured_output
    layer.destroy()
    dist.barrier()
    if rank == 0:
        print(
            f"PASS transport={args.transport} EP={world} H={args.hidden} I={args.intermediate} E={args.experts} topk={args.top_k} tokens_per_rank={args.tokens_per_rank} transport_width={transport_width} graph={args.cuda_graph} initial_sample_rel_l2={initial_error}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # Preserve argparse's normal --help and usage-error exit codes.
        raise
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        # Do not hide a failed rank behind collective teardown while peers wait.
        os._exit(1)
