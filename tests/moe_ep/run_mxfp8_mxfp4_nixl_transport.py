"""EP2 opaque MXFP8/E8M0 transport test, without a compute kernel.

Run with torchrun --nproc-per-node=2 and a built NIXL-EP backend.
This harness reports same-node transport only; it does not validate IB.
"""

from datetime import timedelta
import os
from pathlib import Path
import sys

# Do not shadow the installed/staged nixl_ep with tests/moe_ep/nixl_ep.
_here = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() != _here]


def transport_libraries():
    return sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if any(name in line for name in ("libnixl", "libucp", "libuct", "libucs"))
        }
    )


def main():
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        CombineInputParams,
        DispatchInputParams,
        EpAlgorithm,
        FleetParams,
        HandleAlgoKnobTopKWeights,
        HandleParams,
        create_fleet,
    )

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2, "This deterministic fixture requires EP2"
    store = dist.TCPStore(
        os.environ["MASTER_ADDR"],
        int(os.environ["MASTER_PORT"]) + 1,
        world,
        rank == 0,
        timeout=timedelta(seconds=120),
    )
    tokens, experts = 4, 4
    ids_cpu = torch.tensor([[t % 2, 2 + (t + 1) % 2] for t in range(tokens)])
    ids = ids_cpu.cuda()
    scores = torch.tensor(
        [[0.125 + t / 16, 0.75 - t / 16] for t in range(tokens)],
        device="cuda",
        dtype=torch.float32,
    )
    for hidden, packed_width, transport_width in (
        (2048, 2048, 2048),
        (6656, 4096, 7168),
    ):
        print(f"rank={rank} H={hidden} create", flush=True)
        fleet = create_fleet(
            BootstrapConfig(
                world_size=world,
                rank=rank,
                tcp_store=dist.PrefixStore(f"hidden-{hidden}", store),
                stream=torch.cuda.current_stream().cuda_stream,
            ),
            FleetParams(
                num_experts=experts,
                max_tokens_per_rank=tokens,
                token_hidden_size=hidden,
                transport_hidden_size=transport_width,
                algorithm=EpAlgorithm.LOW_LATENCY,
            ),
            [],
            backend="nixl_ep",
        )
        if rank == 0:
            print(f"loaded transport libraries={transport_libraries()}", flush=True)
        handle = fleet.create_handle(
            HandleParams(topk_ids=ids), [HandleAlgoKnobTopKWeights(weights=scores)]
        )

        def payload(source):
            # Finite E4M3 activation patterns and positive finite E8M0 scales.
            # First activation byte identifies the source token unambiguously.
            data = torch.zeros(tokens, packed_width * 2, dtype=torch.uint8)
            for token in range(tokens):
                identity = source * tokens + token + 1
                data[token, :hidden] = (torch.arange(hidden) + identity) % 120
                data[token, hidden : hidden + hidden // 32] = (
                    torch.arange(hidden // 32) + identity
                ) % 5 + 124
            return data

        all_payloads = [payload(source) for source in range(world)]
        wire = all_payloads[rank].cuda().view(torch.bfloat16)
        print(f"rank={rank} H={hidden} dispatch", flush=True)
        received = handle.dispatch(DispatchInputParams(x=[wire]))
        handle.complete()
        torch.cuda.synchronize()
        counts = received.expert_counts.cpu().tolist()
        raw = received.expert_tensors.contiguous().view(torch.uint8).cpu()
        terms = torch.zeros(
            (*received.expert_tensors.shape[:2], hidden),
            dtype=torch.bfloat16,
            device="cuda",
        )
        feature = (torch.arange(hidden, device="cuda") % 13).float() / 16
        for local in range(experts // world):
            expert = rank * (experts // world) + local
            expected = {}
            for source in range(world):
                for token in range(tokens):
                    if expert in ids_cpu[token].tolist():
                        identity = source * tokens + token + 1
                        expected[identity] = all_payloads[source][token]
            assert counts[local] == len(expected), (rank, expert, counts)
            seen = set()
            for row in range(counts[local]):
                identity = int(raw[local, row, 0])
                assert identity in expected and identity not in seen
                torch.testing.assert_close(
                    raw[local, row], expected[identity], rtol=0, atol=0
                )
                seen.add(identity)
                terms[local, row] = (identity / 8 + expert / 4 + feature).to(
                    torch.bfloat16
                )
        print(
            f"rank={rank} H={hidden} exact dispatch bytes passed; combine", flush=True
        )
        actual = handle.combine(CombineInputParams(x=[terms])).x
        handle.complete()
        expected = torch.zeros(tokens, hidden, dtype=torch.float32, device="cuda")
        for slot in range(2):
            for token in range(tokens):
                identity = rank * tokens + token + 1
                term = (identity / 8 + int(ids_cpu[token, slot]) / 4 + feature).to(
                    torch.bfloat16
                )
                expected[token] += term.float() * scores[token, slot]
        torch.testing.assert_close(actual, expected.to(torch.bfloat16), rtol=0, atol=0)
        torch.cuda.synchronize()
        print(
            f"rank={rank} H={hidden} exact combine passed (same-node only)", flush=True
        )
        handle.destroy()
        fleet.destroy()
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        # Preserve the actual runtime library identity when native transport
        # fails (wheel and UCX headers can otherwise silently disagree).
        print(
            f"rank={os.environ.get('RANK')} loaded transport libraries={transport_libraries()}",
            file=sys.stderr,
            flush=True,
        )
        sys.stderr.flush()
        os._exit(1)
