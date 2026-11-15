#!/usr/bin/env python3
"""BF16 causal-transformer DDP/FSDP training on generated token batches.

This is a controlled training-step benchmark, not a pretrained-model quality
evaluation. Timing includes forward, loss, backward, DDP communication, and SGD.
Model creation, input generation, initialization, and warmups are excluded.
"""

import argparse
import ctypes
from functools import partial
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._comm_hooks.default_hooks import DefaultState, reduce_scatter_hook


class Transformer(nn.Module):
    def __init__(self, width, layers, vocab, sequence):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.position = nn.Embedding(sequence, width)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                width, width // 128, 4 * width, dropout=0,
                batch_first=True, norm_first=True, bias=False,
            ) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(width, bias=False)
        self.head = nn.Linear(width, vocab, bias=False)

    def forward(self, tokens, mask):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        value = self.embedding(tokens) + self.position(positions)
        for layer in self.layers:
            value = layer(value, src_mask=mask, is_causal=True)
        return self.head(self.norm(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--sharding", choices=["ddp", "fsdp"], default="ddp")
    parser.add_argument("--fsdp-rs-group", choices=["separate", "shared"], default="separate")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--vocab", type=int, default=8192)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--bucket-mib", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    if min(args.width, args.layers, args.vocab, args.sequence, args.batch,
           args.bucket_mib, args.warmup, args.steps) < 1 or args.width % 128:
        parser.error("dimensions and counts must be positive; width must be a multiple of 128")
    rank = int(os.environ["LOCAL_RANK"])
    # Match the serving launcher's rank affinity and reserve the same worker cores.
    base = rank * 6
    cpus = set(range(base, base + 4)) | set(range(base + 48, base + 52))
    for task in Path("/proc/self/task").iterdir():
        try:
            os.sched_setaffinity(int(task.name), cpus)
        except ProcessLookupError:
            pass
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    if args.sharding == "fsdp":
        # Leave the default PG unbound to prevent ncclCommSplit for the RS
        # group: the experimental proxy tracks full communicator initialization.
        dist.init_process_group("nccl")
    else:
        dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    if dist.get_world_size() != 8:
        raise ValueError("this benchmark requires eight local ranks")
    torch.manual_seed(args.seed)
    model = Transformer(args.width, args.layers, args.vocab, args.sequence).to(
        device="cuda", dtype=torch.bfloat16
    ).train()
    parameters = sum(p.numel() for p in model.parameters())
    gradient_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    rs_group = None
    if args.sharding == "ddp":
        model = DistributedDataParallel(model, device_ids=[rank],
                                        bucket_cap_mb=args.bucket_mib,
                                        gradient_as_bucket_view=True)
    else:
        model = FullyShardedDataParallel(
            model, device_id=rank, auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={nn.TransformerEncoderLayer},
            ),
        )
        if args.fsdp_rs_group == "separate":
            # The proxy permits one stream per communicator. FSDP's AG and
            # RS use different NCCL streams, so give RS its own communicator.
            # Use the same averaging hook and group layout for both backends.
            rs_group = dist.new_group(ranks=list(range(8)), backend="nccl")
            model.register_comm_hook(DefaultState(rs_group), reduce_scatter_hook)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, foreach=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed + rank + 1)
    batches = [torch.randint(args.vocab, (args.batch, args.sequence + 1),
                            device="cuda", generator=generator) for _ in range(4)]
    mask = torch.ones(args.sequence, args.sequence, device="cuda", dtype=torch.bool).triu(1)
    losses = []
    times = []
    for step in range(args.warmup + args.steps):
        tokens = batches[step % len(batches)]
        # A barrier and synchronization precede timing; no host enqueue interval is omitted.
        dist.barrier(device_ids=[rank])
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens[:, :-1], mask)
        loss = F.cross_entropy(logits.flatten(0, 1).float(), tokens[:, 1:].flatten())
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        losses.append(loss.item())
        if step >= args.warmup:
            times.append(elapsed)
    rank_times = [None] * 8
    rank_losses = [None] * 8
    dist.all_gather_object(rank_times, times)
    dist.all_gather_object(rank_losses, losses)
    if args.sharding == "fsdp":
        # Verification must preserve the trained communicator's AG stream too.
        with torch.cuda.stream(model._unshard_stream):
            with FullyShardedDataParallel.summon_full_params(model, writeback=False):
                probes = torch.cat([p.detach().flatten()[:8].float() for p in model.parameters()])
        torch.cuda.synchronize()
    else:
        probes = torch.cat([p.detach().flatten()[:8].float() for p in model.parameters()])
    reference = probes.clone()
    dist.broadcast(reference, src=0)
    if not torch.equal(probes, reference):
        raise RuntimeError("parameter samples differ across DDP ranks")
    if rank == 0:
        maximums = [max(values[i] for values in rank_times) for i in range(args.steps)]
        nccl_version = ctypes.c_int()
        library = ctypes.CDLL(os.environ["UCCL_LITE_REAL_NCCL"])
        if library.ncclGetVersion(ctypes.byref(nccl_version)):
            raise RuntimeError("could not read runtime NCCL version")
        mapped_nccl = sorted({line.split()[-1] for line in
                              Path("/proc/self/maps").read_text().splitlines()
                              if "/libnccl.so" in line})
        if len(mapped_nccl) != 1:
            raise RuntimeError(f"expected one real NCCL library, found {mapped_nccl}")
        row = {
            **vars(args), "out": str(args.out), "dtype": "bfloat16",
            "torch": torch.__version__, "torch_compiled_nccl": torch.cuda.nccl.version(),
            "runtime_nccl_version": nccl_version.value, "mapped_nccl": mapped_nccl,
            "parameters": parameters, "gradient_bytes": gradient_bytes,
            "ddp": model._get_ddp_logging_data() if args.sharding == "ddp" else None,
            "max_rank_step_s": maximums,
            "median_step_s": statistics.median(maximums),
            "tokens_per_s": 8 * args.batch * args.sequence / statistics.median(maximums),
            "rank_losses": rank_losses,
            "parameter_samples": probes.cpu().tolist(),
            "rank_parameter_samples_identical": True,
        }
        with args.out.open("a") as output:
            output.write(json.dumps(row) + "\n")
        print(json.dumps({k: v for k, v in row.items()
                          if k not in ("parameter_samples", "rank_losses", "ddp")}), flush=True)
    if rs_group is not None:
        dist.destroy_process_group(rs_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
