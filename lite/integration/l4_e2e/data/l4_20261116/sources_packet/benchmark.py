#!/usr/bin/env python3
"""Experimental weight-sharded GPT-2 HTTP serving and full training steps.

One MPI process owns four GPUs on each of one or two nodes. Blocks have sharded BF16 trainable weights;
embeddings/final norm are frozen replicas. This is a controlled adapter, not
SGLang or stock FSDP. Each block gathers weights; autograd scatters their sums.
"""
import argparse
import ctypes
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import statistics
import threading
import time
import urllib.request

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_block
from safetensors.torch import load_file

NODE = 0
NODES = 1
WORLD = 4
REPLICAS = False

ROOT = Path(__file__).resolve().parents[2]


def synchronize():
    for r in range(4):
        torch.cuda.synchronize(r)


class Collectives:
    def __init__(self, backend, replicas=False):
        self.backend = backend
        self.lib = ctypes.CDLL(str(ROOT / "build/l4_e2e/libuccl_lite_peer_e2e.so"))
        pointer_array = ctypes.POINTER(ctypes.c_void_p)
        self.lib.uccl_lite_e2e_create.argtypes = [ctypes.c_int]
        self.lib.uccl_lite_e2e_create.restype = ctypes.c_void_p
        self.lib.uccl_lite_e2e_error.restype = ctypes.c_char_p
        signature = [ctypes.c_void_p, ctypes.c_size_t, pointer_array, pointer_array, pointer_array]
        self.lib.uccl_lite_e2e_prepare.argtypes = signature
        self.lib.uccl_lite_e2e_enqueue.argtypes = signature[:1] + [ctypes.c_int] + signature[1:]
        self.lib.uccl_lite_e2e_destroy.argtypes = [ctypes.c_void_p]
        self.context = self.lib.uccl_lite_e2e_create(int(backend != "nccl") | (2 if replicas else 0) | (4 if backend.startswith("packet") else 0))
        if not self.context:
            raise RuntimeError(self.lib.uccl_lite_e2e_error().decode())
        global NODE, NODES, WORLD, REPLICAS
        for name in ("node", "nodes"):
            fn = getattr(self.lib, "uccl_lite_e2e_" + name)
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = ctypes.c_int
        NODE = self.lib.uccl_lite_e2e_node(self.context)
        NODES = self.lib.uccl_lite_e2e_nodes(self.context)
        REPLICAS = replicas
        WORLD = 4 if replicas else NODES * 4
        self.lib.uccl_lite_e2e_max.argtypes = [ctypes.c_double]
        self.lib.uccl_lite_e2e_max.restype = ctypes.c_double
        self.histogram = {}

    def pointers(self, values):
        return (ctypes.c_void_p * 4)(*[v.data_ptr() for v in values])

    def streams(self):
        return (ctypes.c_void_p * 4)(*[torch.cuda.current_stream(r).cuda_stream for r in range(4)])

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.uccl_lite_e2e_error().decode())

    def prepare(self, count):
        inputs = [torch.empty(max(count * WORLD, 32 << 20) if self.backend.startswith("packet") else count * WORLD, dtype=torch.bfloat16, device=r) for r in range(4)]
        outputs = [torch.empty(max(count, 32 << 20) if self.backend.startswith("packet") else count, dtype=torch.bfloat16, device=r) for r in range(4)]
        synchronize()
        self.check(self.lib.uccl_lite_e2e_prepare(
            self.context, count, self.pointers(inputs), self.pointers(outputs), self.streams()))

    def call(self, operation, inputs):
        if len(inputs) != 4 or any(x.dtype != torch.bfloat16 or not x.is_contiguous()
                                    or x.device.index != r for r, x in enumerate(inputs)):
            raise ValueError("requires four contiguous BF16 inputs on GPUs 0..3")
        if inputs[0].numel() != inputs[1].numel():
            raise ValueError("rank extents differ")
        count = inputs[0].numel() if operation == 1 else inputs[0].numel() // WORLD
        outputs = [torch.empty(count * (WORLD if operation == 1 else 1),
                               device=r, dtype=torch.bfloat16) for r in range(4)]
        self.check(self.lib.uccl_lite_e2e_enqueue(
            self.context, operation, count, self.pointers(inputs), self.pointers(outputs), self.streams()))
        key = f"{'AG' if operation == 1 else 'RS'}:{count * WORLD * 2}"
        self.histogram[key] = self.histogram.get(key, 0) + 1
        return outputs

    def close(self):
        synchronize()
        self.lib.uccl_lite_e2e_destroy(self.context)


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, collectives, *shards):
        ctx.collectives = collectives
        return tuple(collectives.call(1, list(shards)))

    @staticmethod
    def backward(ctx, *gradients):
        shards = ctx.collectives.call(2, [g.contiguous() for g in gradients])
        return None, *shards


class Model:
    def __init__(self, checkpoint, collectives, compile_blocks=False):
        self.checkpoint_blocks = False
        self.config = json.loads((checkpoint / "config.json").read_text())
        state = load_file(checkpoint / "model.safetensors")
        self.collectives = collectives
        self.layers = self.config["n_layer"]
        self.width = self.config["n_embd"]
        self.heads = self.config["n_head"]
        self.names = []
        self.shapes = []
        self.blocks = []
        self.replicas = []
        self.parameter_count = 0
        for layer in range(self.layers):
            prefix = f"h.{layer}."
            parameter_names = {
                "ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias",
                "attn.c_attn.weight", "attn.c_attn.bias", "attn.c_proj.weight", "attn.c_proj.bias",
                "mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias",
            }
            keys = sorted(prefix + name for name in parameter_names)
            if any(k not in state for k in keys):
                raise ValueError("missing GPT-2 block parameters")
            # GPT-2's safetensors uses transformer keys without 'transformer.'.
            if not keys:
                raise ValueError(f"missing {prefix}: {list(state)[:5]}")
            values = [state[k].to(torch.bfloat16) for k in keys]
            self.names.append([k[len(prefix):] for k in keys])
            self.shapes.append([v.shape for v in values])
            flat = torch.cat([v.flatten() for v in values])
            self.parameter_count += flat.numel()
            if flat.numel() % WORLD:
                raise ValueError("block parameters must split equally")
            shards = [torch.nn.Parameter(piece.to(r)) for r, piece in enumerate(flat.chunk(WORLD)[(0 if REPLICAS else NODE)*4:((0 if REPLICAS else NODE)+1)*4])]
            self.blocks.append(shards)
        for r in range(4):
            self.replicas.append({k: state[k].to(device=r, dtype=torch.bfloat16)
                                  for k in ["wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"]})
        from transformers.activations import ACT2FN
        self.activation = ACT2FN[self.config["activation_function"]]
        if compile_blocks:
            torch._inductor.config.emulate_precision_casts = True
            self.block = torch.compile(self.block, fullgraph=True)
        self.parameters = [p for block in self.blocks for p in block]
        self.complete_block_bytes = [sum(p.numel() * 2 for p in block) * (WORLD // 4) for block in self.blocks]
        for count in sorted({p.numel() for p in self.parameters}):
            collectives.prepare(count)

    def unpack(self, layer, flat):
        sizes = [shape.numel() for shape in self.shapes[layer]]
        return {name: value.view(shape) for name, value, shape in
                zip(self.names[layer], flat.split(sizes), self.shapes[layer])}

    def norm(self, value, parameters, prefix):
        return F.layer_norm(value, (self.width,), parameters[prefix + ".weight"],
                            parameters[prefix + ".bias"], self.config["layer_norm_epsilon"])

    @staticmethod
    def linear(value, parameters, prefix):
        weight, bias = parameters[prefix + ".weight"], parameters[prefix + ".bias"]
        return torch.addmm(bias, value.reshape(-1, value.shape[-1]), weight).view(
            *value.shape[:-1], weight.shape[-1])

    def block(self, value, parameters, past):
        residual = value
        value = self.norm(value, parameters, "ln_1")
        qkv = self.linear(value, parameters, "attn.c_attn")
        query, key, val = [x.view(*x.shape[:2], self.heads, self.width // self.heads)
                          .transpose(1, 2) for x in qkv.chunk(3, dim=-1)]
        if past is not None:
            key = torch.cat([past[0], key], dim=2)
            val = torch.cat([past[1], val], dim=2)
        value = F.scaled_dot_product_attention(query, key, val,
                                               is_causal=past is None)
        value = value.transpose(1, 2).contiguous().view(*residual.shape)
        value = residual + self.linear(value, parameters, "attn.c_proj")
        residual = value
        value = self.norm(value, parameters, "ln_2")
        value = self.linear(value, parameters, "mlp.c_fc")
        value = self.activation(value)
        value = residual + self.linear(value, parameters, "mlp.c_proj")
        return value, (key, val)

    def forward(self, tokens, caches=None, last_only=False):
        hidden = []
        for r in range(4):
            with torch.cuda.device(r):
                position = 0 if caches is None else caches[r][0][0].shape[2]
                positions = torch.arange(position, position + tokens[r].shape[1], device=r)
                hidden.append(F.embedding(tokens[r], self.replicas[r]["wte.weight"]) +
                              F.embedding(positions, self.replicas[r]["wpe.weight"]))
        new_caches = [[] for _ in range(4)]
        for layer, shards in enumerate(self.blocks):
            if self.checkpoint_blocks and torch.is_grad_enabled():
                def rematerialize(*values, block_index=layer):
                    weights = Gather.apply(self.collectives, *values[4:])
                    result = []
                    for rank in range(4):
                        with torch.cuda.device(rank):
                            value, _ = self.block(values[rank], self.unpack(block_index, weights[rank]), None)
                            result.append(value)
                    return tuple(result)
                hidden = list(checkpoint_block(rematerialize, *hidden, *shards,
                                               use_reentrant=True, preserve_rng_state=False))
                for rank in range(4):
                    new_caches[rank].append(None)
                continue
            full = Gather.apply(self.collectives, *shards)
            for r in range(4):
                with torch.cuda.device(r):
                    hidden[r], cache = self.block(hidden[r], self.unpack(layer, full[r]),
                                                   None if caches is None else caches[r][layer])
                    new_caches[r].append(cache)
        logits = []
        for r in range(4):
            with torch.cuda.device(r):
                value = self.norm(hidden[r], self.replicas[r], "ln_f")
                if last_only:
                    value = value[:, -1:, :]
                logits.append(F.linear(value, self.replicas[r]["wte.weight"]))
        return logits, new_caches

    @torch.no_grad()
    def generate(self, tokens, outputs):
        caches = None
        result = [[] for _ in range(4)]
        for _ in range(outputs):
            logits, caches = self.forward(tokens, caches, last_only=True)
            tokens = [x[:, -1].argmax(dim=-1, keepdim=True) for x in logits]
            for r in range(4):
                result[r].append(tokens[r])
        return [torch.cat(x, dim=1).cpu().tolist() for x in result]


def correctness(collectives, count):
    """Full-tensor AG and destination-dependent signed RS oracle."""
    collectives.prepare(count)
    shards = [(torch.arange(count, device=r) % 29 - 14 + (0 if REPLICAS else NODE)*4+r).bfloat16() for r in range(4)]
    full = collectives.call(1, shards)
    synchronize()
    expected = torch.cat([(torch.arange(count) % 29 - 14 + r).bfloat16() for r in range(WORLD)])
    assert all(torch.equal(x.cpu(), expected) for x in full), "AG oracle failed"
    inputs = [(torch.arange(count * WORLD, device=r) % 31 - 15 + (NODE*4+r)*2).bfloat16() for r in range(4)]
    reduced = collectives.call(2, inputs)
    synchronize()
    expected = sum((torch.arange(count*WORLD)%31-15+r*2).float() for r in range(NODES*4)).bfloat16()
    assert all(torch.equal(x.cpu(), part) for x,part in zip(reduced,expected.chunk(WORLD)[(0 if REPLICAS else NODE)*4:((0 if REPLICAS else NODE)+1)*4])), "RS oracle failed"


def serving(args, model):
    # The HTTP handler owns all GPU submissions. The client includes JSON,
    # loopback HTTP, token generation, GPU completion and response completion.
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            tokens = torch.tensor(request["tokens"], dtype=torch.long).chunk(NODES*4)[NODE*4:(NODE+1)*4]
            result = model.generate([t.to(r) for r, t in enumerate(tokens)], request["outputs"])
            payload = json.dumps({"tokens": result}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained(args.checkpoint, local_files_only=True)
    corpus = (
        "The engineering team is investigating a distributed system. "
        "Describe the observations, compare the measurements and propose the next experiment. "
        "The service receives requests from several clients and processes them on two GPUs. "
        "We record response latency and check that all generated tokens are returned. "
    )
    rows = []
    try:
        for batch, sequence, outputs in args.cases:
            tokens = [tokenizer.encode(f"Experiment {i}: " + corpus * 32, truncation=True, max_length=sequence) for i in range(batch)]
            assert all(len(t) == sequence for t in tokens)
            payload = json.dumps({"tokens": tokens, "outputs": outputs}).encode()
            samples, hashes, generated = [], [], None
            for i in range(args.warmup + args.repeats):
                model.collectives.lib.uccl_lite_e2e_barrier()
                start = time.perf_counter()
                request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}", data=payload,
                                                  headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=300) as response:
                    data = response.read()
                elapsed = model.collectives.lib.uccl_lite_e2e_max(time.perf_counter() - start)
                result = json.loads(data)["tokens"]
                assert sum(len(part) for part in result) == batch // NODES
                assert all(len(seq) == outputs for part in result for seq in part)
                if i >= args.warmup:
                    samples.append(elapsed)
                    hashes.append(hashlib.sha256(data).hexdigest())
                    generated = result
            row = {"kind": "serving", "case": f"{batch}:{sequence}:{outputs}",
                   "samples_s": samples, "median_s": statistics.median(samples),
                   "prompt_sha256": hashlib.sha256(payload).hexdigest(),
                   "response_sha256": hashes, "generated_tokens": generated}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "generated_tokens"}), flush=True)
    finally:
        server.shutdown()
        worker.join()
        server.server_close()
    return rows


def training(args, model):
    rows = []
    initial = [p.detach().clone() for p in model.parameters]
    for batch, sequence in args.train_cases:
        with torch.no_grad():
            for parameter, value in zip(model.parameters, initial):
                parameter.copy_(value)
        generator = torch.Generator().manual_seed(args.seed)
        batches = [torch.randint(model.config["vocab_size"], (batch, sequence + 1), generator=generator)
                   for _ in range(4)]
        batches = [[x.to(r) for r, x in enumerate(batch.chunk(NODES*4)[NODE*4:(NODE+1)*4])] for batch in batches]
        optimizer = torch.optim.SGD(model.parameters, lr=1e-3, foreach=True)
        samples, losses = [], []
        for i in range(args.warmup + args.repeats):
            synchronize()
            model.collectives.lib.uccl_lite_e2e_barrier()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            tokens = batches[i % len(batches)]
            logits, _ = model.forward([x[:, :-1] for x in tokens])
            loss = [F.cross_entropy(x.flatten(0, 1).float(), t[:, 1:].flatten())
                    for x, t in zip(logits, tokens)]
            torch.autograd.backward(loss, [torch.full((), 1 / (NODES*4), device=r) for r in range(4)])
            optimizer.step()
            synchronize()
            elapsed = model.collectives.lib.uccl_lite_e2e_max(time.perf_counter() - start)
            values = [x.item() for x in loss]
            assert all(torch.isfinite(x) for x in loss)
            losses.append(values)
            if i >= args.warmup:
                samples.append(elapsed)
        if args.save_parameters:
            args.parameter_dir.mkdir(parents=True, exist_ok=True)
            path = args.parameter_dir / f"{args.out.parent.name}_{args.out.stem}_{batch}_{sequence}_parameters.pt"
            torch.save([p.detach().cpu() for p in model.parameters], path)
        probes = [p.detach().float().cpu()[::max(1, p.numel() // 32)].tolist() for p in model.parameters]
        # Full tensor hashes record exact cross-backend equality, not just probes.
        parameter_hashes = [hashlib.sha256(p.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                            for p in model.parameters]
        row = {"kind": "training", "case": f"{batch}:{sequence}", "samples_s": samples,
               "median_s": statistics.median(samples), "losses": losses,
               "parameter_samples": probes, "parameter_sha256": parameter_hashes}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("parameter_samples", "parameter_sha256")}), flush=True)
    return rows


@torch.no_grad()
def reference_check(args, model):
    """Compare every logit of an unsharded Transformers forward outside timing."""
    from transformers import GPT2LMHeadModel
    reference = GPT2LMHeadModel.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True).to(0).eval()
    generator = torch.Generator().manual_seed(args.seed)
    tokens = torch.randint(model.config["vocab_size"], (4, 16), generator=generator)
    actual, _ = model.forward([tokens[r:r + 1].to(r) for r in range(4)])
    expected = torch.cat([reference(t[None].to(0)).logits.float().cpu() for t in tokens])
    actual = torch.cat([x.float().cpu() for x in actual])
    delta = (actual - expected).abs()
    torch.testing.assert_close(actual, expected, atol=0.25, rtol=0.02)
    greedy_equal = torch.equal(actual.argmax(-1), expected.argmax(-1))
    # The decode path must also agree with the reference KV-cache implementation.
    generated = model.generate([tokens[r:r + 1].to(r) for r in range(4)], 4)
    reference_tokens = [reference.generate(t[None].to(0), max_new_tokens=4, do_sample=False,
                                          eos_token_id=None, pad_token_id=50256)[0, -4:].cpu().tolist() for t in tokens]
    assert sum(generated, []) == reference_tokens, "reference decode tokens differ"
    return {"max_logit_abs_diff": delta.max().item(),
            "mean_logit_abs_diff": delta.mean().item(), "greedy_tokens_equal": greedy_equal,
            "four_token_kv_decode_equal": True, "logit_atol": 0.25, "logit_rtol": 0.02}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replica-nodes", action="store_true")
    parser.add_argument("--backend", choices=["nccl", "peer", "packet"], required=True)
    parser.add_argument("--kind", choices=["serving", "training", "check"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["8:32:16", "8:128:16", "16:256:16"])
    parser.add_argument("--train-cases", nargs="+", default=["8:32", "8:64", "16:128"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--checkpoint-blocks", action="store_true")
    parser.add_argument("--compile-blocks", action="store_true")
    parser.add_argument("--save-parameters", action="store_true")
    parser.add_argument("--parameter-dir", type=Path, default=Path("/dev/shm/uccl-lite-e2e-validation"))
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    args.cases = [tuple(map(int, x.split(":"))) for x in args.cases]
    args.train_cases = [tuple(map(int, x.split(":"))) for x in args.train_cases]
    if args.repeats < 1 or args.warmup < 1 or any(b % 4 or min(b, s, o) < 1 or s + o > 1024
            for b, s, o in args.cases) or any(b % 4 or min(b, s) < 1 or s > 1024 for b, s in args.train_cases):
        parser.error("positive counts, global batches divisible by four and GPT-2 context <=1024 required")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    collective = Collectives(args.backend, args.replica_nodes)
    if any(b % (NODES*4) for b, _, _ in args.cases) or any(b % (NODES*4) for b, _ in args.train_cases):
        parser.error("batch must divide global GPU count")
    args.out = args.out.with_name(args.out.stem + f"_node{NODE}.json")
    if args.out.exists():
        parser.error("choose a fresh output file")
    model = Model(args.checkpoint, collective, args.compile_blocks)
    model.checkpoint_blocks = args.checkpoint_blocks
    synchronize()
    # Full checks are outside all timing intervals.
    for count in sorted({p.numel() for p in model.parameters} | {(m << 20) // (WORLD * 2) for m in (4, 8, 16, 32, 64)}):
        correctness(collective, count)
    reference = reference_check(args, model)
    collective.histogram.clear()
    for r in range(4):
        torch.cuda.reset_peak_memory_stats(r)
    if args.kind == "serving":
        rows = serving(args, model)
    elif args.kind == "training":
        rows = training(args, model)
    else:
        rows = []
    record = {"arguments": {**vars(args), "checkpoint": str(args.checkpoint), "out": str(args.out), "parameter_dir": str(args.parameter_dir)},
              "node": NODE, "nodes": NODES, "world": NODES*4, "shard_world": WORLD, "replica_nodes": REPLICAS, "torch": torch.__version__, "runtime_nccl_version": collective.lib.uccl_lite_e2e_version(),
              "gpu": [torch.cuda.get_device_name(r) for r in range(4)],
              "checkpoint_config": model.config, "trainable_parameters": model.parameter_count,
              "complete_block_bytes": model.complete_block_bytes,
              "compute": "checkpointed blocks, eager collectives" if args.checkpoint_blocks else ("compiled blocks, eager collectives" if args.compile_blocks else "eager"),
              "histogram_including_warmups": collective.histogram, "collective_oracle": "PASS", "reference_check": reference,
              "nccl_environment": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
              "mapped_nccl": sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "/libnccl.so" in line}),
              "peak_allocated_bytes": [torch.cuda.max_memory_allocated(r) for r in range(4)],
              "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    collective.close()
    collective.lib.uccl_lite_e2e_finalize()


if __name__ == "__main__":
    main()
