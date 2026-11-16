"""Unscored finite-input bitwise investigation; run existing snapshots separately.

This is not an accuracy oracle or a change to any acceptance threshold. Compare
rows keyed by case/rank/dtype/mode/generation only when input hashes match.
The source and installed binary are identified by run_alias.py and a header.
"""

import argparse
import array
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


WORLD = 8
BASE_SEED = 2026111501
EXPONENTS = (-24, -12, -5, -1, 0, 3, 9, 17, 25)
SIZES = (128, 6 << 10, 12 << 10, 24 << 10, 32 << 10,
         (32 << 10) + 128, 48 << 10, (64 << 10) - 128, 64 << 10,
         (64 << 10) + 128, 96 << 10, 192 << 10, 384 << 10, 1 << 20)
GRAPH_GENERATIONS = (1, 2, 3)
PRODUCER_DELAY_CYCLES = 250000


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def seed_for(size, dtype_name, rank, offset, generation, scope):
    description = f"{BASE_SEED}/{size}/{dtype_name}/{rank}/{offset}/{generation}/{scope}"
    return int.from_bytes(hashlib.sha256(description.encode()).digest()[:8], "little")


def finite_payload(size, dtype_name, rank, offset, generation):
    """CPU-only raw IEEE bits with bounded exponents and cancellation strata.

    Adjacent rank pairs share magnitudes in half the lanes. The four strata
    are exact cancellation, one-mantissa-step cancellation, exponent-shifted
    cancellation, and same-sign addition. Other lanes use independent rank
    seeds. No input is a NaN, infinity, subnormal or zero. Individual magnitudes
    are below 2**27, so any eight-rank rounded sum remains far from overflow.
    """
    if sys.byteorder != "little":
        raise RuntimeError("This reproducible raw-bit protocol requires little endian")
    scalar_bytes, mantissa_bits, code = (2, 7, "H") if dtype_name == "bf16" else (4, 23, "I")
    if dtype_name not in ("bf16", "fp32") or size % scalar_bytes:
        raise ValueError("Unsupported payload specification")
    pair_seed = seed_for(size, dtype_name, rank // 2, offset, generation, "pair")
    rank_seed = seed_for(size, dtype_name, rank, offset, generation, "rank")
    pair_rng, rank_rng = random.Random(pair_seed), random.Random(rank_seed)
    words = array.array(code)
    if words.itemsize != scalar_bytes:
        raise RuntimeError("Unexpected native integer width")
    mask = (1 << mantissa_bits) - 1
    for index in range(size // scalar_bytes):
        common = pair_rng.getrandbits(64)
        independent = rank_rng.getrandbits(64)
        stratum = index % 8
        word = common if stratum < 4 else independent
        exponent = EXPONENTS[(word >> 32) % len(EXPONENTS)]
        mantissa = word & mask
        sign = (word >> 63) & 1
        if stratum < 4:
            sign ^= rank & 1
            if rank & 1:
                if stratum == 1:
                    mantissa = min(mantissa + 1, mask)
                elif stratum == 2:
                    exponent += 1
                elif stratum == 3:
                    sign ^= 1
        words.append((sign << (mantissa_bits + 8))
                     | ((127 + exponent) << mantissa_bits) | mantissa)
    return words.tobytes(), pair_seed, rank_seed


def atomic_emit(record):
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    limit = min(4096, os.fpathconf(1, "PC_PIPE_BUF"))
    if len(payload) > limit:
        raise RuntimeError(f"Diagnostic JSON row exceeds atomic write limit {limit}")
    if os.write(1, payload) != len(payload):
        raise RuntimeError("Short atomic diagnostic write")


def raw_cpu_bytes(tensor):
    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


class FiniteCase:
    def __init__(self, rank, dtype_name, size, offset):
        self.rank, self.dtype_name, self.size, self.offset = rank, dtype_name, size, offset
        self.dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
        self.count = size // self.dtype.itemsize
        self.prefix = 128 // self.dtype.itemsize + offset
        self.suffix = 128 // self.dtype.itemsize
        capacity = self.prefix + self.count + self.suffix
        self.device = torch.device("cuda", rank)
        self.input_storage = torch.empty(capacity, dtype=self.dtype, device=self.device)
        self.output_storage = torch.empty_like(self.input_storage)
        self.source = self.input_storage[self.prefix:self.prefix + self.count]
        self.output = self.output_storage[self.prefix:self.prefix + self.count]
        self.cpu_output_initial = torch.full((capacity,), -119, dtype=self.dtype)
        if (self.source.data_ptr() % 16 != offset * self.dtype.itemsize
                or self.output.data_ptr() % 16 != offset * self.dtype.itemsize):
            raise RuntimeError("Unexpected caller alignment")

    def prepare(self, generation):
        payload, self.pair_seed, self.rank_seed = finite_payload(
            self.size, self.dtype_name, self.rank, self.offset, generation)
        # A writable bytearray avoids frombuffer's immutable-buffer warning.
        cpu_source = torch.frombuffer(bytearray(payload), dtype=torch.uint8).view(self.dtype)
        if not bool(torch.isfinite(cpu_source).all()):
            raise RuntimeError("Finite-input generator produced a nonfinite value")
        self.cpu_initial = torch.full(self.input_storage.shape, -113, dtype=self.dtype,
                                      pin_memory=True)
        self.cpu_initial[self.prefix:self.prefix + self.count].copy_(cpu_source)
        self.input_bytes = payload
        self.input_sha = digest_bytes(payload)
        self.generation = generation

    def produce(self):
        torch.cuda._sleep(PRODUCER_DELAY_CYCLES)
        self.input_storage.copy_(self.cpu_initial, non_blocking=True)
        self.output_storage.fill_(-119)

    def enqueue(self, comm, stream):
        comm.all_reduce(self.source.data_ptr(), self.output.data_ptr(), self.count,
                        0 if self.dtype_name == "bf16" else 1, stream.cuda_stream)

    def check_and_emit(self, mode):
        record = {"type": "finite_bits", "case": f"bytes{self.size}-offset{self.offset}",
                  "rank": self.rank, "dtype": self.dtype_name, "mode": mode,
                  "generation": self.generation, "bytes": self.size,
                  "scalar_offset": self.offset, "pair_seed": self.pair_seed,
                  "rank_seed": self.rank_seed, "input_sha256": self.input_sha}
        try:
            input_bytes = raw_cpu_bytes(self.input_storage.cpu())
            expected_input = raw_cpu_bytes(self.cpu_initial)
            output_cpu = self.output_storage.cpu()
            output_bytes = raw_cpu_bytes(output_cpu)
            expected_output = raw_cpu_bytes(self.cpu_output_initial)
            begin = self.prefix * self.dtype.itemsize
            end = begin + self.size
            input_preserved = input_bytes == expected_input
            guards = (input_bytes[:begin] == expected_input[:begin]
                      and input_bytes[end:] == expected_input[end:]
                      and output_bytes[:begin] == expected_output[:begin]
                      and output_bytes[end:] == expected_output[end:])
            finite = bool(torch.isfinite(output_cpu[self.prefix:self.prefix + self.count]).all())
            record.update(output_sha256=digest_bytes(output_bytes[begin:end]),
                          input_preserved=input_preserved, boundary_guard_pass=guards,
                          output_finite=finite, status="pass" if input_preserved and guards and finite else "fail")
            atomic_emit(record)
            if not input_preserved or not guards or not finite:
                raise RuntimeError("Finite bitwise diagnostic detected mutation, guard damage or nonfinite output")
        except Exception as error:
            if "status" not in record:
                record.update(status="fail", error=f"{type(error).__name__}: {str(error)[:160]}")
                atomic_emit(record)
            raise


def worker(rank, port):
    torch.set_num_threads(1)
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD,
                            timeout=timedelta(minutes=10))
    from vllm import _pcieccl_C
    unique_id = [_pcieccl_C.create_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(unique_id, src=0)
    comm = _pcieccl_C.Communicator(rank, WORLD, unique_id[0], rank, 64 << 20)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    checked = 0
    for dtype_name in ("bf16", "fp32"):
        for size in SIZES:
            for offset in (0, 1):
                case = FiniteCase(rank, dtype_name, size, offset)
                stream = streams[(checked // 4) % 2]
                case.prepare(0)
                torch.cuda.synchronize()
                dist.barrier()
                with torch.cuda.stream(stream):
                    case.produce()
                    case.enqueue(comm, stream)
                stream.synchronize()
                case.check_and_emit("eager")
                checked += 1
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                with torch.cuda.graph(graph, stream=stream):
                    case.enqueue(comm, stream)
                dist.barrier()
                for generation in GRAPH_GENERATIONS:
                    case.prepare(generation)
                    dist.barrier()
                    with torch.cuda.stream(stream):
                        case.produce()
                        graph.replay()
                    stream.synchronize()
                    case.check_and_emit("graph")
                    checked += 1
                del graph, case
    dist.barrier()
    comm.close()
    dist.destroy_process_group()
    atomic_emit({"type": "finite_bits_complete", "rank": rank,
                 "checked_collectives": checked, "scored": False})


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if torch.cuda.device_count() != WORLD:
        raise RuntimeError("Expose exactly eight physical GPUs in order 0 through 7")
    root = Path(__file__).resolve().parents[1]
    package = Path(importlib.util.find_spec("vllm").origin).parent
    extension, = package.glob("_pcieccl_C*.so")
    atomic_emit({"type": "finite_bits_header", "protocol": "finite-bits-v1",
                 "test_sha256": digest_bytes(Path(__file__).read_bytes()),
                 "source_sha256": digest_bytes((root / "source/vllm-0.26.0/csrc/pcieccl/pcieccl.cu").read_bytes()),
                 "binary_sha256": digest_bytes(extension.read_bytes()),
                 "python": sys.version.split()[0], "torch": torch.__version__,
                 "base_seed": BASE_SEED, "byte_order": sys.byteorder,
                 "cases_per_rank": len(SIZES) * 2 * 2,
                 "checked_collectives_per_rank": len(SIZES) * 2 * 2 * 4,
                 "scored": False})
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(port,), nprocs=WORLD, join=True)


if __name__ == "__main__":
    main()
