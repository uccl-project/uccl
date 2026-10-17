"""Independent slot-order oracle and paired GPU expert-routing measurements."""
import json
import statistics
import time
from typing import Literal

import torch
from expert_routes import expert_routes
from ht_bridge import Dispatched


def consume(value: Dispatched, experts: int, strategy: Literal['where', 'sorted']) -> None:
    for route in expert_routes(value, experts, strategy):
        assert route.rows.numel() == route.probability.numel()


@torch.inference_mode()
def main() -> None:
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(20261013)
    cases = 0
    for experts in (1, 16, 128):
        for rows in (1, 31, 64, 8192):
            for pattern in ('uniform', 'hot', 'masked', 'empty'):
                indices = torch.randint(experts, (rows, 8), generator=generator)
                if pattern == 'hot':
                    indices.remainder_(min(experts, 3))
                elif pattern == 'masked':
                    indices[:, ::2] = -1
                elif pattern == 'empty':
                    indices.fill_(-1)
                weights = torch.randn(rows, 8, generator=generator).bfloat16().float()
                expected: list[list[tuple[int, float]]] = [[] for _ in range(experts)]
                for slot, (index, probability) in enumerate(zip(indices.flatten().tolist(), weights.flatten().tolist(), strict=True)):
                    if index >= 0:
                        expected[index].append((slot // 8, probability))
                value = Dispatched(torch.empty((rows, 1), device='cuda', dtype=torch.bfloat16),
                                   weights.cuda(), indices.cuda())
                for strategy in ('where', 'sorted'):
                    actual = [[] for _ in range(experts)]
                    for route in expert_routes(value, experts, strategy):
                        actual[route.expert] = list(zip(route.rows.cpu().tolist(), route.probability.cpu().tolist(), strict=True))
                    assert actual == expected, (experts, rows, pattern, strategy)
                cases += 1
                print(json.dumps({'test': 'expert_route_oracle', 'experts': experts,
                                  'rows': rows, 'topk': 8, 'pattern': pattern, 'status': 'PASS'}), flush=True)
                if experts == 128 and pattern in ('uniform', 'hot', 'masked'):
                    samples = {'where': [], 'sorted': []}
                    for strategy in samples:
                        consume(value, experts, strategy)
                    for repeat in range(10):
                        order = ('where', 'sorted') if repeat % 2 == 0 else ('sorted', 'where')
                        for strategy in order:
                            torch.cuda.synchronize()
                            started = time.perf_counter()
                            consume(value, experts, strategy)
                            torch.cuda.synchronize()
                            samples[strategy].append((time.perf_counter() - started) * 1000)
                    print(json.dumps({'test': 'expert_route_timing', 'rows': rows,
                                      'pattern': pattern, 'samples_ms': samples,
                                      'speedup': statistics.median(samples['where']) / statistics.median(samples['sorted']),
                                      'scope': 'routing only, complete host+GPU time; not model or communication',
                                      'status': 'PASS'}), flush=True)
    print(json.dumps({'test': 'expert_route_summary', 'cases': cases, 'status': 'PASS'}), flush=True)


if __name__ == '__main__':
    main()
