#!/usr/bin/env python3

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp

from service_benchmark.config import load_config
from service_benchmark.runner import BenchmarkRunner, atomic_json
from service_benchmark.workloads import prompt_stream


class Task2Runner(BenchmarkRunner):
    async def wait_ready(self, process, log_path: Path) -> None:
        await super().wait_ready(process, log_path)
        assert self.tokenizer is not None
        case = self.config.cases[0]
        prompt = next(prompt_stream(self.tokenizer, case, self.config.workload))
        payload = {
            "model": self.config.inference.served_model_name,
            "prompt": prompt.value,
            "max_tokens": 64,
            "min_tokens": 64,
            "temperature": 0.0,
            "ignore_eos": True,
            "return_token_ids": True,
            "stream": False,
        }
        timeout = aiohttp.ClientTimeout(
            total=self.config.inference.request_timeout_seconds
        )
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.base_url}/v1/completions", json=payload
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"token diagnostic HTTP {response.status}: "
                        f"{(await response.text())[:2000]}"
                    )
                body = await response.json()
        choice = body["choices"][0]
        token_ids = choice.get("token_ids")
        if not isinstance(token_ids, list) or len(token_ids) != 64:
            raise RuntimeError(
                f"token diagnostic did not return 64 token IDs: {choice}"
            )
        atomic_json(
            self.config.result_dir / "parity.json",
            {
                "prompt_tokens": prompt.token_count,
                "completion_tokens": len(token_ids),
                "token_ids": token_ids,
                "text": choice.get("text", ""),
            },
        )
        print("[token-diagnostic] captured 64 greedy output tokens", flush=True)


async def run(config_paths: list[str], case_names: set[str] | None) -> None:
    configs = [load_config(path) for path in config_paths]
    for config in configs:
        await Task2Runner(config, case_names=case_names).run()

    if len(configs) != 2:
        return

    baseline = json.loads(
        (configs[0].result_dir / "parity.json").read_text(encoding="utf-8")
    )
    optimized = json.loads(
        (configs[1].result_dir / "parity.json").read_text(encoding="utf-8")
    )
    match = baseline["token_ids"] == optimized["token_ids"]
    comparison = {
        "status": "pass" if match else "warning",
        "severity": "diagnostic",
        "blocking": False,
        "exact_token_id_match": match,
        "prompt_tokens": baseline["prompt_tokens"],
        "completion_tokens": baseline["completion_tokens"],
        "baseline_token_ids": baseline["token_ids"],
        "pcieccl_token_ids": optimized["token_ids"],
    }
    output = configs[0].result_dir.parent / "parity_comparison.json"
    atomic_json(output, comparison)
    if not match:
        print(
            f"[token-diagnostic] warning: NCCL and PCIeCCL output token IDs "
            f"differ (non-blocking); "
            f"wrote {output}",
            flush=True,
        )
        return
    print(f"[token-diagnostic] exact output token match; wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument(
        "--case",
        dest="case_names",
        action="append",
        help="run only the named case while preserving the full config signature",
    )
    args = parser.parse_args()
    asyncio.run(run(args.configs, set(args.case_names) if args.case_names else None))


if __name__ == "__main__":
    main()
