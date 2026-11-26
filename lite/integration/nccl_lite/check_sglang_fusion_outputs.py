"""Capture deterministic GLM EP8 outputs for packet/fusion comparison."""

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp


PROMPTS = [
    "Explain packet routing in one sentence.",
    "A server's p95 latency rose after a deployment. List two likely causes.",
    "Translate to Chinese: The database connection pool is exhausted.",
    "Compute 37 * 19 and show the arithmetic.",
    "Write a Python function that returns the median of three numbers.",
    "Summarize why a GPU collective needs ordered stream execution.",
    "A customer saw intermittent timeouts but no data loss. What should the operator check first?",
    "Describe the difference between prefill and decode in language model serving.",
]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30089/generate")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def generate(prompt):
            payload = {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                    "min_new_tokens": 32,
                    "ignore_eos": True,
                },
            }
            async with session.post(args.url, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                if "error" in result:
                    raise RuntimeError(result["error"])
                return {"prompt": prompt, "text": result["text"]}

        results = await asyncio.gather(*(generate(prompt) for prompt in PROMPTS))
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(f"captured {len(results)} deterministic outputs in {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
