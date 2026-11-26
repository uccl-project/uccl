#!/usr/bin/env python3
"""Measure request-level streaming latency on an SGLang /generate endpoint."""

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer


CORPUS = (
    "The customer reported intermittent timeouts after the service update. "
    "The incident log shows that latency rose during the afternoon traffic peak. "
    "Engineers checked request queues, host utilization, and the database connection pool. "
    "They found no data loss, but several requests needed a retry before completion. "
    "Summarize the likely causes, the user impact, and the next diagnostic steps. "
)


async def one_request(session, url, ids, output_len):
    start = time.perf_counter()
    payload = {
        "input_ids": ids,
        "sampling_params": {
            "max_new_tokens": output_len,
            "min_new_tokens": output_len,
            "ignore_eos": True,
            "temperature": 0,
        },
        "stream": True,
    }
    first = None
    last = None
    tokens = 0
    output_text = ""
    async with session.post(url, json=payload) as response:
        response.raise_for_status()
        async for line in response.content:
            if not line.startswith(b"data: ") or line == b"data: [DONE]\n":
                continue
            data = json.loads(line[6:])
            if "error" in data:
                raise RuntimeError(data["error"])
            count = data.get("meta_info", {}).get("completion_tokens", 0)
            output_text = data.get("text", output_text)
            if count > tokens:
                now = time.perf_counter()
                first = now if first is None else first
                last = now
                tokens = count
    if tokens != output_len or first is None:
        raise RuntimeError(f"Expected {output_len} output tokens, received {tokens}")
    return {"start": start, "first": first, "last": last, "tokens": tokens,
            "complete": time.perf_counter(), "text": output_text}


async def run_batch(session, url, prompts, output_len):
    submitted = time.perf_counter()
    results = await asyncio.gather(
        *(one_request(session, url, ids, output_len) for ids in prompts)
    )
    firsts = [x["first"] for x in results]
    lasts = [x["last"] for x in results]
    last_ttft = max(firsts) - submitted
    whole_batch = max(lasts) - submitted
    client_batch = max(x["complete"] for x in results) - submitted
    return {
        "last_ttft_s": last_ttft,
        "whole_batch_s": whole_batch,
        "batch_tail_tpot_ms": (
            1000 * (whole_batch - last_ttft) / (output_len - 1)
            if output_len > 1 else None
        ),
        "client_batch_s": client_batch,
        "input_tokens_per_s": sum(map(len, prompts)) / client_batch,
        "output_sha256": hashlib.sha256(json.dumps(
            [x["text"] for x in results], ensure_ascii=False
        ).encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        "request_median_ttft_s": statistics.median(x["first"] - x["start"] for x in results),
        "request_median_tpot_ms": statistics.median(
            1000 * (x["last"] - x["first"]) / (x["tokens"] - 1)
            for x in results
        ) if output_len > 1 else None,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30089/generate")
    parser.add_argument("--model", default="/data/wxzheng/cache/huggingface/models--zai-org--GLM-4.5-Air-FP8/snapshots/f9a9c5acf5e543cd24d659a056c5dbcda78ffcfc")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "32:128:128",
            "16:512:64",
            "32:512:64",
            "8:64:256",
            "32:64:256",
            "64:256:32",
        ],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=0)
    parser.add_argument("--warmup-case", default="8:64:16")
    parser.add_argument("--case-warmups", type=int, default=0)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.repeats < 1 or min(args.warmup_repeats, args.case_warmups) < 0:
        parser.error("repeats must be positive and warmup-repeats nonnegative")

    def make_case(case):
        batch, input_len, output_len = map(int, case.split(":"))
        if min(batch, input_len, output_len) < 1:
            raise ValueError("Batch size and token counts must be positive")
        base = tokenizer.encode(CORPUS, add_special_tokens=False)
        base *= (input_len + len(base) - 1) // len(base)
        prompts = []
        for i in range(batch):
            prefix = tokenizer.encode(f"Request {i}: ", add_special_tokens=False)
            prompts.append((prefix + base)[:input_len])
        assert all(len(ids) == input_len for ids in prompts)
        return prompts, output_len

    cases = [(case, *make_case(case)) for case in args.cases]
    timeout = aiohttp.ClientTimeout(total=900)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async with session.get(args.url.rsplit("/", 1)[0] + "/health") as response:
            response.raise_for_status()
        warmup_prompts, warmup_output_len = make_case(args.warmup_case)
        for _ in range(args.warmup_repeats):
            await run_batch(session, args.url, warmup_prompts, warmup_output_len)
        for case, prompts, output_len in cases:
            for _ in range(args.case_warmups):
                await run_batch(session, args.url, prompts, output_len)
            for repeat in range(args.repeats):
                row = {"backend": args.backend, "case": case, "repeat": repeat,
                       "run_id": args.run_id}
                row.update(await run_batch(session, args.url, prompts, output_len))
                with args.out.open("a") as out:
                    out.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
