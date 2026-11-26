#!/usr/bin/env python3
"""Replay selected BurstGPT arrival/length traces against SGLang /generate."""

import argparse
import asyncio
import csv
import json
import statistics
import time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

from bench_e2e_workloads import one_request


def load_prompts(path, tokenizer, min_tokens):
    # A range-fetched JSON prefix ends in the middle of an item; parse only
    # complete conversations. The BurstGPT release contains no prompt text.
    content = path.read_bytes().decode("utf-8", "ignore")
    decoder = json.JSONDecoder()
    pos = content.index("[") + 1
    prompts = []
    while True:
        while pos < len(content) and content[pos] in " \t\r\n,":
            pos += 1
        try:
            conversation, pos = decoder.raw_decode(content, pos)
        except json.JSONDecodeError:
            break
        for turn in conversation.get("conversations", []):
            text = turn.get("value")
            if turn.get("from") == "human" and isinstance(text, str) and len(text) >= min_tokens:
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) >= min_tokens:
                    prompts.append(ids)
    return prompts


def read_trace(path, minutes):
    with path.open(newline="") as handle:
        rows = [
            {
                "source_row": int(row["source_row"]),
                "minute": int(row["minute"]),
                "offset_s": float(row["timestamp_s"]) - 60 * int(row["minute"]),
                "input_len": int(row["request_tokens"]),
                "output_len": int(row["response_tokens"]),
            }
            for row in csv.DictReader(handle, delimiter="\t")
            if int(row["minute"]) in minutes
        ]
    rows.sort(key=lambda row: (row["minute"], row["offset_s"], row["source_row"]))
    return rows


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, int(len(ordered) * fraction + 0.999999) - 1)]


async def replay_one(session, url, row, ids, base):
    await asyncio.sleep(max(0, base + row["offset_s"] - time.perf_counter()))
    result = await one_request(session, url, ids, row["output_len"])
    return row, result


async def replay(session, url, rows, prompts):
    base = time.perf_counter()
    results = await asyncio.gather(
        *(replay_one(session, url, row, ids, base) for row, ids in zip(rows, prompts))
    )
    ttfts = [result["first"] - result["start"] for _, result in results]
    tpots = [
        1000 * (result["last"] - result["first"]) / (result["tokens"] - 1)
        for _, result in results
    ]
    e2es = [result["last"] - result["start"] for _, result in results]
    lags = [result["start"] - (base + row["offset_s"]) for row, result in results]
    makespan = max(result["last"] for _, result in results) - base
    return {
        "requests": len(rows),
        "input_tokens": sum(row["input_len"] for row in rows),
        "output_tokens": sum(row["output_len"] for row in rows),
        "ttft_p50_s": statistics.median(ttfts),
        "ttft_p95_s": percentile(ttfts, 0.95),
        "tpot_p50_ms": statistics.median(tpots),
        "tpot_p95_ms": percentile(tpots, 0.95),
        "e2e_p50_s": statistics.median(e2es),
        "e2e_p95_s": percentile(e2es, 0.95),
        "makespan_s": makespan,
        "output_tokens_per_s": sum(row["output_len"] for row in rows) / makespan,
        "dispatch_lag_max_ms": 1000 * max(lags),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30089/generate")
    parser.add_argument("--model", default="/data/wxzheng/cache/huggingface/models--zai-org--GLM-4.5-Air-FP8/snapshots/f9a9c5acf5e543cd24d659a056c5dbcda78ffcfc")
    parser.add_argument("--trace", type=Path, default=Path(__file__).with_name("burstgpt_windows_20261112.tsv"))
    parser.add_argument("--sharegpt-prefix", type=Path, required=True)
    parser.add_argument("--minutes", type=int, nargs="+", default=[9242, 14622])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    trace = read_trace(args.trace, set(args.minutes))
    pool = load_prompts(args.sharegpt_prefix, tokenizer, min(row["input_len"] for row in trace))
    inputs = {}
    for index, row in enumerate(trace):
        candidates = [ids for ids in pool if len(ids) >= row["input_len"]]
        if not candidates:
            raise RuntimeError(f"No real prompt reaches {row['input_len']} tokens")
        inputs[row["source_row"]] = candidates[(index * 37) % len(candidates)][: row["input_len"]]

    timeout = aiohttp.ClientTimeout(total=900)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        async with session.get(args.url.rsplit("/", 1)[0] + "/health") as response:
            response.raise_for_status()
        for minute in args.minutes:
            rows = [row for row in trace if row["minute"] == minute]
            prompts = [inputs[row["source_row"]] for row in rows]
            for repeat in range(args.repeats):
                result = await replay(session, args.url, rows, prompts)
                record = {"backend": args.backend, "minute": minute, "repeat": repeat, **result}
                with args.out.open("a") as output:
                    output.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
