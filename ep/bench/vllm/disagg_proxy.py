#!/usr/bin/env python3
"""
Minimal PD Disaggregation Proxy for vLLM + NixlConnector.

Routes client requests through prefill → decode with KV transfer metadata.

Usage:
    python3 disagg_proxy.py --prefill-url http://<PREFILL_HEAD>:8100 \
                            --decode-url http://<DECODE_HEAD>:8000 \
                            --port 9000

Client sends requests to this proxy (port 9000). The proxy:
  1. Sends the prompt to prefill (max_tokens=1) to populate KV cache
  2. Extracts kv_transfer_params from prefill response
  3. Forwards request + kv_transfer_params to decode for token generation
"""

import argparse
import json
import sys

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()

PREFILL_URL = ""
DECODE_URL = ""
VERBOSE = False


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    # Step 1: Send to prefill with max_tokens=1 to populate KV cache.
    # do_remote_decode=True tells prefill's NixlConnector that a remote
    # node will handle decode. On request completion,
    # it holds KV blocks in GPU memory and returns kv_transfer_params with
    # block IDs, engine ID, and side channel address for the decode node.
    prefill_body = dict(body)
    prefill_body["max_tokens"] = 1
    prefill_body["stream"] = False
    prefill_body.pop("stream_options", None)
    prefill_body["kv_transfer_params"] = {"do_remote_decode": True}

    async with aiohttp.ClientSession() as session:
        # Prefill request
        async with session.post(
            f"{PREFILL_URL}/v1/chat/completions",
            json=prefill_body,
            headers={"Content-Type": "application/json"},
        ) as prefill_resp:
            if prefill_resp.status != 200:
                error = await prefill_resp.text()
                print(
                    f"[ERROR] Prefill {prefill_resp.status}: {error}", file=sys.stderr
                )
                return {"error": f"Prefill failed: {error}"}
            prefill_result = await prefill_resp.json()

        # Step 2: Extract kv_transfer_params from prefill response.
        # NixlConnector returns {do_remote_prefill=True, do_remote_decode=False,
        # remote_block_ids, remote_engine_id, remote_host, remote_port, ...}.
        # The do_remote_prefill=True flag tells the decode node to RDMA-read
        # the KV cache from prefill instead of recomputing it.
        kv_transfer_params = prefill_result.get("kv_transfer_params")
        if not kv_transfer_params:
            # Fallback: check in usage or metadata
            usage = prefill_result.get("usage", {})
            kv_transfer_params = usage.get("kv_transfer_params")

        if not kv_transfer_params:
            print(
                "[WARN] No kv_transfer_params in prefill response. "
                "Decode will run without KV transfer (full recompute).",
                file=sys.stderr,
            )
        elif VERBOSE:
            print(
                f"[KV] kv_transfer_params: {json.dumps(kv_transfer_params, indent=2)}",
                file=sys.stderr,
            )

        # Step 3: Forward to decode with kv_transfer_params
        decode_body = dict(body)
        if kv_transfer_params:
            decode_body["kv_transfer_params"] = kv_transfer_params

        is_stream = body.get("stream", False)

        if is_stream:

            async def stream_decode():
                async with aiohttp.ClientSession() as s:
                    async with s.post(
                        f"{DECODE_URL}/v1/chat/completions",
                        json=decode_body,
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        async for chunk in resp.content.iter_any():
                            yield chunk

            return StreamingResponse(stream_decode(), media_type="text/event-stream")
        else:
            async with session.post(
                f"{DECODE_URL}/v1/chat/completions",
                json=decode_body,
                headers={"Content-Type": "application/json"},
            ) as decode_resp:
                return await decode_resp.json()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    # Proxy decode node's metrics since that's where generation happens
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{DECODE_URL}/metrics") as resp:
            from fastapi.responses import Response

            return Response(
                content=await resp.read(),
                media_type=resp.content_type,
            )


@app.get("/v1/models")
async def models(request: Request):
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{DECODE_URL}/v1/models") as resp:
            return await resp.json()


def main():
    parser = argparse.ArgumentParser(
        description="PD Disaggregation Proxy for vLLM + NixlConnector"
    )
    parser.add_argument(
        "--prefill-url",
        required=True,
        help="Prefill head URL (e.g., http://10.173.44.108:8100)",
    )
    parser.add_argument(
        "--decode-url",
        required=True,
        help="Decode head URL (e.g., http://10.173.44.104:8000)",
    )
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print kv_transfer_params for each request",
    )
    args = parser.parse_args()

    global PREFILL_URL, DECODE_URL, VERBOSE
    PREFILL_URL = args.prefill_url.rstrip("/")
    DECODE_URL = args.decode_url.rstrip("/")
    VERBOSE = args.verbose

    print(f"Disagg Proxy: prefill={PREFILL_URL}, decode={DECODE_URL}")
    print(f"Listening on http://{args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
