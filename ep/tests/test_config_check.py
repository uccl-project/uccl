"""CPU-only tests for the cross-rank Buffer configuration check."""

import multiprocessing as mp
import os
import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "bench"))
import config_check as cc  # noqa: E402

BASE = {
    "build": {
        "num_max_nvl_peers": 8,
        "num_proxy_threads": 4,
        "per_expert_batching": False,
        "sm90_features": True,
        "disable_aggressive_ptx_instrs": False,
    },
    "args": {"low_latency_mode": False, "num_nvl_bytes": 1 << 30, "num_rdma_bytes": 0},
    "config": {"dispatch": (20, 6, 256, 6, 128), "combine": (20, 9, 256, 6, 128)},
}


def signature(env=None, **over):
    with patch.dict(os.environ, env or {}, clear=True):
        sig = {k: dict(v) for k, v in BASE.items()}
        sig["env"] = cc.effective_env()
    for path, value in over.items():
        section, key = path.split("__")
        sig[section][key] = value
    return sig


def entry(sig, mode="on", error=""):
    return {
        "version": cc.SIGNATURE_VERSION,
        "mode": mode,
        "signature": sig,
        "error": error,
    }


class EffectiveEnvTest(unittest.TestCase):
    def test_unset_equals_explicit_default(self):
        explicit = {
            "UCCL_EP_TRANSPORT": "ib",
            "UCCL_EP_DEDUP_CE": "0",
            "UCCL_EP_LL_NUM_SMS": "0",
            "UCCL_EP_GPU_MR_RO": "0",
        }
        self.assertEqual(signature()["env"], signature(explicit)["env"])

    def test_values_follow_the_cpp_parsing(self):
        env = signature(
            {
                "UCCL_EP_TRANSPORT": "cxi",
                "UCCL_EP_DEDUP_CE": " 2x",
                "UCCL_EP_DEDUP_SEND": "warp",
                "UCCL_EP_LL_NUM_SMS": "-3",
                "UCCL_EP_GPU_MR_RO": "1yes",
            }
        )["env"]
        self.assertEqual(env["UCCL_EP_TRANSPORT"], "cxi")
        self.assertTrue(env["UCCL_EP_DEDUP_CE"])
        self.assertEqual(env["UCCL_EP_DEDUP_SEND"], "warp")
        self.assertEqual(env["UCCL_EP_LL_NUM_SMS"], 0)
        self.assertTrue(env["UCCL_EP_GPU_MR_RO"])


class CompareTest(unittest.TestCase):
    def decide(self, entries):
        return cc.compare(entries)[0]

    def test_consistent_passes(self):
        self.assertEqual(self.decide([entry(signature())] * 4), "pass")

    def test_hard_mismatch_is_an_error(self):
        cases = [
            signature(config__dispatch=(24, 6, 256, 6, 128)),
            signature(build__num_max_nvl_peers=4),
            signature(args__num_nvl_bytes=1 << 29),
            signature({"UCCL_EP_TRANSPORT": "cxi"}),
        ]
        for odd in cases:
            with self.subTest(odd=odd):
                entries = [entry(signature())] * 3 + [entry(odd)]
                decision, _, lines = cc.compare(entries)
                self.assertEqual(decision, "error")
                self.assertIn("on ranks [3]", "\n".join(lines))

    def test_warn_only_mismatch_warns(self):
        odd = signature({"UCCL_EP_DEDUP_SEND": "warp"})
        self.assertEqual(self.decide([entry(signature()), entry(odd)]), "warn")

    def test_unrelated_env_is_ignored(self):
        odd = signature({"UCCL_IB_GID_INDEX": "3", "UCCL_EP_CPU_TIMEOUT_SECS": "9"})
        self.assertEqual(self.decide([entry(signature()), entry(odd)]), "pass")

    def test_unknown_key_is_treated_as_hard(self):
        odd = signature()
        odd["build"] = dict(odd["build"], new_layout_flag=True)
        self.assertEqual(self.decide([entry(signature()), entry(odd)]), "error")

    def test_signature_error_is_an_error(self):
        entries = [entry(signature()), entry({}, error="ValueError: bad config")]
        self.assertEqual(self.decide(entries), "error")

    def test_most_permissive_mode_wins(self):
        odd = signature(config__dispatch=(24, 6, 256, 6, 128))
        off = [entry(signature(), mode="off"), entry(odd)]
        warn = [entry(signature()), entry(odd, mode="warn")]
        self.assertEqual(self.decide(off), "pass")
        self.assertEqual(self.decide(warn), "warn")

    def test_mode_parsing(self):
        for value, mode in (("0", "off"), ("warn", "warn"), ("1", "on"), ("", "on")):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {cc.CHECK_ENV: value}, clear=True),
            ):
                self.assertEqual(cc.check_mode(), mode)


def _gloo_worker(rank, world, port, dispatch, mode, queue):
    import torch.distributed as dist

    os.environ.pop(cc.CHECK_ENV, None)
    if mode is not None:
        os.environ[cc.CHECK_ENV] = mode
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    try:
        result = cc.check_consistency(
            dist.group.WORLD, signature(config__dispatch=dispatch)
        )
    except RuntimeError as exc:
        result = "raised: " + str(exc)
    queue.put((rank, result))
    dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class GlooTest(unittest.TestCase):
    """Every rank must reach the same decision, including ranks that opted out."""

    WORLD = 4

    def run_case(self, dispatch_by_rank, mode_by_rank):
        try:
            import torch.distributed  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        ctx = mp.get_context("spawn")
        queue, port = ctx.Queue(), _free_port()
        procs = [
            ctx.Process(
                target=_gloo_worker,
                args=(r, self.WORLD, port, dispatch_by_rank[r], mode_by_rank[r], queue),
            )
            for r in range(self.WORLD)
        ]
        for p in procs:
            p.start()
        results = dict(queue.get(timeout=60) for _ in procs)
        for p in procs:
            p.join(timeout=30)
        return [results[r] for r in range(self.WORLD)]

    def test_mismatch_raises_the_same_error_on_every_rank(self):
        good, odd = (20, 6, 256, 6, 128), (24, 6, 256, 6, 128)
        results = self.run_case([good, good, odd, good], [None] * 4)
        self.assertEqual(len(set(results)), 1)
        self.assertTrue(results[0].startswith("raised: "))
        self.assertIn("config.dispatch", results[0])

    def test_opt_out_on_one_rank_does_not_hang_the_others(self):
        good, odd = (20, 6, 256, 6, 128), (24, 6, 256, 6, 128)
        results = self.run_case([good, good, odd, good], ["0", None, None, None])
        self.assertEqual(results, ["pass"] * 4)


if __name__ == "__main__":
    unittest.main()
