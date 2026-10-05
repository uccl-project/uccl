"""CPU-only config selection tests"""

import ast
import os
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch


class Config:
    def __init__(self, num_sms, *args):
        self.num_sms = num_sms
        self.values = (num_sms, *args)


def load_buffer_config_methods():
    # Execute the real config methods without importing CUDA-dependent Buffer.
    source = Path(__file__).parents[1] / "bench" / "buffer.py"
    tree = ast.parse(source.read_text())
    buffer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Buffer"
    )
    methods = {
        "_config_from_env",
        "_validated_env_configs",
        "get_dispatch_config",
        "get_combine_config",
    }
    buffer.body = [
        node
        for node in buffer.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    namespace = {"os": os, "Config": Config, "Optional": Optional, "Tuple": tuple}
    exec(  # noqa: S102 -- execute only trusted methods from this repository
        compile(ast.Module(body=[buffer], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    result = namespace["Buffer"]
    result.num_sms = 20
    result._is_efa = staticmethod(lambda: False)
    return result


Buffer = load_buffer_config_methods()


class RuntimeConfigTest(unittest.TestCase):
    def test_defaults_and_matching_overrides(self):
        cases = [
            ({}, (20, 6, 256, 6, 128), (20, 4, 256, 6, 128)),
            (
                {"UCCL_EP_DISPATCH_CONFIG": "20,8,256,6,128"},
                (20, 8, 256, 6, 128),
                (20, 4, 256, 6, 128),
            ),
            (
                {"UCCL_EP_COMBINE_CONFIG": "20,9,256,6,128"},
                (20, 6, 256, 6, 128),
                (20, 9, 256, 6, 128),
            ),
            (
                {
                    "UCCL_EP_DISPATCH_CONFIG": "24,8,256,6,128",
                    "UCCL_EP_COMBINE_CONFIG": "24,9,256,6,128",
                },
                (24, 8, 256, 6, 128),
                (24, 9, 256, 6, 128),
            ),
        ]
        for env, dispatch, combine in cases:
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                self.assertEqual(Buffer.get_dispatch_config(8).values, dispatch)
                self.assertEqual(Buffer.get_combine_config(8).values, combine)

    def test_mismatched_overrides_rejected_by_both_getters(self):
        cases = [
            {"UCCL_EP_DISPATCH_CONFIG": "24,6,256,6,128"},
            {"UCCL_EP_COMBINE_CONFIG": "24,4,256,6,128"},
            {
                "UCCL_EP_DISPATCH_CONFIG": "24,6,256,6,128",
                "UCCL_EP_COMBINE_CONFIG": "22,4,256,6,128",
            },
        ]
        for env in cases:
            for getter in (Buffer.get_dispatch_config, Buffer.get_combine_config):
                with (
                    self.subTest(env=env, getter=getter.__name__),
                    patch.dict(os.environ, env, clear=True),
                    self.assertRaisesRegex(ValueError, "same SM count"),
                ):
                    getter(8)


if __name__ == "__main__":
    unittest.main()
