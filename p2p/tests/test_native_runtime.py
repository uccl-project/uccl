"""The test allocator must use the runtime selected by the compiled module."""

import ctypes
import os
import unittest
from unittest.mock import Mock, patch
import native_runtime


class NativeRuntimeTests(unittest.TestCase):
    def test_compiled_identity_drives_symbols_and_copy_enums(self):
        for kind, prefix, library, h2d, init in (
            ("cuda", "cuda", "libcudart.so", 1, ""),
            ("hip", "hip", "libamdhip64.so", 1, "hipInit"),
            ("cnrt", "cnrt", "libcnrt.so", 0, ""),
            ("musa", "musa", "libmusart.so", 1, ""),
        ):
            with self.subTest(kind=kind):
                info = dict(
                    device=kind,
                    runtime_prefix=prefix,
                    runtime_library=library,
                    runtime_init=init,
                    memcpy_h2d=h2d,
                    memcpy_d2h=2,
                )
                lib = Mock()
                for suffix in ("Malloc", "Free", "Memcpy", "SetDevice"):
                    getattr(lib, prefix + suffix).return_value = 0
                lib.hipInit.return_value = 0
                with (
                    patch.object(native_runtime, "backend_info", return_value=info),
                    patch.object(ctypes, "CDLL", return_value=lib) as loader,
                    patch.dict(os.environ, {"UCCL_GPU_RT": "unrelated"}, clear=True),
                ):
                    rt = native_runtime.Rt()
                    loader.assert_called_once_with(library)
                    rt.set_device(3)
                    rt.memcpy_h2d(1, 2, 64)
                    rt.memcpy_d2h(3, 4, 64)
                getattr(lib, prefix + "SetDevice").assert_called_once_with(3)
                self.assertEqual(
                    getattr(lib, prefix + "Memcpy").call_args_list[0].args,
                    (1, 2, 64, h2d),
                )
                self.assertEqual(
                    getattr(lib, prefix + "Memcpy").call_args_list[1].args,
                    (3, 4, 64, 2),
                )
                if init:
                    lib.hipInit.assert_called_once_with(0)
                else:
                    lib.hipInit.assert_not_called()

    def test_library_override_does_not_change_device(self):
        info = dict(
            device="musa",
            runtime_prefix="musa",
            runtime_library="libmusart.so",
            runtime_init="",
            memcpy_h2d=1,
            memcpy_d2h=2,
        )
        with (
            patch.object(native_runtime, "backend_info", return_value=info),
            patch.object(ctypes, "CDLL", return_value=Mock()) as loader,
            patch.dict(
                os.environ, {"UCCL_GPU_RT_LIB": "/test/libmusart.so"}, clear=True
            ),
        ):
            self.assertEqual(native_runtime.Rt().kind, "musa")
            loader.assert_called_once_with("/test/libmusart.so")

    def test_runtime_errors_are_not_silently_ignored(self):
        with self.assertRaisesRegex(RuntimeError, "failed err=5"):
            native_runtime.Rt._chk(5, "allocation")


if __name__ == "__main__":
    unittest.main()
