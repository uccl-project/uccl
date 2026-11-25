"""Load the production signal helper without libpython, then exercise its API."""

import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PY_INCLUDE = Path(sysconfig.get_path("include"))
LOADER = r"""
#include <cassert>
#include <cstring>
#include <dlfcn.h>
int main(int argc, char** argv) {
  if (argc == 4) assert(dlopen(argv[3], RTLD_NOW | RTLD_GLOBAL));
  void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  assert(library); // Must load in a native process with no Python symbols.
  auto poll = reinterpret_cast<void (*)()>(dlsym(library, "poll_signals"));
  assert(poll);
  if (std::strcmp(argv[2], "poll") == 0) { poll(); poll(); }
  dlclose(library);
}
"""
PYTHON_API = r"""
#include <Python.h>
#include <cassert>
#include <cstdlib>
static int stage = 0;
extern "C" int Py_IsInitialized() {
  return !std::getenv("TEST_PY_UNINITIALIZED");
}
extern "C" PyGILState_STATE PyGILState_Ensure() {
  assert(stage == 0); stage = 1; return PyGILState_UNLOCKED;
}
extern "C" int PyErr_CheckSignals() {
  assert(stage == 1); stage = 2;
  return std::getenv("TEST_PY_SIGNAL") ? -1 : 0;
}
extern "C" void PyGILState_Release(PyGILState_STATE state) {
  assert(state == PyGILState_UNLOCKED && stage == 2); stage = 0;
}
"""


@unittest.skipUnless((PY_INCLUDE / "Python.h").is_file(), "Python headers required")
class PythonSignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.root = Path(directory.name)
        (cls.root / "util").mkdir()
        # Preserve fatal control flow without generating core dumps in tests.
        (cls.root / "util/debug.h").write_text(
            "#pragma once\n#include <cstdlib>\n#include <iostream>\n"
            "struct FatalLog {\n"
            "  ~FatalLog() { std::exit(99); }\n"
            "  std::ostream& stream() { return std::cerr; }\n"
            "};\n#define UCCL_LOG(...) FatalLog().stream()\n"
        )
        (cls.root / "signals.cc").write_text(
            '#include "util/python_signal.h"\n'
            'extern "C" void poll_signals() { check_python_signals(); }\n'
        )
        (cls.root / "loader.cc").write_text(LOADER)
        (cls.root / "python_api.cc").write_text(PYTHON_API)
        compiler = [
            os.environ.get("CXX", "c++"),
            "-std=c++17",
            "-I",
            str(cls.root),
            "-I",
            str(ROOT / "p2p"),
            "-I",
            str(PY_INCLUDE),
        ]
        link = [] if sys.platform == "darwin" else ["-ldl"]
        shared = ["-shared", "-fPIC"]
        if sys.platform != "darwin":
            shared += ["-Wl,--no-undefined"]
        for name in ("signals", "python_api", "loader"):
            flags = [] if name == "loader" else shared
            output = name if name == "loader" else f"{name}.so"
            result = subprocess.run(
                compiler
                + flags
                + [str(cls.root / f"{name}.cc"), "-o", str(cls.root / output)]
                + link,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode:
                raise AssertionError(result.stderr)

    def run_loader(self, mode, provider=False, **env):
        args = [str(self.root / "loader"), str(self.root / "signals.so"), mode]
        if provider:
            args.append(str(self.root / "python_api.so"))
        return subprocess.run(
            args,
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_native_load_does_not_require_an_interpreter(self):
        result = self.run_loader("load")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.run_loader("poll")
        self.assertEqual(result.returncode, 99, result.stderr)
        self.assertIn("requires an active interpreter", result.stderr)

    def test_active_interpreter_with_and_without_the_gil(self):
        source = (
            "import ctypes, sys\n"
            "for loader in (ctypes.CDLL, ctypes.PyDLL):\n"
            "    lib = loader(sys.argv[1])\n"
            "    lib.poll_signals.restype = None\n"
            "    lib.poll_signals.argtypes = []\n"
            "    lib.poll_signals()\n"
            "    lib.poll_signals()\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", source, str(self.root / "signals.so")],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_gil_lifetime_and_fatal_signal_handling(self):
        for env, code, message in (
            ({}, 0, ""),
            ({"TEST_PY_SIGNAL": "1"}, 99, "Python signal caught"),
            ({"TEST_PY_UNINITIALIZED": "1"}, 99, "requires an active interpreter"),
        ):
            with self.subTest(env=env):
                result = self.run_loader("poll", provider=True, **env)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
