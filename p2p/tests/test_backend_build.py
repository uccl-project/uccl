"""Check profile isolation, optional SDK boundaries, and object invalidation."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class BackendBuildTests(unittest.TestCase):
    def make(self, directory, *args):
        result = subprocess.run(
            [
                "make",
                "PYTHON=true",
                "PYTHON_CONFIG=true",
                "NB_DIR=/test/nanobind",
                "NB_OBJECTS=",
                *args,
            ],
            cwd=directory,
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_profiles_share_sources_without_crossing_device_sdks(self):
        runtimes = {
            "cuda": "-lcudart",
            "rocm": "-lamdhip64",
            "dtk": "-lamdhip64",
            "therock": "-lamdhip64",
            "cnrt": "-lcnrt",
            "musa": "-lmusart",
        }
        for profile, library in runtimes.items():
            with self.subTest(profile=profile):
                entry = "Makefile" if profile == "cuda" else f"Makefile.{profile}"
                output = self.make(ROOT / "p2p", "-n", "-f", entry, "all")
                self.assertIn(library, output)
                self.assertIn(f".build/{profile}/", output)
                for other in set(runtimes.values()) - {library}:
                    self.assertNotIn(other, output)
                if profile == "musa":
                    for forbidden in (
                        "cxi/cxi_endpoint.cc",
                        "nccl/nccl_endpoint.cc",
                        "-I./nccl",
                        "-I/opt/libfabric/include",
                    ):
                        self.assertNotIn(forbidden, output)
                if profile == "cnrt":
                    # Both Python and C entry points must link the staging TU.
                    links = [
                        line
                        for line in output.splitlines()
                        if " -o p2p" in line or " -o libuccl_p2p.so" in line
                    ]
                    self.assertEqual(len(links), 2)
                    self.assertTrue(
                        all("device/cnrt/staging.o" in line for line in links)
                    )

    def test_disabling_optional_transports_removes_their_sources(self):
        output = self.make(ROOT / "p2p", "-n", "ENABLE_NCCL=0", "ENABLE_CXI=0", "all")
        for forbidden in (
            "nccl_endpoint.cc",
            "cxi_endpoint.cc",
            "fabric_dl.cc",
            "nccl_dl.cc",
        ):
            self.assertNotIn(forbidden, output)

    def test_native_link_does_not_require_python_embedding_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "python-config"
            calls = Path(str(config) + ".calls")
            config.write_text(
                "#!/usr/bin/env python3\nimport sys\n"
                "with open(__file__ + '.calls', 'a') as f:\n"
                "    f.write(' '.join(sys.argv[1:]) + '\\n')\n"
                "print('-lpython-contract' if '--embed' in sys.argv else '')\n"
            )
            config.chmod(0o755)
            for profile in ("cuda", "rocm", "dtk", "therock", "cnrt", "musa"):
                with self.subTest(profile=profile):
                    output = self.make(
                        ROOT / "p2p",
                        "-n",
                        f"PROFILE={profile}",
                        f"PYTHON_CONFIG={config}",
                        "libuccl_p2p.so",
                    )
                    self.assertNotIn("-lpython", output)
                    self.assertIn("--no-undefined", output)
                    self.assertEqual(calls.read_text(), "--ldflags\n")
                    calls.unlink()

    def test_same_config_reuses_object_but_changed_flags_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "p2p/make", root / "make")
            shutil.copyfile(ROOT / "p2p/Makefile", root / "Makefile")
            (root / "engine.cc").write_text("// compiler double input\n")
            compiler = root / "compiler"
            compiler.write_text(
                "#!/usr/bin/env python3\nimport sys, pathlib\n"
                "pathlib.Path(sys.argv[sys.argv.index('-o') + 1]).touch()\n"
                "with open('calls', 'a') as f: f.write('compile\\n')\n"
            )
            compiler.chmod(0o755)
            args = (f"CXX={compiler}", "BUILD_DIR=objects", "objects/engine.o")
            self.make(root, *args)
            self.make(root, *args)
            self.assertEqual((root / "calls").read_text().count("compile"), 1)
            # GNU make 3.81 compares mtimes at whole-second resolution.
            for artifact in ("objects/config", "objects/engine.o"):
                os.utime(root / artifact, (100, 100))
            self.make(root, *args, "CXXFLAGS=-DCHANGED=1")
            self.assertEqual((root / "calls").read_text().count("compile"), 2)


if __name__ == "__main__":
    unittest.main()
