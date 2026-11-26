"""Build the NCCL proxy and its validated EP8 profile as a Linux wheel."""

from pathlib import Path
import shutil
import subprocess

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


HERE = Path(__file__).resolve().parent
PACKAGE = "uccl_lite_nccl_ep8"
NATIVE_LIBRARY = HERE / "build" / "libuccl_lite_nccl_compat.so"
SCRIPTS = (
    "run_packet_8l4.sh",
    "run_sglang_ag_gate_8l4.sh",
    "run_sglang_ag_ready_rs_8l4.sh",
    "run_sglang_ready_combine_8l4.sh",
    "run_sglang_ready_rs_8l4.sh",
    "run_sglang_rs_sum2_8l4.sh",
)


class PlatformDistribution(Distribution):
    def has_ext_modules(self):
        return True


class BuildWithProxy(build_py):
    def run(self):
        subprocess.run(
            ["make", "-C", str(HERE), "build/libuccl_lite_nccl_compat.so"],
            check=True,
        )
        super().run()
        destination = Path(self.build_lib) / PACKAGE
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(NATIVE_LIBRARY, destination / NATIVE_LIBRARY.name)
        for name in SCRIPTS:
            shutil.copy2(HERE / name, destination / name)
        hooks = destination / "sglang_hooks"
        hooks.mkdir(exist_ok=True)
        for source in (HERE / "sglang_hooks").glob("*.py"):
            shutil.copy2(source, hooks / source.name)


class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, platform = super().get_tag()
        return "py3", "none", platform


setup(
    name="uccl-lite-nccl",
    version="0.1.1",
    description="UCCL Lite NCCL launcher with an experimental L4 EP8 profile",
    long_description=(HERE / "WHEEL.md").read_text(),
    long_description_content_type="text/markdown",
    packages=[PACKAGE],
    package_data={PACKAGE: ["*.so", "*.sh", "sglang_hooks/*.py"]},
    include_package_data=True,
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "uccl-lite-nccl=uccl_lite_nccl_ep8.cli:main",
            "uccl-lite-ep8=uccl_lite_nccl_ep8.cli:main",
        ]
    },
    distclass=PlatformDistribution,
    cmdclass={"build_py": BuildWithProxy, "bdist_wheel": PlatformWheel},
    zip_safe=False,
)
