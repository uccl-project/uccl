import pathlib
import sys
import os


_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "python"))

from uccl_gin.microbench import MicrobenchConfig, run_microbench  # noqa: E402


def test_uccl_gin_microbench_correctness():
    if os.environ.get("UCCL_GIN_RUN_MICROBENCH") != "1":
        return

    result = run_microbench(MicrobenchConfig.from_env(_ROOT))
    print(result.stdout)
    result.assert_correct()


if __name__ == "__main__":
    test_uccl_gin_microbench_correctness()
