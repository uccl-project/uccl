"""Regression for shutting down passive acceptance before memory registration."""

import subprocess
import sys


def test_lazy_passive_accept_destruction():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import time; from uccl import p2p; "
            "ep = p2p.Endpoint(); ep.start_passive_accept(); "
            "time.sleep(0.1); del ep",
        ],
        check=True,
        timeout=20,
    )
