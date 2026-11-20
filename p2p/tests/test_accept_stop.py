"""Regressions for passive acceptance and shutdown."""

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


def test_blocking_accept_after_passive_accept():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import time\n"
            "from uccl import p2p\n"
            "receiver = p2p.Endpoint(0)\n"
            "sender = p2p.Endpoint(0)\n"
            "receiver.start_passive_accept()\n"
            "ip, port, bdf = receiver.parse_metadata(receiver.get_metadata())\n"
            "assert sender.connect(ip, bdf, port)[0]\n"
            # Let the passive worker publish the peer before claiming it.
            "time.sleep(1.1)\n"
            "assert receiver.accept()[0]\n"
            "del sender\n"
            "del receiver\n",
        ],
        check=True,
        timeout=30,
    )
