"""UKernel P2P - Point-to-point communication via transport communicator."""

# Import torch first: the extension links torch libraries (libc10 etc.)
# which are only resolvable once torch's own loader has run.
import torch  # noqa: F401

from ukernel_p2p._C import Communicator

__all__ = ["Communicator"]
