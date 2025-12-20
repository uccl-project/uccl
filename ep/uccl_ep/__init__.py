import torch

try:
    from uccl import ep
    from uccl.ep import Config
except ImportError as exc:
    import sys

    sys.stderr.write("Failed to import uccl.ep\n")
    raise

from .utils import EventOverlap
from .buffer import Buffer
