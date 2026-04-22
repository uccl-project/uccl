"""Config dataclass and recommended dispatch/combine configs.

The field layout matches ``uccl.ep.Config`` so instances can be passed
straight through to the C++ runtime.
"""

from typing import NamedTuple

try:
    from uccl import ep as _uccl_ep
except ImportError as exc:  # pragma: no cover - import error surfaced lazily
    _uccl_ep = None
    _import_error = exc
else:
    _import_error = None


class Config(NamedTuple):
    """Performance-tuning configuration for dispatch/combine kernels.

    Attributes match the constructor of ``uccl.ep.Config``.
    """

    num_sms: int
    num_max_nvl_chunked_send_tokens: int
    num_max_nvl_chunked_recv_tokens: int
    num_max_rdma_chunked_send_tokens: int
    num_max_rdma_chunked_recv_tokens: int

    def to_uccl_config(self):
        if _uccl_ep is None:
            raise RuntimeError(
                "uccl.ep is not importable; cannot build a runtime Config. "
                f"Original error: {_import_error}"
            )
        return _uccl_ep.Config(
            int(self.num_sms),
            int(self.num_max_nvl_chunked_send_tokens),
            int(self.num_max_nvl_chunked_recv_tokens),
            int(self.num_max_rdma_chunked_send_tokens),
            int(self.num_max_rdma_chunked_recv_tokens),
        )


_default_num_sms = 20


def set_default_num_sms(num_sms: int) -> None:
    """Set the default number of SMs used by the recommended configs.

    Larger values trade SM occupancy against communication bandwidth.
    20 is the default matching ``deep_ep_wrapper``; 64/80 are typical for
    larger intranode-only configurations.
    """
    global _default_num_sms
    assert num_sms > 0 and num_sms % 2 == 0, "num_sms must be a positive even number"
    _default_num_sms = int(num_sms)


def _require_num_ranks(num_ranks: int) -> int:
    if num_ranks is None:
        # Late import so this module can be imported without JAX being
        # fully initialized.
        import jax

        num_ranks = jax.device_count()
    return int(num_ranks)


# Recommended configs mirror the tables in
# `deep_ep_wrapper/deep_ep/buffer.py`.
_DISPATCH_TABLE = {
    2: (24, 256, 6, 128),
    4: (6, 256, 6, 128),
    8: (6, 256, 6, 128),
    16: (36, 288, 20, 128),
    24: (8, 288, 32, 128),
    32: (32, 288, 32, 128),
    64: (20, 288, 28, 128),
    128: (20, 560, 32, 128),
    144: (32, 720, 12, 128),
    160: (28, 720, 12, 128),
}

_COMBINE_TABLE = {
    2: (10, 256, 6, 128),
    4: (9, 256, 6, 128),
    8: (4, 256, 6, 128),
    16: (4, 288, 12, 128),
    24: (1, 288, 8, 128),
    32: (1, 288, 8, 128),
    64: (1, 288, 20, 128),
    128: (1, 560, 12, 128),
    144: (2, 720, 8, 128),
    160: (2, 720, 8, 128),
}


def get_dispatch_config(num_ranks: int = None) -> Config:
    """Return the recommended dispatch config for ``num_ranks``.

    If ``num_ranks`` is omitted, ``jax.device_count()`` is consulted, which
    returns the global number of devices across all processes when the JAX
    distributed client has been initialized.
    """
    num_ranks = _require_num_ranks(num_ranks)
    if num_ranks not in _DISPATCH_TABLE:
        raise ValueError(f"Unsupported number of EP ranks: {num_ranks}")
    return Config(_default_num_sms, *_DISPATCH_TABLE[num_ranks])


def get_combine_config(num_ranks: int = None) -> Config:
    """Return the recommended combine config for ``num_ranks``."""
    num_ranks = _require_num_ranks(num_ranks)
    if num_ranks not in _COMBINE_TABLE:
        raise ValueError(f"Unsupported number of EP ranks: {num_ranks}")
    return Config(_default_num_sms, *_COMBINE_TABLE[num_ranks])
