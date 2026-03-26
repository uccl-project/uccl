#!/usr/bin/env python3
"""
Verify from Python that repeated register_memory() calls on the same tensor
reuse the underlying registered MR state while still returning distinct
API-level mr_ids.
"""

from __future__ import annotations

import sys
import torch

try:
    from uccl import p2p
except ImportError as exc:
    sys.stderr.write(f"Failed to import p2p: {exc}\n")
    raise


def _field(desc, name: str):
    if isinstance(desc, dict):
        return desc[name]
    return getattr(desc, name)


def _print_desc(tag: str, desc) -> None:
    print(
        f"{tag}: addr={_field(desc, 'addr')} size={_field(desc, 'size')} "
        f"mr_id={_field(desc, 'mr_id')} lkeys={_field(desc, 'lkeys')} "
        f"rkeys={_field(desc, 'rkeys')}"
    )


def _make_test_tensor() -> torch.Tensor:
    # Prefer pinned host memory when available, but plain CPU memory is enough
    # to drive ibv_reg_mr in environments where GPU buffers fall back to IPC.
    base = torch.arange(4096, dtype=torch.float32)
    try:
        return base.pin_memory()
    except RuntimeError:
        return base


def test_register_memory_cache() -> bool:
    device_idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
    tensor = _make_test_tensor()
    ep = p2p.Endpoint(device_idx)

    desc1 = ep.register_memory([tensor])[0]
    desc2 = ep.register_memory([tensor])[0]

    _print_desc("first", desc1)
    _print_desc("second", desc2)

    assert _field(desc1, "addr") == tensor.data_ptr()
    assert _field(desc2, "addr") == tensor.data_ptr()
    assert _field(desc1, "size") == tensor.numel() * tensor.element_size()
    assert _field(desc2, "size") == tensor.numel() * tensor.element_size()

    if _field(desc1, "mr_id") == (2**64 - 1) or not _field(desc1, "lkeys"):
        print("Skipping test: register_memory fell back to IPC-only metadata")
        return True

    # register_memory() still creates a fresh API-level handle per call.
    assert _field(desc1, "mr_id") != _field(
        desc2, "mr_id"
    ), "register_memory should return distinct mr_id handles"

    # The underlying cached MR should be reused, so NIC keys stay identical.
    assert _field(desc1, "lkeys") == _field(
        desc2, "lkeys"
    ), "expected identical lkeys for repeated registration of the same tensor"
    assert _field(desc1, "rkeys") == _field(
        desc2, "rkeys"
    ), "expected identical rkeys for repeated registration of the same tensor"

    # Releasing one descriptor should not evict the cached MR while another
    # registration for the same buffer is still alive.
    ep.deregister_memory([desc1])
    desc3 = ep.register_memory([tensor])[0]
    _print_desc("third", desc3)

    assert _field(desc2, "lkeys") == _field(
        desc3, "lkeys"
    ), "expected cache reuse while an existing registration is still alive"
    assert _field(desc2, "rkeys") == _field(
        desc3, "rkeys"
    ), "expected cache reuse while an existing registration is still alive"

    ep.deregister_memory([desc2, desc3])
    print("✓ register_memory cache verification passed")
    return True


def test_register_memory_subregion_cache() -> bool:
    device_idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
    tensor = _make_test_tensor()
    sub_tensor = tensor[256:1280]
    ep = p2p.Endpoint(device_idx)

    base_desc = ep.register_memory([tensor])[0]
    sub_desc = ep.register_memory([sub_tensor])[0]

    _print_desc("base", base_desc)
    _print_desc("sub", sub_desc)

    assert _field(base_desc, "addr") == tensor.data_ptr()
    assert _field(sub_desc, "addr") == sub_tensor.data_ptr()
    assert _field(base_desc, "size") == tensor.numel() * tensor.element_size()
    assert _field(sub_desc, "size") == sub_tensor.numel() * sub_tensor.element_size()

    if _field(base_desc, "mr_id") == (2**64 - 1) or not _field(base_desc, "lkeys"):
        print("Skipping subregion test: register_memory fell back to IPC-only metadata")
        return True

    assert _field(base_desc, "mr_id") != _field(
        sub_desc, "mr_id"
    ), "subregion registration should still return a distinct API handle"
    assert _field(base_desc, "lkeys") == _field(
        sub_desc, "lkeys"
    ), "expected subregion registration to reuse cached lkeys from the base region"
    assert _field(base_desc, "rkeys") == _field(
        sub_desc, "rkeys"
    ), "expected subregion registration to reuse cached rkeys from the base region"

    ep.deregister_memory([base_desc])
    sub_desc_2 = ep.register_memory([sub_tensor])[0]
    _print_desc("sub_again", sub_desc_2)

    assert _field(sub_desc, "lkeys") == _field(
        sub_desc_2, "lkeys"
    ), "expected subregion cache entry to stay alive while references remain"
    assert _field(sub_desc, "rkeys") == _field(
        sub_desc_2, "rkeys"
    ), "expected subregion cache entry to stay alive while references remain"

    ep.deregister_memory([sub_desc, sub_desc_2])
    print("✓ register_memory subregion cache verification passed")
    return True


def test_register_memory_partial_overlap_no_cache() -> bool:
    device_idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
    tensor = _make_test_tensor()
    overlap_tensor = tensor[768:1792]
    ep = p2p.Endpoint(device_idx)

    base_desc = ep.register_memory([tensor[:1024]])[0]
    overlap_desc = ep.register_memory([overlap_tensor])[0]

    _print_desc("partial_base", base_desc)
    _print_desc("partial_overlap", overlap_desc)

    if _field(base_desc, "mr_id") == (2**64 - 1) or not _field(base_desc, "lkeys"):
        print(
            "Skipping partial-overlap test: register_memory fell back to IPC-only metadata"
        )
        return True

    assert _field(base_desc, "addr") != _field(
        overlap_desc, "addr"
    ), "partial-overlap buffers should start at different addresses"
    assert _field(base_desc, "lkeys") != _field(
        overlap_desc, "lkeys"
    ), "partial overlap without containment should not reuse cached lkeys"
    assert _field(base_desc, "rkeys") != _field(
        overlap_desc, "rkeys"
    ), "partial overlap without containment should not reuse cached rkeys"

    ep.deregister_memory([base_desc, overlap_desc])
    print("✓ register_memory partial-overlap no-cache verification passed")
    return True


def test_register_memory_non_overlap_no_cache() -> bool:
    device_idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
    tensor_a = _make_test_tensor()
    tensor_b = _make_test_tensor().clone()
    ep = p2p.Endpoint(device_idx)

    desc_a = ep.register_memory([tensor_a])[0]
    desc_b = ep.register_memory([tensor_b])[0]

    _print_desc("non_overlap_a", desc_a)
    _print_desc("non_overlap_b", desc_b)

    if _field(desc_a, "mr_id") == (2**64 - 1) or not _field(desc_a, "lkeys"):
        print(
            "Skipping non-overlap test: register_memory fell back to IPC-only metadata"
        )
        return True

    assert _field(desc_a, "addr") != _field(
        desc_b, "addr"
    ), "non-overlap buffers should have different addresses"
    assert _field(desc_a, "lkeys") != _field(
        desc_b, "lkeys"
    ), "independent buffers should not reuse cached lkeys"
    assert _field(desc_a, "rkeys") != _field(
        desc_b, "rkeys"
    ), "independent buffers should not reuse cached rkeys"

    ep.deregister_memory([desc_a, desc_b])
    print("✓ register_memory non-overlap no-cache verification passed")
    return True


def main() -> int:
    try:
        ok = test_register_memory_cache()
        ok = test_register_memory_subregion_cache() and ok
        ok = test_register_memory_partial_overlap_no_cache() and ok
        ok = test_register_memory_non_overlap_no_cache() and ok
        return 0 if ok else 1
    except Exception as exc:
        print(
            f"✗ register_memory cache verification failed: {type(exc).__name__}: {exc}"
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
