"""Regression tests for advertisev() argument validation."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
p2p = pytest.importorskip("uccl.p2p")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="advertisev requires a CUDA-capable endpoint"
)


@pytest.fixture(scope="module")
def endpoint():
    return p2p.Endpoint(local_gpu_idx=torch.cuda.current_device())


@pytest.mark.parametrize(
    ("mr_ids", "ptrs", "sizes", "num_iovs"),
    [
        ([], [0], [0], 1),
        ([0], [], [0], 1),
        ([0], [0], [], 1),
        ([0, 1], [0], [0], 1),
        ([0], [0, 1], [0], 1),
        ([0], [0], [0, 1], 1),
        ([], [], [], 1),
        ([0], [0], [0], 0),
    ],
    ids=[
        "mr-ids-short",
        "pointers-short",
        "sizes-short",
        "mr-ids-long",
        "pointers-long",
        "sizes-long",
        "all-empty-with-nonzero-count",
        "nonempty-with-zero-count",
    ],
)
def test_advertisev_rejects_mismatched_vector_lengths(
    endpoint, mr_ids, ptrs, sizes, num_iovs
):
    """Mismatched inputs must raise before advertisev indexes the vectors."""
    with pytest.raises(
        RuntimeError, match="All input vectors/lists must have length num_iovs"
    ):
        endpoint.advertisev(mr_ids, ptrs, sizes, num_iovs)


def test_advertisev_accepts_an_empty_batch(endpoint):
    assert endpoint.advertisev([], [], [], 0) == (True, [])
