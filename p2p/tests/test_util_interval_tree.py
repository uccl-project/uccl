import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import ClosedIntervalTree


# ---------------------------------------------------------------------------
# add / basic iteration
# ---------------------------------------------------------------------------


def test_add_and_iterate():
    tree = ClosedIntervalTree()
    tree.add(1, 10, "a")
    tree.add(2, 5, "b")
    intervals = list(tree)
    assert len(intervals) == 2
    datas = {d for _, _, d in intervals}
    assert datas == {"a", "b"}


def test_add_single_point_interval():
    """start == end is a valid degenerate interval."""
    tree = ClosedIntervalTree()
    tree.add(5, 5, "point")
    assert len(list(tree)) == 1


def test_add_rejects_inverted_interval():
    tree = ClosedIntervalTree()
    with pytest.raises(ValueError):
        tree.add(10, 5, "bad")


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


def test_remove_existing_interval():
    tree = ClosedIntervalTree()
    tree.add(1, 10, "a")
    tree.add(2, 5, "b")
    removed = tree.remove(2, 5, "b")
    assert removed == 1
    assert len(list(tree)) == 1


def test_remove_nonexistent_returns_zero():
    tree = ClosedIntervalTree()
    tree.add(1, 10, "a")
    removed = tree.remove(999, 1000, "ghost")
    assert removed == 0
    assert len(list(tree)) == 1  # original interval still present


def test_remove_by_boundaries_only():
    """Passing data=None removes any interval with those exact boundaries."""
    tree = ClosedIntervalTree()
    tree.add(1, 10, "a")
    removed = tree.remove(1, 10)
    assert removed == 1
    assert len(list(tree)) == 0


# ---------------------------------------------------------------------------
# query_containing
# ---------------------------------------------------------------------------


def test_query_containing_basic():
    tree = ClosedIntervalTree()
    tree.add(1, 10, "large")
    tree.add(2, 5, "small")

    # [3, 4] is contained by both
    result = tree.query_containing(3, 4)
    datas = {d for _, _, d in result}
    assert "large" in datas
    assert "small" in datas


def test_query_containing_exact_boundary():
    """A query that exactly matches an interval's boundaries is contained."""
    tree = ClosedIntervalTree()
    tree.add(5, 15, "exact")
    result = tree.query_containing(5, 15)
    assert len(result) == 1
    assert result[0][2] == "exact"


def test_query_containing_no_match():
    tree = ClosedIntervalTree()
    tree.add(1, 10, "buf")
    # Query extends beyond the stored interval — not contained
    result = tree.query_containing(5, 20)
    assert result == []


def test_query_containing_empty_tree():
    tree = ClosedIntervalTree()
    assert tree.query_containing(0, 100) == []


# ---------------------------------------------------------------------------
# query_overlap
# ---------------------------------------------------------------------------


def test_query_overlap_partial():
    """Partial overlap should be returned."""
    tree = ClosedIntervalTree()
    tree.add(100, 200, "buf")
    result = tree.query_overlap(180, 300)
    assert len(result) == 1
    assert result[0][2] == "buf"


def test_query_overlap_full_containment():
    """A query fully inside a stored interval should match."""
    tree = ClosedIntervalTree()
    tree.add(100, 200, "buf")
    result = tree.query_overlap(120, 150)
    assert len(result) == 1


def test_query_overlap_no_match():
    tree = ClosedIntervalTree()
    tree.add(100, 200, "buf")
    result = tree.query_overlap(300, 400)
    assert result == []


def test_query_overlap_multiple_matches():
    tree = ClosedIntervalTree()
    tree.add(1, 10, "a")
    tree.add(5, 15, "b")
    tree.add(20, 30, "c")
    # [8, 12] overlaps a and b but not c
    result = tree.query_overlap(8, 12)
    datas = {d for _, _, d in result}
    assert datas == {"a", "b"}


def test_query_overlap_empty_tree():
    tree = ClosedIntervalTree()
    assert tree.query_overlap(0, 100) == []


# ---------------------------------------------------------------------------
# query_exact_match
# ---------------------------------------------------------------------------


def test_query_exact_match_any_data():
    tree = ClosedIntervalTree()
    tree.add(1, 10, "x")
    tree.add(1, 10, "y")
    result = tree.query_exact_match(1, 10)
    assert len(result) == 2


def test_query_exact_match_specific_data():
    tree = ClosedIntervalTree()
    tree.add(1, 10, "x")
    tree.add(1, 10, "y")
    result = tree.query_exact_match(1, 10, "x")
    assert len(result) == 1
    assert result[0][2] == "x"


def test_query_exact_match_no_result():
    tree = ClosedIntervalTree()
    tree.add(1, 10, "x")
    # Different boundaries — should not match
    result = tree.query_exact_match(1, 11)
    assert result == []


def test_query_exact_match_empty_tree():
    tree = ClosedIntervalTree()
    assert tree.query_exact_match(0, 10) == []


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_clear_empties_tree():
    tree = ClosedIntervalTree()
    tree.add(1, 10, "a")
    tree.add(20, 30, "b")
    tree.clear()
    assert list(tree) == []


def test_clear_then_add():
    """Tree should be fully usable after clear."""
    tree = ClosedIntervalTree()
    tree.add(1, 10, "a")
    tree.clear()
    tree.add(5, 15, "b")
    result = list(tree)
    assert len(result) == 1
    assert result[0][2] == "b"


def test_clear_empty_tree_is_safe():
    tree = ClosedIntervalTree()
    tree.clear()  # should not raise
    assert list(tree) == []
