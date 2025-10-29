import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import ClosedIntervalTree


def test_complete_example():
    """Complete example test for closed intervals"""
    print("=== Complete Closed Interval IntervalTree Example ===")

    # Create closed interval tree
    closed_tree = ClosedIntervalTree()

    # Add closed intervals
    intervals = [
        (1, 10, "large_dataset"),
        (2, 5, "sub_dataset_1"),
        (3, 4, "core_data"),
        (15, 25, "another_region"),
        (20, 30, "overlapping_region"),
    ]

    for start, end, data in intervals:
        closed_tree.add(start, end, data)

    print("All closed intervals:")
    print(closed_tree)

    # Test containing queries
    print("\n=== Containing Queries ===")
    test_queries = [(3, 4), (16, 18), (22, 24), (12, 14)]

    for query_start, query_end in test_queries:
        containing = closed_tree.query_containing(query_start, query_end)
        print(f"\nQuery [{query_start}, {query_end}]:")

        if containing:
            print(f"containing: {containing[0][2]}")
            for start, end, data in containing:
                print(f"  Contained by [{start}, {end}] -> {data}")
        else:
            print("  Not contained by any interval")

    # Test removal operations
    print("\n=== Removal Operations ===")
    print("Before removal:", len(list(closed_tree)), "intervals")

    # Remove interval [2, 5]
    closed_tree.remove(2, 5, "sub_dataset_1")
    print("After removing [2, 5]:", len(list(closed_tree)), "intervals")

    # Display remaining intervals
    print("\nRemaining intervals:")
    for start, end, data in closed_tree:
        print(f"  [{start}, {end}] -> {data}")


def test_exact_match_function():
    """Test the exact match query function"""
    print("=== Testing Exact Match Query Function ===")

    # Create and populate the tree
    closed_tree = ClosedIntervalTree()

    intervals = [
        (1, 10, "dataset_1"),
        (2, 5, "dataset_2"),
        (2, 5, "dataset_2_duplicate"),  # Same boundaries, different data
        (3, 4, "dataset_3"),
        (15, 25, "dataset_4"),
        (15, 25, "dataset_4_copy"),  # Same boundaries, different data
    ]

    for start, end, data in intervals:
        closed_tree.add(start, end, data)

    print("All intervals:")
    print(closed_tree)

    # Test exact match queries
    test_cases = [
        (1, 10, None),  # Match any data with boundaries [1,10]
        (2, 5, None),  # Match any data with boundaries [2,5]
        (2, 5, "dataset_2"),  # Match specific data with boundaries [2,5]
        (3, 4, None),  # Match any data with boundaries [3,4]
        (15, 25, "dataset_4"),  # Match specific data with boundaries [15,25]
        (100, 200, None),  # Non-existent interval
        (2, 6, None),  # Non-existent boundaries
    ]

    print("\n=== Exact Match Query Results ===")
    for query_start, query_end, data_filter in test_cases:
        if data_filter is None:
            query_desc = f"[{query_start}, {query_end}] (any data)"
        else:
            query_desc = f"[{query_start}, {query_end}] -> {data_filter}"

        matches = closed_tree.query_exact_match(query_start, query_end, data_filter)

        print(f"\nQuery: {query_desc}")
        if matches:
            print(f"Found {len(matches)} exact match(es):")
            for start, end, data in matches:
                print(f"  [{start}, {end}] -> {data}")
        else:
            print("No exact matches found")

    # Compare with other query types
    print("\n=== Comparison with Other Query Types ===")
    query_start, query_end = 2, 5

    exact_matches = closed_tree.query_exact_match(query_start, query_end)
    containing = closed_tree.query_containing(query_start, query_end)
    overlapping = closed_tree.query_overlap(query_start, query_end)

    print(f"Query: [{query_start}, {query_end}]")
    print(f"Exact matches: {len(exact_matches)} intervals")
    print(f"Containing: {len(containing)} intervals")
    print(f"Overlapping: {len(overlapping)} intervals")


if __name__ == "__main__":
    test_complete_example()
    test_exact_match_function()
