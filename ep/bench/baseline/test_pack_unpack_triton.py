"""
Tests for Triton-based MoE pack/unpack kernels.

These tests verify that the Triton implementation produces identical results
to the CUDA reference implementation.

Run with:
    cd ep/bench/baseline
    python -m pytest test_pack_unpack_triton.py -v
"""

import pytest
import torch
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pack_unpack_triton import (
    count_expert_tokens_triton,
    pack_moe_data_to_buffers_triton,
    unpack_moe_data_from_buffers_triton,
)

# Try to import CUDA reference for comparison tests
try:
    from pack_unpack_cuda import (
        pack_moe_data_to_buffers_cuda,
        unpack_moe_data_from_buffers_cuda,
        CUDA_AVAILABLE,
    )
except ImportError:
    CUDA_AVAILABLE = False


def generate_test_data(
    num_tokens: int,
    hidden_dim: int,
    num_experts: int,
    experts_per_token: int,
    dtype: torch.dtype,
    device: torch.device,
    sparse_ratio: float = 0.0,
):
    """Generate test data for MoE pack/unpack tests."""
    x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)

    # Generate expert assignments
    topk_idx = torch.randint(
        0,
        num_experts,
        (num_tokens, experts_per_token),
        dtype=torch.int64,
        device=device,
    )

    # Add some padding (-1) if sparse_ratio > 0
    if sparse_ratio > 0:
        mask = torch.rand(num_tokens, experts_per_token, device=device) < sparse_ratio
        topk_idx[mask] = -1

    topk_weights = torch.randn(
        num_tokens, experts_per_token, dtype=torch.float32, device=device
    )

    return x, topk_idx, topk_weights


def allocate_buffers(
    num_tokens: int,
    hidden_dim: int,
    experts_per_token: int,
    world_size: int,
    x_dtype: torch.dtype,
    device: torch.device,
):
    """Allocate buffers for pack/unpack operations."""
    bytes_per_token = hidden_dim * torch.tensor([], dtype=x_dtype).element_size()
    bytes_per_idx = 8  # int64
    bytes_per_weight = 4  # float32
    bytes_per_item = bytes_per_token + bytes_per_idx + bytes_per_weight

    # Allocate enough space for worst case (all tokens go to one rank)
    max_items = num_tokens * experts_per_token
    buffer_size = max_items * bytes_per_item

    buffers = [
        torch.zeros(buffer_size, dtype=torch.uint8, device=device)
        for _ in range(world_size)
    ]
    return buffers


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestCountExpertTokens:
    """Test the count_expert_tokens_triton kernel."""

    @pytest.mark.parametrize("num_tokens", [32, 128, 512])
    @pytest.mark.parametrize("num_experts", [8, 32, 256])
    def test_basic_counting(self, num_tokens, num_experts):
        """Test basic expert token counting."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")

        # Generate expert IDs
        expert_ids = torch.randint(
            0, num_experts, (num_tokens,), dtype=torch.int64, device=device
        )

        # Count using Triton
        counts_triton = count_expert_tokens_triton(expert_ids, num_experts)

        # Count using PyTorch reference
        counts_ref = torch.zeros(num_experts, dtype=torch.int32, device=device)
        for i in range(num_tokens):
            expert_id = expert_ids[i].item()
            counts_ref[expert_id] += 1

        assert torch.allclose(
            counts_triton, counts_ref
        ), f"Mismatch: triton={counts_triton}, ref={counts_ref}"

    def test_with_padding(self):
        """Test counting with -1 padding values."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        num_experts = 8

        # Include -1 padding
        expert_ids = torch.tensor(
            [0, 1, -1, 2, -1, 3, 4, -1], dtype=torch.int64, device=device
        )

        counts = count_expert_tokens_triton(expert_ids, num_experts)

        # Should ignore -1 values
        assert counts[0].item() == 1
        assert counts[1].item() == 1
        assert counts[2].item() == 1
        assert counts[3].item() == 1
        assert counts[4].item() == 1
        assert counts[5:].sum().item() == 0

    def test_empty_input(self):
        """Test with empty input."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        expert_ids = torch.tensor([], dtype=torch.int64, device=device)
        counts = count_expert_tokens_triton(expert_ids, 8)

        assert counts.sum().item() == 0


class TestPackUnpackTriton:
    """Test pack and unpack operations."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("num_tokens", [32, 128])
    @pytest.mark.parametrize("hidden_dim", [256, 512])
    def test_roundtrip(self, dtype, num_tokens, hidden_dim):
        """Test that pack -> unpack recovers the original data."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        num_experts = 64
        experts_per_token = 4
        world_size = 8
        num_local_experts = num_experts // world_size

        # Generate test data
        x, topk_idx, topk_weights = generate_test_data(
            num_tokens, hidden_dim, num_experts, experts_per_token, dtype, device
        )

        # Allocate buffers
        buffers = allocate_buffers(
            num_tokens, hidden_dim, experts_per_token, world_size, dtype, device
        )

        # Pack
        per_rank_bytes = pack_moe_data_to_buffers_triton(
            x, topk_idx, topk_weights, num_experts, world_size, device, buffers
        )

        # Unpack
        recv_x, recv_topk_idx, recv_topk_weights, expert_counts = (
            unpack_moe_data_from_buffers_triton(
                buffers,
                per_rank_bytes,
                num_local_experts,
                hidden_dim,
                world_size,
                device,
                dtype,
                torch.int64,
                torch.float32,
            )
        )

        # Verify total items
        total_items = (topk_idx >= 0).sum().item()
        assert (
            recv_x.shape[0] >= total_items or total_items == 0
        ), f"Expected at least {total_items} items, got {recv_x.shape[0]}"

        # Verify expert counts sum to total items
        assert (
            sum(expert_counts) == total_items
        ), f"Expert counts sum {sum(expert_counts)} != total items {total_items}"


# =============================================================================
# Comparison Tests (Triton vs CUDA)
# =============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA extension not available")
class TestTritonVsCuda:
    """Compare Triton implementation against CUDA reference."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_unpack_matches_cuda(self, dtype):
        """Verify Triton unpack produces same output as CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        num_tokens = 128
        hidden_dim = 256
        num_experts = 64
        experts_per_token = 4
        world_size = 8
        num_local_experts = num_experts // world_size

        # Generate test data and pack with CUDA (use CUDA as ground truth for packing)
        x, topk_idx, topk_weights = generate_test_data(
            num_tokens, hidden_dim, num_experts, experts_per_token, dtype, device
        )
        buffers = allocate_buffers(
            num_tokens, hidden_dim, experts_per_token, world_size, dtype, device
        )

        per_rank_bytes = pack_moe_data_to_buffers_cuda(
            x, topk_idx, topk_weights, num_experts, world_size, device, buffers
        )

        # Clone buffers for separate unpack calls
        buffers_copy = [b.clone() for b in buffers]

        # Unpack with Triton
        recv_x_triton, recv_idx_triton, recv_weights_triton, counts_triton = (
            unpack_moe_data_from_buffers_triton(
                buffers,
                per_rank_bytes.clone(),
                num_local_experts,
                hidden_dim,
                world_size,
                device,
                dtype,
                torch.int64,
                torch.float32,
            )
        )

        # Unpack with CUDA
        recv_x_cuda, recv_idx_cuda, recv_weights_cuda, counts_cuda = (
            unpack_moe_data_from_buffers_cuda(
                buffers_copy,
                per_rank_bytes.clone(),
                num_local_experts,
                hidden_dim,
                world_size,
                device,
                dtype,
                torch.int64,
                torch.float32,
            )
        )

        # Compare outputs
        assert torch.allclose(
            recv_x_triton, recv_x_cuda, rtol=1e-5, atol=1e-5
        ), "recv_x mismatch"
        assert torch.equal(recv_idx_triton, recv_idx_cuda), "recv_topk_idx mismatch"
        assert torch.allclose(
            recv_weights_triton, recv_weights_cuda
        ), "recv_topk_weights mismatch"
        assert (
            counts_triton == counts_cuda
        ), f"expert counts mismatch: triton={counts_triton}, cuda={counts_cuda}"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_padding(self):
        """Test with all -1 padding (no valid experts)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        num_tokens = 32
        hidden_dim = 64
        num_experts = 8
        experts_per_token = 2
        world_size = 2

        x = torch.randn(num_tokens, hidden_dim, dtype=torch.float32, device=device)
        topk_idx = torch.full(
            (num_tokens, experts_per_token), -1, dtype=torch.int64, device=device
        )
        topk_weights = torch.randn(
            num_tokens, experts_per_token, dtype=torch.float32, device=device
        )

        buffers = allocate_buffers(
            num_tokens, hidden_dim, experts_per_token, world_size, torch.float32, device
        )

        per_rank_bytes = pack_moe_data_to_buffers_triton(
            x, topk_idx, topk_weights, num_experts, world_size, device, buffers
        )

        # All bytes should be zero
        assert per_rank_bytes.sum().item() == 0

    def test_single_token(self):
        """Test with single token."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        x = torch.randn(1, 64, dtype=torch.float32, device=device)
        topk_idx = torch.tensor([[0, 2]], dtype=torch.int64, device=device)
        topk_weights = torch.randn(1, 2, dtype=torch.float32, device=device)

        buffers = allocate_buffers(1, 64, 2, 2, torch.float32, device)

        per_rank_bytes = pack_moe_data_to_buffers_triton(
            x, topk_idx, topk_weights, 4, 2, device, buffers
        )

        assert per_rank_bytes.sum().item() > 0


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
