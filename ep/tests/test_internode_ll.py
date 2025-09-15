#!/usr/bin/env python3
"""
Test for internode_ll kernels with support for both single-node and multi-node configurations.
This test allows users to run intra-node only (single node) or mixed inter/intra-node workloads.

Usage:
    # Single node mode (intra-node only)
    python test_internode_ll.py --single-node --world-size 8 --gpus-per-node 8

    # Multi-node mode (inter-node + intra-node)
    torchrun --nproc_per_node=8 --nnodes=2 test_internode_ll.py --world-size 16 --num-nodes 2
"""

import os
import sys
import argparse
import time
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.distributed as dist

# Import UCCL EP module
try:
    from uccl.ep import Buffer
    import uccl.ep as ep
except ImportError as e:
    print(f"Error: Failed to import uccl.ep: {e}")
    print("Make sure UCCL is properly installed and in PYTHONPATH")
    sys.exit(1)


@dataclass
class TestConfig:
    """Configuration for the internode_ll kernel test"""

    # Distribution settings
    world_size: int = 8  # Total number of GPUs
    rank: int = 0  # Current rank (node index when single-node)
    num_nodes: int = 1  # Number of nodes
    gpus_per_node: int = 8  # GPUs per node
    single_node: bool = False  # Run single-node mode (intra-node only)

    # Model parameters
    num_tokens: int = 4096  # Number of input tokens
    hidden_size: int = 7168  # Hidden dimension
    num_experts: int = 8  # Total number of experts
    num_topk: int = 2  # Top-K expert selection
    max_tokens_per_rank: int = 8192  # Max tokens per rank for buffering

    # Kernel options
    use_fp8: bool = False  # Use FP8 quantization
    use_logfmt: bool = False  # Use LogFMT compression
    round_scale: bool = False  # Round FP8 scales to power of 2
    use_ue8m0: bool = False  # Use UE8M0 scaling format
    zero_copy: bool = False  # Use zero-copy mode

    # Test options
    num_iterations: int = 10  # Number of test iterations
    validate_results: bool = True  # Validate correctness
    verbose: bool = False  # Verbose output
    warmup_iterations: int = 3  # Warmup iterations

    # Buffer sizes
    rdma_buffer_gb: float = 2.0  # RDMA buffer size in GB
    nvlink_buffer_gb: float = 1.0  # NVLink buffer size in GB


class InternodeLLTester:
    """Tester for internode_ll kernels with mixed workload support"""

    def __init__(self, config: TestConfig):
        self.config = config
        self.buffer: Optional[Buffer] = None
        self.device = None

        # Set up distributed environment
        self._setup_distributed()

        # Initialize CUDA device
        self._setup_device()

        # Initialize EP buffer
        self._setup_buffer()

        # Generate test data
        self._generate_test_data()

    def _setup_distributed(self):
        """Setup distributed environment"""
        if self.config.single_node:
            # For single-node mode, we simulate distribution without actual MPI
            if "RANK" in os.environ:
                self.config.rank = int(os.environ["RANK"])
            if "WORLD_SIZE" in os.environ:
                self.config.world_size = int(os.environ["WORLD_SIZE"])

            # Initialize process group for single node
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl",
                    world_size=self.config.world_size,
                    rank=self.config.rank,
                )
        else:
            # Multi-node mode with torchrun
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")

            self.config.rank = dist.get_rank()
            self.config.world_size = dist.get_world_size()
            self.config.num_nodes = self.config.world_size // self.config.gpus_per_node

    def _setup_device(self):
        """Setup CUDA device"""
        local_rank = self.config.rank % self.config.gpus_per_node
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)

        if self.config.verbose and self.config.rank == 0:
            print(f"Using device: {self.device}")

    def _setup_buffer(self):
        """Setup EP communication buffer"""
        # Calculate buffer sizes
        rdma_buffer_bytes = int(self.config.rdma_buffer_gb * 1024**3)
        nvlink_buffer_bytes = (
            int(self.config.nvlink_buffer_gb * 1024**3)
            if not self.config.single_node
            else 0
        )

        # Get minimum RDMA buffer size requirement
        min_rdma_size = Buffer.get_low_latency_rdma_size_hint(
            self.config.max_tokens_per_rank,
            self.config.hidden_size,
            self.config.world_size,
            self.config.num_experts,
        )

        if rdma_buffer_bytes < min_rdma_size:
            rdma_buffer_bytes = min_rdma_size
            if self.config.rank == 0:
                print(
                    f"Warning: Adjusted RDMA buffer size to minimum required: {rdma_buffer_bytes / 1024**3:.2f} GB"
                )

        # Create process group
        group = dist.group.WORLD

        # Initialize buffer with appropriate settings for single/multi-node
        self.buffer = Buffer(
            group=group,
            num_nvl_bytes=nvlink_buffer_bytes,
            num_rdma_bytes=rdma_buffer_bytes,
            low_latency_mode=True,  # Always use low-latency mode for this test
            num_qps_per_rank=max(24, self.config.num_experts // self.config.world_size),
            allow_nvlink_for_low_latency_mode=True,
            allow_mnnvl=not self.config.single_node,  # Allow MNNVL only for multi-node
            explicitly_destroy=True,
        )

        if self.config.verbose and self.config.rank == 0:
            print(
                f"Initialized buffer - RDMA: {rdma_buffer_bytes / 1024**3:.2f} GB, "
                f"NVLink: {nvlink_buffer_bytes / 1024**3:.2f} GB"
            )

    def _generate_test_data(self):
        """Generate test input data"""
        # Input tokens
        self.input_x = (
            torch.randn(
                self.config.num_tokens,
                self.config.hidden_size,
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.1
        )  # Small values to avoid overflow

        # Generate expert assignments
        self.topk_idx = self._generate_expert_assignments()

        # Generate expert weights (normalized)
        self.topk_weights = torch.rand(
            self.config.num_tokens,
            self.config.num_topk,
            dtype=torch.float,
            device=self.device,
        )
        # Normalize weights per token
        self.topk_weights = self.topk_weights / self.topk_weights.sum(
            dim=1, keepdim=True
        )

        if self.config.verbose and self.config.rank == 0:
            print(
                f"Generated test data - tokens: {self.config.num_tokens}, "
                f"hidden: {self.config.hidden_size}, experts: {self.config.num_experts}"
            )

    def _generate_expert_assignments(self) -> torch.Tensor:
        """Generate expert assignments based on single/multi-node mode"""
        topk_idx = torch.zeros(
            self.config.num_tokens,
            self.config.num_topk,
            dtype=torch.long,
            device=self.device,
        )

        num_local_experts = self.config.num_experts // self.config.world_size

        for i in range(self.config.num_tokens):
            if self.config.single_node:
                # Single-node mode: only assign to local experts on this node
                local_start = self.config.rank * num_local_experts
                available_experts = list(
                    range(local_start, local_start + num_local_experts)
                )

                # For intra-node testing, we can also include experts from other GPUs on the same node
                # This simulates the mixed workload where some tokens stay local, others go to peer GPUs
                node_id = self.config.rank // self.config.gpus_per_node
                node_start = node_id * self.config.gpus_per_node * num_local_experts
                node_end = (node_id + 1) * self.config.gpus_per_node * num_local_experts
                available_experts = list(range(node_start, node_end))
            else:
                # Multi-node mode: can assign to any expert (mix of inter/intra-node)
                available_experts = list(range(self.config.num_experts))

            # Randomly select topk experts
            selected = torch.randperm(len(available_experts))[: self.config.num_topk]
            for k in range(self.config.num_topk):
                topk_idx[i, k] = available_experts[selected[k]]

        return topk_idx

    def _run_single_iteration(self) -> Tuple[float, float]:
        """Run a single iteration of dispatch + combine"""
        torch.cuda.synchronize()
        start_time = time.time()

        # Dispatch phase
        dispatch_start = time.time()
        recv_data, recv_count, handle, dispatch_event, dispatch_hook = (
            self.buffer.low_latency_dispatch(
                x=self.input_x,
                topk_idx=self.topk_idx,
                num_max_dispatch_tokens_per_rank=self.config.max_tokens_per_rank,
                num_experts=self.config.num_experts,
                use_fp8=self.config.use_fp8,
                round_scale=self.config.round_scale,
                use_ue8m0=self.config.use_ue8m0,
                async_finish=False,
                return_recv_hook=False,
            )
        )

        torch.cuda.synchronize()
        dispatch_time = time.time() - dispatch_start

        # For simplicity, create dummy expert output (in real use, this would come from MLP forward)
        if self.config.use_fp8:
            expert_output_shape = recv_data[0].shape
        else:
            expert_output_shape = recv_data.shape

        expert_output = (
            torch.randn(expert_output_shape, dtype=torch.bfloat16, device=self.device)
            * 0.1
        )

        # Combine phase
        combine_start = time.time()
        combined_output, combine_event, combine_hook = self.buffer.low_latency_combine(
            x=expert_output,
            topk_idx=self.topk_idx,
            topk_weights=self.topk_weights,
            handle=handle,
            use_logfmt=self.config.use_logfmt,
            zero_copy=self.config.zero_copy,
            async_finish=False,
            return_recv_hook=False,
        )

        torch.cuda.synchronize()
        combine_time = time.time() - combine_start

        total_time = time.time() - start_time

        return dispatch_time, combine_time

    def run_performance_test(self):
        """Run performance benchmarking"""
        if self.config.rank == 0:
            print(f"\n=== Running Performance Test ===")
            print(
                f"Mode: {'Single-Node (Intra-node only)' if self.config.single_node else 'Multi-Node (Inter + Intra-node)'}"
            )
            print(f"World size: {self.config.world_size}")
            print(
                f"Tokens: {self.config.num_tokens}, Hidden: {self.config.hidden_size}"
            )
            print(f"Experts: {self.config.num_experts}, Top-K: {self.config.num_topk}")
            print(f"FP8: {self.config.use_fp8}, LogFMT: {self.config.use_logfmt}")

        # Warmup
        if self.config.rank == 0:
            print(f"Warming up for {self.config.warmup_iterations} iterations...")

        for i in range(self.config.warmup_iterations):
            self._run_single_iteration()

        # Benchmark
        dispatch_times = []
        combine_times = []

        if self.config.rank == 0:
            print(f"Running {self.config.num_iterations} benchmark iterations...")

        for i in range(self.config.num_iterations):
            dispatch_time, combine_time = self._run_single_iteration()
            dispatch_times.append(dispatch_time)
            combine_times.append(combine_time)

            if self.config.verbose and self.config.rank == 0 and i % 5 == 0:
                print(
                    f"Iteration {i}: dispatch={dispatch_time*1000:.2f}ms, combine={combine_time*1000:.2f}ms"
                )

        # Gather results from all ranks
        all_dispatch_times = [None] * self.config.world_size
        all_combine_times = [None] * self.config.world_size

        dist.all_gather_object(all_dispatch_times, dispatch_times)
        dist.all_gather_object(all_combine_times, combine_times)

        if self.config.rank == 0:
            self._print_performance_results(all_dispatch_times, all_combine_times)

    def _print_performance_results(
        self, all_dispatch_times: List, all_combine_times: List
    ):
        """Print performance analysis"""
        print(f"\n=== Performance Results ===")

        # Calculate statistics
        avg_dispatch = sum(sum(times) for times in all_dispatch_times) / (
            len(all_dispatch_times) * len(all_dispatch_times[0])
        )
        avg_combine = sum(sum(times) for times in all_combine_times) / (
            len(all_combine_times) * len(all_combine_times[0])
        )

        print(f"Average dispatch time: {avg_dispatch*1000:.2f} ms")
        print(f"Average combine time: {avg_combine*1000:.2f} ms")
        print(f"Average total time: {(avg_dispatch + avg_combine)*1000:.2f} ms")

        # Calculate throughput
        total_data_bytes = (
            self.config.num_tokens * self.config.hidden_size * 2
        )  # bfloat16
        throughput_gbps = (total_data_bytes / (1024**3)) / (avg_dispatch + avg_combine)
        print(f"Throughput: {throughput_gbps:.2f} GB/s")

        # Per-rank analysis
        print(f"\nPer-rank analysis:")
        for rank in range(self.config.world_size):
            rank_dispatch_avg = sum(all_dispatch_times[rank]) / len(
                all_dispatch_times[rank]
            )
            rank_combine_avg = sum(all_combine_times[rank]) / len(
                all_combine_times[rank]
            )
            print(
                f"  Rank {rank}: dispatch={rank_dispatch_avg*1000:.2f}ms, combine={rank_combine_avg*1000:.2f}ms"
            )

        # Workload analysis
        if self.config.single_node:
            print(f"\nWorkload Analysis (Single-Node Mode):")
            print(f"  - All communication is intra-node (NVLink/PCIe)")
            print(f"  - No inter-node RDMA traffic")
            print(f"  - {self.config.gpus_per_node} GPUs per node")
        else:
            print(f"\nWorkload Analysis (Multi-Node Mode):")
            print(f"  - Mixed inter-node (RDMA) + intra-node (NVLink) communication")
            print(
                f"  - {self.config.num_nodes} nodes with {self.config.gpus_per_node} GPUs each"
            )
            experts_per_node = self.config.num_experts // self.config.num_nodes
            print(f"  - {experts_per_node} experts per node")

    def run_correctness_test(self):
        """Run correctness validation"""
        if not self.config.validate_results:
            return

        if self.config.rank == 0:
            print(f"\n=== Running Correctness Test ===")

        # Run one iteration and check basic properties
        recv_data, recv_count, handle, _, _ = self.buffer.low_latency_dispatch(
            x=self.input_x,
            topk_idx=self.topk_idx,
            num_max_dispatch_tokens_per_rank=self.config.max_tokens_per_rank,
            num_experts=self.config.num_experts,
            use_fp8=self.config.use_fp8,
            round_scale=self.config.round_scale,
            use_ue8m0=self.config.use_ue8m0,
            async_finish=False,
            return_recv_hook=False,
        )

        # Basic shape validation
        num_local_experts = self.config.num_experts // self.config.world_size
        expected_shape = (
            num_local_experts,
            self.config.max_tokens_per_rank * self.config.world_size,
            self.config.hidden_size,
        )

        if self.config.use_fp8:
            actual_shape = recv_data[0].shape
        else:
            actual_shape = recv_data.shape

        assert (
            actual_shape == expected_shape
        ), f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
        assert recv_count.shape == (
            num_local_experts,
        ), f"Count shape mismatch: {recv_count.shape}"

        # Check that received counts are reasonable
        total_received = recv_count.sum().item()
        expected_tokens = self.config.num_tokens * self.config.num_topk

        if self.config.rank == 0:
            print(f"Correctness check passed:")
            print(f"  - Data shape: {actual_shape}")
            print(f"  - Count shape: {recv_count.shape}")
            print(
                f"  - Total tokens received: {total_received} (expected ~{expected_tokens})"
            )

    def cleanup(self):
        """Clean up resources"""
        if self.buffer:
            self.buffer.destroy()

        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Test internode_ll kernels with mixed workload support"
    )

    # Mode selection
    parser.add_argument(
        "--single-node",
        action="store_true",
        help="Run in single-node mode (intra-node only)",
    )

    # Distribution settings
    parser.add_argument(
        "--world-size", type=int, default=8, help="Total number of GPUs (default: 8)"
    )
    parser.add_argument(
        "--num-nodes", type=int, default=1, help="Number of nodes (default: 1)"
    )
    parser.add_argument(
        "--gpus-per-node", type=int, default=8, help="GPUs per node (default: 8)"
    )

    # Model parameters
    parser.add_argument(
        "--tokens", type=int, default=4096, help="Number of tokens (default: 4096)"
    )
    parser.add_argument(
        "--hidden", type=int, default=7168, help="Hidden size (default: 7168)"
    )
    parser.add_argument(
        "--experts", type=int, default=8, help="Number of experts (default: 8)"
    )
    parser.add_argument("--topk", type=int, default=2, help="Top-K value (default: 2)")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max tokens per rank (default: 8192)",
    )

    # Kernel options
    parser.add_argument("--fp8", action="store_true", help="Use FP8 quantization")
    parser.add_argument("--logfmt", action="store_true", help="Use LogFMT compression")
    parser.add_argument("--zero-copy", action="store_true", help="Use zero-copy mode")

    # Test options
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of iterations (default: 10)"
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Warmup iterations (default: 3)"
    )
    parser.add_argument(
        "--no-validate", action="store_true", help="Skip correctness validation"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    # Buffer sizes
    parser.add_argument(
        "--rdma-buffer-gb",
        type=float,
        default=2.0,
        help="RDMA buffer size in GB (default: 2.0)",
    )
    parser.add_argument(
        "--nvlink-buffer-gb",
        type=float,
        default=1.0,
        help="NVLink buffer size in GB (default: 1.0)",
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Create test configuration
    config = TestConfig(
        world_size=args.world_size,
        num_nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        single_node=args.single_node,
        num_tokens=args.tokens,
        hidden_size=args.hidden,
        num_experts=args.experts,
        num_topk=args.topk,
        max_tokens_per_rank=args.max_tokens,
        use_fp8=args.fp8,
        use_logfmt=args.logfmt,
        zero_copy=args.zero_copy,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
        validate_results=not args.no_validate,
        verbose=args.verbose,
        rdma_buffer_gb=args.rdma_buffer_gb,
        nvlink_buffer_gb=args.nvlink_buffer_gb,
    )

    # Validate configuration
    if config.single_node:
        config.num_nodes = 1
        config.world_size = config.gpus_per_node
    else:
        config.world_size = config.num_nodes * config.gpus_per_node

    if config.num_experts % config.world_size != 0:
        print(
            f"Error: Number of experts ({config.num_experts}) must be divisible by world size ({config.world_size})"
        )
        sys.exit(1)

    # Initialize tester
    tester = InternodeLLTester(config)

    try:
        # Run correctness test
        tester.run_correctness_test()

        # Run performance test
        tester.run_performance_test()

        if tester.config.rank == 0:
            print(f"\n=== Test Completed Successfully ===")

    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
    finally:
        # Cleanup
        tester.cleanup()


if __name__ == "__main__":
    main()
