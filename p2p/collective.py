"""
Collective API for UCCL P2P Engine

This module provides a high-level collective interface that wraps the low-level
p2p engine and provides simple send/recv operations similar to NCCL.

The API automatically handles:
- Connection setup and metadata exchange via torch.distributed
- Memory registration and management
- Async operation polling
- Resource cleanup
"""

from __future__ import annotations

import struct
import socket
from typing import Dict, Tuple, Optional, List
import warnings

import torch
import torch.distributed as dist

try:
    from . import p2p
except ImportError:
    import p2p


class CollectiveContext:
    """
    High-level collective communication context that wraps the UCCL p2p engine.

    Provides NCCL-like send/recv interface with automatic connection management.

    Features:
    - Automatically detects local vs remote peers based on IP addresses
    - Uses IPC (Inter-Process Communication) for local GPU-to-GPU transfers
    - Uses RDMA for remote GPU-to-GPU transfers
    - Memory registration is only required for remote connections
    - Transparent API - applications don't need to know if peers are local or remote
    """

    def __init__(self, num_cpus: int = 4, local_gpu_idx: Optional[int] = None):
        """
        Initialize collective context with automatic GPU index derivation from torch.distributed.

        Args:
            num_cpus: Number of CPU threads for RDMA operations
            local_gpu_idx: Optional override for local GPU index. If None, will be derived from torch.distributed
        """
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before creating CollectiveContext"
            )

        if dist.get_backend() != "gloo":
            raise RuntimeError(
                "CollectiveContext requires torch.distributed to use 'gloo' backend"
            )

        self.num_cpus = num_cpus
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Derive local GPU index from distributed context
        if local_gpu_idx is not None:
            self.local_gpu_idx = local_gpu_idx
        else:
            self.local_gpu_idx = self._get_local_gpu_idx()

        self.ep = None
        self.send_connections = None  # array indexed by rank for sending TO that rank
        self.recv_connections = (
            None  # array indexed by rank for receiving FROM that rank
        )
        self.local_connections = (
            None  # array indexed by rank - True if local, False if remote
        )
        self.memory_regions: Dict[int, int] = {}  # ptr -> mr_id
        self.initialized = False

    def _get_local_gpu_idx(self) -> int:
        """
        Derive local GPU index from torch.distributed context.

        This method attempts to get the local rank in the following order:
        1. LOCAL_RANK environment variable (common in torchrun/distributed launchers)
        2. Calculate from global rank assuming equal distribution across nodes
        3. Fall back to global rank (single node case)
        """
        import os

        # Method 1: Check LOCAL_RANK environment variable (most reliable)
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            try:
                return int(local_rank_env)
            except ValueError:
                pass

        # Method 2: Try to get from torch.distributed if available
        try:
            # This requires torch.distributed to have local rank support
            if hasattr(dist, "get_local_rank"):
                return dist.get_local_rank()
        except (AttributeError, RuntimeError):
            pass

        # Method 3: Check RANK and LOCAL_WORLD_SIZE for calculation
        rank_env = os.environ.get("RANK")
        local_world_size_env = os.environ.get("LOCAL_WORLD_SIZE")
        if rank_env is not None and local_world_size_env is not None:
            try:
                global_rank = int(rank_env)
                local_world_size = int(local_world_size_env)
                return global_rank % local_world_size
            except ValueError:
                pass

        # Method 4: Fall back to global rank (assumes single node or user manages GPU mapping)
        return self.rank

    def init(self):
        """Initialize the collective context and establish connections with all peers."""
        if self.initialized:
            warnings.warn("CollectiveContext already initialized")
            return

        # Create endpoint
        self.ep = p2p.Endpoint(self.local_gpu_idx, self.num_cpus)
        local_metadata = self.ep.get_endpoint_metadata()

        # Initialize connection arrays
        self.send_connections = [None] * self.world_size  # indexed by rank
        self.recv_connections = [None] * self.world_size  # indexed by rank
        self.local_connections = [False] * self.world_size  # indexed by rank

        # Exchange metadata with all peers using torch.distributed
        all_metadata = [None] * self.world_size

        # Gather all metadata
        metadata_tensor = torch.ByteTensor(list(local_metadata))
        gathered_tensors = [
            torch.zeros_like(metadata_tensor) for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_tensors, metadata_tensor)

        for i, tensor in enumerate(gathered_tensors):
            all_metadata[i] = bytes(tensor.tolist())

        # Establish connections with all other ranks
        self._establish_connections(all_metadata)
        self.initialized = True

    def _establish_connections(self, all_metadata: list):
        """Establish full mesh connections with all peers using concurrent threads."""
        import concurrent.futures

        # First, determine which connections are local vs remote
        local_ip, _, _ = p2p.Endpoint.parse_metadata(all_metadata[self.rank])

        for peer_rank in range(self.world_size):
            if peer_rank != self.rank:
                peer_ip, _, _ = p2p.Endpoint.parse_metadata(all_metadata[peer_rank])
                self.local_connections[peer_rank] = peer_ip == local_ip
                connection_type = (
                    "local" if self.local_connections[peer_rank] else "remote"
                )
                print(
                    f"[Rank {self.rank}] Rank {peer_rank} is {connection_type} (IP: {peer_ip})"
                )

        connect_errors = []
        accept_errors = []

        def connect_to_peer(peer_rank):
            """Connect to a specific peer for sending data TO that peer."""
            try:
                if self.local_connections[peer_rank]:
                    # Use local IPC connection
                    _, _, gpu_idx = p2p.Endpoint.parse_metadata(all_metadata[peer_rank])
                    ok, conn_id = self.ep.connect_local(gpu_idx)
                    if not ok:
                        raise RuntimeError(
                            f"Failed to connect locally to rank {peer_rank}"
                        )
                    self.send_connections[peer_rank] = conn_id
                    print(
                        f"[Rank {self.rank}] Connected locally to rank {peer_rank} for sending (conn_id={conn_id})"
                    )
                else:
                    # Use remote RDMA connection
                    ip, port, gpu_idx = p2p.Endpoint.parse_metadata(
                        all_metadata[peer_rank]
                    )
                    ok, conn_id = self.ep.connect(ip, gpu_idx, remote_port=port)
                    if not ok:
                        raise RuntimeError(
                            f"Failed to connect remotely to rank {peer_rank}"
                        )
                    self.send_connections[peer_rank] = conn_id
                    print(
                        f"[Rank {self.rank}] Connected remotely to rank {peer_rank} for sending (conn_id={conn_id})"
                    )
            except Exception as e:
                connect_errors.append(f"Connect to rank {peer_rank}: {e}")

        def accept_from_peer():
            """Accept connection from a specific peer for receiving data FROM that peer."""
            try:
                # We need to accept both local and remote connections
                # For local connections, we can only accept local ones
                # For remote connections, we accept regular ones
                # We'll need to track which type we're expecting based on remaining connections

                # Count how many local vs remote connections we still need to accept
                remaining_local = sum(
                    1
                    for rank in range(self.world_size)
                    if rank != self.rank
                    and self.local_connections[rank]
                    and self.recv_connections[rank] is None
                )
                remaining_remote = sum(
                    1
                    for rank in range(self.world_size)
                    if rank != self.rank
                    and not self.local_connections[rank]
                    and self.recv_connections[rank] is None
                )

                if remaining_local > 0:
                    # Try to accept a local connection first
                    ok, r_gpu, conn_id = self.ep.accept_local()
                    if ok:
                        # Find the rank that corresponds to this GPU index
                        for rank in range(self.world_size):
                            if rank != self.rank and self.local_connections[rank]:
                                _, _, gpu_idx = p2p.Endpoint.parse_metadata(
                                    all_metadata[rank]
                                )
                                if gpu_idx == r_gpu:
                                    self.recv_connections[rank] = conn_id
                                    print(
                                        f"[Rank {self.rank}] Accepted local connection from rank {rank} (GPU {r_gpu}) for receiving (conn_id={conn_id})"
                                    )
                                    return
                        raise RuntimeError(f"Could not map local GPU {r_gpu} to a rank")

                if remaining_remote > 0:
                    # Accept a remote connection
                    ok, r_ip, r_gpu, conn_id = self.ep.accept()
                    if ok:
                        # Find the rank that corresponds to this IP/GPU combination
                        for rank in range(self.world_size):
                            if rank != self.rank and not self.local_connections[rank]:
                                ip, _, gpu_idx = p2p.Endpoint.parse_metadata(
                                    all_metadata[rank]
                                )
                                if ip == r_ip and gpu_idx == r_gpu:
                                    self.recv_connections[rank] = conn_id
                                    print(
                                        f"[Rank {self.rank}] Accepted remote connection from rank {rank} (IP {r_ip}, GPU {r_gpu}) for receiving (conn_id={conn_id})"
                                    )
                                    return
                        raise RuntimeError(
                            f"Could not map remote IP {r_ip}, GPU {r_gpu} to a rank"
                        )

                raise RuntimeError("No connections to accept or accept failed")
            except Exception as e:
                accept_errors.append(f"Accept connection: {e}")

        # Create thread pools for concurrent operations
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.world_size
        ) as executor:
            # Submit connect tasks for all other ranks (for sending TO them)
            connect_futures = []
            for peer_rank in range(self.world_size):
                if peer_rank != self.rank:
                    future = executor.submit(connect_to_peer, peer_rank)
                    connect_futures.append(future)

            # Submit accept tasks for all other ranks (for receiving FROM them)
            accept_futures = []
            for peer_rank in range(self.world_size):
                if peer_rank != self.rank:
                    future = executor.submit(accept_from_peer)
                    accept_futures.append(future)

            # Wait for all connections to complete
            concurrent.futures.wait(connect_futures + accept_futures)

        # Check for any errors
        if connect_errors:
            raise RuntimeError(f"Connect errors: {'; '.join(connect_errors)}")
        if accept_errors:
            raise RuntimeError(f"Accept errors: {'; '.join(accept_errors)}")

        # Count local vs remote connections
        local_count = sum(1 for is_local in self.local_connections if is_local)
        remote_count = self.world_size - 1 - local_count
        print(
            f"[Rank {self.rank}] Full mesh established: {local_count} local connections, {remote_count} remote connections"
        )

    def _get_buffer_info(self, tensor: torch.Tensor) -> Tuple[int, int]:
        """Get buffer pointer and size from tensor."""
        if not tensor.is_contiguous():
            raise ValueError("Tensor must be contiguous")
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        return ptr, size

    def _register_memory(self, ptr: int, size: int) -> int:
        """Register memory and cache the memory region ID."""
        if ptr in self.memory_regions:
            return self.memory_regions[ptr]

        ok, mr_id = self.ep.reg(ptr, size)
        if not ok:
            raise RuntimeError("Failed to register memory")
        self.memory_regions[ptr] = mr_id
        return mr_id

    def register_tensor(self, tensor: torch.Tensor):
        """
        Register a tensor for efficient memory access.

        Note: Registration is only required for tensors used with remote (RDMA) connections.
        Local IPC connections do not require memory registration.

        Args:
            tensor: Tensor to register
        """
        if not self.initialized:
            raise RuntimeError("CollectiveContext not initialized. Call init() first.")

        ptr, size = self._get_buffer_info(tensor)
        self._register_memory(ptr, size)

    def send(self, tensor: torch.Tensor, dst: int):
        """
        Send tensor to destination rank (synchronous).

        Args:
            tensor: Tensor to send
            dst: Destination rank
        """
        if not self.initialized:
            raise RuntimeError("CollectiveContext not initialized. Call init() first.")

        if dst == self.rank:
            return  # No-op for self-send

        if dst >= self.world_size or self.send_connections[dst] is None:
            raise ValueError(f"No send connection to rank {dst}")

        ptr, size = self._get_buffer_info(tensor)
        conn_id = self.send_connections[dst]

        if self.local_connections[dst]:
            # Use IPC for local connection (no memory registration needed)
            ok = self.ep.send_ipc(conn_id, ptr, size)
            if not ok:
                raise RuntimeError(f"Failed to initiate IPC send to rank {dst}")
        else:
            # Use RDMA for remote connection (requires memory registration)
            if ptr not in self.memory_regions:
                raise RuntimeError(
                    f"Tensor memory not registered for remote communication with rank {dst}. "
                    f"Call register_tensor() first for tensors used with remote ranks."
                )
            mr_id = self.memory_regions[ptr]
            ok = self.ep.send(conn_id, mr_id, ptr, size)
            if not ok:
                raise RuntimeError(f"Failed to initiate RDMA send to rank {dst}")

    def recv(self, tensor: torch.Tensor, src: int):
        """
        Receive tensor from source rank (synchronous).

        Args:
            tensor: Tensor to receive into
            src: Source rank
        """
        if not self.initialized:
            raise RuntimeError("CollectiveContext not initialized. Call init() first.")

        if src == self.rank:
            return  # No-op for self-recv

        if src >= self.world_size or self.recv_connections[src] is None:
            raise ValueError(f"No recv connection from rank {src}")

        ptr, size = self._get_buffer_info(tensor)
        conn_id = self.recv_connections[src]

        if self.local_connections[src]:
            # Use IPC for local connection (no memory registration needed)
            ok = self.ep.recv_ipc(conn_id, ptr, size)
            if not ok:
                raise RuntimeError(f"Failed to initiate IPC recv from rank {src}")
        else:
            # Use RDMA for remote connection (requires memory registration)
            if ptr not in self.memory_regions:
                raise RuntimeError(
                    f"Tensor memory not registered for remote communication with rank {src}. "
                    f"Call register_tensor() first for tensors used with remote ranks."
                )
            mr_id = self.memory_regions[ptr]
            ok = self.ep.recv(conn_id, mr_id, ptr, size)
            if not ok:
                raise RuntimeError(f"Failed to initiate RDMA recv from rank {src}")

    def isend(self, tensor: torch.Tensor, dst: int) -> int:
        """
        Initiate asynchronous send (non-blocking).

        Args:
            tensor: Tensor to send
            dst: Destination rank

        Returns:
            transfer_id: ID to poll for completion
        """
        if not self.initialized:
            raise RuntimeError("CollectiveContext not initialized. Call init() first.")

        if dst == self.rank:
            return -1  # Return dummy ID for self-send

        if dst >= self.world_size or self.send_connections[dst] is None:
            raise ValueError(f"No send connection to rank {dst}")

        ptr, size = self._get_buffer_info(tensor)
        conn_id = self.send_connections[dst]

        if self.local_connections[dst]:
            # Use IPC async for local connection (no memory registration needed)
            ok, transfer_id = self.ep.send_ipc_async(conn_id, ptr, size)
            if not ok:
                raise RuntimeError(f"Failed to initiate async IPC send to rank {dst}")
            return transfer_id
        else:
            # Use RDMA async for remote connection (requires memory registration)
            if ptr not in self.memory_regions:
                raise RuntimeError(
                    f"Tensor memory not registered for remote communication with rank {dst}. "
                    f"Call register_tensor() first for tensors used with remote ranks."
                )
            mr_id = self.memory_regions[ptr]
            ok, transfer_id = self.ep.send_async(conn_id, mr_id, ptr, size)
            if not ok:
                raise RuntimeError(f"Failed to initiate async RDMA send to rank {dst}")
            return transfer_id

    def irecv(self, tensor: torch.Tensor, src: int) -> int:
        """
        Initiate asynchronous receive (non-blocking).

        Args:
            tensor: Tensor to receive into
            src: Source rank

        Returns:
            transfer_id: ID to poll for completion
        """
        if not self.initialized:
            raise RuntimeError("CollectiveContext not initialized. Call init() first.")

        if src == self.rank:
            return -1  # Return dummy ID for self-recv

        if src >= self.world_size or self.recv_connections[src] is None:
            raise ValueError(f"No recv connection from rank {src}")

        ptr, size = self._get_buffer_info(tensor)
        conn_id = self.recv_connections[src]

        if self.local_connections[src]:
            # Use IPC async for local connection (no memory registration needed)
            ok, transfer_id = self.ep.recv_ipc_async(conn_id, ptr, size)
            if not ok:
                raise RuntimeError(f"Failed to initiate async IPC recv from rank {src}")
            return transfer_id
        else:
            # Use RDMA async for remote connection (requires memory registration)
            if ptr not in self.memory_regions:
                raise RuntimeError(
                    f"Tensor memory not registered for remote communication with rank {src}. "
                    f"Call register_tensor() first for tensors used with remote ranks."
                )
            mr_id = self.memory_regions[ptr]
            ok, transfer_id = self.ep.recv_async(conn_id, mr_id, ptr, size)
            if not ok:
                raise RuntimeError(
                    f"Failed to initiate async RDMA recv from rank {src}"
                )
            return transfer_id

    def wait(self, transfer_id: int):
        """
        Wait for asynchronous operation to complete.

        Args:
            transfer_id: Transfer ID returned by isend/irecv
        """
        if transfer_id == -1:
            return  # No-op for dummy self-transfer

        while True:
            ok, is_done = self.ep.poll_async(transfer_id)
            if not ok:
                raise RuntimeError(f"Error polling transfer {transfer_id}")
            if is_done:
                break

    def test(self, transfer_id: int) -> bool:
        """
        Test if asynchronous operation is complete (non-blocking).

        Args:
            transfer_id: Transfer ID returned by isend/irecv

        Returns:
            True if complete, False otherwise
        """
        if transfer_id == -1:
            return True  # Self-transfers are always complete

        ok, is_done = self.ep.poll_async(transfer_id)
        if not ok:
            raise RuntimeError(f"Error polling transfer {transfer_id}")
        return is_done

    def finalize(self):
        """Clean up resources and close connections."""
        if not self.initialized:
            return

        # Note: The p2p endpoint should handle cleanup automatically
        # when it goes out of scope, but we can add explicit cleanup here if needed
        self.send_connections = None
        self.recv_connections = None
        self.local_connections = None
        self.memory_regions.clear()
        self.ep = None
        self.initialized = False


# Global collective context instance
_default_context: Optional[CollectiveContext] = None


def init_collective(num_cpus: int = 4, local_gpu_idx: Optional[int] = None):
    """
    Initialize the default collective context with automatic GPU index derivation.

    Args:
        num_cpus: Number of CPU threads for RDMA operations
        local_gpu_idx: Optional override for local GPU index. If None, will be derived from torch.distributed
    """
    global _default_context
    _default_context = CollectiveContext(num_cpus, local_gpu_idx)
    _default_context.init()


def get_collective() -> CollectiveContext:
    """Get the default collective context."""
    if _default_context is None:
        raise RuntimeError("Collective not initialized. Call init_collective() first.")
    return _default_context


def send(tensor: torch.Tensor, dst: int):
    """Send tensor using the default collective context."""
    get_collective().send(tensor, dst)


def recv(tensor: torch.Tensor, src: int):
    """Receive tensor using the default collective context."""
    get_collective().recv(tensor, src)


def isend(tensor: torch.Tensor, dst: int) -> int:
    """Async send tensor using the default collective context."""
    return get_collective().isend(tensor, dst)


def irecv(tensor: torch.Tensor, src: int) -> int:
    """Async receive tensor using the default collective context."""
    return get_collective().irecv(tensor, src)


def wait(transfer_id: int):
    """Wait for async operation using the default collective context."""
    get_collective().wait(transfer_id)


def wait_all(transfer_ids: List[int]):
    """Wait for all async operations using the default collective context."""
    for transfer_id in transfer_ids:
        get_collective().wait(transfer_id)


def test(transfer_id: int) -> bool:
    """Test async operation using the default collective context."""
    return get_collective().test(transfer_id)


def register_tensor(tensor: torch.Tensor):
    """Register tensor using the default collective context."""
    get_collective().register_tensor(tensor)


def finalize_collective():
    """Finalize the default collective context."""
    global _default_context
    if _default_context is not None:
        _default_context.finalize()
        _default_context = None
