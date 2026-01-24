"""
Collective API for UCCL P2P Engine
"""

from __future__ import annotations

import struct
import socket
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Union
import warnings

import torch
import torch.distributed as dist

try:
    from . import p2p
    from . import utils
except ImportError:
    import p2p
    import utils


@dataclass
class P2POp:
    """
    Descriptor for a point-to-point operation.
    Created by CollectiveContext.P2POp() or collective.P2POp().
    """

    op: str
    tensor: torch.Tensor
    peer: int

    def __repr__(self):
        return f"P2POp({self.op}, tensor={self.tensor.shape}, peer={self.peer})"


# Alias to avoid shadowing by module-level P2POp function
_P2POpClass = P2POp


class CollectiveContext:
    """
    High-level collective communication context that wraps the UCCL p2p engine.
    Provides NCCL-like send/recv interface with automatic connection management.
    """

    def __init__(
        self,
        num_cpus: int = 4,
        local_gpu_idx: Optional[int] = None,
        use_copy_engine_for_intra: Optional[bool] = False,
    ):
        """
        Initialize collective context. Requires torch.distributed to be initialized.

        Args:
            num_cpus: Number of CPU threads for RDMA operations
            local_gpu_idx: Optional override for local GPU index. If None, will be derived from torch.distributed
            use_copy_engine_for_intra: Whether to use the copy engine for intra-node communication
        """
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before creating CollectiveContext"
            )

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.dist_backend = "gloo" if use_copy_engine_for_intra else "nccl"
        self.group = dist.new_group(
            ranks=range(self.world_size), backend=self.dist_backend
        )
        self.num_cpus = num_cpus
        self.use_copy_engine_for_intra = use_copy_engine_for_intra

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
        self.memory_regions = utils.ClosedIntervalTree()
        self.initialized = False

        # check and setup fd limit and somaxconn for UDS
        utils.set_files_limit()

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
                return dist.get_local_rank(group=self.group)
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
        print(f"[Rank {self.rank}] Created p2p.Endpoint on GPU {self.local_gpu_idx}")
        local_metadata = self.ep.get_metadata()

        # Initialize connection arrays
        self.send_connections = [None] * self.world_size  # indexed by rank
        self.recv_connections = [None] * self.world_size  # indexed by rank
        self.local_connections = [False] * self.world_size  # indexed by rank

        device = torch.device(f"cuda:{self.local_gpu_idx}")
        # Exchange metadata with all peers using torch.distributed
        all_metadata = [None] * self.world_size
        # Gather all metadata
        metadata_tensor = torch.ByteTensor(list(local_metadata)).to(device)
        gathered_tensors = [
            torch.zeros_like(metadata_tensor, device=device)
            for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_tensors, metadata_tensor, group=self.group)

        for i, tensor in enumerate(gathered_tensors):
            if tensor.is_cuda:
                tensor_cpu = tensor.cpu()
                all_metadata[i] = bytes(tensor_cpu.tolist())
            else:
                all_metadata[i] = bytes(tensor.tolist())

        if self.dist_backend == "nccl":
            torch.cuda.synchronize(device)

        # Establish connections with all other ranks
        print(f"[Rank {self.rank}] Establishing connections with all peers...")
        self._establish_connections(all_metadata)
        print(f"[Rank {self.rank}] All connections established.")
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

        def accept_from_peer(peer_rank):
            print("Accept connection from a specific peer")
            """Accept connection from a specific peer for receiving data FROM that peer."""
            try:
                # We need to accept both local and remote connections
                # For local connections, we can only accept local ones
                # For remote connections, we accept regular ones
                # We'll need to track which type we're expecting based on remaining connections

                # Count how many local vs remote connections we still need to accept
                if self.local_connections[peer_rank]:
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
                else:
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
            max_workers=self.world_size * 2
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
                    future = executor.submit(accept_from_peer, peer_rank)
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

    def float_type_from_tensor(self, t: torch.Tensor) -> p2p.FloatType:
        if t.dtype == torch.float16:
            return p2p.FloatType.kFloat16
        elif t.dtype == torch.bfloat16:
            return p2p.FloatType.kBFloat16
        elif t.dtype == torch.float32:
            return p2p.FloatType.kFloat32
        else:
            raise TypeError(f"Unsupported tensor dtype: {t.dtype}")

    def _register_memory(self, ptr: int, size: int, float_type: p2p.FloatType) -> int:
        """Register memory and cache the memory region information."""
        existing_mr_id = self._check_register(ptr, size)
        if existing_mr_id is not None:
            return existing_mr_id
        ok, mr_id = self.ep.reg(ptr, size, float_type)
        if not ok:
            raise RuntimeError("Failed to register memory")
        self.memory_regions.add(ptr, ptr + size, mr_id)
        return mr_id

    def _check_register(self, ptr: int, size: int) -> Optional[int]:
        end_ptr = ptr + size
        containing = self.memory_regions.query_containing(ptr, end_ptr)
        if containing:
            return containing[0][2]
        return None

    def check_tensor_registered(self, tensor: torch.Tensor) -> Optional[int]:
        if not self.initialized:
            raise RuntimeError("CollectiveContext not initialized. Call init() first.")

        ptr, size = self._get_buffer_info(tensor)
        return self._check_register(ptr, size)

    def register_tensor(self, tensor: torch.Tensor) -> int:
        """Register a tensor for RDMA. Not required for local IPC connections."""
        if not self.initialized:
            raise RuntimeError("CollectiveContext not initialized. Call init() first.")
        ptr, size = self._get_buffer_info(tensor)
        return self._register_memory(ptr, size, self.float_type_from_tensor(tensor))

    def deregister_tensor(self, tensor: torch.Tensor) -> bool:
        """Deregister a previously registered tensor."""
        if not self.initialized:
            raise RuntimeError("CollectiveContext not initialized. Call init() first.")

        ptr, mem_size = self._get_buffer_info(tensor)
        start, end = ptr, ptr + mem_size

        matching_regions = self.memory_regions.query_exact_match(start, end)

        if not matching_regions:
            return False

        for region_start, region_end, mr_id in matching_regions:
            try:
                self.ep.dereg(mr_id)
                self.memory_regions.remove(region_start, region_end, mr_id)
            except Exception as e:
                print(f"Failed to deregister memory region {mr_id}: {e}")
                continue

        return True

    def send(self, tensor: torch.Tensor, dst: int):
        """Send tensor to destination rank (synchronous)."""
        if not self.initialized:
            raise RuntimeError("CollectiveContext not initialized. Call init() first.")

        if dst == self.rank:
            return  # No-op for self-send

        if dst >= self.world_size or self.send_connections[dst] is None:
            raise ValueError(f"No send connection to rank {dst}")

        ptr, size = self._get_buffer_info(tensor)
        conn_id = self.send_connections[dst]

        if self.local_connections[dst]:
            if self.use_copy_engine_for_intra:
                # Use IPC for local connection (no memory registration needed)
                ok = self.ep.send_ipc(conn_id, ptr, size)
                if not ok:
                    raise RuntimeError(f"Failed to initiate IPC send to rank {dst}")
            else:
                dist.send(tensor, dst, group=self.group)
        else:
            # Use RDMA for remote connection (requires memory registration)
            mr_id = self.check_tensor_registered(tensor)
            if mr_id == None:
                raise RuntimeError(
                    f"Tensor memory not registered for remote communication with rank {dst}. "
                    f"Call register_tensor() first for tensors used with remote ranks."
                )
            ok = self.ep.send(conn_id, mr_id, ptr, size)
            if not ok:
                raise RuntimeError(f"Failed to initiate RDMA send to rank {dst}")

    def recv(self, tensor: torch.Tensor, src: int):
        """Receive tensor from source rank (synchronous)."""
        if not self.initialized:
            raise RuntimeError("CollectiveContext not initialized. Call init() first.")

        if src == self.rank:
            return  # No-op for self-recv

        if src >= self.world_size or self.recv_connections[src] is None:
            raise ValueError(f"No recv connection from rank {src}")

        ptr, size = self._get_buffer_info(tensor)
        conn_id = self.recv_connections[src]

        if self.local_connections[src]:
            if self.use_copy_engine_for_intra:
                # Use IPC for local connection (no memory registration needed)
                ok = self.ep.recv_ipc(conn_id, ptr, size)
                if not ok:
                    raise RuntimeError(f"Failed to initiate IPC recv from rank {src}")
            else:
                dist.recv(tensor, src, group=self.group)
        else:
            # Use RDMA for remote connection (requires memory registration)
            mr_id = self.check_tensor_registered(tensor)
            if mr_id == None:
                raise RuntimeError(
                    f"Tensor memory not registered for remote communication with rank {src}. "
                    f"Call register_tensor() first for tensors used with remote ranks."
                )
            ok = self.ep.recv(conn_id, mr_id, ptr, size)
            if not ok:
                raise RuntimeError(f"Failed to initiate RDMA recv from rank {src}")

    def isend(self, tensor: torch.Tensor, dst: int) -> Union[int, dist.Work]:
        """Initiate asynchronous send (non-blocking)."""
        if not self.initialized:
            raise RuntimeError("CollectiveContext not initialized. Call init() first.")

        if dst == self.rank:
            return -1  # Return dummy ID for self-send

        if dst >= self.world_size or self.send_connections[dst] is None:
            raise ValueError(f"No send connection to rank {dst}")

        ptr, size = self._get_buffer_info(tensor)
        conn_id = self.send_connections[dst]

        if self.local_connections[dst]:
            if self.use_copy_engine_for_intra:
                # Use IPC async for local connection (no memory registration needed)
                ok, transfer_handle = self.ep.send_ipc_async(conn_id, ptr, size)
                if not ok:
                    raise RuntimeError(
                        f"Failed to initiate async IPC send to rank {dst}"
                    )
                return transfer_handle
            else:
                # Use NCCL - start immediately and return Work object
                op = dist.P2POp(dist.isend, tensor, dst, group=self.group)
                reqs = dist.batch_isend_irecv([op])
                return reqs[0]
        else:
            # Use RDMA async for remote connection (requires memory registration)
            mr_id = self.check_tensor_registered(tensor)
            if mr_id == None:
                raise RuntimeError(
                    f"Tensor memory not registered for remote communication with rank {dst}. "
                    f"Call register_tensor() first for tensors used with remote ranks."
                )
            ok, transfer_handle = self.ep.send_async(conn_id, mr_id, ptr, size)
            if not ok:
                raise RuntimeError(f"Failed to initiate async RDMA send to rank {dst}")
            return transfer_handle

    def irecv(self, tensor: torch.Tensor, src: int) -> Union[int, dist.Work]:
        """Initiate asynchronous receive (non-blocking)."""
        if not self.initialized:
            raise RuntimeError("CollectiveContext not initialized. Call init() first.")

        if src == self.rank:
            return -1  # Return dummy ID for self-recv

        if src >= self.world_size or self.recv_connections[src] is None:
            raise ValueError(
                f"No recv connection from rank {src} world_size {self.world_size}"
            )

        ptr, size = self._get_buffer_info(tensor)
        conn_id = self.recv_connections[src]

        if self.local_connections[src]:
            if self.use_copy_engine_for_intra:
                # Use IPC async for local connection (no memory registration needed)
                ok, transfer_handle = self.ep.recv_ipc_async(conn_id, ptr, size)
                if not ok:
                    raise RuntimeError(
                        f"Failed to initiate async IPC recv from rank {src}"
                    )
                return transfer_handle
            else:
                # Use NCCL - start immediately and return Work object
                op = dist.P2POp(dist.irecv, tensor, src, group=self.group)
                reqs = dist.batch_isend_irecv([op])
                return reqs[0]
        else:
            # Use RDMA async for remote connection (requires memory registration)
            mr_id = self.check_tensor_registered(tensor)
            if mr_id == None:
                raise RuntimeError(
                    f"Tensor memory not registered for remote communication with rank {src}. "
                    f"Call register_tensor() first for tensors used with remote ranks."
                )
            ok, transfer_handle = self.ep.recv_async(conn_id, mr_id, ptr, size)
            if not ok:
                raise RuntimeError(
                    f"Failed to initiate async RDMA recv from rank {src}"
                )
            return transfer_handle

    def test(self, transfer_handle: Union[int, dist.Work]) -> bool:
        """Test if asynchronous operation is complete (non-blocking)."""
        if isinstance(transfer_handle, int):
            if transfer_handle == -1:
                return True  # Self-transfers are always complete
            ok, is_done = self.ep.poll_async(transfer_handle)
            if not ok:
                raise RuntimeError(f"Error polling transfer {transfer_handle}")
            return is_done
        else:
            # Handle Work object - use is_completed() for non-blocking check
            return transfer_handle.is_completed()

    def wait(self, transfer_handle: Union[int, dist.Work]):
        """Wait for asynchronous operation to complete."""
        if isinstance(transfer_handle, int):
            if transfer_handle == -1:
                return  # No-op for dummy self-transfer
            while True:
                ok, is_done = self.ep.poll_async(transfer_handle)
                if not ok:
                    raise RuntimeError(f"Error polling transfer {transfer_handle}")
                if is_done:
                    break
        else:
            # Handle Work object - wait for completion
            transfer_handle.wait()
            torch.cuda.synchronize()

    def wait_all(self, transfer_handles: List[Union[int, dist.Work]]):
        """Wait for all asynchronous operations to complete."""
        work_objects = []
        for item in transfer_handles:
            if isinstance(item, int):
                self.wait(item)
            else:
                work_objects.append(item)
        # Wait for all Work objects
        for work in work_objects:
            work.wait()
        if len(work_objects) > 0:
            # Synchronize CUDA to ensure NCCL operations are actually complete
            torch.cuda.synchronize()

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

    def P2POp(self, op, tensor: torch.Tensor, peer: int) -> _P2POpClass:
        """
        Create a P2P operation descriptor for use with batch_isend_irecv.

        Usage:
            ctx.P2POp(collective.isend, tensor, peer)
            ctx.P2POp(collective.irecv, tensor, peer)
            ctx.P2POp("send", tensor, peer)
            ctx.P2POp("recv", tensor, peer)
        """
        # Normalize op to string
        if callable(op):
            op_name = getattr(op, "__name__", str(op))
            if "send" in op_name.lower():
                op_str = "send"
            elif "recv" in op_name.lower():
                op_str = "recv"
            else:
                raise ValueError(f"Unknown op function: {op_name}")
        elif isinstance(op, str):
            if op not in ("send", "recv"):
                raise ValueError(f"op must be 'send' or 'recv', got {op}")
            op_str = op
        else:
            raise ValueError(f"op must be str or callable, got {type(op)}")

        return _P2POpClass(op_str, tensor, peer)

    def batch_isend_irecv(
        self, ops: List[Union[_P2POpClass, Tuple[str, torch.Tensor, int]]]
    ) -> List[Union[int, dist.Work]]:
        """
        Batch execute isend/irecv operations efficiently.

        ops: List of P2POp descriptors (from ctx.P2POp()) or (op_type, tensor, peer) tuples
        """
        if not self.initialized:
            raise RuntimeError("CollectiveContext not initialized. Call init() first.")

        p2p_ops = []
        int_handles = []

        for op in ops:
            # Support both P2POp objects and tuples
            if isinstance(op, _P2POpClass):
                op_type, tensor, peer = op.op, op.tensor, op.peer
            else:
                op_type, tensor, peer = op
            if peer == self.rank:
                continue  # Skip self-communication

            if op_type == "send":
                if self.local_connections[peer]:
                    if self.use_copy_engine_for_intra:
                        ptr, size = self._get_buffer_info(tensor)
                        conn_id = self.send_connections[peer]
                        ok, handle = self.ep.send_ipc_async(conn_id, ptr, size)
                        if not ok:
                            raise RuntimeError(f"Failed IPC send to {peer}")
                        int_handles.append(handle)
                    else:
                        p2p_ops.append(
                            dist.P2POp(dist.isend, tensor, peer, group=self.group)
                        )
                else:
                    ptr, size = self._get_buffer_info(tensor)
                    conn_id = self.send_connections[peer]
                    mr_id = self.check_tensor_registered(tensor)
                    if mr_id is None:
                        raise RuntimeError(f"Tensor not registered for rank {peer}")
                    ok, handle = self.ep.send_async(conn_id, mr_id, ptr, size)
                    if not ok:
                        raise RuntimeError(f"Failed RDMA send to {peer}")
                    int_handles.append(handle)
            else:  # recv
                if self.local_connections[peer]:
                    if self.use_copy_engine_for_intra:
                        ptr, size = self._get_buffer_info(tensor)
                        conn_id = self.recv_connections[peer]
                        ok, handle = self.ep.recv_ipc_async(conn_id, ptr, size)
                        if not ok:
                            raise RuntimeError(f"Failed IPC recv from {peer}")
                        int_handles.append(handle)
                    else:
                        p2p_ops.append(
                            dist.P2POp(dist.irecv, tensor, peer, group=self.group)
                        )
                else:
                    ptr, size = self._get_buffer_info(tensor)
                    conn_id = self.recv_connections[peer]
                    mr_id = self.check_tensor_registered(tensor)
                    if mr_id is None:
                        raise RuntimeError(f"Tensor not registered for rank {peer}")
                    ok, handle = self.ep.recv_async(conn_id, mr_id, ptr, size)
                    if not ok:
                        raise RuntimeError(f"Failed RDMA recv from {peer}")
                    int_handles.append(handle)

        # Batch execute NCCL P2POps
        work_handles = []
        if len(p2p_ops) > 0:
            work_handles = dist.batch_isend_irecv(p2p_ops)

        return int_handles + work_handles

    def allgather(
        self,
        send_tensor: torch.Tensor,
        recv_tensor: torch.Tensor,
    ):
        """All-gather operation: gather data from all ranks to all ranks (synchronous)."""
        handles = self.iallgather(send_tensor, recv_tensor)
        self.wait_all(handles)

    def iallgather(
        self,
        send_tensor: torch.Tensor,
        recv_tensor: torch.Tensor,
    ) -> List[Union[int, dist.Work]]:
        """Initiate asynchronous all-gather operation (non-blocking)."""
        if not self.initialized:
            raise RuntimeError("CollectiveContext not initialized. Call init() first.")

        # Validate tensor sizes
        send_size = send_tensor.numel() * send_tensor.element_size()
        recv_size = recv_tensor.numel() * recv_tensor.element_size()
        expected_recv_size = send_size * self.world_size

        if recv_size < expected_recv_size:
            raise ValueError(
                f"recv_tensor too small: got {recv_size} bytes, need at least {expected_recv_size} bytes "
                f"(send_size={send_size} * world_size={self.world_size})"
            )

        # Calculate chunk size (same as send_tensor size)
        chunk_numel = send_tensor.numel()

        # Split recv_tensor into chunks for each rank
        recv_chunks = [
            recv_tensor.view(-1)[i * chunk_numel : (i + 1) * chunk_numel].view(
                send_tensor.shape
            )
            for i in range(self.world_size)
        ]

        # Copy own data to appropriate position in recv buffer
        recv_chunks[self.rank].copy_(send_tensor)

        # Build P2POp list for batch_isend_irecv
        ops = []
        for peer_rank in range(self.world_size):
            if peer_rank != self.rank:
                ops.append(self.P2POp("send", send_tensor, peer_rank))

        for peer_rank in range(self.world_size):
            if peer_rank != self.rank:
                ops.append(self.P2POp("recv", recv_chunks[peer_rank], peer_rank))

        return self.batch_isend_irecv(ops)


# Global collective context instance
_default_context: Optional[CollectiveContext] = None


def init_collective(
    num_cpus: int = 4,
    local_gpu_idx: Optional[int] = None,
    use_copy_engine_for_intra: Optional[bool] = False,
):
    """Initialize the default collective context."""
    global _default_context
    _default_context = CollectiveContext(
        num_cpus, local_gpu_idx, use_copy_engine_for_intra
    )
    _default_context.init()


def get_collective() -> CollectiveContext:
    """Get the default collective context."""
    if _default_context is None:
        raise RuntimeError("Collective not initialized. Call init_collective() first.")
    return _default_context


def register_tensor(tensor: torch.Tensor):
    """Register tensor using the default collective context."""
    get_collective().register_tensor(tensor)


def deregister_tensor(tensor: torch.Tensor):
    """Deregister tensor using the default collective context."""
    get_collective().deregister_tensor(tensor)


def send(tensor: torch.Tensor, dst: int):
    """Send tensor using the default collective context."""
    get_collective().send(tensor, dst)


def recv(tensor: torch.Tensor, src: int):
    """Receive tensor using the default collective context."""
    get_collective().recv(tensor, src)


def isend(tensor: torch.Tensor, dst: int) -> Union[int, dist.Work]:
    """Async send tensor using the default collective context."""
    return get_collective().isend(tensor, dst)


def irecv(tensor: torch.Tensor, src: int) -> Union[int, dist.Work]:
    """Async receive tensor using the default collective context."""
    return get_collective().irecv(tensor, src)


def test(transfer_handle: Union[int, dist.Work]) -> bool:
    """Test async operation using the default collective context."""
    return get_collective().test(transfer_handle)


def wait(transfer_handle: Union[int, dist.Work]):
    """Wait for async operation using the default collective context."""
    get_collective().wait(transfer_handle)


def wait_all(transfer_handles: List[Union[int, dist.Work]]):
    """Wait for all async operations using the default collective context."""
    get_collective().wait_all(transfer_handles)


def P2POp(op, tensor: torch.Tensor, peer: int) -> _P2POpClass:
    """Create a P2P operation descriptor for use with batch_isend_irecv."""
    return get_collective().P2POp(op, tensor, peer)


def batch_isend_irecv(
    ops: List[Union[_P2POpClass, Tuple[str, torch.Tensor, int]]],
) -> List[Union[int, dist.Work]]:
    """Batch execute isend/irecv operations efficiently."""
    return get_collective().batch_isend_irecv(ops)


def finalize_collective():
    """Finalize the default collective context."""
    global _default_context
    if _default_context is not None:
        _default_context.finalize()
        _default_context = None


def allgather(send_tensor: torch.Tensor, recv_tensor: torch.Tensor):
    """All-gather using the default collective context."""
    get_collective().allgather(send_tensor, recv_tensor)


def iallgather(
    send_tensor: torch.Tensor, recv_tensor: torch.Tensor
) -> List[Union[int, dist.Work]]:
    """Async all-gather using the default collective context."""
    return get_collective().iallgather(send_tensor, recv_tensor)
