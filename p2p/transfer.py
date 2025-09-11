import socket
import torch
from .utils import create_socket_and_connect, send_obj, recv_obj

try:
    from . import p2p
except ImportError:
    import p2p


class TransferManager:

    class ConnState:
        def __init__(
            self,
            local_gpu_idx: int,
            remote_gpu_idx: int,
            socket: socket.socket,
            conn_id: int,
            is_local: bool,
        ):
            self.local_gpu_idx = local_gpu_idx
            self.remote_gpu_idx = remote_gpu_idx
            self.socket = socket
            self.conn_id = conn_id
            self.is_local = is_local

    class TransferState:
        def __init__(
            self,
            data: int,
            size: int,
            mr_id: int,
            conn_state: "TransferManager.ConnState",
        ):
            self.data = data
            self.size = size
            self.mr_id = mr_id
            self.conn_state = conn_state

    def __init__(self, local_gpu_idx: int, num_cpus: int, listen_port: int):
        self.local_gpu_idx = local_gpu_idx
        self.num_cpus = num_cpus
        self.listen_port = listen_port

        self.ep = p2p.Endpoint(local_gpu_idx, num_cpus)
        # The C++ Endpoint listens on this port.
        self.local_ep_port = p2p.Endpoint.parse_metadata(self.ep.get_metadata())[1]
        # Used to determine if the connection is local or remote
        self.local_ep_ip = p2p.get_oob_ip()

        # Mapping remote GPU index to ConnState
        self.conn_table = {}
        # Mapping transfer id to TransferState
        self.transfer_table = {}
        self.next_transfer_id = 0

        # This Python TransferManager listens on this port.
        self.listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listen_socket.bind(("", listen_port))
        self.listen_socket.listen(128)

    # Connecting to a remote GPU.
    def connect(self, remote_ip: str, remote_listen_port: int) -> int:
        socket = create_socket_and_connect(remote_ip, remote_listen_port)

        send_obj(socket, [self.local_gpu_idx, self.local_ep_port, self.local_ep_ip])
        remote_gpu_idx, remote_ep_port, remote_ep_ip = recv_obj(socket)

        is_local = self.local_ep_ip == remote_ep_ip

        if is_local:
            success, conn_id = self.ep.connect_local(remote_gpu_idx)
            assert success, f"Failed to connect to local GPU {remote_gpu_idx}"
        else:
            success, conn_id = self.ep.connect(
                remote_ep_ip,
                remote_gpu_idx,
                remote_ep_port,
            )
            assert (
                success
            ), f"Failed to connect to remote GPU {remote_gpu_idx} on {remote_ep_ip}"

        self.conn_table[conn_id] = self.ConnState(
            self.local_gpu_idx,
            remote_gpu_idx,
            socket,
            conn_id,
            is_local,
        )

        return conn_id

    def accept(self) -> int:
        socket, addr = self.listen_socket.accept()

        send_obj(socket, [self.local_gpu_idx, self.local_ep_port, self.local_ep_ip])
        remote_gpu_idx, remote_ep_port, remote_ep_ip = recv_obj(socket)

        is_local = self.local_ep_ip == remote_ep_ip

        if is_local:
            success, _remote_gpu_idx, conn_id = self.ep.accept_local()
            assert (
                success
            ), f"Failed to accept connection from local GPU {remote_gpu_idx}"
        else:
            success, _remote_ep_addr, _remote_gpu_idx, conn_id = self.ep.accept()
            assert (
                success
            ), f"Failed to accept connection from remote GPU {remote_gpu_idx} on {remote_ep_ip}"

        self.conn_table[conn_id] = self.ConnState(
            self.local_gpu_idx,
            remote_gpu_idx,
            socket,
            conn_id,
            is_local,
        )

        return conn_id

    def register_transfer(
        self,
        conn_id: int,
        tensor: torch.Tensor,
    ) -> int:
        assert tensor.is_contiguous()
        data = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()

        success, mr_id = self.ep.reg(data, size)
        assert success, f"Failed to register tensor on GPU {self.local_gpu_idx}"

        conn_state = self.conn_table[conn_id]
        self.transfer_table[self.next_transfer_id] = self.TransferState(
            data,
            size,
            mr_id,
            conn_state,
        )
        self.next_transfer_id += 1
        return self.next_transfer_id - 1

    def deregister_transfer(self, transfer_id: int) -> bool:
        transfer_state = self.transfer_table[transfer_id]
        success = self.ep.dereg(transfer_state.mr_id)
        assert success, f"Failed to cleanup tensor on GPU {self.local_gpu_idx}"

        del self.transfer_table[transfer_id]
        return True

    def post_transfer_metadata(self, transfer_id: int) -> bool:
        transfer_state = self.transfer_table[transfer_id]
        conn_state = transfer_state.conn_state

        if conn_state.is_local:
            success, transfer_metadata = self.ep.advertise_ipc(
                conn_state.conn_id, transfer_state.data, transfer_state.size
            )
            assert (
                success
            ), f"Failed to advertise tensor on GPU {self.local_gpu_idx} for IPC"
        else:
            success, transfer_metadata = self.ep.advertise(
                conn_state.conn_id,
                transfer_state.mr_id,
                transfer_state.data,
                transfer_state.size,
            )
            assert (
                success
            ), f"Failed to advertise tensor on GPU {self.local_gpu_idx} for RDMA"

        send_obj(conn_state.socket, transfer_metadata)
        return True

    def fetch_transfer_metadata(self, transfer_id: int) -> bytes:
        transfer_state = self.transfer_table[transfer_id]
        conn_state = transfer_state.conn_state

        transfer_metadata = recv_obj(conn_state.socket)
        assert (
            transfer_metadata is not None
        ), f"Failed to fetch transfer metadata on GPU {self.local_gpu_idx}"

        return transfer_metadata

    def do_transfer_async(self, transfer_id: int, transfer_metadata: bytes) -> int:
        transfer_state = self.transfer_table[transfer_id]
        conn_state = transfer_state.conn_state
        if conn_state.is_local:
            success, poll_id = self.ep.write_ipc_async(
                conn_state.conn_id,
                transfer_state.data,
                transfer_state.size,
                transfer_metadata,
            )
        else:
            success, poll_id = self.ep.write_async(
                conn_state.conn_id,
                transfer_state.mr_id,
                transfer_state.data,
                transfer_state.size,
                transfer_metadata,
            )
        assert success, f"Failed to transfer tensor on GPU {self.local_gpu_idx}"
        return poll_id

    def check_transfer_done(self, transfer_id: int, poll_id: int) -> bool:
        success, is_done = self.ep.poll_async(poll_id)
        assert (
            success
        ), f"Failed to check if transfer is done on GPU {self.local_gpu_idx}"

        return is_done
