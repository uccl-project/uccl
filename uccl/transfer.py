import zmq
import torch
import pybind11 as py

try:
    from . import p2p
except ImportError:
    import p2p


class TransferManager:

    class Connection:
        def __init__(
            self,
            local_gpu_idx: int,
            remote_gpu_idx: int,
            socket: zmq.Socket,
            conn_id: int,
            is_local: bool,
        ):
            self.local_gpu_idx = local_gpu_idx
            self.remote_gpu_idx = remote_gpu_idx
            self.conn_id = conn_id
            self.is_local = is_local

    class TransferState:
        def __init__(
            self,
            data_ptr: int,
            size: int,
            mr_id: int,
            conn: "TransferManager.Connection",
        ):
            self.data_ptr = data_ptr
            self.size = size
            self.mr_id = mr_id
            self.conn = conn

    def __init__(
        self, gpu_idx_list: list[int], num_cpus: int, zmq_port: int
    ):
        assert len(gpu_idx_list) > 0
        self.gpu_idx_list = gpu_idx_list
        self.num_cpus = num_cpus
        self.zmq_port = zmq_port

        max_gpu_idx = max(gpu_idx_list)
        self.eps = [None for _ in range(max_gpu_idx + 1)]
        self.ep_ports = [None for _ in range(max_gpu_idx + 1)]
        for gpu_idx in gpu_idx_list:
            self.eps[gpu_idx] = p2p.Endpoint(gpu_idx, num_cpus)
            self.ep_ports[gpu_idx] = self.eps[gpu_idx].get_metadata()[1]

        # Setup ZMQ socket for listening if port specified
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.zmq_port}")

        self.conn_ids = {gpu_idx: {} for gpu_idx in self.gpu_idx_list}
        # Used to determine if the connection is local or remote
        self.local_ip = p2p.get_oob_ip()
        self.transfer_id = 0
        self.transfer_table = {}

    # Connecting to a remote server GPU-to-GPU.
    def connect(self, remote_ip: str, remote_zmq_port: int) -> bool:
        socket = self.zmq_context.socket(zmq.REQ)
        socket.connect(f"tcp://{remote_ip}:{remote_zmq_port}")

        socket.send_pyobj(self.gpu_idx_list)
        remote_gpu_idx_list = socket.recv_pyobj()
        # Assume the same GPU indexes in both sides
        assert remote_gpu_idx_list == self.gpu_idx_list

        socket.send_pyobj(self.ep_ports)
        remote_ep_ports = socket.recv_pyobj()
        assert len(remote_ep_ports) == len(self.ep_ports)

        socket.send_pyobj(self.local_ip)
        remote_ip = socket.recv_pyobj()

        for local_gpu_idx in self.gpu_idx_list:
            remote_gpu_idx = local_gpu_idx
            if self.local_ip == remote_ip:
                success, conn_id = self.eps[local_gpu_idx].connect_local(
                    remote_gpu_idx
                )
            else:
                success, conn_id = self.eps[local_gpu_idx].connect(
                    remote_ip,
                    remote_gpu_idx,
                    self.remote_ep_ports[remote_gpu_idx],
                )
            assert (
                success
            ), f"Failed to connect to {remote_ip} on GPU {remote_gpu_idx}"
            self.conn_ids[local_gpu_idx][remote_gpu_idx] = self.Connection(
                local_gpu_idx,
                remote_gpu_idx,
                socket,
                conn_id,
                self.local_ip == remote_ip,
            )

        return True

    def accept(self) -> bool:
        remote_gpu_idx_list = self.socket.recv_pyobj()
        assert remote_gpu_idx_list == self.gpu_idx_list
        self.socket.send_pyobj(self.gpu_idx_list)

        remote_ep_ports = self.socket.recv_pyobj()
        assert len(remote_ep_ports) == len(self.ep_ports)
        self.socket.send_pyobj(self.ep_ports)

        remote_ip = self.socket.recv_pyobj()
        self.socket.send_pyobj(self.local_ip)

        for local_gpu_idx in self.gpu_idx_list:
            if self.local_ip == remote_ip:
                success, remote_gpu_idx, conn_id = self.eps[
                    local_gpu_idx
                ].accept_local()
            else:
                success, remote_ip_addr, remote_gpu_idx, conn_id = self.eps[
                    local_gpu_idx
                ].accept()
            assert (
                success
            ), f"Failed to accept connection from {remote_ip} on GPU {remote_gpu_idx}"
            self.conn_ids[local_gpu_idx][remote_gpu_idx] = self.Connection(
                local_gpu_idx,
                remote_gpu_idx,
                self.socket,
                conn_id,
                self.local_ip == remote_ip,
            )

        return True

    def register_transfer(
        self, local_gpu_idx: int, remote_gpu_idx: int, tensor: torch.Tensor
    ) -> int:
        assert tensor.is_contiguous()
        success, mr_id = self.eps[local_gpu_idx].reg(
            tensor.data_ptr(), tensor.numel() * tensor.element_size()
        )
        assert success, f"Failed to register tensor on GPU {local_gpu_idx}"
        self.transfer_table[self.transfer_id] = self.TransferState(
            tensor.data_ptr(),
            tensor.numel() * tensor.element_size(),
            mr_id,
            self.conn_ids[local_gpu_idx][remote_gpu_idx],
        )
        self.transfer_id += 1
        return self.transfer_id - 1

    def post_transfer_metadata(self, transfer_id: int) -> bool:
        transfer_state = self.transfer_table[transfer_id]
        conn = transfer_state.conn
        if conn.is_local:
            success, transfer_metadata = self.eps[
                conn.local_gpu_idx
            ].advertise_ipc(
                conn.conn_id, transfer_state.data_ptr, transfer_state.size
            )
        else:
            success, transfer_metadata = self.eps[conn.local_gpu_idx].advertise(
                conn.conn_id,
                transfer_state.mr_id,
                transfer_state.data_ptr,
                transfer_state.size,
            )
        assert (
            success
        ), f"Failed to advertise tensor on GPU {conn.local_gpu_idx}"
        conn.socket.send_pyobj(transfer_metadata)
        return True

    def fetch_transfer_metadata(self, transfer_id: int) -> bytes:
        transfer_state = self.transfer_table[transfer_id]
        conn = transfer_state.conn
        transfer_metadata = conn.socket.recv_pyobj()
        assert (
            transfer_metadata is not None
        ), f"Failed to fetch transfer metadata on GPU {conn.local_gpu_idx}"
        return transfer_metadata

    def transfer_tensor(
        self, transfer_id: int, transfer_metadata: bytes
    ) -> bool:
        transfer_state = self.transfer_table[transfer_id]
        conn = transfer_state.conn
        if conn.is_local:
            success = self.eps[conn.local_gpu_idx].write_ipc(
                conn.conn_id,
                transfer_state.data_ptr,
                transfer_state.size,
                transfer_metadata,
            )
        else:
            success = self.eps[conn.local_gpu_idx].write(
                conn.conn_id,
                transfer_state.mr_id,
                transfer_state.data_ptr,
                transfer_state.size,
                transfer_metadata,
            )
        assert success, f"Failed to transfer tensor on GPU {conn.local_gpu_idx}"
        return True

    def cleanup_transfer(self, transfer_id: int) -> bool:
        transfer_state = self.transfer_table[transfer_id]
        conn = transfer_state.conn
        success = self.eps[conn.local_gpu_idx].dereg(transfer_state.mr_id)
        assert success, f"Failed to cleanup tensor on GPU {conn.local_gpu_idx}"
        del self.transfer_table[transfer_id]
        return True
