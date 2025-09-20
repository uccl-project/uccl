import sys
import resource
import socket
import time
import struct
import pickle
from typing import Any, Tuple

import torch

try:
    from . import p2p
except ImportError:
    import p2p


def set_files_limit():
    """
    Configure files limit for high-performance communication.
    """
    print("Setting up files limit...", file=sys.stderr)

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"Current fd limit: soft={soft}, hard={hard}", file=sys.stderr)

        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            print(f"fd limit raised to: {hard}", file=sys.stderr)
        else:
            print("fd limit already at maximum", file=sys.stderr)

    except Exception as e:
        print(f"Failed to set fd limit: {e}", file=sys.stderr)


def create_socket_and_connect(
    host,
    port,
    max_retries=None,
    initial_delay=0.5,
    backoff=2,
    max_delay=10,
    timeout=None,
):
    """
    Try to connect to (host, port) with retry logic.

    :param host: Server hostname or IP
    :param port: Server port
    :param max_retries: Maximum number of retries (None = retry forever)
    :param initial_delay: Delay (seconds) before first retry
    :param backoff: Exponential backoff multiplier
    :param max_delay: Maximum delay between retries
    :param timeout: Optional socket timeout (seconds)
    :return: Connected socket object
    :raises: OSError if cannot connect after max_retries
    """
    attempt = 0
    delay = initial_delay

    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if timeout:
                s.settimeout(timeout)
            s.connect((host, port))
            print(f"[retry_connect] Connected to {host}:{port} after {attempt} retries")
            return s
        except (ConnectionRefusedError, TimeoutError, OSError) as e:
            attempt += 1
            if max_retries is not None and attempt > max_retries:
                raise OSError(
                    f"[retry_connect] Failed to connect after {max_retries} retries: {e}"
                )
            print(
                f"[retry_connect] Attempt {attempt} failed: {e}, retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
            delay = min(delay * backoff, max_delay)


_LEN_FMT = "!Q"  # 8-byte unsigned length, network byte order
_LEN_SIZE = struct.calcsize(_LEN_FMT)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes or raise ConnectionError on EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        buf += chunk
    return bytes(buf)


def send_obj(
    sock: socket.socket, obj: Any, *, protocol: int = pickle.HIGHEST_PROTOCOL
) -> None:
    """Pickle + length-prefix + sendall."""
    payload = pickle.dumps(obj, protocol=protocol)
    header = struct.pack(_LEN_FMT, len(payload))
    # sendall guarantees the entire buffer is sent (or raises)
    sock.sendall(header)
    sock.sendall(payload)


def recv_obj(sock: socket.socket) -> Any:
    """Recv length-prefix + unpickle."""
    header = _recv_exact(sock, _LEN_SIZE)
    (length,) = struct.unpack(_LEN_FMT, header)
    if length == 0:
        # Support empty payload edge-case (rare)
        return None
    payload = _recv_exact(sock, length)
    return pickle.loads(payload)


def create_tensor(
    shape: Tuple[int, ...], dtype: torch.dtype, device: str
) -> Tuple[torch.Tensor, int]:
    """
    Create an empty tensor and register GPU memory if applicable.

    Args:
        shape (Tuple[int, ...]): e.g., (2, 100, 4)
        dtype (torch.dtype): e.g., torch.float32
        device (str): 'cuda:0', 'cuda:1', or 'cpu'

    Returns:
        Tuple[torch.Tensor, int]: tensor, tensor_id
    """
    if device.startswith("cuda"):
        gpu_id = int(device.split(":")[1])
    elif device.startswith("cpu"):
        gpu_id = -1  # CPU -1
    else:
        raise RuntimeError(f"Invalid device: ", device)

    tensor = torch.empty(size=shape, dtype=dtype, device=device)

    addr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()

    tensor_id = p2p.reg_mem(gpu_id, addr, size)

    if tensor_id < 0:
        raise RuntimeError(
            f"Failed to register memory: tensor_id={tensor_id}, gpu_id={gpu_id}"
        )

    return tensor, tensor_id


def free_tensor(tensor_id: int):
    """
    Free the registered GPU/CPU memory for a tensor.

    Args:
        tensor_id (int)
    """
    if tensor_id < 0:
        raise RuntimeError(f"Invalid tensor_id: {tensor_id}")

    p2p.dereg_mem(tensor_id)
