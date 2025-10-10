import sys
import resource
import socket
import time
import struct
import pickle
from typing import Any, Tuple, Dict
import torch
import weakref

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


_GLOBAL_TENSOR_IDS: Dict[int, int] = {}


def get_tensor_id_by_tensor(tensor: torch.Tensor):
    if not tensor.is_contiguous():
        raise ValueError("Tensor must be contiguous")
    ptr = tensor.data_ptr()
    if ptr not in _GLOBAL_TENSOR_IDS:
        raise RuntimeError(
            f"Tensor memory not registered for communication. "
            f"Call create_tensor() to create a tensor."
        )
    return _GLOBAL_TENSOR_IDS[ptr]


def _auto_free_tensor(ptr: int, tensor_id: int):
    if ptr not in _GLOBAL_TENSOR_IDS:
        return
    print(f"[Auto Free] ptr={ptr}, tensor_id={tensor_id}")
    try:
        p2p.dereg_mem(tensor_id)
    finally:
        _GLOBAL_TENSOR_IDS.pop(ptr, None)


def create_tensor(
    shape: Tuple[int, ...], dtype: torch.dtype, device: str = "cuda:0"
) -> Tuple[torch.Tensor, int]:
    """
    Create an empty tensor and register GPU memory if applicable.
    Only support GPU.
    Args:
        shape (Tuple[int, ...]): e.g., (2, 100, 4)
        dtype (torch.dtype): e.g., torch.float32
        device (str): 'cuda:0', 'cuda:1'
    Returns:
        Tuple[torch.Tensor, int]: tensor, tensor_id
    """
    if not device.startswith("cuda"):
        raise ValueError(f"Only GPU device is supported, got: {device}")

    try:
        gpu_id = int(device.split(":")[1])
    except (IndexError, ValueError):
        raise ValueError(f"Invalid cuda device: {device}")

    tensor = torch.empty(size=shape, dtype=dtype, device=device)
    addr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()

    if addr in _GLOBAL_TENSOR_IDS:
        raise RuntimeError(f"Tensor at address {addr} is already registered.")

    tensor_id = p2p.reg_mem(gpu_id, addr, size)
    if tensor_id < 0:
        raise RuntimeError(f"Failed to register memory: tensor_id={tensor_id}")

    _GLOBAL_TENSOR_IDS[addr] = tensor_id

    weakref.finalize(tensor, _auto_free_tensor, addr, tensor_id)

    _GLOBAL_TENSOR_IDS[addr] = tensor_id
    return tensor, tensor_id


def free_tensor(tensor: torch.Tensor):
    """
    Note that the GPU memory allocated for a tensor is automatically freed
    when the tensor object is garbage collected. Manual intervention to free
    GPU memory is generally not required unless immediate resource release
    is necessary.
    Args:
        tensor: torch.Tensor
    """
    ptr = tensor.data_ptr()
    if ptr not in _GLOBAL_TENSOR_IDS:
        print(f"Tensor at address {ptr} has been deregistered")
        return

    tensor_id = _GLOBAL_TENSOR_IDS[ptr]
    _auto_free_tensor(ptr, tensor_id)
