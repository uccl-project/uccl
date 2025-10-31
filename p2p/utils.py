import sys
import resource
import socket
import time
import struct
import pickle
from typing import Any
from intervaltree import Interval, IntervalTree


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


class ClosedIntervalTree:
    def __init__(self):
        self.tree = IntervalTree()

    def add(self, start, end, data):
        if end < start:
            raise ValueError(f"Invalid closed interval: end ({end}) < start ({start})")
        self.tree.add(Interval(start, end + 1, data))

    def remove(self, start, end, data=None):
        interval_start = start
        interval_end = end + 1

        intervals_to_remove = []
        for interval in self.tree[interval_start:interval_end]:
            if (
                interval.begin == interval_start
                and interval.end == interval_end
                and (data is None or interval.data == data)
            ):
                intervals_to_remove.append(interval)

        for interval in intervals_to_remove:
            self.tree.remove(interval)

        return len(intervals_to_remove)

    def query_containing(self, query_start, query_end):
        query_interval_start = query_start
        query_interval_end = query_end + 1

        results = []
        for interval in self.tree[query_interval_start:query_interval_end]:
            if (
                interval.begin <= query_interval_start
                and interval.end >= query_interval_end
            ):
                closed_interval = (interval.begin, interval.end - 1, interval.data)
                results.append(closed_interval)

        return results

    def query_overlap(self, query_start, query_end):
        query_interval_start = query_start
        query_interval_end = query_end + 1

        results = []
        for interval in self.tree[query_interval_start:query_interval_end]:
            closed_interval = (interval.begin, interval.end - 1, interval.data)
            results.append(closed_interval)

        return results

    def query_exact_match(self, query_start, query_end, data=None):
        """
        Find intervals that exactly match the given [query_start, query_end]

        Args:
            query_start: Start of the query interval
            query_end: End of the query interval
            data: Optional data to match (if None, matches any data)

        Returns:
            List of matching intervals as (start, end, data) tuples
        """
        # Convert closed interval to half-open interval for internal representation
        interval_start = query_start
        interval_end = query_end + 1

        results = []
        for interval in self.tree[interval_start:interval_end]:
            # Check for exact boundary match
            if interval.begin == interval_start and interval.end == interval_end:
                # Check data match if specified
                if data is None or interval.data == data:
                    closed_interval = (interval.begin, interval.end - 1, interval.data)
                    results.append(closed_interval)

        return results

    def __iter__(self):
        for interval in sorted(self.tree):
            yield (interval.begin, interval.end - 1, interval.data)

    def __str__(self):
        lines = []
        for start, end, data in self:
            lines.append(f"[{start}, {end}] -> {data}")
        return "\n".join(lines)

    def clear(self):
        """Remove all intervals from the tree"""
        self.tree.clear()
