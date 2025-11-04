# Epoll Client-Server with Connection Reuse

This directory contains an epoll-based client-server implementation that supports running across multiple nodes with connection reuse capabilities.

## Files

- `epoll_server.h` - Server implementation header with EpollServer class
- `server_main.cpp` - Server main program
- `epoll_client.h` - Client implementation header with EpollClient class (supports connection reuse)
- `client_main.cpp` - Client main program

## Compilation

```bash
# Compile server
g++ -std=c++17 server_main.cpp -o epoll_server -pthread

# Compile client
g++ -std=c++17 client_main.cpp -o epoll_client -pthread
```

## Usage

### Server

```bash
# Start server with default settings (port 9000, 8 threads)
./epoll_server

# Start server on custom port with custom thread pool size
./epoll_server [port] [thread_count]

# Examples:
./epoll_server 12345 16              # Port 12345, 16 worker threads
./epoll_server 9000                  # Port 9000, default 8 threads

# Help
./epoll_server -h
```

The server binds to `0.0.0.0` (all network interfaces), so it can accept connections from any network interface.

### Client

```bash
# Connect to server with default settings (127.0.0.1:9000, send 10 messages)
./epoll_client

# Connect to remote server
./epoll_client [server_ip] [port] [num_messages] [client_id_start]

# Examples:
./epoll_client 192.168.1.100 9000 20 0     # Connect to 192.168.1.100:9000, send 20 messages
./epoll_client 10.0.0.5 12345 100 1000     # Connect to 10.0.0.5:12345, send 100 messages, client IDs 1000-1099
./epoll_client 172.31.0.10 9000            # Connect to 172.31.0.10:9000, send default 10 messages

# Help
./epoll_client -h
```

## Running on Two Nodes

### Setup

**Node 1 (Server):**
1. Find your IP address:
   ```bash
   hostname -I
   # or
   ip addr show
   ```
   Example output: `192.168.1.100`

2. Start the server:
   ```bash
   ./epoll_server 9000 8
   ```

**Node 2 (Client):**
1. Run the client pointing to Node 1's IP:
   ```bash
   ./epoll_client 192.168.1.100 9000 20
   ```

### Firewall Configuration

If you encounter connection issues, make sure the firewall allows connections:

```bash
# On the server node (Ubuntu/Debian)
sudo ufw allow 9000/tcp

# Or for RHEL/CentOS
sudo firewall-cmd --permanent --add-port=9000/tcp
sudo firewall-cmd --reload

# Check if port is listening
netstat -tuln | grep 9000
# or
ss -tuln | grep 9000
```

### AWS/Cloud Considerations

If running on AWS EC2 or other cloud environments:

1. **Security Groups:** Ensure the security group allows inbound TCP traffic on the server's port
2. **Network ACLs:** Check that network ACLs permit the traffic
3. **Use Private IPs:** For better performance and security, use private IPs when both nodes are in the same VPC
4. **EFA Support:** This implementation works over standard TCP/IP. For RDMA support, see the EFA examples in other files

## Key Features

### Server Features
- Epoll-based event loop for high performance
- Edge-triggered mode for efficiency
- Thread pool for parallel message processing
- Non-blocking I/O
- Multiple concurrent client support
- Automatic connection management

### Client Features
- **Connection Reuse:** Single connection for multiple messages
- Epoll-based asynchronous I/O
- Callback support for responses
- Multiple server connection management
- Automatic reconnection handling
- Thread-safe send operations

## Protocol

Messages are framed as:
```
[4 bytes: length in network byte order][payload]
```

The payload is a serialized `MetaInfo` struct:
```cpp
struct MetaInfo {
    int32_t client_id;
    double value;
    char message[64];
};
```

## Performance Tips

1. **Thread Pool Size:** Adjust based on workload and CPU cores
2. **Message Batching:** Send multiple messages on the same connection
3. **Network Tuning:** Consider TCP buffer sizes for high-throughput scenarios
4. **Connection Pooling:** Client automatically reuses connections

## Troubleshooting

### Connection Refused
- Verify server is running: `ps aux | grep epoll_server`
- Check if port is listening: `netstat -tuln | grep [port]`
- Verify IP address is correct
- Check firewall rules

### Connection Timeout
- Verify network connectivity: `ping [server_ip]`
- Check if correct network interface is being used
- Ensure no network ACLs blocking traffic

### Partial Message Loss
- Check server logs for errors
- Verify network stability
- Consider increasing timeout values

## Testing Locally

For local testing on a single machine:

```bash
# Terminal 1 - Server
./epoll_server 9000 8

# Terminal 2 - Client
./epoll_client 127.0.0.1 9000 10

# Or test with multiple clients in parallel
for i in {1..5}; do
    ./epoll_client 127.0.0.1 9000 5 $((i*1000)) &
done
wait
```

## Monitoring

Monitor active connections:
```bash
# On server node
watch -n 1 'netstat -an | grep :9000 | grep ESTABLISHED | wc -l'

# Show all connections
netstat -an | grep :9000
```
