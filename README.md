# Unified CCL

### Run TCP testing

```
# On both server and client
./run.sh ens6 1
sudo ifconfig ens6 mtu 9000

# On server
./server_tcp_ep

# On client
./client_tcp_ep -a 172.31.22.249
```