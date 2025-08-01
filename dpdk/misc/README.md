# AF_XDP MISC

### Run TCP testing

```bash
cd afxdp; make -j "CXXFLAGS=-DAWS_ENA"
or 
cd afxdp; make -j "CXXFLAGS=-DCLAB_MLX5"

# On both server and client
./setup_nic.sh ens6 4 4 9001 tcp aws
or
./setup_nic.sh ens1f1np1 4 4 1500 tcp clab

# On server, edit nodes.txt to include all node ips
python rsync.sh
./server_tcp_main

# On client
./client_tcp_main -a 192.168.6.1
```

### Run AFXDP testing

```bash
cd afxdp; make -j "CXXFLAGS=-DAWS_ENA"
or 
cd afxdp; make -j "CXXFLAGS=-DCLAB_MLX5"

# On both server and client
./setup_nic.sh ens6 1 1 3498 afxdp aws
or
./setup_nic.sh ens1f1np1 1 1 1500 afxdp clab

# On server, edit nodes.txt to include all node ips
python rsync.sh
sudo ./server_main

# On client
sudo ./client_main
```
