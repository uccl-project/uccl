
KERNEL = $(shell uname -r)
INC = -I ../lib/install/include 
LIBS = ../lib/install/lib/libxdp.a ../lib/install/lib/libbpf.a -lglog
CXXFLAGS = -O3 -g -std=c++17 -Wno-pointer-arith
CLANGFLAGS = -O3 -g
DEPS = util*.h transport*.h
LIB_OBJ = util_umem.o util_afxdp.o

.PHONY: build
build: client server server_direct client_tcp server_tcp client_tcp_ep server_tcp_ep \
	transport_test util_afxdp_test

ebpf_client.o: ebpf_client.c $(DEPS)
	clang $(INC) -target bpf -c ebpf_client.c -o ebpf_client.o $(CLANGFLAGS)

ebpf_server.o: ebpf_server.c $(DEPS)
	clang $(INC) -target bpf -c ebpf_server.c -o ebpf_server.o $(CLANGFLAGS)

ebpf_server_direct.o: ebpf_server_direct.c $(DEPS)
	clang $(INC) -target bpf -c ebpf_server_direct.c -o ebpf_server_direct.o $(CLANGFLAGS)

client_tcp: client_tcp.cc $(DEPS)
	g++ client_tcp.cc -o client_tcp $(INC) $(LIBS) $(CXXFLAGS) -Wno-write-strings

server_tcp: server_tcp.cc $(DEPS) 
	g++ server_tcp.cc -o server_tcp $(INC) $(LIBS) $(CXXFLAGS) -Wno-write-strings

client_tcp_ep: client_tcp_ep.cc $(DEPS)
	g++ client_tcp_ep.cc -o client_tcp_ep $(INC) $(LIBS) $(CXXFLAGS) -Wno-write-strings

server_tcp_ep: server_tcp_ep.cc $(DEPS)
	g++ server_tcp_ep.cc -o server_tcp_ep $(INC) $(LIBS) $(CXXFLAGS) -Wno-write-strings

client: client.cc ebpf_client.o $(DEPS) $(LIB_OBJ)
	g++ client.cc -o client $(LIB_OBJ) $(INC) $(LIBS) $(CXXFLAGS) -lz -lelf

server: server.cc ebpf_server.o $(DEPS) $(LIB_OBJ)
	g++ server.cc -o server $(LIB_OBJ) $(INC) $(LIBS) $(CXXFLAGS) -lz -lelf

server_direct: server_direct.cc ebpf_server_direct.o $(DEPS) $(LIB_OBJ)
	g++ server_direct.cc -o server_direct $(LIB_OBJ) $(INC) $(LIBS) $(CXXFLAGS) -lz -lelf

transport_test: transport_test.cc ebpf_client.o $(DEPS) $(LIB_OBJ)
	g++ transport_test.cc -o transport_test $(LIB_OBJ) $(INC) $(LIBS) $(CXXFLAGS) -lz -lelf

util_afxdp_test: util_afxdp_test.cc ebpf_client.o $(DEPS) $(LIB_OBJ)
	g++ util_afxdp_test.cc -o util_afxdp_test $(LIB_OBJ) $(INC) $(LIBS) $(CXXFLAGS) -lz -lelf

%.o: %.cc $(DEPS)
	g++ -c $< -o $@ $(INC) $(CXXFLAGS)

.PHONY: clean
clean:
	@rm -f *.o client server server_direct client_tcp server_tcp client_tcp_ep server_tcp_ep \
		transport_test util_afxdp_test
