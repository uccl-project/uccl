
KERNEL = $(shell uname -r)
INC = -I ../lib/install/include 
LIBS = ../lib/install/lib/libxdp.a ../lib/install/lib/libbpf.a -lglog
CXXFLAGS = -O3 -g -std=c++17 -Wno-pointer-arith
CLANGFLAGS = -O3 -g
DEPS = util*.h transport*.h
LIB_OBJ = util_umem.o util_afxdp.o

.PHONY: build
build: client server server1 client_tcp server_tcp client_tcp_ep server_tcp_ep

client_xdp.o: client_xdp.c $(DEPS)
	clang $(INC) -target bpf -c client_xdp.c -o client_xdp.o $(CLANGFLAGS)

server_xdp.o: server_xdp.c $(DEPS)
	clang $(INC) -target bpf -c server_xdp.c -o server_xdp.o $(CLANGFLAGS)

server1_xdp.o: server1_xdp.c $(DEPS)
	clang $(INC) -target bpf -c server1_xdp.c -o server1_xdp.o $(CLANGFLAGS)

client_tcp: client_tcp.cc $(DEPS)
	g++ client_tcp.cc -o client_tcp $(INC) $(LIBS) $(CXXFLAGS) -Wno-write-strings

server_tcp: server_tcp.cc $(DEPS) 
	g++ server_tcp.cc -o server_tcp $(INC) $(LIBS) $(CXXFLAGS) -Wno-write-strings

client_tcp_ep: client_tcp_ep.cc $(DEPS)
	g++ client_tcp_ep.cc -o client_tcp_ep $(INC) $(LIBS) $(CXXFLAGS) -Wno-write-strings

server_tcp_ep: server_tcp_ep.cc $(DEPS)
	g++ server_tcp_ep.cc -o server_tcp_ep $(INC) $(LIBS) $(CXXFLAGS) -Wno-write-strings

client: client.cc client_xdp.o $(DEPS) $(LIB_OBJ)
	g++ client.cc -o client $(LIB_OBJ) $(INC) $(LIBS) $(CXXFLAGS) -lz -lelf

server: server.cc server_xdp.o $(DEPS) $(LIB_OBJ)
	g++ server.cc -o server $(LIB_OBJ) $(INC) $(LIBS) $(CXXFLAGS) -lz -lelf

server1: server1.cc server1_xdp.o $(DEPS) $(LIB_OBJ)
	g++ server1.cc -o server1 $(LIB_OBJ) $(INC) $(LIBS) $(CXXFLAGS) -lz -lelf

%.o: %.cc
	g++ -c $< -o $@ $(INC) $(CXXFLAGS)

.PHONY: clean
clean:
	@rm -f client server server1 client_tcp server_tcp client_tcp_ep server_tcp_ep *.o
