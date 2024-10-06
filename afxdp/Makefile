
KERNEL = $(shell uname -r)
INC = -I ../lib/install/include 
LIBS = ../lib/install/lib/libxdp.a ../lib/install/lib/libbpf.a -lglog
FLAGS = -std=c++17 -Wno-pointer-arith
DEPS = util_*.h transport_*.h
LIB_OBJ = transport_umem.o

.PHONY: build
build: client server server1 client_tcp server_tcp client_tcp_ep server_tcp_ep

client_xdp.o: client_xdp.c $(DEPS)
	clang -O3 -g $(INC) -target bpf -c client_xdp.c -o client_xdp.o

server_xdp.o: server_xdp.c $(DEPS)
	clang -O3 -g $(INC) -target bpf -c server_xdp.c -o server_xdp.o

server1_xdp.o: server1_xdp.c $(DEPS)
	clang -O3 -g $(INC) -target bpf -c server1_xdp.c -o server1_xdp.o

client: client.cc client_xdp.o $(DEPS) $(LIB_OBJ)
	g++ -O3 -g client.cc -o client $(LIB_OBJ) $(INC) $(LIBS) $(FLAGS) -lz -lelf

server: server.cc server_xdp.o $(DEPS) $(LIB_OBJ)
	g++ -O3 -g server.cc -o server $(LIB_OBJ) $(INC) $(LIBS) $(FLAGS) -lz -lelf

server1: server1.cc server1_xdp.o $(DEPS) $(LIB_OBJ)
	g++ -O3 -g server1.cc -o server1 $(LIB_OBJ) $(INC) $(LIBS) $(FLAGS) -lz -lelf

client_tcp: client_tcp.cc $(DEPS) $(LIB_OBJ)
	g++ -O3 -g client_tcp.cc -o client_tcp $(LIB_OBJ) $(INC) $(LIBS) $(FLAGS) -Wno-write-strings

server_tcp: server_tcp.cc $(DEPS) $(LIB_OBJ)
	g++ -O3 -g server_tcp.cc -o server_tcp $(LIB_OBJ) $(INC) $(LIBS) $(FLAGS) -Wno-write-strings

client_tcp_ep: client_tcp_ep.cc $(DEPS) $(LIB_OBJ)
	g++ -O3 -g client_tcp_ep.cc -o client_tcp_ep $(LIB_OBJ) $(INC) $(LIBS) $(FLAGS) -Wno-write-strings

server_tcp_ep: server_tcp_ep.cc $(DEPS) $(LIB_OBJ)
	g++ -O3 -g server_tcp_ep.cc -o server_tcp_ep $(LIB_OBJ) $(INC) $(LIBS) $(FLAGS) -Wno-write-strings

%.o: %.c
	g++ $(FLAGS) -c $< -o $@

.PHONY: clean
clean:
	@rm -f client server server1 client_tcp server_tcp client_tcp_ep server_tcp_ep *.o
