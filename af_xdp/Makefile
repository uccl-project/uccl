
KERNEL = $(shell uname -r)
INC = -I ../lib/install/include 
DEP = ../lib/install/lib/libxdp.a ../lib/install/lib/libbpf.a
FLAGS = -Wno-pointer-arith

.PHONY: build
build: client server

client: client.cc client_xdp.o
	g++ -O3 -g client.cc -o client $(INC) $(DEP) $(FLAGS) -lz -lelf

client_xdp.o: client_xdp.c
	clang -O3 -g $(INC) -target bpf -c client_xdp.c -o client_xdp.o

server: server.cc server_xdp.o
	g++ -O3 -g server.cc -o server $(INC) $(DEP) $(FLAGS) -lz -lelf

server_xdp.o: server_xdp.c
	clang -O3 -g $(INC) -target bpf -c server_xdp.c -o server_xdp.o

.PHONY: clean
clean:
	rm -f client server *.o
