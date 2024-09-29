
KERNEL = $(shell uname -r)
INC = -I ../../lib/install/include 
DEP = ../../lib/install/lib/libxdp.a ../../lib/install/lib/libbpf.a

.PHONY: build
build: client server

client: client.c client_xdp.o
	gcc -O2 -g client.c -o client $(INC) $(DEP) -lz -lelf

client_xdp.o: client_xdp.c
	clang -O2 -g $(INC) -target bpf -c client_xdp.c -o client_xdp.o

server: server.c server_xdp.o
	gcc -O2 -g server.c -o server $(INC) $(DEP) -lz -lelf

server_xdp.o: server_xdp.c
	clang -O2 -g $(INC) -target bpf -c server_xdp.c -o server_xdp.o

.PHONY: clean
clean:
	rm -f client server *.o
