
KERNEL = $(shell uname -r)
INC = -I ../lib/install/include 
LIBS = ../lib/install/lib/libxdp.a ../lib/install/lib/libbpf.a
FLAGS = -Wno-pointer-arith
DEPS = util.h util_xdp.h util_tcp.h

.PHONY: build
build: client server server1 client_tcp server_tcp client_tcp_ep server_tcp_ep

client: client.cc client_xdp.o $(DEPS)
	g++ -O3 -g client.cc -o client $(INC) $(LIBS) $(FLAGS) -lz -lelf

client_xdp.o: client_xdp.c $(DEPS)
	clang -O3 -g $(INC) -target bpf -c client_xdp.c -o client_xdp.o

server: server.cc server_xdp.o $(DEPS)
	g++ -O3 -g server.cc -o server $(INC) $(LIBS) $(FLAGS) -lz -lelf

server_xdp.o: server_xdp.c $(DEPS)
	clang -O3 -g $(INC) -target bpf -c server_xdp.c -o server_xdp.o

server1: server1.cc server1_xdp.o $(DEPS)
	g++ -O3 -g server1.cc -o server1 $(INC) $(LIBS) $(FLAGS) -lz -lelf

server1_xdp.o: server1_xdp.c $(DEPS)
	clang -O3 -g $(INC) -target bpf -c server1_xdp.c -o server1_xdp.o

client_tcp: client_tcp.cc $(DEPS)
	g++ -O3 -g client_tcp.cc -o client_tcp $(FLAGS) -Wno-write-strings

server_tcp: server_tcp.cc $(DEPS)
	g++ -O3 -g server_tcp.cc -o server_tcp $(FLAGS) -Wno-write-strings

client_tcp_ep: client_tcp_ep.cc $(DEPS)
	g++ -O3 -g client_tcp_ep.cc -o client_tcp_ep $(FLAGS) -Wno-write-strings

server_tcp_ep: server_tcp_ep.cc $(DEPS)
	g++ -O3 -g server_tcp_ep.cc -o server_tcp_ep $(FLAGS) -Wno-write-strings

.PHONY: clean
clean:
	@rm -f client server server1 client_tcp server_tcp client_tcp_ep server_tcp_ep *.o
