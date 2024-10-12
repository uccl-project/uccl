KERNEL = $(shell uname -r)
INC = -I ../lib/install/include 
LIBS = ../lib/install/lib/libxdp.a ../lib/install/lib/libbpf.a -lglog -lgflags
CXXFLAGS = -O3 -g -std=c++17 -Wno-pointer-arith
CLANGFLAGS = -O3 -g
DEPS = *.h

lib_src = $(wildcard *.cc)
lib_src := $(filter-out %_main.cc,$(lib_src))
lib_src := $(filter-out %_test.cc,$(lib_src))
lib_obj = $(lib_src:.cc=.o)

main_src = $(wildcard *_main.cc)
main_src += $(wildcard *_test.cc)
main_bin = $(main_src:.cc=)

ebpf_obj = ebpf_client.o ebpf_server.o ebpf_server_direct.o ebpf_transport.o ebpf_transport_pktloss.o

.PHONY: build
build: $(main_bin) $(ebpf_obj)

ebpf_client.o: ebpf_client.c $(DEPS)
	clang $(INC) -target bpf -c ebpf_client.c -o ebpf_client.o $(CLANGFLAGS)

ebpf_server.o: ebpf_server.c $(DEPS)
	clang $(INC) -target bpf -c ebpf_server.c -o ebpf_server.o $(CLANGFLAGS)

ebpf_server_direct.o: ebpf_server_direct.c $(DEPS)
	clang $(INC) -target bpf -c ebpf_server_direct.c -o ebpf_server_direct.o $(CLANGFLAGS)

ebpf_transport.o: ebpf_transport.c $(DEPS)
	clang $(INC) -target bpf -c ebpf_transport.c -o ebpf_transport.o $(CLANGFLAGS)

ebpf_transport_pktloss.o: ebpf_transport_pktloss.c $(DEPS)
	clang $(INC) -target bpf -c ebpf_transport_pktloss.c -o ebpf_transport_pktloss.o $(CLANGFLAGS)

client_tcp_main: client_tcp_main.cc $(DEPS)
	g++ client_tcp_main.cc -o client_tcp_main $(INC) $(LIBS) $(CXXFLAGS) -Wno-write-strings

server_tcp_main: server_tcp_main.cc $(DEPS) 
	g++ server_tcp_main.cc -o server_tcp_main $(INC) $(LIBS) $(CXXFLAGS) -Wno-write-strings

client_tcp_ep_main: client_tcp_ep_main.cc $(DEPS)
	g++ client_tcp_ep_main.cc -o client_tcp_ep_main $(INC) $(LIBS) $(CXXFLAGS) -Wno-write-strings

server_tcp_ep_main: server_tcp_ep_main.cc $(DEPS)
	g++ server_tcp_ep_main.cc -o server_tcp_ep_main $(INC) $(LIBS) $(CXXFLAGS) -Wno-write-strings

client_main: client_main.cc ebpf_client.o $(DEPS) $(lib_obj)
	g++ client_main.cc -o client_main $(lib_obj) $(INC) $(LIBS) $(CXXFLAGS) -lz -lelf

server_main: server_main.cc ebpf_server.o $(DEPS) $(lib_obj)
	g++ server_main.cc -o server_main $(lib_obj) $(INC) $(LIBS) $(CXXFLAGS) -lz -lelf

server_direct_main: server_direct_main.cc ebpf_server_direct.o $(DEPS) $(lib_obj)
	g++ server_direct_main.cc -o server_direct_main $(lib_obj) $(INC) $(LIBS) $(CXXFLAGS) -lz -lelf

transport_test: transport_test.cc ebpf_client.o $(DEPS) $(lib_obj)
	g++ transport_test.cc -o transport_test $(lib_obj) $(INC) $(LIBS) $(CXXFLAGS) -lz -lelf

util_afxdp_test: util_afxdp_test.cc ebpf_client.o $(DEPS) $(lib_obj)
	g++ util_afxdp_test.cc -o util_afxdp_test $(lib_obj) $(INC) $(LIBS) $(CXXFLAGS) -lz -lelf

%.o: %.cc $(DEPS)
	g++ -c $< -o $@ $(INC) $(CXXFLAGS)

.PHONY: clean
clean:
	rm -f *.o $(main_bin)
