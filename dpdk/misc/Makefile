ABS_ROOT_PATH := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

KERNEL = $(shell uname -r)
INC = -I ../../lib/install/include -I ../
LIBS = ../../lib/install/lib/libxdp.a ../../lib/install/lib/libbpf.a -lglog -lgflags -lgtest -lz -lelf
LIBS_SHARED = -Wl,-rpath=$(ABS_ROOT_PATH)/../../lib/xdp-tools/lib/libxdp/ \
	   		  -Wl,-rpath=$(ABS_ROOT_PATH)/../../lib/libbpf/src/ \
	   		  -L../../lib/xdp-tools/lib/libxdp -lxdp -L../../lib/libbpf/src -lbpf \
	   		  -lglog -lgflags -lgtest -lz -lelf
override CXXFLAGS += -O3 -g -std=c++17 -Wno-pointer-arith -Wno-interference-size -fPIC
CLANGFLAGS = -O3 -g
DEPS = *.h

lib_src = $(wildcard *.cc)
lib_src := $(filter-out %_main.cc,$(lib_src))
lib_src := $(filter-out %_test.cc,$(lib_src))
lib_src := $(filter-out %_plugin.cc,$(lib_src))
lib_obj = $(lib_src:.cc=.o)

main_src = $(wildcard *_main.cc)
main_src += $(wildcard *_test.cc)
main_bin = $(main_src:.cc=)

ebpf_obj = ebpf_client.o ebpf_server.o ebpf_server_direct.o

.PHONY: build
build: $(main_bin) $(ebpf_obj)

ebpf_client.o: ebpf_client.c $(DEPS)
	clang $(INC) -target bpf -c ebpf_client.c -o ebpf_client.o $(CLANGFLAGS)

ebpf_server.o: ebpf_server.c $(DEPS)
	clang $(INC) -target bpf -c ebpf_server.c -o ebpf_server.o $(CLANGFLAGS)

ebpf_server_direct.o: ebpf_server_direct.c $(DEPS)
	clang $(INC) -target bpf -c ebpf_server_direct.c -o ebpf_server_direct.o $(CLANGFLAGS)

client_tcp_main: client_tcp_main.cc $(DEPS)
	g++ client_tcp_main.cc -o client_tcp_main $(INC) $(LIBS) $(CXXFLAGS) -Wno-write-strings

server_tcp_main: server_tcp_main.cc $(DEPS) 
	g++ server_tcp_main.cc -o server_tcp_main $(INC) $(LIBS) $(CXXFLAGS) -Wno-write-strings

client_tcp_ep_main: client_tcp_ep_main.cc $(DEPS)
	g++ client_tcp_ep_main.cc -o client_tcp_ep_main $(INC) $(LIBS) $(CXXFLAGS) -Wno-write-strings

server_tcp_ep_main: server_tcp_ep_main.cc $(DEPS)
	g++ server_tcp_ep_main.cc -o server_tcp_ep_main $(INC) $(LIBS) $(CXXFLAGS) -Wno-write-strings

tcp_latency_main: tcp_latency_main.cc $(DEPS)
	g++ tcp_latency_main.cc -o tcp_latency_main $(INC) $(LIBS) $(CXXFLAGS) -Wno-write-strings

client_main: client_main.cc ebpf_client.o $(DEPS) $(lib_obj)
	g++ client_main.cc -o client_main $(lib_obj) $(INC) $(LIBS) $(CXXFLAGS)

server_main: server_main.cc ebpf_server.o $(DEPS) $(lib_obj)
	g++ server_main.cc -o server_main $(lib_obj) $(INC) $(LIBS) $(CXXFLAGS)

server_direct_main: server_direct_main.cc ebpf_server_direct.o $(DEPS) $(lib_obj)
	g++ server_direct_main.cc -o server_direct_main $(lib_obj) $(INC) $(LIBS) $(CXXFLAGS)

afxdp_setup_main: afxdp_setup_main.cc $(DEPS) $(lib_obj)
	g++ afxdp_setup_main.cc -o afxdp_setup_main $(lib_obj) $(INC) $(LIBS) $(CXXFLAGS)

afxdp_use_main: afxdp_use_main.cc $(DEPS) $(lib_obj)
	g++ afxdp_use_main.cc -o afxdp_use_main $(lib_obj) $(INC) $(LIBS) $(CXXFLAGS)

membw_main: membw_main.cc
	g++ membw_main.cc -o membw_main $(CXXFLAGS)

%.o: %.cc $(DEPS)
	g++ -c $< -o $@ $(INC) $(CXXFLAGS)

.PHONY: clean
clean:
	rm -f *.o $(main_bin) $(PLUGIN_SO)
