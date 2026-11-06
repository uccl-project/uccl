MUSA_HOME ?= /usr/local/musa
MCCL_HOME ?= /usr
MPICXX ?= mpicxx

.PHONY: probe
probe: mccl_p2p_probe

mccl_p2p_probe: mccl_p2p_probe.cc
	$(MPICXX) -std=c++17 -O2 -g -Wall -Wextra \
		-I$(MUSA_HOME)/include -I$(MCCL_HOME)/include $< \
		-L$(MUSA_HOME)/lib -L$(MCCL_HOME)/lib \
		-Wl,-rpath,$(MUSA_HOME)/lib -Wl,-rpath,$(MCCL_HOME)/lib \
		-lmccl -lmusart -o $@
