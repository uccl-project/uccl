# SPDX-License-Identifier: (GPL-2.0 OR BSD-2-Clause)

XDP_TARGETS  := af_xdp_kern
USER_TARGETS := af_xdp_user
LDLIBS += -lpthread

COMMON_DIR := ../common

include $(COMMON_DIR)/common.mk
COMMON_OBJS := $(COMMON_DIR)/common_params.o
COMMON_OBJS += $(COMMON_DIR)/common_user_bpf_xdp.o

dgram_gen: dgram_gen.c
	gcc dgram_gen.c -o dgram_gen -I /opt/amazon/efa/include -L /opt/amazon/efa/lib -lfabric

clean_dgram_gen:
	rm dgram_gen