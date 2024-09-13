# SPDX-License-Identifier: (GPL-2.0 OR BSD-2-Clause)

XDP_TARGETS  := af_xdp_kern
USER_TARGETS := af_xdp_user
LDLIBS += -lpthread

COMMON_DIR := ../common

include $(COMMON_DIR)/common.mk
COMMON_OBJS := $(COMMON_DIR)/common_params.o
COMMON_OBJS += $(COMMON_DIR)/common_user_bpf_xdp.o

fi_pingpong: fi_pingpong.c
	gcc fi_pingpong.c -o fi_pingpong -I /opt/amazon/efa/include -L /opt/amazon/efa/lib -lfabric

fi_udpgen: fi_udpgen.c
	gcc fi_udpgen.c -o fi_udpgen -I /opt/amazon/efa/include -L /opt/amazon/efa/lib -lfabric

clean_fi:
	rm fi_pingpong fi_udpgen