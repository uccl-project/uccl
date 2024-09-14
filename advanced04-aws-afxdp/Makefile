# SPDX-License-Identifier: (GPL-2.0 OR BSD-2-Clause)

XDP_TARGETS  := af_xdp_kern
USER_TARGETS := af_xdp_user
FI_FLAGS = -I /opt/amazon/efa/include -L /opt/amazon/efa/lib -lfabric
LDLIBS += -lpthread $(FI_FLAGS)

COMMON_DIR := ../common

include $(COMMON_DIR)/common.mk
COMMON_OBJS := $(COMMON_DIR)/common_params.o
COMMON_OBJS += $(COMMON_DIR)/common_user_bpf_xdp.o

fi: fi_pingpong fi_info fi_udpgen fi_efagen

fi_pingpong: fi_pingpong.c
	gcc -O3 fi_pingpong.c -o fi_pingpong $(FI_FLAGS)

fi_info: fi_info.c
	gcc -O3 fi_info.c -o fi_info $(FI_FLAGS)

fi_udpgen: fi_udpgen.c
	gcc -O3 fi_udpgen.c -o fi_udpgen $(FI_FLAGS)

fi_efagen: fi_efagen.c
	gcc -O3 fi_efagen.c -o fi_efagen $(FI_FLAGS)

fi_clean:
	rm fi_pingpong fi_info fi_udpgen fi_efagen