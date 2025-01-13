ifeq ("$(origin V)", "command line")
VERBOSE = $(V)
endif
ifndef VERBOSE
VERBOSE = 0
endif

ifeq ($(VERBOSE),0)
MAKEFLAGS += --no-print-directory
Q = @
endif

# LESSONS = $(wildcard afxdp*) $(wildcard aws*)
LESSONS = $(wildcard aws*)
LESSONS_CLEAN = $(addsuffix _clean,$(LESSONS))

.PHONY: clean clobber distclean $(LESSONS) $(LESSONS_CLEAN)

all: lib $(LESSONS)

clean: $(LESSONS_CLEAN)
	@echo; echo lib/xdp-tools; $(MAKE) -C lib/xdp-tools clean
	@echo; echo lib; $(MAKE) -C lib clean
	@echo; echo common; $(MAKE) -C common clean

lib: check_submodule config
	@echo; echo lib; $(MAKE) -C lib
	@echo; echo lib/xdp-tools; $(MAKE) -C lib/xdp-tools

check_submodule:
	@git submodule update --init

config: configure
	@sh configure
	@sh cd lib/xdp-tools; ./configure

$(LESSONS):
	@echo; echo $@; $(MAKE) -C $@

$(LESSONS_CLEAN):
	@echo; echo $@; $(MAKE) -C $(subst _clean,,$@) clean

clobber:
	@touch config.mk
	$(Q)$(MAKE) clean
	$(Q)rm -f config.mk

distclean:	clobber

