# mk/arch_defaults.mk - GPU-arch detection + per-arch reduce-kernel defaults.
#
# Shared by the top-level, src/ccl, and src/device Makefiles so the arch
# list and the reduce-kernel knobs (REDUCE_ILP / TMA_REDUCE /
# REDUCE_SMEM_KB) come from one table.
#
# Rules:
#   * SM=<n> given        -> build exactly that arch; derive defaults from it.
#   * SM unset            -> detect the GPU(s) present via nvidia-smi and
#     build only those archs (the old fixed 4-arch list missed sm_103).
#     No GPU / no nvidia-smi -> fall back to DEFAULT_SMS.
#   * Per-arch defaults: sm_100/103 -> REDUCE_ILP=16, sm_90 -> 8, <90 -> 4
#     on the vector path. TMA bulk reduce + 224KB smem default on for
#     sm_90+ perf builds, which switch ILP back to 4 (the bulk chunks are
#     TMA-driven, ILP only feeds the tail/vector fallback — ILP=16+TMA
#     just quadruples cicc/ptxas time for no large-task gain).
#     VALIDATE=1 forces the fast build (no TMA, 4KB smem, arch ILP).
#   * Explicit REDUCE_ILP / TMA_REDUCE / REDUCE_SMEM_KB (command line or
#     env) win. The legacy ENABLE_TMA knob is honored as an alias for
#     TMA_REDUCE so old `ENABLE_TMA=0` commands still disable TMA.

DEFAULT_SMS := 80 86 89 100

ifeq ($(origin SM), undefined)
  # Auto-detect compute capabilities of the GPUs present (8.6 -> 86).
  _SMI_LIST := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed 's/\.//' | sort -u)
  ifneq ($(strip $(_SMI_LIST)),)
    SM_LIST := $(_SMI_LIST)
  else
    SM_LIST := $(DEFAULT_SMS)
  endif
  NVCC_ARCH_FLAGS := $(foreach sm,$(SM_LIST),-gencode arch=compute_$(sm),code=sm_$(sm))
  SUBMAKE_SM :=
else
  SM_NUM := $(shell printf '%s\n' "$(SM)" | sed 's/[^0-9].*//')
  SM_LIST := $(SM_NUM)
  NVCC_ARCH_FLAGS := -arch=sm_$(SM_NUM)
  SUBMAKE_SM := SM=$(SM)
endif

# --- Per-arch reduce-kernel defaults -----------------------------------
# <90 (Ampere sm_80/86, Ada sm_89): ILP=4, no TMA. sm_90 (Hopper): ILP=8.
# sm_100/103 (Blackwell): ILP=16 (measured +24%/block over ILP=4 on B300
# for the vector path). TMA bulk reduce needs sm_90+ hardware and is only
# defaulted on for non-VALIDATE builds; TMA builds drop ILP to 4 (the
# bulk chunks are cp.async.bulk-driven, so ILP only pays for the tail).
_GE90 := $(filter-out 80 86 89,$(SM_LIST))
ifeq ($(strip $(_GE90)),)
  _AUTO_ILP := 4
  _AUTO_TMA := 0
  _AUTO_SMEM_KB := 4
else ifeq ($(words $(SM_LIST)),1)
  ifeq ($(filter-out 90,$(SM_LIST)),)
    _AUTO_ILP := 8
  else
    _AUTO_ILP := 16
  endif
  _AUTO_TMA := 1
  _AUTO_SMEM_KB := 224
else
  # Mixed-arch single binary: stay conservative (higher ILP spills on the
  # <90 passes; TMA only helps the >=90 passes but costs compile time).
  _AUTO_ILP := 4
  _AUTO_TMA := 0
  _AUTO_SMEM_KB := 4
endif

ifeq ($(VALIDATE),1)
  _AUTO_TMA := 0
  _AUTO_SMEM_KB := 4
else
  # TMA does the bulk work in perf builds, so ILP=4 keeps the sm_103
  # build in the 15-25min window instead of ~1h (ILP=16+TMA quadruples
  # cicc/ptxas time for no large-task gain). VALIDATE keeps the arch ILP
  # because it exercises the vector path that ships without TMA.
  ifneq ($(strip $(_GE90)),)
    ifeq ($(words $(SM_LIST)),1)
      _AUTO_ILP := 4
    endif
  endif
endif

REDUCE_ILP ?= $(_AUTO_ILP)
ifeq ($(origin TMA_REDUCE), undefined)
  ifneq ($(origin ENABLE_TMA), undefined)
    TMA_REDUCE := $(ENABLE_TMA)
  endif
endif
TMA_REDUCE ?= $(_AUTO_TMA)
REDUCE_SMEM_KB ?= $(_AUTO_SMEM_KB)
TMA_WARPSPEC ?= 0

# Rebuild stamp: make tracks timestamps, not compile flags, so switching
# VALIDATE <-> perf or ILP/TMA/SMEM values used to silently reuse stale
# objects. Each Makefile refreshes this file (with the final NVCCFLAGS +
# CXXFLAGS) after its flag lines; every object rule depends on it, so a
# flag change rebuilds the affected objects.
BUILD_FLAGS_STAMP := .uk_build_flags
