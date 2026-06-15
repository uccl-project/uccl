// Minimal EFA stub header for platforms without the AWS EFA SDK (e.g., Hygon
// DTK). Provides only the type definitions and constants required for
// compilation. EFA transport will be unavailable at runtime on non-EFA
// hardware.
#pragma once
#include <infiniband/verbs.h>
#include <stdint.h>

#define EFADV_QP_DRIVER_TYPE_SRD 0

struct efadv_qp_init_attr {
  uint64_t comp_mask;
  uint8_t driver_qp_type;
  uint8_t sl;
  uint16_t flags;
  uint8_t reserved[4];
};

#ifdef __cplusplus
extern "C" {
#endif

struct ibv_qp* efadv_create_qp_ex(struct ibv_context* ibvctx,
                                  struct ibv_qp_init_attr_ex* attr_ex,
                                  struct efadv_qp_init_attr* efa_attr,
                                  uint32_t inlen);

#ifdef __cplusplus
}
#endif
