#ifndef RDMA_UTIL_HPP
#define RDMA_UTIL_HPP

#include "rdma.hpp"
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_set>
#include <vector>
#include <fcntl.h>
#include <limits.h>
#include <sys/socket.h>
#include <unistd.h>

void fill_local_gid(ProxyCtx& S, RDMAConnectionInfo* local_info) {
  if (!S.context) {
    fprintf(stderr, "Error: context not initialized when filling GID\n");
    exit(1);
  }

  // Query port attributes to determine if this is RoCE (Ethernet) or InfiniBand
  struct ibv_port_attr port_attr;
  if (ibv_query_port(S.context, 1, &port_attr)) {
    perror("Failed to query port for GID");
    exit(1);
  }

  // For RoCE (Ethernet), we need to fill the GID
  if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET ||
      port_attr.link_layer == IBV_LINK_LAYER_UNSPECIFIED) {
    union ibv_gid local_gid;
    int gid_index = S.gid_index;
    // EFA:
    if (port_attr.link_layer == IBV_LINK_LAYER_UNSPECIFIED) gid_index = 0;

    if (ibv_query_gid(S.context, 1, gid_index, &local_gid)) {
      perror("Failed to query GID");
      exit(1);
    }

    // Copy the GID to the connection info
    memcpy(local_info->gid, &local_gid, 16);
    // printf(
    //     "[RDMA] Local GID filled for RoCE (Ethernet) connection: "
    //     "%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%"
    //     "02x\n",
    //     local_info->gid[0], local_info->gid[1], local_info->gid[2],
    //     local_info->gid[3], local_info->gid[4], local_info->gid[5],
    //     local_info->gid[6], local_info->gid[7], local_info->gid[8],
    //     local_info->gid[9], local_info->gid[10], local_info->gid[11],
    //     local_info->gid[12], local_info->gid[13], local_info->gid[14],
    //     local_info->gid[15]);
  } else {
    // For InfiniBand, GID is not strictly required, but we can still fill it
    union ibv_gid local_gid;
    if (ibv_query_gid(S.context, 1, 0, &local_gid) == 0) {
      memcpy(local_info->gid, &local_gid, 16);
      // printf(
      //     "[RDMA] Local GID filled for InfiniBand connection: "
      //     "%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%"
      //     "02x\n",
      //     local_info->gid[0], local_info->gid[1], local_info->gid[2],
      //     local_info->gid[3], local_info->gid[4], local_info->gid[5],
      //     local_info->gid[6], local_info->gid[7], local_info->gid[8],
      //     local_info->gid[9], local_info->gid[10], local_info->gid[11],
      //     local_info->gid[12], local_info->gid[13], local_info->gid[14],
      //     local_info->gid[15]);
    } else {
      // If GID query fails for InfiniBand, zero it out
      memset(local_info->gid, 0, 16);
      // printf(
      //     "[RDMA] GID zeroed for InfiniBand connection (GID query
      //     failed)\n");
    }
  }
}

// Helper functions for ncclIbGetGidIndex
static inline int get_gid_index_from_env() {
  static int gid_index = -1;
  char const* env = getenv("NCCL_IB_GID_INDEX");
  if (env) gid_index = atoi(env);
  return gid_index;
}

static inline int get_routable_flid_gid_index_from_env() {
  static int routable_flid_gid_index = 1;
  char const* env = getenv("NCCL_IB_ROUTABLE_FLID_GID_INDEX");
  if (env) routable_flid_gid_index = atoi(env);
  return routable_flid_gid_index;
}

static inline int get_roce_version_from_env() {
  static int roce_version = 2;
  char const* env = getenv("NCCL_IB_ROCE_VERSION_NUM");
  if (env) roce_version = atoi(env);
  return roce_version;
}

static sa_family_t envIbAddrFamily(void) {
  sa_family_t family = AF_INET;
  char const* env = getenv("NCCL_IB_ADDR_FAMILY");
  if (env == NULL || strlen(env) == 0) {
    return family;
  }

  if (strcmp(env, "AF_INET") == 0) {
    family = AF_INET;
  } else if (strcmp(env, "AF_INET6") == 0) {
    family = AF_INET6;
  }

  return family;
}

static void* envIbAddrRange(sa_family_t af, int* mask) {
  *mask = 0;
  static struct in_addr addr;
  static struct in6_addr addr6;
  void* ret = (af == AF_INET) ? (void*)&addr : (void*)&addr6;

  char const* env = getenv("NCCL_IB_ADDR_RANGE");
  if (NULL == env || strlen(env) == 0) {
    return NULL;
  }

  char addrString[128] = {0};
  snprintf(addrString, 128, "%s", env);
  char* addrStrPtr = addrString;
  char* maskStrPtr = strstr(addrString, "/");
  if (NULL == maskStrPtr) {
    return NULL;
  }
  *(maskStrPtr++) = '\0';

  if (inet_pton(af, addrStrPtr, ret) == 0) {
    return NULL;
  }

  *mask = (int)strtol(maskStrPtr, NULL, 10);
  if (af == AF_INET && *mask > 32) {
    *mask = 0;
    ret = NULL;
  } else if (af == AF_INET6 && *mask > 128) {
    *mask = 0;
    ret = NULL;
  }

  return ret;
}

static sa_family_t getGidAddrFamily(union ibv_gid* gid) {
  const struct in6_addr* a = (struct in6_addr*)gid->raw;
  bool isIpV4Mapped = ((a->s6_addr32[0] | a->s6_addr32[1]) |
                       (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL;
  bool isIpV4MappedMulticast =
      (a->s6_addr32[0] == htonl(0xff0e0000) &&
       ((a->s6_addr32[1] | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL));
  return (isIpV4Mapped || isIpV4MappedMulticast) ? AF_INET : AF_INET6;
}

static bool matchGidAddrPrefix(sa_family_t af, void* prefix, int prefixlen,
                               union ibv_gid* gid) {
  struct in_addr* base = NULL;
  struct in6_addr* base6 = NULL;
  struct in6_addr* addr6 = NULL;

  if (af == AF_INET) {
    base = (struct in_addr*)prefix;
  } else {
    base6 = (struct in6_addr*)prefix;
  }
  addr6 = (struct in6_addr*)gid->raw;

#define NETMASK(bits) (htonl(0xffffffff ^ ((1 << (32 - bits)) - 1)))

  int i = 0;
  while (prefixlen > 0 && i < 4) {
    if (af == AF_INET) {
      int mask = NETMASK(prefixlen);
      if ((base->s_addr & mask) ^ (addr6->s6_addr32[3] & mask)) {
        break;
      }
      prefixlen = 0;
      break;
    } else {
      if (prefixlen >= 32) {
        if (base6->s6_addr32[i] ^ addr6->s6_addr32[i]) {
          break;
        }
        prefixlen -= 32;
        ++i;
      } else {
        int mask = NETMASK(prefixlen);
        if ((base6->s6_addr32[i] & mask) ^ (addr6->s6_addr32[i] & mask)) {
          break;
        }
        prefixlen = 0;
      }
    }
  }

  return (prefixlen == 0) ? true : false;
}

static bool configuredGid(union ibv_gid* gid) {
  const struct in6_addr* a = (struct in6_addr*)gid->raw;
  int trailer = (a->s6_addr32[1] | a->s6_addr32[2] | a->s6_addr32[3]);
  if (((a->s6_addr32[0] | trailer) == 0UL) ||
      ((a->s6_addr32[0] == htonl(0xfe800000)) && (trailer == 0UL))) {
    return false;
  }
  return true;
}

static bool linkLocalGid(union ibv_gid* gid) {
  const struct in6_addr* a = (struct in6_addr*)gid->raw;
  if (a->s6_addr32[0] == htonl(0xfe800000) && a->s6_addr32[1] == 0UL) {
    return true;
  }
  return false;
}

static bool validGid(union ibv_gid* gid) {
  return (configuredGid(gid) && !linkLocalGid(gid));
}

static bool ncclIbRoceGetVersionNum(char const* deviceName, int portNum,
                                    int gidIndex, int* version) {
  char gidRoceVerStr[16] = {0};
  char roceTypePath[PATH_MAX] = {0};
  sprintf(roceTypePath, "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d",
          deviceName, portNum, gidIndex);

  int fd = open(roceTypePath, O_RDONLY);
  if (fd == -1) {
    return false;
  }
  int ret = read(fd, gidRoceVerStr, 15);
  close(fd);

  if (ret == -1) {
    return false;
  }

  if (strlen(gidRoceVerStr)) {
    if (strncmp(gidRoceVerStr, "IB/RoCE v1", strlen("IB/RoCE v1")) == 0 ||
        strncmp(gidRoceVerStr, "RoCE v1", strlen("RoCE v1")) == 0) {
      *version = 1;
    } else if (strncmp(gidRoceVerStr, "RoCE v2", strlen("RoCE v2")) == 0) {
      *version = 2;
    }
  }

  return true;
}

static bool ncclUpdateGidIndex(struct ibv_context* context, uint8_t portNum,
                               sa_family_t af, void* prefix, int prefixlen,
                               int roceVer, int gidIndexCandidate,
                               int* gidIndex) {
  union ibv_gid gid, gidCandidate;
  if (ibv_query_gid(context, portNum, *gidIndex, &gid) != 0) {
    return false;
  }
  if (ibv_query_gid(context, portNum, gidIndexCandidate, &gidCandidate) != 0) {
    return false;
  }

  sa_family_t usrFam = af;
  sa_family_t gidFam = getGidAddrFamily(&gid);
  sa_family_t gidCandidateFam = getGidAddrFamily(&gidCandidate);
  bool gidCandidateMatchSubnet =
      matchGidAddrPrefix(usrFam, prefix, prefixlen, &gidCandidate);

  if (gidCandidateFam != gidFam && gidCandidateFam == usrFam &&
      gidCandidateMatchSubnet) {
    *gidIndex = gidIndexCandidate;
  } else {
    if (gidCandidateFam != usrFam || !validGid(&gidCandidate) ||
        !gidCandidateMatchSubnet) {
      return true;
    }
    int usrRoceVer = roceVer;
    int gidRoceVerNum, gidRoceVerNumCandidate;
    char const* deviceName = ibv_get_device_name(context->device);
    if (!ncclIbRoceGetVersionNum(deviceName, portNum, *gidIndex,
                                 &gidRoceVerNum)) {
      return false;
    }
    if (!ncclIbRoceGetVersionNum(deviceName, portNum, gidIndexCandidate,
                                 &gidRoceVerNumCandidate)) {
      return false;
    }
    if ((gidRoceVerNum != gidRoceVerNumCandidate || !validGid(&gid)) &&
        gidRoceVerNumCandidate == usrRoceVer) {
      *gidIndex = gidIndexCandidate;
    }
  }

  return true;
}

static int ncclIbExtractFlid(union ibv_gid* gid) {
  return ntohs(*((uint16_t*)((uintptr_t)(gid->raw) + 4)));
}

static bool ncclIbGetGidIndex(struct ibv_context* context, uint8_t portNum,
                              struct ibv_port_attr* portAttr, int* gidIndex) {
  int gidTblLen = portAttr->gid_tbl_len;

  // for IB, choose GID Index that will have routable FLID if present
  if (portAttr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
    union ibv_gid gid;
    int routableGidIndex = get_routable_flid_gid_index_from_env();
    if (routableGidIndex < gidTblLen) {
      if (ibv_query_gid(context, portNum, routableGidIndex, &gid) != 0) {
        return false;
      }
      if (ncclIbExtractFlid(&gid) != 0) {
        *gidIndex = routableGidIndex;
        return true;
      }
    }
    *gidIndex = 0;
    return true;
  }

  // for ROCE
  *gidIndex = get_gid_index_from_env();
  if (*gidIndex >= 0) {
    return true;
  }

  sa_family_t userAddrFamily = envIbAddrFamily();
  int userRoceVersion = get_roce_version_from_env();
  int prefixlen;
  void* prefix = envIbAddrRange(userAddrFamily, &prefixlen);

  *gidIndex = 0;
  for (int gidIndexNext = 1; gidIndexNext < gidTblLen; ++gidIndexNext) {
    if (!ncclUpdateGidIndex(context, portNum, userAddrFamily, prefix, prefixlen,
                            userRoceVersion, gidIndexNext, gidIndex)) {
      return false;
    }
  }

  return true;
}

#endif  // RDMA_UTIL_HPP