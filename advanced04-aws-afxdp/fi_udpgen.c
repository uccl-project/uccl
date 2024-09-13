#include <config.h>

#include <time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <assert.h>
#include <getopt.h>
#include <inttypes.h>
#include <netdb.h>
#include <poll.h>
#include <limits.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/time.h>

#include <ofi_mem.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_tagged.h>

#ifndef OFI_MR_BASIC_MAP
#define OFI_MR_BASIC_MAP (FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_VIRT_ADDR)
#endif

#define PP_PRINTERR(call, retv)                                     \
    fprintf(stderr, "%s(): %s:%-4d, ret=%d (%s)\n", call, __FILE__, \
            __LINE__, (int)retv, fi_strerror((int)-retv))

#define PP_ERR(fmt, ...)                                          \
    fprintf(stderr, "[%s] %s:%-4d: " fmt "\n", "error", __FILE__, \
            __LINE__, ##__VA_ARGS__)

static int print_short_info(struct fi_info *info)
{
    struct fi_info *cur;

    for (cur = info; cur; cur = cur->next)
    {
        printf("provider: %s\n", cur->fabric_attr->prov_name);
        printf("    fabric: %s\n", cur->fabric_attr->name),
            printf("    domain: %s\n", cur->domain_attr->name),
            printf("    version: %d.%d\n", FI_MAJOR(cur->fabric_attr->prov_version),
                   FI_MINOR(cur->fabric_attr->prov_version));
        printf("    type: %s\n", fi_tostr(&cur->ep_attr->type, FI_TYPE_EP_TYPE));
        printf("    protocol: %s\n", fi_tostr(&cur->ep_attr->protocol, FI_TYPE_PROTOCOL));
    }
    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
    int ret = EXIT_SUCCESS;

    char *dst_addr = "172.31.76.70";
    uint16_t dst_port = 8889;

    struct fi_info *fi_pep, *fi, *hints;
    struct fid_fabric *fabric;
    struct fi_eq_attr eq_attr;
    struct fid_eq *eq;
    struct fid_domain *domain;
    struct fi_cq_attr cq_attr;
    struct fid_cq *txcq, *rxcq;
    struct fi_av_attr av_attr;
    struct fid_av *av;
    fi_addr_t local_fi_addr, remote_fi_addr;
    void *local_name, *rem_name;
    struct fi_context tx_ctx[2], rx_ctx[2];

    hints = fi_allocinfo();
    if (!hints)
        return EXIT_FAILURE;

    hints->ep_attr->type = FI_EP_DGRAM;
    hints->caps = FI_MSG;
    hints->mode = FI_CONTEXT;
    hints->domain_attr->mr_mode = FI_MR_UNSPEC;
    hints->domain_attr->name = "enp39s0";
    hints->fabric_attr->name = "172.31.64.0/20";
    hints->fabric_attr->prov_name = "udp"; // "sockets" -> TCP

    ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
                     NULL, NULL, 0, hints, &fi);
    if (ret)
    {
        PP_PRINTERR("fi_getinfo", ret);
        return ret;
    }
    print_short_info(fi);

    ret = fi_fabric(fi->fabric_attr, &(fabric), NULL);
    if (ret)
    {
        PP_PRINTERR("fi_fabric", ret);
        return ret;
    }

    ret = fi_eq_open(fabric, &(eq_attr), &(eq), NULL);
    if (ret)
    {
        PP_PRINTERR("fi_eq_open", ret);
        return ret;
    }

    ret = fi_domain(fabric, fi, &(domain), NULL);
    if (ret)
    {
        PP_PRINTERR("fi_domain", ret);
        return ret;
    }

    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    cq_attr.wait_obj = FI_WAIT_NONE;
    cq_attr.size = 1024;
    ret = fi_cq_open(domain, &(cq_attr), &(txcq), &(txcq));
    if (ret)
    {
        PP_PRINTERR("fi_cq_open", ret);
        return ret;
    }
    ret = fi_cq_open(domain, &(cq_attr), &(rxcq), &(rxcq));
    if (ret)
    {
        PP_PRINTERR("fi_cq_open", ret);
        return ret;
    }

    av_attr.type = FI_AV_MAP;
    ret = fi_av_open(domain, &(av_attr), &(av), NULL);
    if (ret)
    {
        PP_PRINTERR("fi_av_open", ret);
        return ret;
    }

    struct fid_ep *ep;
    ret = fi_endpoint(domain, fi, &(ep), NULL);
    if (ret)
    {
        PP_PRINTERR("fi_endpoint", ret);
        return ret;
    }

#define PP_EP_BIND(ep, fd, flags)                        \
    do                                                   \
    {                                                    \
        int ret;                                         \
        if ((fd))                                        \
        {                                                \
            ret = fi_ep_bind((ep), &(fd)->fid, (flags)); \
            if (ret)                                     \
            {                                            \
                PP_PRINTERR("fi_ep_bind", ret);          \
                return ret;                              \
            }                                            \
        }                                                \
    } while (0)

    PP_EP_BIND(ep, eq, 0);
    PP_EP_BIND(ep, av, 0);
    PP_EP_BIND(ep, txcq, FI_TRANSMIT);
    PP_EP_BIND(ep, rxcq, FI_RECV);

    ret = fi_enable(ep);
    if (ret)
    {
        PP_PRINTERR("fi_enable", ret);
        return ret;
    }

    size_t addrlen = 0;
    local_name = NULL;
    ret = fi_getname(&ep->fid, local_name, &addrlen);
    if ((ret != -FI_ETOOSMALL) || (addrlen <= 0))
    {
        PP_ERR("fi_getname didn't return length\n");
        return -EMSGSIZE;
    }
    local_name = malloc(addrlen);

    ret = fi_getname(&ep->fid, local_name, &addrlen);
    if (ret)
    {
        PP_PRINTERR("fi_getname", ret);
        return ret;
    }

    ret = fi_av_insert(av, local_name, 1, &local_fi_addr, 0, NULL);
    if (ret < 0)
    {
        PP_PRINTERR("fi_av_insert", ret);
        return ret;
    }
    else if (ret != 1)
    {
        PP_ERR("fi_av_insert: number of addresses inserted = %d;"
               " number of addresses given = %zd\n",
               ret, 1);
        return -EXIT_FAILURE;
    }

    char ep_name_buf[128];
    size_t size = 0;
    fi_av_straddr(av, local_name, NULL, &size);

    fi_av_straddr(av, local_name, ep_name_buf, &size);

    printf("OFI EP prov %s name %s straddr %s\n",
           fi->fabric_attr->prov_name,
           fi->fabric_attr->name, ep_name_buf);

    return 0;
}