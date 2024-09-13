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

#define PP_PRINTERR(call, retv)                                                \
	fprintf(stderr, "%s(): %s:%-4d, ret=%d (%s)\n", call, __FILE__,        \
		__LINE__, (int)retv, fi_strerror((int) -retv))

#define PP_ERR(fmt, ...)                                                       \
	fprintf(stderr, "[%s] %s:%-4d: " fmt "\n", "error", __FILE__,          \
		__LINE__, ##__VA_ARGS__)

int main(int argc, char **argv)
{
    int ret = EXIT_SUCCESS;

    char *dst_addr = "172.31.76.70";
    uint16_t dst_port = 8889;

    struct fi_info *fi_pep, *fi, *hints;
    hints = fi_allocinfo();
    if (!hints)
        return EXIT_FAILURE;

    hints->ep_attr->type = FI_EP_DGRAM;
    hints->caps = FI_MSG;
    hints->mode = FI_CONTEXT;
    hints->domain_attr->mr_mode = FI_MR_UNSPEC;

    uint64_t flags = 0;
    ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
                     NULL, NULL, flags, hints, &fi);
    if (ret)
    {
        PP_PRINTERR("fi_getinfo", ret);
        return ret;
    }
    struct fi_context tx_ctx[2], rx_ctx[2];

    struct fid_fabric *fabric;
    ret = fi_fabric(fi->fabric_attr, &(fabric), NULL);
    if (ret)
    {
        PP_PRINTERR("fi_fabric", ret);
        return ret;
    }

    struct fi_eq_attr eq_attr;
    struct fid_eq *eq;
    ret = fi_eq_open(fabric, &(eq_attr), &(eq), NULL);
    if (ret)
    {
        PP_PRINTERR("fi_eq_open", ret);
        return ret;
    }

    struct fid_domain *domain;
    ret = fi_domain(fabric, fi, &(domain), NULL);
    if (ret)
    {
        PP_PRINTERR("fi_domain", ret);
        return ret;
    }

    struct fi_av_attr av_attr;
    struct fi_cq_attr cq_attr;
    struct fid_cq *txcq, *rxcq;

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

    struct fid_av *av;
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

    void *local_name, *rem_name;
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

    fi_addr_t local_fi_addr, remote_fi_addr;
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

    char buf[128];
    size_t reslen = addrlen;
    char* res = fi_av_straddr(av, local_name, buf, &reslen);
    printf("reslen %lu, local_name: %s\n", reslen, res);

    return 0;
}