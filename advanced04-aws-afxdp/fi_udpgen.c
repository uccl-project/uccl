#include <arpa/inet.h>
#include <assert.h>
#include <config.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <netdb.h>
#include <netinet/in.h>
#include <ofi_mem.h>
#include <poll.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_tagged.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef OFI_MR_BASIC_MAP
#define OFI_MR_BASIC_MAP (FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_VIRT_ADDR)
#endif

#define PP_PRINTERR(call, retv)                                     \
    fprintf(stderr, "%s(): %s:%-4d, ret=%d (%s)\n", call, __FILE__, \
            __LINE__, (int)retv, fi_strerror((int)-retv))

#define PP_ERR(fmt, ...)                                          \
    fprintf(stderr, "[%s] %s:%-4d: " fmt "\n", "error", __FILE__, \
            __LINE__, ##__VA_ARGS__)

static int print_short_info(struct fi_info *info) {
    struct fi_info *cur;

    for (cur = info; cur; cur = cur->next) {
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

/*******************************************************************************
 *                                      Data Messaging
 ******************************************************************************/

static int pp_cq_readerr(struct fid_cq *cq) {
    struct fi_cq_err_entry cq_err = {0};
    int ret;

    ret = fi_cq_readerr(cq, &cq_err, 0);
    if (ret < 0) {
        PP_PRINTERR("fi_cq_readerr", ret);
    } else {
        PP_ERR("cq_readerr: %s",
               fi_cq_strerror(cq, cq_err.prov_errno, cq_err.err_data,
                              NULL, 0));
        ret = -cq_err.err;
    }
    return ret;
}

static int pp_get_cq_comp(struct fid_cq *cq, uint64_t *cur, uint64_t total,
                          int timeout_sec) {
    struct fi_cq_err_entry comp;
    uint64_t a = 0, b = 0;
    int ret = 0;

    if (timeout_sec >= 0)
        a = pp_gettime_us();

    do {
        ret = fi_cq_read(cq, &comp, 1);
        if (ret > 0) {
            if (timeout_sec >= 0)
                a = pp_gettime_us();

            (*cur)++;
        } else if (ret < 0 && ret != -FI_EAGAIN) {
            if (ret == -FI_EAVAIL) {
                ret = pp_cq_readerr(cq);
                (*cur)++;
            } else {
                PP_PRINTERR("pp_get_cq_comp", ret);
            }

            return ret;
        } else if (timeout_sec >= 0) {
            b = pp_gettime_us();
            if ((b - a) / 1000000 > timeout_sec) {
                fprintf(stderr, "%ds timeout expired\n",
                        timeout_sec);
                return -FI_ENODATA;
            }
        }
    } while (total - *cur > 0);

    return 0;
}

uint64_t tx_seq, rx_seq, tx_cq_cntr, rx_cq_cntr;
long cnt_ack_msg;

static int pp_get_rx_comp(struct fid_cq *rxcq, uint64_t total) {
    int ret = FI_SUCCESS;
    ret = pp_get_cq_comp(rxcq, &(rx_cq_cntr), total, -1);
    return ret;
}

static int pp_get_tx_comp(struct fid_cq *txcq, uint64_t total) {
    int ret;
    ret = pp_get_cq_comp(txcq, &(tx_cq_cntr), total, -1);
    return ret;
}

static ssize_t pp_tx(struct fid_ep *ep, void *tx_buf, size_t size, fi_addr_t remote_fi_addr, void *tx_ctx, struct fid_cq *txcq) {
    ssize_t ret, rc;

    while (1) {
        ret = (int)fi_send(ep, tx_buf, size, NULL, remote_fi_addr, tx_ctx);
        if (!ret)
            break;

        if (ret != -FI_EAGAIN) {
            PP_PRINTERR("transmit", ret);
            return ret;
        }

        rc = pp_get_tx_comp(txcq, tx_seq);
        if (rc && rc != -FI_EAGAIN) {
            PP_ERR("Failed to get transmit completion");
            return rc;
        }
    }
    tx_seq++;

    ret = pp_get_tx_comp(txcq, tx_seq);

    return ret;
}

static ssize_t pp_inject(struct fid_ep *ep, void *tx_buf, size_t size, fi_addr_t remote_fi_addr, struct fid_cq *txcq) {
    ssize_t ret, rc;

    while (1) {
        ret = (int)fi_inject(ep, tx_buf, size, remote_fi_addr);
        if (!ret)
            break;

        if (ret != -FI_EAGAIN) {
            PP_PRINTERR('inject', ret);
            return ret;
        }

        rc = pp_get_tx_comp(txcq, rx_seq);
        if (rc && rc != -FI_EAGAIN) {
            PP_ERR("Failed to get inject completion");
            return rc;
        }
    }
    tx_seq++;

    if (ret)
        return ret;

    tx_cq_cntr++;
    return ret;
}

static ssize_t pp_rx(struct fid_ep *ep, void *rx_buf, size_t size, void *rx_ctx, struct fid_cq *rxcq) {
    ssize_t ret, rc;

    ret = pp_get_rx_comp(rxcq, rx_seq);
    if (ret)
        return ret;

    /* Ignore the size arg. Post a buffer large enough to handle all message
     * sizes. pp_sync() makes use of pp_rx() and gets called in tests just
     * before message size is updated. The recvs posted are always for the
     * next incoming message.
     */

    while (1) {
        ret = (int)fi_recv(ep, rx_buf, size, NULL, 0, rxcq);
        if (!ret)
            break;

        if (ret != -FI_EAGAIN) {
            PP_PRINTERR('receive', ret);
            return ret;
        }

        rc = pp_get_rx_comp(rxcq, rx_seq);
        if (rc && rc != -FI_EAGAIN) {
            PP_ERR("Failed to get receive completion");
            return rc;
        }
    }
    rx_seq++;

    if (!ret)
        cnt_ack_msg++;

    return ret;
}

char *dst_addr = "172.31.38.12";
uint16_t dst_port = 8889;
uint16_t oob_dst_port = 8890;
size_t msg_size = 128;

int main(int argc, char **argv) {
    int ret = EXIT_SUCCESS;

    struct fi_info *fi_pep, *fi, *hints;
    struct fid_fabric *fabric;
    struct fi_eq_attr eq_attr;
    struct fid_eq *eq;
    struct fid_domain *domain;
    struct fid_ep *ep;
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
    // hints->fabric_attr->name = "172.31.64.0/20";
    hints->fabric_attr->prov_name = "udp";  // "sockets" -> TCP

    ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
                     NULL, NULL, 0, hints, &fi);
    if (ret) {
        PP_PRINTERR("fi_getinfo", ret);
        return ret;
    }
    print_short_info(fi);

    ret = fi_fabric(fi->fabric_attr, &(fabric), NULL);
    if (ret) {
        PP_PRINTERR("fi_fabric", ret);
        return ret;
    }

    ret = fi_eq_open(fabric, &(eq_attr), &(eq), NULL);
    if (ret) {
        PP_PRINTERR("fi_eq_open", ret);
        return ret;
    }

    ret = fi_domain(fabric, fi, &(domain), NULL);
    if (ret) {
        PP_PRINTERR("fi_domain", ret);
        return ret;
    }

    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    cq_attr.wait_obj = FI_WAIT_NONE;
    cq_attr.size = 1024;
    ret = fi_cq_open(domain, &(cq_attr), &(txcq), &(txcq));
    if (ret) {
        PP_PRINTERR("fi_cq_open", ret);
        return ret;
    }
    ret = fi_cq_open(domain, &(cq_attr), &(rxcq), &(rxcq));
    if (ret) {
        PP_PRINTERR("fi_cq_open", ret);
        return ret;
    }

    av_attr.type = FI_AV_MAP;
    ret = fi_av_open(domain, &(av_attr), &(av), NULL);
    if (ret) {
        PP_PRINTERR("fi_av_open", ret);
        return ret;
    }

    ret = fi_endpoint(domain, fi, &(ep), NULL);
    if (ret) {
        PP_PRINTERR("fi_endpoint", ret);
        return ret;
    }

#define PP_EP_BIND(ep, fd, flags)                        \
    do {                                                 \
        int ret;                                         \
        if ((fd)) {                                      \
            ret = fi_ep_bind((ep), &(fd)->fid, (flags)); \
            if (ret) {                                   \
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
    if (ret) {
        PP_PRINTERR("fi_enable", ret);
        return ret;
    }

    size_t addrlen = 0;
    local_name = NULL;
    ret = fi_getname(&ep->fid, local_name, &addrlen);
    if ((ret != -FI_ETOOSMALL) || (addrlen <= 0)) {
        PP_ERR("fi_getname didn't return length\n");
        return -EMSGSIZE;
    }
    local_name = malloc(addrlen);

    ret = fi_getname(&ep->fid, local_name, &addrlen);
    if (ret) {
        PP_PRINTERR("fi_getname", ret);
        return ret;
    }

    ret = fi_av_insert(av, local_name, 1, &local_fi_addr, 0, NULL);
    if (ret < 0) {
        PP_PRINTERR("fi_av_insert", ret);
        return ret;
    } else if (ret != 1) {
        PP_ERR(
            "fi_av_insert: number of addresses inserted = %d;"
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

    // OOB communication to get the remote address
    int sockfd, connfd;
    struct sockaddr_in servaddr, cli;

    // socket create and verification
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        printf("socket creation failed...\n");
        exit(0);
    } else
        printf("Socket successfully created..\n");
    bzero(&servaddr, sizeof(servaddr));

    // assign IP, PORT
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = inet_addr(dst_addr);
    servaddr.sin_port = htons(oob_dst_port);

    // connect the client socket to server socket
    if (connect(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) != 0) {
        printf("connection with the server failed...\n");
        exit(0);
    } else
        printf("connected to the server..\n");

    ret = read(sockfd, &size, sizeof(size));
    if (ret != sizeof(size)) {
        printf("size reading failure\n");
        exit(0);
    }
    rem_name = malloc(size);
    ret = read(sockfd, rem_name, size);
    if (ret != size) {
        printf("rem_name reading failure\n");
        exit(0);
    }

    ret = fi_av_insert(av, rem_name, 1, &remote_fi_addr, 0, NULL);
    if (ret < 0) {
        PP_PRINTERR("fi_av_insert", ret);
        return ret;
    } else if (ret != 1) {
        PP_ERR(
            "fi_av_insert: number of addresses inserted = %d;"
            " number of addresses given = %zd\n",
            ret, 1);
        return -EXIT_FAILURE;
    }

    size = 0;
    fi_av_straddr(av, rem_name, NULL, &size);
    fi_av_straddr(av, rem_name, ep_name_buf, &size);

    printf("Remote OFI EP prov %s name %s straddr %s\n",
           fi->fabric_attr->prov_name,
           fi->fabric_attr->name, ep_name_buf);

#define MAGIC_NUMBER 0xdeadbeef

    void *sendbuf = malloc(msg_size);
    *(uint64_t*)sendbuf = MAGIC_NUMBER;
    void *recvbuf = malloc(msg_size);

    for (int i = 0; i < 1024; i++) {
        pp_tx(ep, sendbuf, msg_size, remote_fi_addr, &tx_ctx[0], txcq);
        printf("pp_tx %d\n", i);
        // sleep(1);
        pp_rx(ep, recvbuf, msg_size, &rx_ctx[0], rxcq);
        printf("pp_rx %d\n", i);
    }

    return 0;
}