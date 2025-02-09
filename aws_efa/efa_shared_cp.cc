#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>

#include <cstring>
#include <iostream>
#include <vector>

#ifndef OFI_MR_BASIC_MAP
#define OFI_MR_BASIC_MAP (FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_VIRT_ADDR)
#endif
#define NUM_EPS 2         // Number of UD endpoints (EPs)
#define BUFFER_SIZE 4096  // Receive buffer size
#define MAX_RECV_WR 10    // Max receive work requests per EP
#define MR_KEY 0xC0DE

struct EFAContext {
    struct fi_info *hints, *info;
    struct fid_fabric *fabric;
    struct fid_domain *domain;
    struct fid_cq *cq;
    struct fid_av *av;
    struct fid_mr *mr;
    void *mr_desc;
    std::vector<struct fid_ep *> eps;  // List of UD endpoints
    void *recv_buffer_pool;
    std::vector<char *> recv_buffers;
};

// Utility function for error handling
void check_fi_call(int ret, const char *msg) {
    if (ret) {
        std::cerr << msg << ": " << fi_strerror(-ret) << std::endl;
        exit(1);
    }
}

// Initialize the EFA fabric and create shared CQ
void efa_init(EFAContext &efa) {
    // Allocate hints
    efa.hints = fi_allocinfo();

    efa.hints->mode = ~0;
    efa.hints->domain_attr->mode = ~0;
    efa.hints->domain_attr->mr_mode = ~(FI_MR_BASIC | FI_MR_SCALABLE);

    // efa.hints->caps = FI_MSG | FI_RMA;
    // efa.hints->mode = FI_CONTEXT;
    // efa.hints->ep_attr->type = FI_EP_DGRAM;
    // efa.hints->fabric_attr->prov_name = strdup("efa");

    efa.hints->ep_attr->type = FI_EP_DGRAM;
    efa.hints->caps = FI_MSG;
    efa.hints->mode = FI_CONTEXT | FI_CONTEXT2 | FI_MSG_PREFIX;
    efa.hints->domain_attr->mr_mode = FI_MR_LOCAL | OFI_MR_BASIC_MAP;
    efa.hints->domain_attr->name = "rdmap16s27-dgrm";
    // hints->fabric_attr->name = "172.31.64.0/20";
    efa.hints->fabric_attr->prov_name =
        "efa";  // "sockets" -> TCP, "udp" -> UDP

    // Get fabric info
    int ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), NULL,
                         NULL, 0, efa.hints, &efa.info);
    check_fi_call(ret, "fi_getinfo failed");

    // Open fabric
    ret = fi_fabric(efa.info->fabric_attr, &efa.fabric, NULL);
    check_fi_call(ret, "fi_fabric failed");

    // Open domain
    ret = fi_domain(efa.fabric, efa.info, &efa.domain, NULL);
    check_fi_call(ret, "fi_domain failed");

    // Create completion queue (shared across all EPs)
    struct fi_cq_attr cq_attr = {};
    memset(&cq_attr, 0, sizeof(cq_attr));
    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    cq_attr.wait_obj = FI_WAIT_NONE;
    cq_attr.size = 1024;
    ret = fi_cq_open(efa.domain, &cq_attr, &efa.cq, NULL);
    check_fi_call(ret, "fi_cq_open failed");

    // Create address vector (AV)
    struct fi_av_attr av_attr = {};
    memset(&av_attr, 0, sizeof(av_attr));
    av_attr.type = FI_AV_MAP;
    ret = fi_av_open(efa.domain, &av_attr, &efa.av, NULL);
    check_fi_call(ret, "fi_av_open failed");
}

// Create multiple UD endpoints and associate them with the shared CQ
void create_ud_eps(EFAContext &efa) {
    for (int i = 0; i < NUM_EPS; i++) {
        std::cout << "Creating UD EP: " << i << std::endl;
        struct fid_ep *ep;
        int ret = fi_endpoint(efa.domain, efa.info, &ep, NULL);
        check_fi_call(ret, "fi_endpoint failed");

        // Bind CQ to the EP
        ret = fi_ep_bind(ep, &efa.cq->fid, FI_SEND | FI_RECV);
        check_fi_call(ret, "fi_ep_bind to CQ failed");

        // Bind AV to the EP
        ret = fi_ep_bind(ep, &efa.av->fid, 0);
        check_fi_call(ret, "fi_ep_bind to AV failed");

        // Enable the endpoint
        ret = fi_enable(ep);
        check_fi_call(ret, "fi_enable failed");

        efa.eps.push_back(ep);
    }
}

void create_mr_and_buf(EFAContext &efa) {
    int alignment = 4096;
    int ret = posix_memalign(&(efa.recv_buffer_pool), (size_t)alignment,
                             MAX_RECV_WR * BUFFER_SIZE);
    check_fi_call(ret, "ofi_memalign failed");

    ret = fi_mr_reg(efa.domain, efa.recv_buffer_pool, MAX_RECV_WR * BUFFER_SIZE,
                    FI_SEND | FI_RECV, 0, MR_KEY, 0, &(efa.mr), NULL);
    check_fi_call(ret, "fi_mr_reg failed");
    efa.mr_desc = fi_mr_desc(efa.mr);

    for (int i = 0; i < MAX_RECV_WR; i++) {
        char *recv_buf = (char *)efa.recv_buffer_pool + i * BUFFER_SIZE;
        memset(recv_buf, 0, BUFFER_SIZE);
        efa.recv_buffers.push_back(recv_buf);
    }
}

// Post receive buffers for each UD endpoint
void post_receive_buffers(EFAContext &efa) {
    for (auto &ep : efa.eps) {
        for (int i = 0; i < MAX_RECV_WR; i++) {
            char *recv_buf = efa.recv_buffers[i];

            struct fi_msg msg = {};
            struct iovec iov = {recv_buf, BUFFER_SIZE};
            void *desc = NULL;
            msg.msg_iov = &iov;
            msg.iov_count = 1;
            msg.desc = &efa.mr_desc;
            msg.addr = FI_ADDR_UNSPEC;  // Accept from any sender
            msg.context = recv_buf;

            int ret = fi_recvmsg(ep, &msg, FI_COMPLETION);
            check_fi_call(ret, "fi_recvmsg failed");
        }
    }
}

// Poll the shared CQ for completions
void poll_cq(EFAContext &efa) {
    while (true) {
        struct fi_cq_data_entry cq_entry;
        int num_completions = fi_cq_read(efa.cq, &cq_entry, 1);
        if (num_completions > 0) {
            // Find which EP received the message
            struct fid_ep *ep_received = (struct fid_ep *)cq_entry.op_context;
            std::cout << "Received message on UD EP: " << ep_received
                      << std::endl;

            // Process message
            char *msg_data = (char *)cq_entry.buf;
            std::cout << "Received data: " << msg_data << std::endl;

            // Repost the buffer for further reception
            struct fi_msg msg = {};
            struct iovec iov = {msg_data, BUFFER_SIZE};
            void *desc = NULL;
            msg.msg_iov = &iov;
            msg.iov_count = 1;
            msg.desc = &(efa.mr_desc);
            msg.addr = FI_ADDR_UNSPEC;  // Accept from any sender
            msg.context = msg_data;

            int ret = fi_recvmsg(ep_received, &msg, FI_COMPLETION);
            check_fi_call(ret, "fi_recvmsg repost failed");
        }
    }
}

// Cleanup resources
void cleanup(EFAContext &efa) {
    for (auto ep : efa.eps) {
        fi_close(&ep->fid);
    }
    fi_close(&efa.av->fid);
    fi_close(&efa.cq->fid);
    fi_close(&efa.domain->fid);
    fi_close(&efa.fabric->fid);
    fi_freeinfo(efa.info);
    fi_freeinfo(efa.hints);
}

// Main function
int main() {
    EFAContext efa;

    // Initialize the EFA fabric and create shared CQ
    efa_init(efa);

    // Create multiple UD endpoints
    create_ud_eps(efa);

    // Create memory region and receive buffers
    create_mr_and_buf(efa);

    // Post receive buffers for each endpoint
    post_receive_buffers(efa);

    // Poll the shared CQ for completions
    poll_cq(efa);

    // Cleanup (this won't be reached in an infinite polling loop)
    cleanup(efa);

    return 0;
}
