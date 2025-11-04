#pragma once
#include "define.h"
#include "rdma_context.h"


class EFAChannel {
public:
    explicit EFAChannel(std::shared_ptr<RdmaContext> ctx)
        : ctx_(ctx), qp_(nullptr), cq_ex_(nullptr), ah_(nullptr), qpn_(0) {
        initQP();
    }

    void connect(const metadata& remote_meta) {
        remote_meta_ = remote_meta;
        ah_ = ctx_->createAH(remote_meta_.gid);
        qpn_ = remote_meta_.qpn;
        std::cout << "EFAChannel connected to remote qpn=" << qpn_ << std::endl;
    }

    // send() 使用 ibv_wr_rdma_write_imm
    void send(std::shared_ptr<EFASendRequest> req) {
        auto* qpx = ibv_qp_to_qp_ex(qp_);
        ibv_wr_start(qpx);

        qpx->wr_id = 1;
        qpx->comp_mask = 0;
        qpx->wr_flags = IBV_SEND_SIGNALED;

        ibv_wr_rdma_write_imm(qpx, req->rkey, req->addr, req->imm_data);

        struct ibv_sge sge[1] = {
            {(uintptr_t)req->addr, (uint32_t)req->length, ctx_->getMR()->lkey}
        };
        ibv_wr_set_sge_list(qpx, 1, sge);
        ibv_wr_set_ud_addr(qpx, ah_, qpn_, 0x12345);

        if (ibv_wr_complete(qpx)) {
            perror("ibv_wr_complete failed");
        }
    }

    // recv() 使用 ibv_post_recv
    void recv(std::shared_ptr<EFARecvRequest> req) {
        struct ibv_sge sge = {
            .addr = (uintptr_t)req->addr,
            .length = (uint32_t)req->length,
            .lkey = ctx_->getMR()->lkey,
        };
        struct ibv_recv_wr wr = {0}, *bad_wr = nullptr;
        wr.wr_id = 2;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        if (ibv_post_recv(qp_, &wr, &bad_wr)) {
            perror("ibv_post_recv failed");
        }
    }

    struct ibv_cq_ex* getCQ() const { return cq_ex_; }
    struct ibv_qp* getQP() const { return qp_; }

private:
    std::shared_ptr<RdmaContext> ctx_;
    struct ibv_cq_ex* cq_ex_;
    struct ibv_qp* qp_;
    struct ibv_ah* ah_;
    metadata remote_meta_;
    uint32_t qpn_;

    void initQP() {
        struct ibv_cq_init_attr_ex cq_attr = {};
        cq_attr.cqe = 1024;
        cq_attr.wc_flags = IBV_WC_STANDARD_FLAGS;
        cq_attr.comp_mask = 0;

        cq_ex_ = ibv_create_cq_ex(ctx_->getCtx(), &cq_attr);
        assert(cq_ex_);

        struct ibv_qp_init_attr_ex qp_attr = {};
        qp_attr.pd = ctx_->getPD();
        qp_attr.send_cq = ibv_cq_ex_to_cq(cq_ex_);
        qp_attr.recv_cq = ibv_cq_ex_to_cq(cq_ex_);
        qp_attr.cap.max_send_wr = 128;
        qp_attr.cap.max_recv_wr = 128;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        qp_attr.qp_type = IBV_QPT_DRIVER;

        struct efadv_qp_init_attr efa_attr = {};
        efa_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
        efa_attr.sl = 8;

        qp_ = efadv_create_qp_ex(ctx_->getCtx(), &qp_attr, &efa_attr, sizeof(efa_attr));
        assert(qp_);

        // 初始化 QP 状态
        struct ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_INIT;
        attr.port_num = 1;
        attr.qkey = 0x12345;
        if (ibv_modify_qp(qp_, &attr,
                          IBV_QP_STATE | IBV_QP_PORT | IBV_QP_QKEY)) {
            perror("Failed to set QP INIT");
        }
        attr.qp_state = IBV_QPS_RTR;
        ibv_modify_qp(qp_, &attr, IBV_QP_STATE);
        attr.qp_state = IBV_QPS_RTS;
        ibv_modify_qp(qp_, &attr, IBV_QP_STATE);
    }
};
