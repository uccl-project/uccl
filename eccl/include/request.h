#pragma once

#include <memory>
#include <atomic>

enum class ReductionType : int {
    NONE = 0,
    SUM  = 1,
    MIN  = 2,
    MAX  = 3,
    COUNT
};

enum class RequestType : int {
    SEND = 0,
    RECV = 1,
    COUNT
};

struct Request {
    unsigned                id;
    void*                   buf;
    size_t                  len;
    bool                    on_gpu;
    bool                    do_reduction;
    ReductionType           reduction_op;
    RequestType             reqtype;

    std::atomic<bool>       running{false};
    std::atomic<bool>       finished{false};
    static std::atomic<unsigned> global_id_counter;

    Request(void* buf, size_t len,
        bool gpu,
        RequestType reqtype = RequestType::SEND,
        bool reduction = false,
        ReductionType op = ReductionType::NONE)
    : id(global_id_counter.fetch_add(1, std::memory_order_relaxed)),
      buf(buf),
      len(len),
      on_gpu(gpu),
      do_reduction(reduction),
      reduction_op(op),
      reqtype(reqtype),
      running(false),
      finished(false)
    {}
};


struct CommRequest { // child of the Request, for communication
    std::shared_ptr<Request>        req;
    size_t                          offset;
    size_t                          wrlen;
    std::atomic<bool>               done{false};

    CommRequest(std::shared_ptr<Request> req, size_t offset, size_t wrlen)
        : req(req), offset(offset), wrlen(wrlen) {}

    void onCommCompletion();
};

struct ComputeRequest { // child of the Request, for communication
    std::shared_ptr<Request>        req;
    size_t                          offset;
    size_t                          wrlen;
    std::atomic<bool>               done{false};

    ComputeRequest(std::shared_ptr<Request> req, size_t offset, size_t wrlen)
        : req(req), offset(offset), wrlen(wrlen) {}

    void onComputeCompletion();
};
