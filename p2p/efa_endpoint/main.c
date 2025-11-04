#include "DeviceManager.h"
#include "EFAChannel.h"
#include <thread>

int main() {
    // 初始化 DeviceManager (单例)
    auto& manager = DeviceManager::getInstance();

    // 获取第0个 device 对应的 context
    auto ctx = manager.getContextById(0);

    // 创建 EFAChannel
    auto channel = std::make_shared<EFAChannel>(ctx);

    // 模拟本地和远程 metadata（一般通过 TCP 交换）
    metadata local_meta = {
        .qpn = channel->getQP()->qp_num,
        .rkey = ctx->getMR()->rkey,
        .addr = (uint64_t)ctx->getMR()->addr
    };
    ctx->getGID(0, &local_meta.gid);

    metadata remote_meta = local_meta; // demo 中本地充当远端
    channel->connect(remote_meta);

    // 分配发送缓冲区
    char* buf = (char*)ctx->getMR()->addr;
    strcpy(buf, "Hello from EFAChannel!");

    // 创建请求并发送
    auto sendReq = std::make_shared<EFASendRequest>();
    sendReq->addr = (uint64_t)buf;
    sendReq->rkey = remote_meta.rkey;
    sendReq->length = strlen(buf) + 1;
    sendReq->imm_data = 0x1;

    channel->send(sendReq);
    std::cout << "Message sent successfully." << std::endl;

    // 模拟接收
    auto recvReq = std::make_shared<EFARecvRequest>();
    recvReq->addr = (uint64_t)buf;
    recvReq->rkey = ctx->getMR()->rkey;
    recvReq->length = 1024;

    channel->recv(recvReq);
    std::cout << "Receive posted." << std::endl;

    return 0;
}
