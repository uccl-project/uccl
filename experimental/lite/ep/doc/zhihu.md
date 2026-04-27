最近给 SGLang 提了一个 PR，新增了一个 shm_pinned 的 PD disaggregation 传输后端。PR链接：https://github.com/sgl-project/sglang/pull/23318

这个后端主要面向一种很常见的部署环境：单机多张消费级显卡，比如 RTX 4090。

这类机器算力不弱（4090, FP8 660 TFLOPS），但跨 GPU 通信条件比较受限。没有 NVLink，官方驱动下也很难稳定依赖 GPU P2P，Prefill 和 Decode 之间传 KV cache 时效率很低。

对于普通模型推理来说，这个问题可能没那么明显。但做 PD 分离之后，Prefill 和 Decode 被拆到不同进程，甚至不同 GPU 上，KV cache 传输就会直接影响 TTFT、TPOT 和整体吞吐。



1. 为什么要关注消费级显卡上的 PD 分离
在业务落地里，并不是所有场景都需要 H100、A800 这种高端卡。很多垂类模型本身尺寸不大，例如 8B、4B 级别模型，在 RTX 4090 这类消费级显卡上做PD分离也能提供不错的吞吐和成本表现。

但这里有一个前提：通信路径要足够高效。

问题在于，Prefill 算完之后，需要把 KV cache 交给 Decode。 如果这一步传得慢，Decode 就要等。请求越多，排队越明显，TTFT 就会被拉高。

在 H100、A800、NVLink、InfiniBand、RDMA 这些条件比较完整的环境里，可以依赖 Mooncake、NIXL、NCCL 这类通信后端，通信效率能做的比较好。

但在 RTX 4090 这种单机 PCIe 多卡机器上，情况会变得比较难处理。

开源通信库NIXL的问题
SGLang 里已有 nixl 后端。NIXL 是 NVIDIA 开源的大模型推理通信库，底层接入 UCX，理论上可以覆盖很多通信路径。

但在我测试的 RTX 4090 环境里，GPU 之间没有可用的 P2P：

torch.cuda.can_device_access_peer(0, 1) = False
torch.cuda.can_device_access_peer(1, 0) = False
再看 NIXL 的 debug 日志，可以看到 UCX 最终选到的是 TCP 相关 transport：

ucp_context_0 intra-node cfg#2
rma_am(tcp/eth0) amo_am(tcp/eth0) am(tcp/eth0 cma/memory) ka(tcp/eth0)
每次传输都要经过更重的协议栈和 CPU 处理。对于 PD 分离这种延迟敏感场景，这个开销会直接反映到 TTFT 和吞吐上。



在同一台机器上，Prefill 和 Decode 只是不同进程。 既然没有 GPU P2P，也不想走 TCP 这套路径，那能不能直接用 CPU 内存作为中转区，把这条路径做得更轻？

这就是 shm_pinned 的思路来源。

3. 用 pinned host memory 做更直接的中转
shm_pinned 的核心路径很直接：

Prefill GPU → pinned host memory → Decode GPU
Decode 侧创建一块 POSIX shared memory。 这块共享内存对 Prefill 和 Decode 两个进程都可见。

然后通过 cudaHostRegister 把这段 host memory 注册成 pinned memory。 这样 GPU 和 CPU 内存之间可以通过 DMA 做 D2H / H2D 拷贝，避免普通 pageable memory 带来的额外开销。

数据仍然要经历两段拷贝：

Prefill GPU → CPU pinned memory
CPU pinned memory → Decode GPU
真正的优化点在于两件事：

第一，pinned host memory 让 D2H / H2D 更适合 GPU DMA。 第二，ring buffer 让 Prefill 和 Decode 可以流水线工作。

如果只是申请一块 pinned memory，然后 Prefill 写完，Decode 再读，那整个流程还是串行的。 吞吐提升的关键在于，Prefill 生产新 KV chunk 的同时，Decode 可以消费之前已经 ready 的 KV chunk。

4. 为什么需要 ring buffer
如果 Prefill 和 Decode 共享同一整块内存，那么最直接的问题就是： 谁能写？谁能读？什么时候能复用？会不会互相覆盖？

shm_pinned 的做法是把这块 pinned shared memory 切成多个 slot。 每个 slot 都可以独立承载一段 KV cache 数据。这样一整块共享内存就变成了一个 ring buffer。

每个 slot 有自己的状态，例如：

free
writing
ready
reading
Prefill 只能拿 free 状态的 slot 写入 KV cache。 写完之后，把这个 slot 标记成 ready。

Decode 只消费 ready 状态的 slot。 H2D 拷贝完成后，再把这个 slot 释放回 free。

整个生命周期可以理解成：

free slot → prefill 写入 → ready slot → decode 读取 → free slot
这样做之后，Prefill 和 Decode 并不会同时读写同一个 slot。

比如 Prefill 正在往 slot 5 写新的 KV chunk，Decode 可以同时从 slot 3 读取已经 ready 的旧 KV chunk。 它们访问的是同一块共享内存里的不同区域，因此不会互相覆盖。

这里的关键点是：共享内存只是数据载体，slot 状态才是所有权协议。

一个 slot 处于 writing 时，只属于 Prefill。 一个 slot 处于 reading 时，只属于 Decode。 只有状态切换完成后，另一个进程才能接手。

这就是 shm_pinned 能够同时工作又不发生数据冲突的原因。

5. 背压是怎么形成的
ring buffer 还有一个好处，天然带背压。

如果 Decode 消费慢了，ready slot 会逐渐变多，free slot 会逐渐变少。 当没有 free slot 时，Prefill 就会等待，不能继续往共享内存里塞数据。

如果 Decode 消费变快，slot 会重新释放成 free，Prefill 又可以继续写入。

这样系统不会无限堆积内存，也不需要额外做复杂的流控逻辑。 slot 数量本身就决定了 Prefill 和 Decode 之间的缓冲深度。

slot 太少，流水线很容易断。 Decode 稍微慢一点，Prefill 就会被阻塞。

slot 太多，也会带来问题。 pinned memory 是比较敏感的系统资源，申请太大可能影响系统其他进程，也可能触发系统或驱动层面的限制。

所以 slot count 本质上是在吞吐、延迟和内存占用之间做权衡。

