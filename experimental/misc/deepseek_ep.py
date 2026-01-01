import torch
import torch.distributed as dist
import argparse


def execute_ep_dispatch(
    inputs,
    ep_group,
    num_local_tokens,
    num_experts,
    top_k=8,
):
    """
    使用uint8精度将token均匀分发到跨GPU的专家上。
    
    参数:
        inputs: 形状为 [num_local_tokens, hidden_size] 的uint8类型张量
        ep_group: 用于all-to-all通信的进程组
        num_local_tokens: 本地GPU上的token数量
        num_experts: 专家总数
        top_k: 每个token选取的顶部专家数量
    
    返回:
        dispatched_inputs: 形状为 [num_recv_tokens, hidden_size] 的uint8类型张量
        recv_splits: 从每个GPU接收的token数量列表
        expert_indices: 每个token对应的专家索引
    """
    # 获取GPU的rank和world size
    rank = dist.get_rank(ep_group)
    world_size = dist.get_world_size(ep_group)
    device = inputs.device
    hidden_size = inputs.shape[1]
    
    # 确认num_local_tokens就是inputs的第一维大小
    assert num_local_tokens == inputs.shape[0], "num_local_tokens应等于inputs.shape[0]"
    # 确认输入是uint8类型
    assert inputs.dtype == torch.uint8, "输入张量必须是torch.uint8类型"
    
    # 计算每个GPU上的local expert数量
    local_experts_per_gpu = num_experts // world_size
    
    # 计算每个token发送到每个expert的数量
    # 在均匀分布的假设下，每个expert接收相同数量的token
    num_routed_tokens = num_local_tokens * top_k  # 每个GPU需要路由的token总数
    tokens_per_expert = num_routed_tokens // num_experts  # 每个expert接收的token数量
    
    # 创建expert计数矩阵 [world_size, local_experts_per_gpu]
    # 表示当前GPU有多少token需要路由到第i个GPU的第j个local expert
    exp_counts = torch.ones(world_size, local_experts_per_gpu, 
                          dtype=torch.int32, device=device) * tokens_per_expert
    
    # 第一次all-to-all: 交换expert计数信息
    exp_counts_exchanged = torch.empty_like(exp_counts)
    torch.distributed.all_to_all_single(
        output=exp_counts_exchanged,
        input=exp_counts,
        group=ep_group,
    )
    
    # 计算每个GPU发送和接收的token数量
    dispatch_send_counts = exp_counts.sum(dim=1)  # [world_size]
    # dispatch_recv_counts = exp_counts_exchanged.sum(dim=1)  # [world_size]
    # NOTE(yongji): UCCL结果是错的，不能用传的数据
    dispatch_recv_counts = exp_counts.sum(dim=1)  # [world_size]
    
    # tokens_per_gpu就是每个GPU需要发送的token总数，即num_routed_tokens
    tokens_per_gpu = num_routed_tokens
    
    # 直接将输入重复top_k次，并确保内存连续
    # [num_local_tokens, hidden_size] -> [num_local_tokens * top_k, hidden_size]
    scaled_inputs = inputs.repeat_interleave(top_k, dim=0).contiguous()
    
    # 准备all-to-all通信
    # 每个GPU发送的token数量由dispatch_send_counts决定
    send_splits = dispatch_send_counts.cpu().tolist()
    recv_splits = dispatch_recv_counts.cpu().tolist()
    
    # [sum(recv_splits), hidden_size]
    num_recv_tokens = dispatch_recv_counts.sum().item()
    dispatched_inputs = torch.empty(
        num_recv_tokens,
        hidden_size,
        dtype=torch.uint8,
        device=device,
    )
    
    # 第二次all-to-all: 交换输入数据
    torch.distributed.all_to_all_single(
        output=dispatched_inputs,
        input=scaled_inputs,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
        group=ep_group,
    )
    
    # 为每个接收到的token分配本地专家索引
    # 在均匀分布的情况下，可以按顺序分配专家索引
    local_expert_indices = torch.arange(num_recv_tokens, device=device) % local_experts_per_gpu
    
    return dispatched_inputs, recv_splits, local_expert_indices


def execute_ep_combine(
    expert_outputs,
    ep_group,
    send_splits,
    recv_splits,
    num_local_tokens,
    top_k=8,
):
    """
    使用bf16精度合并跨GPU的专家输出。
    
    参数:
        expert_outputs: 形状为 [num_recv_tokens, hidden_size] 的bf16类型张量
        ep_group: 用于all-to-all通信的进程组
        send_splits: 要发送到每个GPU的token数量列表
        recv_splits: 从每个GPU接收的token数量列表
        num_local_tokens: 本地GPU上的token数量
        top_k: 每个token选取的顶部专家数量
    
    返回:
        combined_outputs: 形状为 [num_local_tokens, hidden_size] 的bf16类型张量
    """
    device = expert_outputs.device
    hidden_size = expert_outputs.shape[1]
    world_size = dist.get_world_size(ep_group)
    
    # 确认输入是bf16类型
    assert expert_outputs.dtype == torch.bfloat16, "专家输出张量必须是torch.bfloat16类型"
    
    # 总共要接收的token数量
    num_routed_tokens = num_local_tokens * top_k
    
    # 准备all-to-all通信
    outputs_to_send = expert_outputs
    outputs_to_recv = torch.empty(
        num_routed_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    
    # 执行all-to-all通信以发送处理后的输出
    torch.distributed.all_to_all_single(
        output=outputs_to_recv,
        input=outputs_to_send,
        # NOTE(yongji): 添加下面俩个参数后，nccl正常，uccl报错
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
        group=ep_group,
    )
    
    return outputs_to_recv


def benchmark_ep(num_local_tokens, num_experts, hidden_size, top_k=8, num_runs=100):
    """对专家并行的分发和合并操作进行基准测试
    
    参数:
        num_local_tokens: 本地GPU上的token数量
        num_experts: 专家总数
        hidden_size: 隐藏层维度大小
        top_k: 每个token选取的顶部专家数量
        num_runs: 测试运行次数，默认为100
    """
    
    # 初始化EP的进程组
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    ep_group = dist.group.WORLD
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 创建一个基于gloo的CPU进程组用于all_gather_object操作
    try:
        gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        if rank == 0:
            print("成功创建gloo进程组用于all_gather_object操作")
    except Exception as e:
        if rank == 0:
            print(f"创建gloo进程组失败: {e}")
            print("将使用默认进程组进行all_gather_object")
        gloo_group = None
    
    # 验证我们是否有预期的16个GPU
    if rank == 0 and world_size != 32:
        print(f"警告：预期32个GPU，但实际得到{world_size}个")
    
    # 验证num_local_tokens * top_k是world_size的倍数
    num_routed_tokens = num_local_tokens * top_k
    if num_routed_tokens % world_size != 0:
        if rank == 0:
            print(f"错误：num_local_tokens({num_local_tokens}) * top_k({top_k}) = {num_routed_tokens}，必须是world_size({world_size})的倍数")
        return
    
    device = torch.device(f"cuda:{rank % 8}")  # 假设每个节点4个GPU
    torch.cuda.set_device(device)
    
    # 创建虚拟输入(uint8类型)
    inputs = torch.randint(0, 255, (num_local_tokens, hidden_size), device=device, dtype=torch.uint8)
    
    # 创建CUDA事件用于计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # 预热
    if rank == 0:
        print(f"开始预热 ({num_local_tokens} tokens)...")
    
    for _ in range(5):
        dispatched_inputs, recv_splits, expert_indices = execute_ep_dispatch(
            inputs, ep_group, num_local_tokens, num_experts, top_k
        )
        
        dispatched_inputs_bf16 = torch.empty_like(dispatched_inputs, dtype=torch.bfloat16)
        expert_outputs = dispatched_inputs_bf16
        combined_outputs = execute_ep_combine(
            expert_outputs, ep_group, recv_splits, recv_splits, num_local_tokens, top_k
        )
    
    # 初始化结果数组
    dispatch_times_all_runs = []  # 每个元素都是一个包含所有rank时间的列表
    combine_times_all_runs = []   # 每个元素都是一个包含所有rank时间的列表
    
    if rank == 0:
        print(f"开始运行 {num_runs} 次测试 ({num_local_tokens} tokens)...")
    
    # 运行多次测试
    for run in range(num_runs):
        # 同步后开始计时
        torch.cuda.synchronize()
        
        # 对分发进行基准测试
        start_event.record()
        dispatched_inputs, recv_splits, expert_indices = execute_ep_dispatch(
            inputs, ep_group, num_local_tokens, num_experts, top_k
        )
        end_event.record()
        torch.cuda.synchronize()
        dispatch_time = start_event.elapsed_time(end_event)
        
        dispatched_inputs_bf16 = torch.empty_like(dispatched_inputs, dtype=torch.bfloat16)
        expert_outputs = dispatched_inputs_bf16
        
        # 对合并进行基准测试
        start_event.record()
        combined_outputs = execute_ep_combine(
            expert_outputs, ep_group, recv_splits, recv_splits, num_local_tokens, top_k
        )
        end_event.record()
        torch.cuda.synchronize()
        combine_time = start_event.elapsed_time(end_event)
        
        # 从所有rank收集计时结果
        dispatch_times = [0.0] * world_size
        combine_times = [0.0] * world_size
        
        # 使用gloo进程组进行all_gather_object操作
        if gloo_group is not None:
            dist.all_gather_object(dispatch_times, dispatch_time, group=gloo_group)
            dist.all_gather_object(combine_times, combine_time, group=gloo_group)
        else:
            # 如果没有gloo进程组，只保存当前rank的时间
            if rank == 0:
                # 只使用本地测量的时间
                dispatch_times = [dispatch_time] * world_size
                combine_times = [combine_time] * world_size
                print(f"Run {run+1}: dispatch={dispatch_time:.3f}ms, combine={combine_time:.3f}ms")
        
        # 存储这次运行的结果
        dispatch_times_all_runs.append(dispatch_times)
        combine_times_all_runs.append(combine_times)
        
        # 每10次运行显示进度
        if rank == 0 and (run + 1) % 10 == 0:
            print(f"完成 {run + 1}/{num_runs} 次测试...")
    
    if rank == 0:
        # 计算统计信息
        
        # 计算每次运行的平均和最大值
        avg_dispatch_per_run = [sum(times) / len(times) for times in dispatch_times_all_runs]
        max_dispatch_per_run = [max(times) for times in dispatch_times_all_runs]
        avg_combine_per_run = [sum(times) / len(times) for times in combine_times_all_runs]
        max_combine_per_run = [max(times) for times in combine_times_all_runs]
        
        # 计算所有运行的平均值
        avg_dispatch_time = sum(avg_dispatch_per_run) / num_runs
        avg_max_dispatch_time = sum(max_dispatch_per_run) / num_runs
        avg_combine_time = sum(avg_combine_per_run) / num_runs
        avg_max_combine_time = sum(max_combine_per_run) / num_runs
        
        # 计算所有运行的标准差
        std_dispatch_time = (sum((t - avg_dispatch_time) ** 2 for t in avg_dispatch_per_run) / num_runs) ** 0.5
        std_max_dispatch_time = (sum((t - avg_max_dispatch_time) ** 2 for t in max_dispatch_per_run) / num_runs) ** 0.5
        std_combine_time = (sum((t - avg_combine_time) ** 2 for t in avg_combine_per_run) / num_runs) ** 0.5
        std_max_combine_time = (sum((t - avg_max_combine_time) ** 2 for t in max_combine_per_run) / num_runs) ** 0.5
        
        # 打印结果
        print(f"\n===== 使用num_local_tokens={num_local_tokens}的基准测试 ({num_runs}次平均) =====")
        print(f"平均分发时间: {avg_dispatch_time:.3f} ± {std_dispatch_time:.3f} 毫秒")
        print(f"最大分发时间: {avg_max_dispatch_time:.3f} ± {std_max_dispatch_time:.3f} 毫秒")
        print(f"平均合并时间: {avg_combine_time:.3f} ± {std_combine_time:.3f} 毫秒")
        print(f"最大合并时间: {avg_max_combine_time:.3f} ± {std_max_combine_time:.3f} 毫秒")
        print(f"总时间: {avg_max_dispatch_time + avg_max_combine_time:.3f} 毫秒")
        print()

def main():
    """运行基准测试的主函数"""
    parser = argparse.ArgumentParser(description="专家并行基准测试")
    parser.add_argument("--hidden-size", type=int, default=4096, help="隐藏层维度大小")
    parser.add_argument("--num-experts", type=int, default=256, help="专家数量")
    parser.add_argument("--top-k", type=int, default=8, help="每个token的顶部专家数量")
    parser.add_argument("--num-runs", type=int, default=100, help="测试运行次数")
    args = parser.parse_args()
    
    # 测试num_local_tokens=4096的情况
    benchmark_ep(4096, args.num_experts, args.hidden_size, args.top_k, args.num_runs)
    
    # 测试num_local_tokens=128的情况, decode
    benchmark_ep(128, args.num_experts, args.hidden_size, args.top_k, args.num_runs)


if __name__ == "__main__":
    main()
