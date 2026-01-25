import torch
import ukernel

x = torch.randn(1024, device="cuda")
y = torch.empty_like(x)

rule = ukernel.ParallelRule(num_tasks=4, tiles_per_task=8)

r = ukernel.moe_routing(0, x, x, rule)
a2a = ukernel.all_to_all(1, x, x, rule, deps=[r.id])
gemm = ukernel.moe_expert_gemm(2, x, x, rule, deps=[a2a.id])
combine = ukernel.moe_combine(3, x, y, rule, deps=[gemm.id])
allreduce = ukernel.all_reduce(4, x, y, ukernel.ReduceKind.Sum, rule, deps=[combine.id])

ukernel.run([r, a2a, gemm, combine, allreduce])
