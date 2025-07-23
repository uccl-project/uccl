# Incast Benchmark

This benchmark is done on a local 2 HGX testbed each with 8xH100 GPUs and 8x400G CX-7 IB NICs. These two servers are under the same rack. 
Our results show severe interference between incast traffic and permutation traffic, see Figure 9 in our [technique report](https://arxiv.org/pdf/2504.17307). 

## Run benchmark

0. Clone this repo on a master node.
1. Build `uccl/rdma` per its [README](../../README.md). 
2. For Permutation Traffic, generate `matrix.txt` using `python gen_permutation_full_bisection.py matrix.txt 16 4`
3. Run `sync_repo.sh` to copy this repo to all nodes. It will also compile `uccl/rdma/incast`.
4. ​Run the test by executing:

```bash
./run.sh
```