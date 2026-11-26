#!/usr/bin/env python3
"""Add optional symmetric windows to the exact figure's NCCL-only harness."""
import difflib
import hashlib
from pathlib import Path
import subprocess

root = Path('/home/xinze/dev/pro6000-integrated-b0ac7b4a-20261115-022114-retry1/source/experimental/lite')
out = root / 'build/nccl_tuned_20261115'
out.mkdir(exist_ok=True)
source = root / 'benchmarks/joint/joint_bench.cu'
original = source.read_text()
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'f208e6e222f1286b5b64f648534b4f1f32ce783a835cfd3403d85122f11a2ed9'
text = original

def replace(before, after):
    global text
    assert text.count(before) == 1, before
    text = text.replace(before, after)

replace('#include "../cpu_router/nccl_identity.hpp"', '#include "nccl_identity.hpp"')
replace('  Buffers(int ranks, size_t bytes) {', '''  bool registered = std::getenv("UCCL_NCCL_REGISTER") != nullptr;
  size_t allocationBytes;
  Buffers(int ranks, size_t bytes) : allocationBytes(bytes + 4 * kGuard) {''')
replace('      ck(cudaMalloc(&a, bytes + 4 * kGuard));\n      ck(cudaMalloc(&b, bytes + 4 * kGuard));', '''      if (registered) {
        nk(ncclMemAlloc(reinterpret_cast<void **>(&a), allocationBytes));
        nk(ncclMemAlloc(reinterpret_cast<void **>(&b), allocationBytes));
      } else {
        ck(cudaMalloc(&a, allocationBytes));
        ck(cudaMalloc(&b, allocationBytes));
      }''')
replace('      cudaFree(in[r] - kGuard);\n      cudaFree(out[r] - kGuard);', '''      if (registered) {
        ncclMemFree(in[r] - kGuard);
        ncclMemFree(out[r] - kGuard);
      } else {
        cudaFree(in[r] - kGuard);
        cudaFree(out[r] - kGuard);
      }''')
replace('  std::vector<ncclComm_t> comm;\n  explicit Nccl',
        '  std::vector<ncclComm_t> comm;\n  std::vector<ncclWindow_t> inputWindows, outputWindows;\n  explicit Nccl')
replace('    nk(ncclCommInitAll(comm.data(), int(comm.size()), b.devices.data()));\n  }', '''    nk(ncclCommInitAll(comm.data(), int(comm.size()), b.devices.data()));
    if (b.registered) {
      inputWindows.resize(comm.size());
      outputWindows.resize(comm.size());
      nk(ncclGroupStart());
      for (size_t r = 0; r < comm.size(); ++r) {
        ck(cudaSetDevice(b.devices[r]));
        nk(ncclCommWindowRegister(comm[r], b.in[r] - kGuard, b.allocationBytes,
                                 &inputWindows[r], NCCL_WIN_COLL_SYMMETRIC));
        nk(ncclCommWindowRegister(comm[r], b.out[r] - kGuard, b.allocationBytes,
                                 &outputWindows[r], NCCL_WIN_COLL_SYMMETRIC));
      }
      nk(ncclGroupEnd());
      for (size_t r = 0; r < comm.size(); ++r)
        if (!inputWindows[r] || !outputWindows[r])
          throw std::runtime_error("symmetric window registration unavailable");
      std::cerr << "Symmetric windows registered on every rank\\n";
    }
  }''')
replace('  ~Nccl() {\n    for (auto c : comm)', '''  ~Nccl() {
    if (b.registered) {
      for (size_t r = 0; r < comm.size(); ++r) {
        cudaSetDevice(b.devices[r]);
        cudaStreamSynchronize(b.streams[r]);
        ncclCommWindowDeregister(comm[r], inputWindows[r]);
        ncclCommWindowDeregister(comm[r], outputWindows[r]);
      }
    }
    for (auto c : comm)''')
replace('Buffers b(cfg.gpus, std::max(cfg.end, size_t(4096)));',
        'Buffers b(cfg.gpus, std::max(cfg.end, size_t(1UL << 30)));')
(out / 'nccl_reference.cu').write_text(text)
(out / 'source.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
    fromfile='benchmarks/joint/joint_bench.cu', tofile='build/nccl_tuned_20261115/nccl_reference.cu')))
(out / 'probe.mk').write_text('''include Makefile
.PHONY: nccl-tuning-reference
nccl-tuning-reference: build/nccl_tuned_20261115/nccl_reference
build/nccl_tuned_20261115/nccl_reference: build/nccl_tuned_20261115/nccl_reference.cu
\t$(NVCC) $(NVCC_FLAGS) -std=c++17 -DNCCL_REFERENCE_ONLY -Ibenchmarks/cpu_router -I$(NCCL_HOME)/include $< -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -o $@
''')
cmd = ['make', '-f', str(out / 'probe.mk'), 'nccl-tuning-reference', 'CUDA_HOME=/opt/pytorch/cuda',
       'CUDA_ARCH=sm_120', 'EXTRA_NVCC_FLAGS=-L/opt/pytorch/cuda/lib',
       'NCCL_HOME=/home/xinze/dev/uccl-lite-spin-pro6000-20261113/build/experiment/deps/nccl']
with (out / 'build.log').open('w') as log:
    subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
print('Matched NCCL reference build PASS', flush=True)
