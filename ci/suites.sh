#!/usr/bin/env bash
set -euo pipefail
suite=${1:?suite required}
name=${2:-}
if [[ $CI_TARGET == cpu && ( $suite == scalar-host || $suite == sack-host ) ]]; then
  sdk="$CI_DEPS/host-cuda-12.8/nvidia"
  export CUDA_HOME="$sdk/cuda_runtime"
  export CPATH="$sdk/cuda_nvcc/include:$sdk/cuda_cccl/include${CPATH:+:$CPATH}"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
case $suite in
  sack-host)
    make -C collective/tests -j"$CI_JOBS" CUDA_HOME="$CUDA_HOME" test
    ;;
  rdma-test)
    make -C collective/rdma -j"$CI_JOBS" "${name}_test"
    args=()
    [[ $name != pcie ]] || args=(0)
    [[ $name != flush ]] || args=(--gpu)
    exec "collective/rdma/${name}_test" "${args[@]}"
    ;;
  ukernel-ccl)
    make -C experimental/ukernel/src/ccl -j"$CI_JOBS" SM="$SM" "$name"
    if [[ $CI_NODE == 0 ]]; then role=server; else role=client; fi
    exec "experimental/ukernel/src/ccl/$name" --role "$role" --gpu 0 \
      --exchanger-ip "$MASTER_ADDR" --exchanger-port "$MASTER_PORT" --transport rdma
    ;;
  ep-rb)
    CPATH="$CI_REPO/include${CPATH:+:$CPATH}" \
      make -C ep/bench/rb -j"$CI_JOBS" SM="$SM" "test_${name}_throughput"
    exec "ep/bench/rb/test_${name}_throughput"
    ;;
  lite-ep)
    cd experimental/lite/lite-ep
    export PYTHONPATH="$PWD"
    export LITE_EP_TRANSPORT=uccl-no-gdr LITE_EP_NVLINK=0 EP_SUPPRESS_NCCL_CHECK=1
    export DISABLE_SM90_FEATURES=1 NCCL_GIN_TYPE=2
    export LD_LIBRARY_PATH="$NCCL_ROOT/lib:$LD_LIBRARY_PATH"
    export LD_PRELOAD="$NCCL_ROOT/lib/libnccl.so.2"
    args=(--num-processes 2)
    case $name in
      ep) args+=(--num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 --test-first-only --do-handle-copy-modes=0 --expert-alignment-modes=128 --fp8-dispatch-modes=0 --num-bench-tests=2);;
      barrier) args+=(--allow-hybrid-mode=0);;
      engram) args+=(--num-entries=4096 --num-tokens=128);;
      pp) args+=(--num-tokens=128 --num-sends=4 --num-stress-iterations=1);;
      agrs) args+=(--num-stress-iterations=1);;
    esac
    exec python3 "tests/elastic/test_${name}.py" "${args[@]}"
    ;;
  scalar-host)
    family=${name%%.*}; binary=${name#*.}
    extra=()
    [[ $binary != util_lrpc ]] || extra=("collective/$family/util_lrpc.cc")
    g++ -O2 -std=c++17 -pthread -DUSE_CUDA -Iinclude -I"collective/$family" \
      -I"$CUDA_HOME/include" "collective/$family/${binary}_test.cc" "${extra[@]}" \
      -L"$CUDA_HOME/lib64" -lcudart -lgtest -lgflags -libverbs -lz -lelf \
      -o "$CI_ARTIFACTS/${family}-${binary}"
    exec "$CI_ARTIFACTS/${family}-${binary}"
    ;;
  compression)
    g++ -O2 -std=c++17 -DUSE_CUDA -DUSE_DIETGPU -Ip2p/rdma -Ip2p/util \
      -Ip2p -Iinclude -Ithirdparty/dietgpu -I"$CUDA_HOME/include" \
      p2p/rdma/tests/test_compression.cpp -L"$CUDA_HOME/lib64" -lcudart -lcuda \
      -Lthirdparty/dietgpu/dietgpu/float -ldietgpu_float \
      -Wl,-rpath,"$CI_REPO/thirdparty/dietgpu/dietgpu/float" \
      -libverbs -lgflags -lpthread -o "$CI_ARTIFACTS/compression"
    for type in fp16 bf16 fp32; do "$CI_ARTIFACTS/compression" --float_type="$type" --iterations=2; done
    ;;
  *) echo "Unknown suite: $suite" >&2; exit 2;;
esac
