#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PRIMUS_PATH=${PRIMUS_PATH:-"${script_dir}/../../../thirdparty/Primus"}
export MODEL_NAME=${MODEL_NAME:-deepseek_v2_lite}
export GPU_ARCH=${GPU_ARCH:-"MI300X"}
export EXP="examples/megatron/configs/${GPU_ARCH}/${MODEL_NAME}-BF16-pretrain.yaml"
###################### Training Docker and Variables ##########################
export DOCKER_IMAGE=${DOCKER_IMAGE:-docker.io/rocm/primus:v25.9_gfx942}
export HF_TOKEN=${HF_TOKEN:-"your_hf_token"}

######################## Cluster Settings ###############################
export NNODES=${NNODES:-1}
export NODE_LISTS=${NODE_LISTS:-""}
export NCCL_IB_HCA=${NCCL_IB_HCA:-}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}

export USING_BNXT=${USING_BNXT:-0}
export PATH_TO_BNXT_TAR_PACKAGE=${PATH_TO_BNXT_TAR_PACKAGE:-}


########################### Training Config ###################################
export MBS=${MBS:-8}
export GBS=${GBS:-256}
export TP=${TP:-1}
export PP=${PP:-1}
export EP=${EP:-1}
export ETP=${ETP:-1}
export CP=${CP:-1}
export VPP=${VPP:-1}
export SEQ_LENGTH=${SEQ_LENGTH:-4096}
export LOG_AVG_SKIP_ITERATIONS=${LOG_AVG_SKIP_ITERATIONS:-2}
export RECOMPUTE_LAYERS=${RECOMPUTE_LAYERS:-0}
export LEGACY_GG=${LEGACY_GG:-False}
export TRAIN_ITERS=${TRAIN_ITERS:-10}
export VPP=${VPP:-1}
export PROFILE=${PROFILE:-False}
export USE_UCCL=${USE_UCCL:-0}
export UCCL_REF=${UCCL_REF:-main}


# Optional pipeline layout: if PIPELINE_LAYOUT is set externally, pass it through;
# otherwise do not configure pipeline_model_parallel_layout at all.
PIPELINE_LAYOUT=${PIPELINE_LAYOUT:-""}

########################### Feature Config ###################################

# Primus Turbo performance optimization: enable turbo attention and grouped_mlp for better throughput
# need to install Primus-Turbo from https://github.com/AMD-AGI/Primus-Turbo
FEATURE_ARGS=(
    --enable_primus_turbo "False"
    --use_turbo_attention "False"
    --use_turbo_grouped_mlp "False"
	--use_turbo_deepep "False"
	--turbo_deepep_num_cu "32"
	--turbo_sync_free_moe_stage "0"
)

if [ -n "$PIPELINE_LAYOUT" ]; then
	FEATURE_ARGS+=("--pipeline_model_parallel_layout" "$PIPELINE_LAYOUT")
elif [ "$VPP" -gt 1 ]; then
	FEATURE_ARGS+=("--num_virtual_stages_per_pipeline_rank" "$VPP")
fi

###################### Training Launch Config #################################
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1

export NUMA_BINDING=${NUMA_BINDING:-False}
if [ "$NUMA_BINDING" = "True" ]; then
	export ENABLE_NUMA_BINDING=1
	export HSA_KERNARG_POOL_SIZE=12582912
fi

####################### Training Experiments ##################################
export PRIMUS_TEAM="date-$(date +%Y%m%d)"
export PRIMUS_EXP_NAME="${MODEL_NAME}_${GPU_ARCH}_NNODES${NNODES}_MBS${MBS}_GBS${GBS}_TP${TP}_PP${PP}_VPP${VPP}_EP${EP}_ETP${ETP}_CP${CP}"

LOG_DIR=${script_dir}/output/$PRIMUS_TEAM/$PRIMUS_EXP_NAME
export LOG_FILE=$LOG_DIR/training.log
export EXPORT_CONFIG=$LOG_DIR/config.yaml
mkdir -p "$LOG_DIR"

########################## Training Job #######################################

run_primus_cli() {
	pushd "${PRIMUS_PATH}"

	bash runner/primus-cli slurm \
		-N "${NNODES}" \
		--nodelist "${NODE_LISTS}" \
		-- --image "${DOCKER_IMAGE}" --clean \
		--env "REBUILD_BNXT=${USING_BNXT}" \
		--env "PATH_TO_BNXT_TAR_PACKAGE=${PATH_TO_BNXT_TAR_PACKAGE}" \
		--env "REBUILD_UCCL=${USE_UCCL}" \
		--env "UCCL_REF=${UCCL_REF}" \
		--env "NCCL_IB_HCA=${NCCL_IB_HCA}" \
		--env "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}" \
		--env "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}" \
		--env "GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}" \
		--env "UCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}" \
		--env "UCCL_IB_HCA=${NCCL_IB_HCA}" \
		--env "UCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}" \
		--volume $script_dir:$script_dir \
		"$@"

	popd
}


run_primus_cli -- --log_file "$LOG_FILE"  train pretrain \
	--config "${EXP}" \
	--micro_batch_size "$MBS" \
	--global_batch_size "$GBS" \
	--seq_length "$SEQ_LENGTH" \
	--tensor_model_parallel_size "$TP" \
	--pipeline_model_parallel_size "$PP" \
	--disable_primus_topk_router "True" \
	--pp_warmup "True" \
	--num_virtual_stages_per_pipeline_rank "$VPP" \
	--expert_model_parallel_size "$EP" \
	--expert_tensor_parallel_size "$ETP" \
	--context_parallel_size "$CP" \
	--moe_use_legacy_grouped_gemm "$LEGACY_GG" \
	--recompute_granularity "full" \
	--recompute_method "block" \
	--recompute_num_layers "${RECOMPUTE_LAYERS}" \
	--cross_entropy_fusion_impl "te" \
	--cross_entropy_loss_fusion "True" \
	--log_avg_skip_iterations "$LOG_AVG_SKIP_ITERATIONS" \
	--profile "${PROFILE}" \
	--use_pytorch_profiler "${PROFILE}" \
	--profile_step_start 3 \
	--profile_step_end 5 \
	"${FEATURE_ARGS[@]}" \
	--train_iters "$TRAIN_ITERS" "$@"
