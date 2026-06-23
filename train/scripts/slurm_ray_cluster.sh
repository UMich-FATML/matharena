#!/usr/bin/env bash
# Shared Ray bootstrap helpers for CSCS Slurm jobs.

slurm_cleanup_ray_cluster() {
  set +e
  for pid in "${SLURM_RAY_SRUN_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}

slurm_start_gpu_util_log() {
  local expected_nodes="${1:?expected node count required}"
  case "${SLURM_GPU_UTIL_LOG:-auto}" in false|False|0|no|off) return 0 ;; esac
  export WANDB_RUN_ID="${WANDB_RUN_ID:-slurm-${SLURM_JOB_ID:-manual}}"
  export WANDB_RESUME="${WANDB_RESUME:-allow}"

  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY not set; cluster GPU utilization will not be logged to W&B."
    return 0
  fi

  local interval="${SLURM_GPU_UTIL_LOG_INTERVAL:-10}"
  local node_csv run_name log_file
  node_csv="$(IFS=,; echo "${SLURM_RAY_NODES[*]:0:${expected_nodes}}")"
  run_name="${EXPERIMENT_NAME:-ray-cluster}"
  log_file="${SLURM_GPU_UTIL_LOG_FILE:-train/logs/gpu-util-${SLURM_JOB_ID:-manual}.log}"
  mkdir -p "$(dirname "${log_file}")"
  echo "Logging cluster GPU utilization to W&B run id=${WANDB_RUN_ID} name=${run_name} every ${interval}s log=${log_file}"

  (
    set -o pipefail
    srun --overlap \
      --nodes="${expected_nodes}" \
      --ntasks="${expected_nodes}" \
      --ntasks-per-node=1 \
      -w "${node_csv}" \
      --mpi=pmix \
      --network=disable_rdzv_get \
      --environment=./train/env/cscs-verl.toml \
      bash -lc '''
        set -euo pipefail
        interval="${1:?interval required}"
        query="index,utilization.gpu,memory.used,memory.total,power.draw"
        while true; do
          ts=$(date -Is)
          node=$(hostname)
          nvidia-smi --query-gpu="${query}" --format=csv,noheader,nounits | sed "s/^/${ts},${node},/"
          sleep "${interval}"
        done
      ''' bash "${interval}" |
    srun --overlap \
      --nodes=1 \
      --ntasks=1 \
      --ntasks-per-node=1 \
      -w "${RAY_HEAD_NODE}" \
      --mpi=pmix \
      --network=disable_rdzv_get \
      --environment=./train/env/cscs-verl.toml \
      bash -lc '''
        set -euo pipefail
        source "${VENV_DIR}/bin/activate"
        if [[ -x "${VENV_DIR}/bin/python3" ]]; then
          "${VENV_DIR}/bin/python3" train/scripts/wandb_gpu_util_log.py --from-stdin "$@"
        else
          python3 train/scripts/wandb_gpu_util_log.py --from-stdin "$@"
        fi
      ''' bash \
      --nodes "${node_csv}" \
      --gpus-per-node "${SLURM_RAY_GPUS_PER_NODE}" \
      --trainer-nodes "${FULL_ASYNC_TRAINER_NNODES:-1}" \
      --interval "${interval}" \
      --project "${PROJECT_NAME:-matharena-arxivmath-grpo}" \
      --group "${EXPERIMENT_NAME:-ray-cluster}" \
      --name "${run_name}"
  ) >"${log_file}" 2>&1 &
  SLURM_RAY_SRUN_PIDS+=("$!")
}

slurm_prepare_ray_cluster_env() {
  local expected_nodes="${1:?expected node count required}"
  local gpus_per_node="${2:?GPU count per node required}"

  mapfile -t SLURM_RAY_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
  if (( expected_nodes < 1 || expected_nodes > ${#SLURM_RAY_NODES[@]} )); then
    echo "Invalid Ray node count: expected=${expected_nodes}, allocated=${#SLURM_RAY_NODES[@]}" >&2
    exit 1
  fi

  export RAY_HEAD_NODE="${SLURM_RAY_NODES[0]}"
  local head_node_ip
  head_node_ip="$(
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 -w "${RAY_HEAD_NODE}" \
      bash -lc 'hostname --ip-address | awk "{for (i=1; i<=NF; i++) if (\$i !~ /:/) {print \$i; exit}; print \$1}"'
  )"
  export RAY_HEAD_IP="${RAY_HEAD_IP:-${head_node_ip}}"
  export RAY_PORT="${RAY_PORT:-6379}"
  export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
  export RAY_ADDRESS="${RAY_HEAD_IP}:${RAY_PORT}"
  export SLURM_RAY_EXPECTED_NODES="${expected_nodes}"
  export SLURM_RAY_GPUS_PER_NODE="${gpus_per_node}"
}

slurm_wait_for_ray_head_ready() {
  local head_pid="${1:?head srun pid required}"
  local timeout="${RAY_HEAD_STARTUP_TIMEOUT:-180}"
  local deadline=$((SECONDS + timeout))

  while [[ ! -f "${SLURM_RAY_HEAD_READY_FILE}" ]]; do
    if ! kill -0 "${head_pid}" 2>/dev/null; then
      wait "${head_pid}" || true
      echo "Ray head step exited before becoming ready." >&2
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out after ${timeout}s waiting for Ray head readiness file: ${SLURM_RAY_HEAD_READY_FILE}" >&2
      exit 1
    fi
    sleep 2
  done
}

slurm_run_with_ray_cluster() {
  local expected_nodes="${1:?expected node count required}"
  local gpus_per_node="${2:?GPU count per node required}"
  local label="${3:-Ray cluster}"
  local driver_command="${4:?driver command required}"

  slurm_prepare_ray_cluster_env "${expected_nodes}" "${gpus_per_node}"

  echo "Starting ${label}: head=${RAY_HEAD_NODE} address=${RAY_ADDRESS} nodes=${expected_nodes} gpus_per_node=${gpus_per_node}"

  SLURM_RAY_SRUN_PIDS=()
  trap slurm_cleanup_ray_cluster EXIT
  slurm_start_gpu_util_log "${expected_nodes}"
  export SLURM_RAY_HEAD_READY_FILE="${SLURM_RAY_HEAD_READY_FILE:-${PWD}/train/logs/ray-head-${SLURM_JOB_ID}.ready}"
  export SLURM_RAY_DRIVER_COMMAND="${driver_command}"
  rm -f "${SLURM_RAY_HEAD_READY_FILE}"

  srun --overlap \
    --nodes=1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    -w "${RAY_HEAD_NODE}" \
    --mpi=pmix \
    --network=disable_rdzv_get \
    --environment=./train/env/cscs-verl.toml \
    bash -lc '
      set -euo pipefail
      ulimit -c 0 || true
      source "${VENV_DIR}/bin/activate"
      unset VLLM_USE_V1
      unset VLLM_VERSION VLLM_FLASH_ATTN_SRC_DIR VLLM_ATTENTION_BACKEND
      unset HIP_VISIBLE_DEVICES
      unset ROCR_VISIBLE_DEVICES
      export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
      export TORCH_CUDNN_V8_API_DISABLED="${TORCH_CUDNN_V8_API_DISABLED:-1}"
      if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
        echo "Unsetting PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}; vLLM memory pool is incompatible with expandable_segments." >&2
        unset PYTORCH_CUDA_ALLOC_CONF
      fi
      ray stop --force >/dev/null 2>&1 || true
      ray start --head \
        --node-ip-address="${RAY_HEAD_IP}" \
        --port="${RAY_PORT}" \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${RAY_DASHBOARD_PORT}" \
        --temp-dir="${RAY_TMPDIR}" \
        --num-cpus="${SLURM_CPUS_PER_TASK}" \
        --num-gpus="${SLURM_RAY_GPUS_PER_NODE}" \
        --disable-usage-stats
      touch "${SLURM_RAY_HEAD_READY_FILE}"
      python -c '"'"'
import os
import time
import ray

expected = int(os.environ["SLURM_RAY_EXPECTED_NODES"])
deadline = time.time() + int(os.environ.get("RAY_CLUSTER_STARTUP_TIMEOUT", "300"))
last = None
while time.time() < deadline:
    try:
        ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True)
        alive = [n for n in ray.nodes() if n["Alive"]]
        last = len(alive)
        print(f"Ray nodes alive: {last}/{expected}", flush=True)
        if last >= expected:
            ray.shutdown()
            break
        ray.shutdown()
    except Exception as exc:
        last = repr(exc)
        print(f"Waiting for Ray cluster: {last}", flush=True)
    time.sleep(5)
else:
    raise SystemExit(f"Timed out waiting for Ray nodes. Last status: {last}")
'"'"'
      status=0
      eval "${SLURM_RAY_DRIVER_COMMAND}" || status=$?
      ray stop --force >/dev/null 2>&1 || true
      exit "${status}"
    ' &
  local head_pid="$!"
  SLURM_RAY_SRUN_PIDS+=("${head_pid}")

  slurm_wait_for_ray_head_ready "${head_pid}"

  for ((i = 1; i < expected_nodes; i++)); do
    local node_i="${SLURM_RAY_NODES[$i]}"
    echo "Starting Ray worker on ${node_i}"
    srun --overlap \
      --nodes=1 \
      --ntasks=1 \
      --ntasks-per-node=1 \
      -w "${node_i}" \
      --mpi=pmix \
      --network=disable_rdzv_get \
      --environment=./train/env/cscs-verl.toml \
      bash -lc '
        set -euo pipefail
        ulimit -c 0 || true
        source "${VENV_DIR}/bin/activate"
        unset VLLM_USE_V1
        unset VLLM_VERSION VLLM_FLASH_ATTN_SRC_DIR VLLM_ATTENTION_BACKEND
        unset HIP_VISIBLE_DEVICES
        unset ROCR_VISIBLE_DEVICES
        export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
        export TORCH_CUDNN_V8_API_DISABLED="${TORCH_CUDNN_V8_API_DISABLED:-1}"
        if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
          echo "Unsetting PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}; vLLM memory pool is incompatible with expandable_segments." >&2
          unset PYTORCH_CUDA_ALLOC_CONF
        fi
        ray stop --force >/dev/null 2>&1 || true
        ray start --address="${RAY_ADDRESS}" \
          --temp-dir="${RAY_TMPDIR}" \
          --num-cpus="${SLURM_CPUS_PER_TASK}" \
          --num-gpus="${SLURM_RAY_GPUS_PER_NODE}" \
          --disable-usage-stats \
          --block
      ' &
    SLURM_RAY_SRUN_PIDS+=("$!")
  done

  wait "${head_pid}"
}
