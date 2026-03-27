#!/usr/bin/env bash
# Used by AMD self-hosted CI: before GPU benchmarks, either find GPUs idle or
# clear prior uccl-ci-sandbox jobs only. Fails if non-CI processes hold the GPU.
#
# Env:
#   UCCL_CI_SANDBOX  (default: $HOME/uccl-ci-sandbox)

set -euo pipefail

SANDBOX="${UCCL_CI_SANDBOX:-$HOME/uccl-ci-sandbox}"
SANDBOX_ABS="$(readlink -f "$SANDBOX" 2>/dev/null || echo "$SANDBOX")"

collect_rocm_gpu_pids() {
  local out pid
  local -a _cands
  out="$(rocm-smi --showpids 2>/dev/null || true)"
  # Prefer explicit PID tokens when present (rocm-smi text output varies by version).
  if grep -qE 'PID[[:punct:][:space:]]+[0-9]+' <<<"$out"; then
    mapfile -t _cands < <(grep -oE 'PID[[:punct:][:space:]]+[0-9]+' <<<"$out" | grep -oE '[0-9]+$' | sort -nu | uniq)
    for pid in "${_cands[@]}"; do
      [[ -n "$pid" && -d "/proc/$pid" ]] || continue
      echo "$pid"
    done
    return
  fi
  # Fallback: long numeric tokens, keep only live PIDs (reduces stray matches in headers).
  mapfile -t _cands < <(grep -oE '\b[0-9]{4,}\b' <<<"$out" | sort -nu | uniq)
  if [[ ${#_cands[@]} -eq 0 ]]; then
    mapfile -t _cands < <(grep -oE '\b[0-9]{3,}\b' <<<"$out" | sort -nu | uniq)
  fi
  for pid in "${_cands[@]}"; do
    [[ -n "$pid" && -d "/proc/$pid" ]] || continue
    echo "$pid"
  done
}

pid_is_uccl_ci_sandbox() {
  local pid=$1
  local cwd exe cmd
  [[ -d "/proc/$pid" ]] || return 1
  cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
  cmd="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
  [[ -n "$cwd" ]] && { [[ "$cwd" == "$SANDBOX_ABS" ]] || [[ "$cwd" == "$SANDBOX_ABS"/* ]]; } && return 0
  [[ -n "$exe" && "$exe" == "$SANDBOX_ABS"/* ]] && return 0
  [[ "$cmd" == *"$SANDBOX_ABS"* ]] && return 0
  return 1
}

describe_pid() {
  local pid=$1
  local cwd exe
  cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || echo "?")"
  exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || echo "?")"
  echo "pid=$pid cwd=$cwd exe=$exe"
}

mapfile -t GPU_PIDS < <(collect_rocm_gpu_pids | sort -nu)
if [[ ${#GPU_PIDS[@]} -eq 0 ]]; then
  exit 0
fi

FOREIGN=()
for pid in "${GPU_PIDS[@]}"; do
  [[ -n "$pid" ]] || continue
  pid_is_uccl_ci_sandbox "$pid" || FOREIGN+=("$pid")
done

if [[ ${#FOREIGN[@]} -gt 0 ]]; then
  echo "GPU busy with processes outside $SANDBOX_ABS — refusing to kill:" >&2
  for pid in "${FOREIGN[@]}"; do
    describe_pid "$pid" >&2
  done
  exit 1
fi

echo "Clearing stale UCCL CI GPU PIDs: ${GPU_PIDS[*]}"

kill -TERM "${GPU_PIDS[@]}" 2>/dev/null || true
for _ in 1 2 3 4 5; do
  mapfile -t GPU_PIDS < <(collect_rocm_gpu_pids | sort -nu)
  [[ ${#GPU_PIDS[@]} -eq 0 ]] && exit 0
  sleep 1
done

kill -KILL "${GPU_PIDS[@]}" 2>/dev/null || true
sleep 2
mapfile -t GPU_PIDS < <(collect_rocm_gpu_pids | sort -nu)
if [[ ${#GPU_PIDS[@]} -eq 0 ]]; then
  exit 0
fi

# After SIGKILL, anything left is unexpected or foreign.
FOREIGN=()
for pid in "${GPU_PIDS[@]}"; do
  pid_is_uccl_ci_sandbox "$pid" || FOREIGN+=("$pid")
done
if [[ ${#FOREIGN[@]} -gt 0 ]]; then
  echo "GPU still busy with non-UCCL-CI processes:" >&2
  for pid in "${FOREIGN[@]}"; do
    describe_pid "$pid" >&2
  done
  exit 1
fi

echo "GPU still in use after kill; remaining PIDs: ${GPU_PIDS[*]}" >&2
for pid in "${GPU_PIDS[@]}"; do
  describe_pid "$pid" >&2
done
exit 1
