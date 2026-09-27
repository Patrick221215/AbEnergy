#!/usr/bin/env bash

###############################################################################
# ABX multi-GPU launcher
#
# 用 GPU=<phys_ids> bash train.sh <config.json | other CLI args>
#
# 逻辑：
#   1. 读取 GPU="4,6" ⇒ export CUDA_VISIBLE_DEVICES=4,6
#   2. 进程内部只看见逻辑索引 0,1 …，因此向 Python 传递 --gpus 0 1 …
#   3. 若单卡 ⇒ torchrun 不启动；多卡 ⇒ torchrun 启动并注入 LOCAL_RANK
###############################################################################
set -euo pipefail

# --------------- 用户输入 ---------------
GPU="${GPU:--1}"                 # GPU=0,2 或 GPU=-1(=CPU)
MASTER_ADDR="${ADDR:-localhost}"
MASTER_PORT="${PORT:-29534}"
export CUDA_VISIBLE_DEVICES=$GPU

# --------------- CLI / JSON 处理 ---------------
if [[ $# -lt 1 ]]; then
  echo "❌ 需提供 <config.json> 或其他 CLI 参数"; exit 1
fi

FIRST_ARG="$1"; shift
ARGS=""
if [[ "$FIRST_ARG" == *.json ]]; then
  [[ ! -f "$FIRST_ARG" ]] && { echo "❌ 找不到 $FIRST_ARG"; exit 1; }
  CONFIG_JSON="$FIRST_ARG"
  # JSON -> CLI
  ARGS=$(python - <<'PY' "$CONFIG_JSON"
import json, sys, shlex
cfg = json.load(open(sys.argv[1]))
args=[]
for k,v in cfg.items():
    if v is False or v is None: continue
    flag='--'+k
    if isinstance(v,bool):
        args.append(flag)
    elif isinstance(v,(list,tuple)):
        args.append(flag); args.extend(map(str,v))
    else:
        args.append(flag); args.append(str(v).strip())
print(" ".join(map(shlex.quote,args)))
PY
)
else
  # 直接透传其余 CLI
  ARGS="$FIRST_ARG $*"
fi

# --------------- GPU 处理 ---------------
IFS=',' read -ra _gpu_phys <<<"$GPU"
_world=${#_gpu_phys[@]}
if [[ $_world -gt 1 ]]; then
  GPU_CLI=$(seq 0 $(($_world-1)) | xargs)   # 0 … N-1
else
  GPU_CLI="0"
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES → pass --gpus $GPU_CLI"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Args: $ARGS"

# --------------- 启动 ---------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $_world -gt 1 ]]; then
  torchrun \
    --nproc_per_node=$_world \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    train_ema.py --use_ema --gpus $GPU_CLI $ARGS
else
  python train_ema.py --use_ema --gpus $GPU_CLI $ARGS
fi
