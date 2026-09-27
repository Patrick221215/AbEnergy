set -euo pipefail

# --------------- 用户输入 ---------------
GPU="${GPU:--1}"                 # 例如 GPU=1,3
MASTER_ADDR="${ADDR:-127.0.0.1}" # 建议固定 127.0.0.1，少踩坑
MASTER_PORT="${PORT:-29525}"     # 建议每次手动换一个空闲端口
export CUDA_VISIBLE_DEVICES="$GPU"

# --------------- CLI / JSON 处理 ---------------
if [[ $# -lt 1 ]]; then
  echo "❌ 需提供 <config.json> 或其他 CLI 参数"
  exit 1
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
    if v is False or v is None:
        continue
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
  GPU_CLI=$(seq 0 $(($_world-1)) | xargs)   # 0 … N-1 (对应 CUDA_VISIBLE_DEVICES 映射)
else
  GPU_CLI="0"
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES → pass --gpu_list $GPU_CLI"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Args: $ARGS"

# --------------- 启动 ---------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 关键：你们 inference.py 内部已经 mp.spawn 了，
# 这里再 torchrun 会二次 DDP，必炸端口。
# 所以统一直接 python 启动，并显式给 MASTER_ADDR/PORT。
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="$MASTER_PORT"

python inference.py --gpu_list $GPU_CLI $ARGS
