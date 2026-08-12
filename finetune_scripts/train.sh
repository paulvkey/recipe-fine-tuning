#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

LLAMAFACTORY_DIR=${LLAMAFACTORY_DIR:-"$HOME/LlamaFactory"}
DATA_FILE=${DATA_FILE:-"$PROJECT_ROOT/training_sample/recipe_train_sample_100000.jsonl"}
MODEL_PATH=${MODEL_PATH:-"$HOME/models/Qwen3-8B-Base"}
GPU_IDS=${GPU_IDS:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"$SCRIPT_DIR/outputs/qwen3-8b-base/recipe-lora-sft"}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-auto}
USE_DORA=${USE_DORA:-0}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-0}
DRY_RUN=${DRY_RUN:-0}
TRAIN_RUN_MODE=${TRAIN_RUN_MODE:-background}
TRAIN_BACKGROUND_WORKER=${TRAIN_BACKGROUND_WORKER:-0}
TRAIN_LOG_DIR=${TRAIN_LOG_DIR:-"$SCRIPT_DIR/logs"}
TRAIN_LOG_FILE=${TRAIN_LOG_FILE:-}
TRAIN_PID_FILE=${TRAIN_PID_FILE:-"$TRAIN_LOG_DIR/train.pid"}

DATA_DIR="$SCRIPT_DIR/data"
DATA_LINK="$DATA_DIR/recipe_train_sample.jsonl"

die() {
  printf '错误：%s\n' "$*" >&2
  exit 1
}

normalize_bool() {
  case "$1" in
    1|true|TRUE|yes|YES) printf 'true' ;;
    0|false|FALSE|no|NO) printf 'false' ;;
    *) die "$2 只支持 0/1、true/false 或 yes/no" ;;
  esac
}

case "$TRAIN_RUN_MODE" in
  background|foreground) ;;
  *) die "TRAIN_RUN_MODE 只支持 background 或 foreground" ;;
esac

CONFIG_FILE="$SCRIPT_DIR/configs/recipe_qwen3_8b_base_lora_sft.yaml"

[[ -d "$LLAMAFACTORY_DIR" ]] || die "找不到 LlamaFactory 安装目录：$LLAMAFACTORY_DIR"
LLAMAFACTORY_DIR=$(cd -- "$LLAMAFACTORY_DIR" && pwd)
[[ -r "$CONFIG_FILE" ]] || die "找不到训练配置：$CONFIG_FILE"
[[ -r "$DATA_DIR/dataset_info.json" ]] || die "找不到数据集注册文件：$DATA_DIR/dataset_info.json"
grep -Rqs -- 'qwen3_nothink' "$LLAMAFACTORY_DIR/src/llamafactory" \
  || die "当前 LlamaFactory 源码不包含 qwen3_nothink，请更新 $LLAMAFACTORY_DIR 后重新安装"
[[ -f "$DATA_FILE" ]] || die "找不到训练数据：${DATA_FILE}；可通过 DATA_FILE 指定服务器上的实际路径"
DATA_FILE=$(cd -- "$(dirname -- "$DATA_FILE")" && pwd)/$(basename -- "$DATA_FILE")
if [[ "$MODEL_PATH" == /* && ! -d "$MODEL_PATH" ]]; then
  die "找不到本地模型：${MODEL_PATH}；请先运行 bash finetune_scripts/download_model.sh"
fi
if [[ -d "$MODEL_PATH" ]]; then
  MODEL_PATH=$(cd -- "$MODEL_PATH" && pwd)
fi
if [[ "$MODEL_PATH" == /* ]] && ! python3 "$SCRIPT_DIR/verify_model.py" "$MODEL_PATH" --quiet; then
  python3 "$SCRIPT_DIR/verify_model.py" "$MODEL_PATH" || true
  die "本地模型文件不完整，请重新运行下载脚本"
fi
command -v llamafactory-cli >/dev/null 2>&1 \
  || die "找不到 llamafactory-cli，请先执行：conda activate llamafactory"

USE_DORA_BOOL=$(normalize_bool "$USE_DORA" USE_DORA)
GRADIENT_CHECKPOINTING_BOOL=$(normalize_bool "$GRADIENT_CHECKPOINTING" GRADIENT_CHECKPOINTING)
if [[ "$GRADIENT_CHECKPOINTING_BOOL" == true ]]; then
  DISABLE_GRADIENT_CHECKPOINTING_BOOL=false
else
  DISABLE_GRADIENT_CHECKPOINTING_BOOL=true
fi

case "$ATTENTION_BACKEND" in
  auto)
    if python3 -c 'import flash_attn' >/dev/null 2>&1; then
      ATTENTION_BACKEND=fa2
    else
      ATTENTION_BACKEND=sdpa
    fi
    ;;
  fa2|sdpa|disabled) ;;
  *) die "ATTENTION_BACKEND 只支持 auto、fa2、sdpa 或 disabled" ;;
esac

if [[ "$ATTENTION_BACKEND" == fa2 ]] && ! python3 -c 'import flash_attn' >/dev/null 2>&1; then
  die "指定了 ATTENTION_BACKEND=fa2，但当前环境无法导入 flash_attn"
fi

mkdir -p -- "$DATA_DIR" "$OUTPUT_DIR"
OUTPUT_DIR=$(cd -- "$OUTPUT_DIR" && pwd)
if [[ -e "$DATA_LINK" && ! -L "$DATA_LINK" ]]; then
  die "数据链接位置已存在普通文件，为避免覆盖已停止：$DATA_LINK"
fi
ln -sfn -- "$DATA_FILE" "$DATA_LINK"

declare -a overrides=(
  "model_name_or_path=$MODEL_PATH"
  "dataset_dir=$DATA_DIR"
  "output_dir=$OUTPUT_DIR"
  "flash_attn=$ATTENTION_BACKEND"
  "use_dora=$USE_DORA_BOOL"
  "disable_gradient_checkpointing=$DISABLE_GRADIENT_CHECKPOINTING_BOOL"
)

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  [[ -d "$RESUME_FROM_CHECKPOINT" ]] \
    || die "找不到断点目录：$RESUME_FROM_CHECKPOINT"
  RESUME_FROM_CHECKPOINT=$(cd -- "$RESUME_FROM_CHECKPOINT" && pwd)
  [[ -r "$RESUME_FROM_CHECKPOINT/adapter_config.json" ]] \
    || die "断点目录缺少 adapter_config.json：$RESUME_FROM_CHECKPOINT"
  CHECKPOINT_USES_DORA=$(python3 -c \
    'import json,sys; print("true" if json.load(open(sys.argv[1], encoding="utf-8")).get("use_dora") else "false")' \
    "$RESUME_FROM_CHECKPOINT/adapter_config.json")
  [[ "$CHECKPOINT_USES_DORA" == "$USE_DORA_BOOL" ]] \
    || die "断点的 DoRA=$CHECKPOINT_USES_DORA，与当前 USE_DORA=$USE_DORA_BOOL 不一致"
  overrides+=("resume_from_checkpoint=$RESUME_FROM_CHECKPOINT")
fi

printf '\n========== LlamaFactory 微调配置 ==========\n'
printf '安装目录：%s\n' "$LLAMAFACTORY_DIR"
if [[ "$USE_DORA_BOOL" == true ]]; then
  printf '训练方式：LoRA + LoRA+ + DoRA（实验模式）\n'
else
  printf '训练方式：LoRA + LoRA+\n'
fi
printf '模型：%s\n' "$MODEL_PATH"
printf '模板：qwen3_nothink\n'
printf '训练数据：%s\n' "$DATA_FILE"
printf 'GPU：%s\n' "$GPU_IDS"
printf '精度：BF16\n'
printf 'Attention 后端：%s\n' "$ATTENTION_BACKEND"
printf 'DoRA：%s\n' "$USE_DORA_BOOL"
printf '梯度检查点：%s\n' "$GRADIENT_CHECKPOINTING_BOOL"
printf '单卡 batch size：4\n'
printf '梯度累积：4\n'
printf '单卡有效 batch size：16\n'
printf '验证 batch size：8\n'
printf '截断长度：2048\n'
printf '日志间隔：100 steps\n'
printf '保存间隔：1400 steps\n'
printf 'Warmup：1000 steps\n'
printf '评估频率：1400 steps\n'
printf '输出目录：%s\n' "$OUTPUT_DIR"
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  printf '恢复断点：%s\n' "$RESUME_FROM_CHECKPOINT"
fi
printf '===========================================\n\n'

declare -a command=(llamafactory-cli train "$CONFIG_FILE" "${overrides[@]}")
if [[ "$DRY_RUN" == 1 ]]; then
  printf '即将执行：\ncd %q\nCUDA_VISIBLE_DEVICES=%q ' "$LLAMAFACTORY_DIR" "$GPU_IDS"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

# 正式训练默认脱离当前终端；后台 worker 本身保持前台执行。
if [[ "$TRAIN_RUN_MODE" == background && "$TRAIN_BACKGROUND_WORKER" != 1 ]]; then
  mkdir -p -- "$TRAIN_LOG_DIR"
  TRAIN_LOG_DIR=$(cd -- "$TRAIN_LOG_DIR" && pwd)
  if [[ -z "$TRAIN_LOG_FILE" ]]; then
    TRAIN_LOG_FILE="$TRAIN_LOG_DIR/train_$(date '+%Y%m%d_%H%M%S').log"
  elif [[ "$TRAIN_LOG_FILE" != /* ]]; then
    TRAIN_LOG_FILE="$PWD/$TRAIN_LOG_FILE"
  fi
  if [[ "$TRAIN_PID_FILE" != /* ]]; then
    TRAIN_PID_FILE="$PWD/$TRAIN_PID_FILE"
  fi
  mkdir -p -- "$(dirname -- "$TRAIN_LOG_FILE")" "$(dirname -- "$TRAIN_PID_FILE")"

  if [[ -s "$TRAIN_PID_FILE" ]]; then
    IFS= read -r existing_pid < "$TRAIN_PID_FILE" || true
    if [[ "$existing_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
      die "检测到仍在运行的训练进程 PID=$existing_pid；PID 文件：$TRAIN_PID_FILE"
    fi
  fi

  declare -a worker_command=(
    env
    TRAIN_BACKGROUND_WORKER=1
    TRAIN_RUN_MODE=foreground
    TRAIN_LOG_DIR="$TRAIN_LOG_DIR"
    TRAIN_LOG_FILE="$TRAIN_LOG_FILE"
    TRAIN_PID_FILE="$TRAIN_PID_FILE"
    bash "$SCRIPT_DIR/train.sh"
  )
  if command -v setsid >/dev/null 2>&1; then
    worker_command=(setsid "${worker_command[@]}")
  fi

  nohup "${worker_command[@]}" </dev/null >>"$TRAIN_LOG_FILE" 2>&1 &
  background_pid=$!
  printf '%s\n' "$background_pid" > "$TRAIN_PID_FILE"

  printf '\n训练已在后台启动。\n'
  printf 'PID：%s\n' "$background_pid"
  printf 'PID 文件：%s\n' "$TRAIN_PID_FILE"
  printf '日志文件：%s\n' "$TRAIN_LOG_FILE"
  printf '查看日志：tail -f %q\n' "$TRAIN_LOG_FILE"
  printf '查看进程：ps -p %q -o pid,etime,stat,cmd\n\n' "$background_pid"
  exit 0
fi

cd -- "$LLAMAFACTORY_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

printf 'LlamaFactory 版本：'
llamafactory-cli version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.free \
    --format=csv,noheader,nounits || true
fi

exec "${command[@]}"
