#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

LLAMAFACTORY_DIR=${LLAMAFACTORY_DIR:-"$HOME/LlamaFactory"}
MODEL_PATH=${MODEL_PATH:-"$HOME/models/Qwen3-8B-Base"}
RESULT_ROOT=${RESULT_ROOT:-"$SCRIPT_DIR/evaluation/qwen3-8b-base-recipe"}
BEST_MODEL_FILE=${BEST_MODEL_FILE:-"$RESULT_ROOT/best_checkpoint.txt"}
ADAPTER_PATH=${ADAPTER_PATH:-}
EXPORT_DIR=${EXPORT_DIR:-"$SCRIPT_DIR/exported_models/qwen3-8b-base-recipe"}
MERGE_DEVICE=${MERGE_DEVICE:-auto}
MIN_FREE_GIB=${MIN_FREE_GIB:-20}
GPU_IDS=${GPU_IDS:-0}
DRY_RUN=${DRY_RUN:-0}

MERGE_CONFIG="$SCRIPT_DIR/configs/recipe_qwen3_8b_base_lora_merge.yaml"

die() {
  printf '错误：%s\n' "$*" >&2
  exit 1
}

[[ -d "$LLAMAFACTORY_DIR" ]] || die "找不到 LlamaFactory：$LLAMAFACTORY_DIR"
LLAMAFACTORY_DIR=$(cd -- "$LLAMAFACTORY_DIR" && pwd)
[[ -r "$MERGE_CONFIG" ]] || die "找不到合并配置：$MERGE_CONFIG"
command -v llamafactory-cli >/dev/null 2>&1 \
  || die "找不到 llamafactory-cli，请先激活 LlamaFactory 环境"

[[ -d "$MODEL_PATH" ]] || die "找不到未量化基础模型：$MODEL_PATH"
MODEL_PATH=$(cd -- "$MODEL_PATH" && pwd)
python3 "$SCRIPT_DIR/verify_model.py" "$MODEL_PATH" --quiet \
  || die "基础模型不完整或不是 Qwen3-8B-Base：$MODEL_PATH"

if [[ -z "$ADAPTER_PATH" ]]; then
  [[ -s "$BEST_MODEL_FILE" ]] \
    || die "找不到最佳模型记录；请先完成评估，或通过 ADAPTER_PATH 指定适配器"
  IFS= read -r ADAPTER_PATH < "$BEST_MODEL_FILE"
fi
[[ -d "$ADAPTER_PATH" ]] || die "找不到适配器目录：$ADAPTER_PATH"
ADAPTER_PATH=$(cd -- "$ADAPTER_PATH" && pwd)
[[ -r "$ADAPTER_PATH/adapter_config.json" ]] \
  || die "适配器目录缺少 adapter_config.json：$ADAPTER_PATH"

case "$MERGE_DEVICE" in
  auto|cpu) ;;
  *) die "MERGE_DEVICE 只支持 auto 或 cpu" ;;
esac
[[ "$MIN_FREE_GIB" =~ ^[1-9][0-9]*$ ]] || die "MIN_FREE_GIB 必须是正整数"

EXPORT_PARENT=$(dirname -- "$EXPORT_DIR")
EXPORT_NAME=$(basename -- "$EXPORT_DIR")
mkdir -p -- "$EXPORT_PARENT"
EXPORT_PARENT=$(cd -- "$EXPORT_PARENT" && pwd)
EXPORT_DIR="$EXPORT_PARENT/$EXPORT_NAME"
if [[ -d "$EXPORT_DIR" && -n "$(find "$EXPORT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  die "导出目录非空，为避免覆盖已停止：$EXPORT_DIR；请换一个 EXPORT_DIR"
fi

available_kib=$(df -Pk "$EXPORT_PARENT" | awk 'NR == 2 {print $4}')
required_kib=$((MIN_FREE_GIB * 1024 * 1024))
[[ "$available_kib" =~ ^[0-9]+$ ]] || die "无法读取导出目录的可用空间"
(( available_kib >= required_kib )) \
  || die "合并导出前至少需要 ${MIN_FREE_GIB}GiB 可用空间：$EXPORT_PARENT"

USES_DORA=$(python3 -c \
  'import json,sys; print("true" if json.load(open(sys.argv[1], encoding="utf-8")).get("use_dora") else "false")' \
  "$ADAPTER_PATH/adapter_config.json")

declare -a command=(
  llamafactory-cli export "$MERGE_CONFIG"
  "model_name_or_path=$MODEL_PATH"
  "adapter_name_or_path=$ADAPTER_PATH"
  "template=qwen3_nothink"
  "export_dir=$EXPORT_DIR"
  "export_device=$MERGE_DEVICE"
)

printf '\n========== 最佳模型合并配置 ==========\n'
printf '基础模型：%s\n' "$MODEL_PATH"
printf '适配器：%s\n' "$ADAPTER_PATH"
printf 'DoRA：%s\n' "$USES_DORA"
printf '导出设备：%s\n' "$MERGE_DEVICE"
printf '导出目录：%s\n' "$EXPORT_DIR"
printf '======================================\n\n'

if [[ "$DRY_RUN" == 1 ]]; then
  printf '即将执行：\ncd %q\nCUDA_VISIBLE_DEVICES=%q ' "$LLAMAFACTORY_DIR" "$GPU_IDS"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

cd -- "$LLAMAFACTORY_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
"${command[@]}"

python3 "$SCRIPT_DIR/verify_model.py" "$EXPORT_DIR" \
  || die "导出完成，但最终模型完整性校验失败：$EXPORT_DIR"
printf '\n最终合并模型已生成：%s\n' "$EXPORT_DIR"
