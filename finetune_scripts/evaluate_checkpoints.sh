#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

LLAMAFACTORY_DIR=${LLAMAFACTORY_DIR:-"$HOME/LlamaFactory"}
MODEL_PATH=${MODEL_PATH:-"$HOME/models/Qwen3-8B-Base"}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"$SCRIPT_DIR/outputs/qwen3-8b-base/recipe-lora-sft"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-}
RESULT_ROOT=${RESULT_ROOT:-"$SCRIPT_DIR/evaluation/qwen3-8b-base-recipe"}
GPU_IDS=${GPU_IDS:-0}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-1000}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
VLLM_BATCH_SIZE=${VLLM_BATCH_SIZE:-128}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.88}
MERGE_DEVICE=${MERGE_DEVICE:-auto}
MIN_MERGE_FREE_GIB=${MIN_MERGE_FREE_GIB:-20}
INCLUDE_BASE_MODEL=${INCLUDE_BASE_MODEL:-1}
INCLUDE_FINAL_MODEL=${INCLUDE_FINAL_MODEL:-1}
FORCE_EVAL=${FORCE_EVAL:-0}
DRY_RUN=${DRY_RUN:-0}

DATA_DIR="$SCRIPT_DIR/data"
EVAL_DATA_FILE="$DATA_DIR/recipe_eval_holdout.jsonl"
DATASET_NAME=recipe_eval_holdout
VLLM_SCRIPT="$LLAMAFACTORY_DIR/scripts/vllm_infer.py"
MERGE_CONFIG="$SCRIPT_DIR/configs/recipe_qwen3_8b_base_lora_merge.yaml"
MERGED_CACHE_ROOT="$RESULT_ROOT/.merged_cache"
ACTIVE_MERGED_DIR=

die() {
  printf '错误：%s\n' "$*" >&2
  exit 1
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

cleanup_active_merge() {
  if [[ -n "$ACTIVE_MERGED_DIR" && -d "$ACTIVE_MERGED_DIR" ]]; then
    case "$ACTIVE_MERGED_DIR" in
      "$MERGED_CACHE_ROOT"/*)
        printf '\n清理临时合并模型：%s\n' "$ACTIVE_MERGED_DIR"
        rm -rf -- "$ACTIVE_MERGED_DIR"
        ;;
      *)
        printf '警告：临时目录不在预期范围内，拒绝清理：%s\n' "$ACTIVE_MERGED_DIR" >&2
        ;;
    esac
  fi
  ACTIVE_MERGED_DIR=
}

trap cleanup_active_merge EXIT INT TERM

[[ -d "$LLAMAFACTORY_DIR" ]] || die "找不到 LlamaFactory：$LLAMAFACTORY_DIR"
LLAMAFACTORY_DIR=$(cd -- "$LLAMAFACTORY_DIR" && pwd)
VLLM_SCRIPT="$LLAMAFACTORY_DIR/scripts/vllm_infer.py"
[[ -r "$VLLM_SCRIPT" ]] || die "找不到官方评估脚本：$VLLM_SCRIPT"
[[ -r "$MERGE_CONFIG" ]] || die "找不到 DoRA 合并配置：$MERGE_CONFIG"
grep -qs -- 'matrix_save_name' "$VLLM_SCRIPT" \
  || die "官方 vllm_infer.py 版本过旧（缺少 matrix_save_name），请更新 LlamaFactory"
grep -qs -- 'enable_thinking' "$VLLM_SCRIPT" \
  || die "官方 vllm_infer.py 版本过旧（缺少 enable_thinking），请更新 LlamaFactory"
[[ -r "$DATA_DIR/dataset_info.json" ]] || die "找不到数据集注册文件：$DATA_DIR/dataset_info.json"
[[ -s "$EVAL_DATA_FILE" ]] \
  || die "找不到固定评估集，请先执行：python3 finetune_scripts/prepare_eval_data.py"
[[ -d "$MODEL_PATH" ]] || die "找不到本地基础模型：$MODEL_PATH"
MODEL_PATH=$(cd -- "$MODEL_PATH" && pwd)
python3 "$SCRIPT_DIR/verify_model.py" "$MODEL_PATH" --quiet \
  || die "基础模型文件校验失败：$MODEL_PATH"
is_positive_integer "$EVAL_MAX_SAMPLES" || die "EVAL_MAX_SAMPLES 必须是正整数"
is_positive_integer "$MAX_NEW_TOKENS" || die "MAX_NEW_TOKENS 必须是正整数"
is_positive_integer "$VLLM_BATCH_SIZE" || die "VLLM_BATCH_SIZE 必须是正整数"
is_positive_integer "$MIN_MERGE_FREE_GIB" || die "MIN_MERGE_FREE_GIB 必须是正整数"
case "$MERGE_DEVICE" in
  auto|cpu) ;;
  *) die "MERGE_DEVICE 只支持 auto 或 cpu" ;;
esac
command -v llamafactory-cli >/dev/null 2>&1 \
  || die "找不到 llamafactory-cli，请先激活 LlamaFactory 环境"

python3 -c 'import av, datasets, fire, jieba, nltk, rouge_chinese, vllm' >/dev/null 2>&1 \
  || die "缺少评估依赖；请在 LlamaFactory 环境安装 metrics 依赖和 vLLM，命令见 README"

declare -a checkpoint_dirs=()
FINAL_ADAPTER_DIR=
if [[ -n "$CHECKPOINT_DIR" ]]; then
  [[ -d "$CHECKPOINT_DIR" ]] || die "找不到 CHECKPOINT_DIR：$CHECKPOINT_DIR"
  checkpoint_dirs+=("$(cd -- "$CHECKPOINT_DIR" && pwd)")
else
  [[ -d "$CHECKPOINT_ROOT" ]] || die "找不到 checkpoint 根目录：$CHECKPOINT_ROOT"
  CHECKPOINT_ROOT=$(cd -- "$CHECKPOINT_ROOT" && pwd)
  while IFS= read -r checkpoint; do
    checkpoint_dirs+=("$checkpoint")
  done < <(find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
  if [[ "$INCLUDE_FINAL_MODEL" == 1 && -r "$CHECKPOINT_ROOT/adapter_config.json" ]]; then
    FINAL_ADAPTER_DIR="$CHECKPOINT_ROOT"
  fi
fi
(( ${#checkpoint_dirs[@]} > 0 )) || die "没有找到 checkpoint-* 目录"

for checkpoint in "${checkpoint_dirs[@]}"; do
  [[ -r "$checkpoint/adapter_config.json" ]] \
    || die "不是完整的 LoRA checkpoint（缺少 adapter_config.json）：$checkpoint"
done

mkdir -p -- "$RESULT_ROOT"
RESULT_ROOT=$(cd -- "$RESULT_ROOT" && pwd)
MERGED_CACHE_ROOT="$RESULT_ROOT/.merged_cache"
mkdir -p -- "$MERGED_CACHE_ROOT"
VLLM_CONFIG=$(printf '{"gpu_memory_utilization":%s,"max_num_seqs":%s,"max_lora_rank":32}' \
  "$VLLM_GPU_MEMORY_UTILIZATION" "$VLLM_BATCH_SIZE")

result_is_complete() {
  local run_name=$1
  [[ -s "$RESULT_ROOT/$run_name/generated_predictions.jsonl" \
    && -s "$RESULT_ROOT/$run_name/metrics.json" ]]
}

adapter_uses_dora() {
  python3 -c \
    'import json,sys; print("1" if json.load(open(sys.argv[1], encoding="utf-8")).get("use_dora") else "0")' \
    "$1/adapter_config.json"
}

merge_dora_checkpoint() {
  local checkpoint_path=$1
  local run_name=$2
  local available_kib required_kib
  available_kib=$(df -Pk "$MERGED_CACHE_ROOT" | awk 'NR == 2 {print $4}')
  required_kib=$((MIN_MERGE_FREE_GIB * 1024 * 1024))
  [[ "$available_kib" =~ ^[0-9]+$ ]] || die "无法读取临时合并目录的可用空间"
  (( available_kib >= required_kib )) \
    || die "临时合并至少需要 ${MIN_MERGE_FREE_GIB}GiB 可用空间：$MERGED_CACHE_ROOT"

  ACTIVE_MERGED_DIR=$(mktemp -d "$MERGED_CACHE_ROOT/${run_name}.XXXXXX")
  declare -a merge_command=(
    llamafactory-cli export "$MERGE_CONFIG"
    "model_name_or_path=$MODEL_PATH"
    "adapter_name_or_path=$checkpoint_path"
    "template=qwen3_nothink"
    "export_dir=$ACTIVE_MERGED_DIR"
    "export_device=$MERGE_DEVICE"
  )

  printf '\n---------- 临时合并 DoRA：%s ----------\n' "$run_name"
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '%q ' "${merge_command[@]}"
    printf '\n'
    return
  fi
  "${merge_command[@]}"
  [[ -r "$ACTIVE_MERGED_DIR/config.json" ]] \
    || die "DoRA 合并完成后缺少 config.json：$ACTIVE_MERGED_DIR"
}

run_evaluation() {
  local run_name=$1
  local inference_model=$2
  local adapter_path=${3:-}
  local source_path=${4:-$inference_model}
  local run_dir="$RESULT_ROOT/$run_name"
  local predictions="$run_dir/generated_predictions.jsonl"
  local metrics="$run_dir/metrics.json"
  mkdir -p -- "$run_dir"

  if [[ "$FORCE_EVAL" != 1 && -s "$predictions" && -s "$metrics" ]]; then
    printf '已有完整结果，跳过：%s\n' "$run_name"
    return
  fi

  declare -a command=(
    python3 "$VLLM_SCRIPT"
    --model_name_or_path "$inference_model"
    --dataset "$DATASET_NAME"
    --dataset_dir "$DATA_DIR"
    --template qwen3_nothink
    --cutoff_len 2048
    --max_samples "$EVAL_MAX_SAMPLES"
    --save_name "$predictions"
    --matrix_save_name "$metrics"
    --temperature 0.0
    --top_p 1.0
    --top_k -1
    --max_new_tokens "$MAX_NEW_TOKENS"
    --repetition_penalty 1.0
    --enable_thinking false
    --seed 42
    --batch_size "$VLLM_BATCH_SIZE"
    --vllm_config "$VLLM_CONFIG"
  )
  if [[ -n "$adapter_path" ]]; then
    command+=(--adapter_name_or_path "$adapter_path")
  fi

  printf '\n---------- 评估：%s ----------\n' "$run_name"
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_IDS"
    printf '%q ' "${command[@]}"
    printf '\n'
    return
  fi

  "${command[@]}"
  python3 -c \
    'import json,sys; p=sys.argv[1]; d=json.load(open(p, encoding="utf-8")); d["checkpoint_path"]=sys.argv[2]; json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)' \
    "$metrics" "$source_path"
}

evaluate_adapter() {
  local run_name=$1
  local adapter_path=$2
  if [[ "$FORCE_EVAL" != 1 ]] && result_is_complete "$run_name"; then
    printf '已有完整结果，跳过：%s\n' "$run_name"
    return
  fi

  if [[ "$(adapter_uses_dora "$adapter_path")" == 1 ]]; then
    merge_dora_checkpoint "$adapter_path" "$run_name"
    run_evaluation "$run_name" "$ACTIVE_MERGED_DIR" "" "$adapter_path"
    cleanup_active_merge
  else
    run_evaluation "$run_name" "$MODEL_PATH" "$adapter_path" "$adapter_path"
  fi
}

printf '\n========== 最终评估配置 ==========\n'
printf '基础模型：%s\n' "$MODEL_PATH"
printf '评估集：%s\n' "$EVAL_DATA_FILE"
printf '样本上限：%s\n' "$EVAL_MAX_SAMPLES"
printf 'Checkpoint 数量：%s\n' "${#checkpoint_dirs[@]}"
printf '基础模型基线：%s\n' "$INCLUDE_BASE_MODEL"
if [[ -n "$FINAL_ADAPTER_DIR" ]]; then
  printf '训练结束模型：%s\n' "$FINAL_ADAPTER_DIR"
else
  printf '训练结束模型：未发现或未启用\n'
fi
printf '生成方式：greedy，非思考模式\n'
printf '适配器处理：普通 LoRA 直接加载；DoRA 自动临时合并\n'
printf '临时合并设备：%s\n' "$MERGE_DEVICE"
printf '最大新 token：%s\n' "$MAX_NEW_TOKENS"
printf 'GPU：%s\n' "$GPU_IDS"
printf '结果目录：%s\n' "$RESULT_ROOT"
printf '==================================\n'

cd -- "$LLAMAFACTORY_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

if [[ "$INCLUDE_BASE_MODEL" == 1 ]]; then
  run_evaluation base_model "$MODEL_PATH"
fi

total=${#checkpoint_dirs[@]}
index=0
for checkpoint in "${checkpoint_dirs[@]}"; do
  index=$((index + 1))
  name=$(basename -- "$checkpoint")
  printf '\n========== CHECKPOINT %s/%s ==========' "$index" "$total"
  printf '\n路径：%s\n' "$checkpoint"
  evaluate_adapter "$name" "$checkpoint"
done

if [[ -n "$FINAL_ADAPTER_DIR" ]]; then
  printf '\n========== FINAL MODEL =========='
  printf '\n路径：%s\n' "$FINAL_ADAPTER_DIR"
  evaluate_adapter final_model "$FINAL_ADAPTER_DIR"
fi

if [[ "$DRY_RUN" == 1 ]]; then
  printf '\nDry-run 完成，未加载模型、未生成评估结果。\n'
  exit 0
fi

printf '\n========== 汇总并排序 =========='
printf '\n'
python3 "$SCRIPT_DIR/rank_checkpoints.py" "$RESULT_ROOT"
printf '\n最终评估完成。\n'
