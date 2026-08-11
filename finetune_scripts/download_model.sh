#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_REPO=${MODEL_REPO:-Qwen/Qwen3-8B-Base}
MODEL_DIR=${MODEL_DIR:-"$HOME/models/Qwen3-8B-Base"}
DOWNLOAD_SOURCE=${DOWNLOAD_SOURCE:-huggingface}
DRY_RUN=${DRY_RUN:-0}
MIN_FREE_GIB=${MIN_FREE_GIB:-20}

die() {
  printf '错误：%s\n' "$*" >&2
  exit 1
}

case "$DOWNLOAD_SOURCE" in
  huggingface|modelscope) ;;
  *) die "DOWNLOAD_SOURCE 只支持 huggingface 或 modelscope" ;;
esac

if python3 "$SCRIPT_DIR/verify_model.py" "$MODEL_DIR" --quiet; then
  printf '模型已经完整，无需重复下载：%s\n' "$MODEL_DIR"
  python3 "$SCRIPT_DIR/verify_model.py" "$MODEL_DIR"
  exit 0
fi

if [[ "$DRY_RUN" == 1 ]]; then
  printf '下载来源：%s\n模型仓库：%s\n保存目录：%s\n' "$DOWNLOAD_SOURCE" "$MODEL_REPO" "$MODEL_DIR"
  if [[ "$DOWNLOAD_SOURCE" == huggingface ]]; then
    printf '将执行：hf download %q --local-dir %q\n' "$MODEL_REPO" "$MODEL_DIR"
  else
    printf '将执行：modelscope download --local_dir %q %q\n' "$MODEL_DIR" "$MODEL_REPO"
  fi
  exit 0
fi

mkdir -p -- "$MODEL_DIR"
available_kib=$(df -Pk "$MODEL_DIR" | awk 'NR == 2 { print $4 }')
required_kib=$((MIN_FREE_GIB * 1024 * 1024))
[[ "$available_kib" =~ ^[0-9]+$ ]] || die "无法读取模型目录的剩余磁盘空间"
if (( available_kib < required_kib )); then
  die "磁盘空间不足：下载前至少需要 ${MIN_FREE_GIB} GiB 可用空间"
fi

printf '\n========== 下载 Qwen3-8B-Base ==========\n'
printf '来源：%s\n' "$DOWNLOAD_SOURCE"
printf '仓库：%s\n' "$MODEL_REPO"
printf '目录：%s\n' "$MODEL_DIR"
printf '可用空间：%.1f GiB\n' "$(awk -v kib="$available_kib" 'BEGIN { print kib / 1024 / 1024 }')"
printf '=========================================\n\n'

if [[ "$DOWNLOAD_SOURCE" == huggingface ]]; then
  export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
  if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$MODEL_REPO" --local-dir "$MODEL_DIR"
  else
    die "找不到 hf 或 huggingface-cli；请在 llamafactory 环境安装 huggingface_hub"
  fi
else
  command -v modelscope >/dev/null 2>&1 \
    || die "找不到 modelscope；请先在 llamafactory 环境安装 modelscope"
  modelscope download --local_dir "$MODEL_DIR" "$MODEL_REPO"
fi

printf '\n下载结束，开始校验模型文件。\n'
python3 "$SCRIPT_DIR/verify_model.py" "$MODEL_DIR" \
  || die "模型文件不完整，可重新运行本脚本继续下载"

printf '\n模型准备完成。现在可以执行：\n'
printf 'bash %q\n' "$SCRIPT_DIR/train.sh"

