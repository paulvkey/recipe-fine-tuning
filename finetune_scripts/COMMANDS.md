# LlamaFactory 训练到评估命令手册

本文档汇总 Qwen3-8B-Base 食谱微调从模型准备、训练、断点恢复、checkpoint 评估到
最终模型合并的常用命令。默认使用单卡 H100 90GB、BF16、LoRA + LoRA+ 和
`qwen3_nothink` 非思考模板。

所有命令均在服务器的项目根目录执行。先按实际位置修改项目路径：

```bash
PROJECT_DIR=$HOME/recipe-fine-tuning
cd "$PROJECT_DIR"
conda activate llamafactory
```

## 1. 训练前检查

```bash
test -d "$HOME/LlamaFactory"
test -f training_sample/recipe_train_sample_100000.jsonl
command -v llamafactory-cli
llamafactory-cli version
nvidia-smi
df -h . "$HOME/models"
```

默认使用以下位置：

```text
LlamaFactory：$HOME/LlamaFactory
基础模型：$HOME/models/Qwen3-8B-Base
训练数据：training_sample/recipe_train_sample_100000.jsonl
训练输出：finetune_scripts/outputs/qwen3-8b-base/recipe-lora-sft/
后台日志：finetune_scripts/logs/
评估输出：finetune_scripts/evaluation/qwen3-8b-base-recipe/
最终模型：finetune_scripts/exported_models/qwen3-8b-base-recipe/
```

## 2. 下载并校验基础模型

只显示下载命令，不实际下载：

```bash
DRY_RUN=1 bash finetune_scripts/download_model.sh
```

从 Hugging Face 下载：

```bash
bash finetune_scripts/download_model.sh
```

无法稳定访问 Hugging Face 时改用 ModelScope：

```bash
python -m pip install modelscope
DOWNLOAD_SOURCE=modelscope bash finetune_scripts/download_model.sh
```

模型下载到其他磁盘：

```bash
MODEL_DIR=/data/models/Qwen3-8B-Base \
bash finetune_scripts/download_model.sh
```

单独校验默认模型目录：

```bash
python3 finetune_scripts/verify_model.py "$HOME/models/Qwen3-8B-Base"
```

## 3. 准备固定评估集

项目已经生成了默认 1000 条评估集。只有全量清洗文件、10 万条训练集、采样数量或随机
种子发生变化时才需要重新执行：

```bash
python3 finetune_scripts/prepare_eval_data.py
```

指定评估样本数量和随机种子：

```bash
python3 finetune_scripts/prepare_eval_data.py \
  --count 2000 \
  --seed 20260811
```

该脚本会流式扫描清洗后的全量数据，排除训练集重合记录和重复记录，再进行固定种子的
蓄水池随机采样。它不执行数据清洗，也不重新生成训练集。

默认输出：

```text
finetune_scripts/data/recipe_eval_holdout.jsonl
finetune_scripts/data/eval_holdout_report.json
```

## 4. 检查并启动正式训练

先执行 dry-run，检查路径、模型、数据、GPU 和最终 LlamaFactory 命令：

```bash
GPU_IDS=0 DRY_RUN=1 bash finetune_scripts/train.sh
```

确认无误后启动正式训练：

```bash
GPU_IDS=0 bash finetune_scripts/train.sh
```

正式训练默认自动转入后台，命令会立即返回并显示 PID、PID 文件和日志文件。脚本使用
`nohup`，系统提供 `setsid` 时还会创建独立会话，因此可以直接关闭 SSH 终端，不需要在
命令末尾额外添加 `&`。

该方式适用于普通 SSH 服务器。如果服务器由 Slurm、Kubernetes 等调度系统管理，或者
管理员启用了退出登录后清理用户进程的策略，应改用平台提供的作业提交命令；Shell 脚本
无法绕过服务器级进程清理策略。

如果默认输出目录已经存在训练文件，脚本会显示明确警告：

```text
输入 yes 将永久清空目录内容并从头训练，其他输入均取消：
```

只有输入完整的小写 `yes` 才会清空旧目录并继续。设置 `RESUME_FROM_CHECKPOINT` 时属于
断点恢复，不执行清空操作。`DRY_RUN=1` 只提示将发生清空确认，不删除文件。

需要在前台调试时：

```bash
GPU_IDS=0 TRAIN_RUN_MODE=foreground bash finetune_scripts/train.sh
```

默认训练参数：

```text
模型：Qwen3-8B-Base
模板：qwen3_nothink
训练方法：LoRA + LoRA+
LoRA rank / alpha：32 / 64
LoRA+ 比例：16
DoRA：关闭
BF16：开启
单卡 batch size：4
梯度累积：4
有效 batch size：16
学习率：1e-5
训练轮数：3
最大样本数：100000
验证集比例：0.1
截断长度：2048
logging / eval / save / warmup steps：100 / 1400 / 1400 / 1000
checkpoint：全部保留
梯度检查点：开启
```

指定其他模型、数据和 checkpoint 输出目录：

```bash
MODEL_PATH=/data/models/Qwen3-8B-Base \
DATA_FILE=/data/recipe/recipe_train_sample_100000.jsonl \
OUTPUT_DIR=/data/checkpoints/recipe-lora-sft \
GPU_IDS=0 \
bash finetune_scripts/train.sh
```

GPU 没有默认值，每次训练和 dry-run 都必须显式指定。使用 GPU 0：

```bash
GPU_IDS=0 bash finetune_scripts/train.sh
```

显式指定 Attention 后端：

```bash
GPU_IDS=0 ATTENTION_BACKEND=fa2 bash finetune_scripts/train.sh
GPU_IDS=0 ATTENTION_BACKEND=sdpa bash finetune_scripts/train.sh
```

默认自动检测 FlashAttention-2；环境不支持时回退到 PyTorch SDPA。

实测关闭梯度检查点时，batch 4 的长序列在交叉熵计算阶段会耗尽约 95GiB 显存，因此
当前默认开启梯度检查点，验证 batch 也已从 8 调为 4。若仍然 OOM：

```bash
TRAIN_BATCH_SIZE=2 \
EVAL_BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=8 \
GPU_IDS=0 \
bash finetune_scripts/train.sh
```

DoRA 默认关闭。如需进行独立对照实验，必须使用不同输出目录：

```bash
USE_DORA=1 \
OUTPUT_DIR=/data/checkpoints/recipe-dora-sft \
GPU_IDS=0 \
bash finetune_scripts/train.sh
```

## 5. 查看训练状态

日常排查只执行这一条命令：

```bash
python3 finetune_scripts/check_training_status.py
```

每 30 秒自动刷新状态：

```bash
watch -n 30 python3 finetune_scripts/check_training_status.py
```

命令会直接汇总：

- `运行中`、`已正常完成`、`异常退出`或`进程已消失`；
- PID、退出码和开始/结束时间；
- 最后训练步数、最近 loss 和学习率；
- checkpoint 数量与最近 checkpoint；
- CUDA OOM、磁盘不足、NCCL、NaN、系统强杀、数据异常或 Python 异常；
- 针对已识别错误的处理建议。

输出机器可读 JSON，便于接入 cron、Webhook 或其他监控：

```bash
python3 finetune_scripts/check_training_status.py --json
```

返回码为 `0` 表示运行中或正常完成，`2` 表示失败或进程异常消失，`3` 表示尚无状态。

查看 GPU 使用情况：

```bash
watch -n 2 nvidia-smi
```

查看已经生成的 checkpoint：

```bash
find finetune_scripts/outputs/qwen3-8b-base/recipe-lora-sft \
  -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V
```

只有自动诊断信息不足时，才查看日志最后 80 行：

```bash
RECIPE_TRAIN_LOG=$(ls -1t finetune_scripts/logs/train_*.log | head -n 1)
tail -n 80 "$RECIPE_TRAIN_LOG"
```

需要主动停止训练时，向后台主进程发送 TERM：

```bash
RECIPE_TRAIN_PID=$(cat finetune_scripts/logs/train.pid)
kill "$RECIPE_TRAIN_PID"
```

停止时当前尚未到达保存点的进度可能丢失，之后应从最近的完整 checkpoint 恢复。

## 6. 从 checkpoint 继续训练

使用默认输出目录中的 checkpoint：

```bash
RESUME_FROM_CHECKPOINT=finetune_scripts/outputs/qwen3-8b-base/recipe-lora-sft/checkpoint-5000 \
GPU_IDS=0 \
bash finetune_scripts/train.sh
```

使用自定义输出目录时，应同时传回相同的 `OUTPUT_DIR`：

```bash
OUTPUT_DIR=/data/checkpoints/recipe-lora-sft \
RESUME_FROM_CHECKPOINT=/data/checkpoints/recipe-lora-sft/checkpoint-5000 \
GPU_IDS=0 \
bash finetune_scripts/train.sh
```

DoRA checkpoint 续训时必须继续设置 `USE_DORA=1`：

```bash
USE_DORA=1 \
OUTPUT_DIR=/data/checkpoints/recipe-dora-sft \
RESUME_FROM_CHECKPOINT=/data/checkpoints/recipe-dora-sft/checkpoint-5000 \
GPU_IDS=0 \
bash finetune_scripts/train.sh
```

脚本会检查 checkpoint 的 `adapter_config.json`，拒绝使用不一致的 DoRA 配置恢复。

## 7. 安装最终评估依赖

评估使用 LlamaFactory 官方 `scripts/vllm_infer.py` 和中文 BLEU/ROUGE。建议使用独立
Conda 环境，避免改变训练环境中的 PyTorch：

```bash
conda create --name llamafactory-eval --clone llamafactory
conda activate llamafactory-eval
cd "$HOME/LlamaFactory"
python -m pip install -r requirements/metrics.txt
python -m pip install vllm
python -m pip check
python -c 'import vllm, jieba, nltk, rouge_chinese; print("评估依赖正常")'
cd "$PROJECT_DIR"
```

如果服务器已有验证过的 vLLM 环境，可以直接使用，不必重复安装或强制升级。

## 8. 评估基础模型和全部 checkpoint

先检查评估命令：

```bash
DRY_RUN=1 bash finetune_scripts/evaluate_checkpoints.sh
```

正式评估默认输出目录中的基础模型、全部 checkpoint 和训练结束适配器：

```bash
bash finetune_scripts/evaluate_checkpoints.sh
```

checkpoint 位于其他目录时：

```bash
CHECKPOINT_ROOT=/data/checkpoints/recipe-lora-sft \
bash finetune_scripts/evaluate_checkpoints.sh
```

只评估一个 checkpoint，并跳过基础模型：

```bash
CHECKPOINT_DIR=/data/checkpoints/recipe-lora-sft/checkpoint-8000 \
INCLUDE_BASE_MODEL=0 \
bash finetune_scripts/evaluate_checkpoints.sh
```

评估中断后可以直接重新执行，已经生成完整预测和指标的 checkpoint 会被跳过。强制覆盖
已有评估结果：

```bash
FORCE_EVAL=1 bash finetune_scripts/evaluate_checkpoints.sh
```

vLLM 显存不足时降低批量和显存占用比例：

```bash
VLLM_BATCH_SIZE=64 \
VLLM_GPU_MEMORY_UTILIZATION=0.80 \
bash finetune_scripts/evaluate_checkpoints.sh
```

普通 LoRA checkpoint 会被 vLLM 直接加载；DoRA checkpoint 会自动临时合并、评估并
清理临时模型。

## 9. 查看评估结果

```bash
sed -n '1,200p' \
  finetune_scripts/evaluation/qwen3-8b-base-recipe/evaluation_report.md

cat finetune_scripts/evaluation/qwen3-8b-base-recipe/best_checkpoint.txt
```

主要结果文件：

```text
evaluation_report.md       Markdown 排名和选择建议
checkpoint_ranking.csv     表格格式的全部 checkpoint 指标
checkpoint_ranking.json    程序可读的排序结果
best_checkpoint.txt        自动排名第一的适配器路径
*/metrics.json             单个 checkpoint 的 BLEU/ROUGE 指标
*/generated_predictions.jsonl  单个 checkpoint 的逐条预测结果
```

自动排名只能作为初筛。最终应人工比较排名前 3 的相同样例，检查食材遗漏、用量合理性、
步骤可执行性和食品安全。

## 10. 合并并导出最终模型

确认最终 checkpoint 后，先检查合并命令：

```bash
DRY_RUN=1 bash finetune_scripts/export_best_model.sh
```

默认读取评估目录中的 `best_checkpoint.txt` 并导出：

```bash
bash finetune_scripts/export_best_model.sh
```

人工指定 checkpoint 和导出目录：

```bash
ADAPTER_PATH=/data/checkpoints/recipe-lora-sft/checkpoint-8000 \
EXPORT_DIR=/data/models/qwen3-8b-base-recipe-final \
bash finetune_scripts/export_best_model.sh
```

模型和 checkpoint 使用自定义路径时：

```bash
MODEL_PATH=/data/models/Qwen3-8B-Base \
ADAPTER_PATH=/data/checkpoints/recipe-lora-sft/checkpoint-8000 \
EXPORT_DIR=/data/models/qwen3-8b-base-recipe-final \
bash finetune_scripts/export_best_model.sh
```

合并必须使用未量化的基础模型。导出脚本默认按 5GB 分片保存 safetensors，并在完成后
校验模型权重。目标目录非空时脚本会停止，避免覆盖已有模型。

## 11. 最短完整执行顺序

训练数据和环境已经准备好的情况下，依次执行：

```bash
cd "$PROJECT_DIR"
conda activate llamafactory

bash finetune_scripts/download_model.sh
GPU_IDS=0 DRY_RUN=1 bash finetune_scripts/train.sh
GPU_IDS=0 bash finetune_scripts/train.sh

conda activate llamafactory-eval
DRY_RUN=1 bash finetune_scripts/evaluate_checkpoints.sh
bash finetune_scripts/evaluate_checkpoints.sh

DRY_RUN=1 bash finetune_scripts/export_best_model.sh
bash finetune_scripts/export_best_model.sh
```

如果评估环境和训练环境相同，可以省略第二次 `conda activate`。

## 12. 常用环境变量

| 环境变量                 | 默认值                                             | 作用                           |
| ------------------------ | -------------------------------------------------- | ------------------------------ |
| `LLAMAFACTORY_DIR`       | `$HOME/LlamaFactory`                               | LlamaFactory 安装目录          |
| `MODEL_PATH`             | `$HOME/models/Qwen3-8B-Base`                       | 训练、评估或合并使用的基础模型 |
| `DATA_FILE`              | `training_sample/recipe_train_sample_100000.jsonl` | 训练数据                       |
| `OUTPUT_DIR`             | `finetune_scripts/outputs/.../recipe-lora-sft`     | 训练和 checkpoint 输出目录     |
| `GPU_IDS`                | 无，必须显式指定                                   | 使用的 CUDA 设备               |
| `ATTENTION_BACKEND`      | `auto`                                             | `fa2`、`sdpa` 或 `disabled`    |
| `GRADIENT_CHECKPOINTING` | `1`                                                | 是否启用梯度检查点             |
| `TRAIN_BATCH_SIZE`       | `4`                                                | 单卡训练 micro-batch           |
| `EVAL_BATCH_SIZE`        | `4`                                                | 单卡验证 batch                 |
| `GRADIENT_ACCUMULATION_STEPS` | `4`                                           | 梯度累积步数                   |
| `USE_DORA`               | `0`                                                | 是否启用实验性 DoRA            |
| `RESUME_FROM_CHECKPOINT` | 空                                                 | 断点恢复目录                   |
| `TRAIN_RUN_MODE`         | `background`                                       | 后台运行或前台调试             |
| `TRAIN_LOG_DIR`          | `finetune_scripts/logs`                            | 后台训练日志目录               |
| `TRAIN_LOG_FILE`         | 自动生成时间戳文件名                               | 自定义本次后台日志文件         |
| `TRAIN_PID_FILE`         | `finetune_scripts/logs/train.pid`                  | 后台训练 PID 文件              |
| `TRAIN_STATUS_FILE`      | `finetune_scripts/logs/train_status.json`          | 状态、退出码和路径记录         |
| `CHECKPOINT_ROOT`        | 默认训练输出目录                                   | 批量评估的 checkpoint 根目录   |
| `CHECKPOINT_DIR`         | 空                                                 | 只评估一个 checkpoint          |
| `RESULT_ROOT`            | `finetune_scripts/evaluation/...`                  | 评估结果目录                   |
| `ADAPTER_PATH`           | `best_checkpoint.txt` 中的路径                     | 最终导出的适配器               |
| `EXPORT_DIR`             | `finetune_scripts/exported_models/...`             | 合并模型输出目录               |
| `DRY_RUN`                | `0`                                                | 设置为 1 时只检查并显示命令    |
