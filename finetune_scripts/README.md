# Qwen3-8B-Base 食谱微调命令

本目录用于在服务器的 `$HOME/LlamaFactory` 中，使用当前项目准备好的 10 万条食谱
指令数据对 `Qwen/Qwen3-8B-Base` 执行非思考模式 LoRA 监督微调（SFT）。

## 目录内容

```text
finetune_scripts/
├── configs/
│   ├── recipe_qwen3_8b_base_lora_merge.yaml
│   └── recipe_qwen3_8b_base_lora_sft.yaml
├── data/
│   ├── dataset_info.json
│   ├── eval_holdout_report.json
│   └── recipe_eval_holdout.jsonl
├── .gitignore
├── COMMANDS.md
├── download_model.sh
├── evaluate_checkpoints.sh
├── export_best_model.sh
├── prepare_eval_data.py
├── rank_checkpoints.py
├── README.md
├── train.sh
└── verify_model.py
```

`train.sh` 启动时会在 `data/` 中创建指向真实 JSONL 的符号链接，不复制训练数据，
也不修改 `$HOME/LlamaFactory/data/dataset_info.json`。

只需要按顺序复制命令时，直接查看 [`COMMANDS.md`](COMMANDS.md)。

## 最终训练参数

| 参数                | 配置值                                            |
| ------------------- | ------------------------------------------------- |
| 基础模型            | `Qwen/Qwen3-8B-Base`                              |
| 训练阶段            | SFT                                               |
| 对话模板            | `qwen3_nothink`（非思考模式）                     |
| 微调方式            | LoRA                                              |
| LoRA target         | `all`                                             |
| LoRA rank           | 32                                                |
| LoRA alpha          | 64                                                |
| LoRA dropout        | 0.0                                               |
| LoRA+ 学习率比例    | 16                                                |
| DoRA                | 默认关闭，可通过环境变量启用                      |
| 最大样本数          | 100000                                            |
| 验证集比例          | 0.1                                               |
| 训练轮数            | 3                                                 |
| 截断长度            | 2048 tokens                                       |
| 单卡训练 batch size | 4                                                 |
| 梯度累积            | 4                                                 |
| 单卡有效 batch size | 16                                                |
| 单卡验证 batch size | 8                                                 |
| 计算精度            | BF16                                              |
| 基础学习率          | `1e-5`                                            |
| 学习率调度          | cosine                                            |
| Logging steps       | 100                                               |
| Save steps          | 1400（约 1/4 epoch）                              |
| Warmup steps        | 1000                                              |
| Checkpoint 保留数量 | 不限制，保留全部                                  |
| 评估频率            | 1400 steps                                        |
| 梯度检查点          | 默认关闭，可在 OOM 时启用                         |
| Prompt loss         | 关闭，只对 output 计算损失                        |
| Packing             | 关闭                                              |
| 随机种子            | 42                                                |
| Attention           | H100 优先 FlashAttention-2，未安装时自动回退 SDPA |

LoRA+ 的比例表示 LoRA B 矩阵学习率与 A 矩阵学习率的比例。当前 A 矩阵学习率为
`1e-5`，B 矩阵学习率为 `1.6e-4`。`lora_alpha: 64` 是 rank 32 的 2 倍，符合
LlamaFactory 参数文档的常用设置。训练最多读取 100000 条样本，再按 0.1 划分验证集。

`train_on_prompt: false` 明确只让答案部分参与损失计算，符合当前
`instruction/input/output` 的食谱问答目标。`packing: false` 是有意保留：短样本打包可能
提高吞吐，但会改变每个 epoch 的优化步数，使现有 `warmup_steps`、`save_steps` 和检查点
规划失去对应关系；建议首轮训练结束后再单独做吞吐对照实验。

## H100 90GB 参数评估

这套配置可以作为正式训练的第一版：

- Qwen3-8B-Base 官方 BF16 权重约 16.4GB；LoRA 不保存全部基础参数的梯度和优化器状态。
- H100 原生适合 BF16，没必要为了显存改成 QLoRA。
- 单卡 micro-batch 4、梯度累积 4，对应有效 batch size 16，适合当前 10 万条 SFT 数据。
- rank 32、alpha 64、全线性层 target 和 LoRA+ 已提供较高的适配容量。
- `1e-5` 是偏保守的基础学习率；LoRA+ 比例 16 会提高 B 矩阵对应的学习率。
- 3 个 epoch 合理，但应观察第 2、3 个 epoch 的验证损失是否继续下降，避免领域数据过拟合。

当前训练文件的字符长度统计如下：

| 分位点 | instruction + input + output 字符数 |
| ------ | ----------------------------------: |
| P50    |                                 320 |
| P90    |                                 672 |
| P95    |                                 841 |
| P99    |                                1323 |
| P99.9  |                                2272 |

只有 0.15% 的记录超过 2048 个字符。字符数不完全等于 token 数，但说明 2048 的截断
长度总体合理。下载模型后如需精确判断，可以用 Qwen3 tokenizer 对全部数据再次统计。

验证集比例 0.1 会得到约 9 万条训练数据和 1 万条验证数据。配置每 1400 个优化步骤
评估并保存一次，约等于每个 epoch 的 1/4；评估点和可供最终生成式评测的 checkpoint
一一对应。不设置 `save_total_limit`，因此不会自动删除中间 checkpoint。

按 9 万条训练数据、单卡有效 batch size 16 和 3 个 epoch 估算，总优化步数约为
16875：`logging_steps: 100` 大约产生 169 次训练日志；`save_steps: 1400` 大约产生
12 次阶段性 checkpoint，并且全部保留；`warmup_steps: 1000` 约占总训练步数的 5.9%。这里使用显式
`warmup_steps`，不再同时配置 `warmup_ratio`，避免两个 warmup 参数产生理解歧义。

当前没有启用 `load_best_model_at_end`，所有评估点都会保存 checkpoint。训练结束后应使用
固定评测集或实际问答样例比较中间 checkpoint，
确定最终版本后再人工删除效果较差的节点。所有 checkpoint 都包含继续训练需要的状态，
可以直接通过 `RESUME_FROM_CHECKPOINT` 恢复。

H100 90GB 上默认关闭梯度检查点，以减少重算并提高吞吐。如实测发生 OOM，可直接启用：

```bash
GRADIENT_CHECKPOINTING=1 bash finetune_scripts/train.sh
```

若仍然 OOM，再把单卡 batch size 从 4 降为 2、梯度累积从 4 提高为 8，以保持有效
batch size 16。YAML 使用 LlamaFactory 当前参数 `disable_gradient_checkpointing`，
`train.sh` 用更易理解的 `GRADIENT_CHECKPOINTING=0/1` 对其反向转换。

## DoRA 的作用和当前取舍

DoRA 把预训练权重分解为“幅度”和“方向”：方向部分由 LoRA 更新，幅度作为独立参数
训练。它的目标是让低秩适配的学习方式更接近全参数微调，在部分任务上可能改善收敛、
稳定性和最终精度；合并进基础权重后不会增加最终模型的推理计算量。

当前正式方案默认关闭 DoRA，原因不是 H100 显存不足，而是工程收益不确定：rank 32、
alpha 64、`lora_target: all` 和 LoRA+ 已经有足够容量；同时 vLLM 当前不能直接加载
`use_dora: true` 的 PEFT 适配器，每个中间 checkpoint 在评估前都必须先合并一次，增加
时间、磁盘临时空间和失败点。普通 LoRA checkpoint 可以被评估脚本直接加载。

如需验证 DoRA 是否真的改善食谱任务，可只做一个严格对照实验，其他参数和随机种子不变：

```bash
USE_DORA=1 OUTPUT_DIR=/data/checkpoints/recipe-dora-sft \
bash finetune_scripts/train.sh
```

断点续训必须保持与原 checkpoint 相同的 `USE_DORA`；脚本会读取
`adapter_config.json` 并阻止不一致的恢复。只有 DoRA 的验证收益稳定超过普通 LoRA，才建议
把它恢复为生产默认值。

`train.sh` 会检测当前 Python 环境能否导入 `flash_attn`：可用时给 LlamaFactory 传入
`flash_attn=fa2`，否则传入 `flash_attn=sdpa`。也可以显式指定：

```bash
ATTENTION_BACKEND=fa2 bash finetune_scripts/train.sh
ATTENTION_BACKEND=sdpa bash finetune_scripts/train.sh
```

指定 `fa2` 但环境无法导入 FlashAttention 时，脚本会在训练前停止。启动时还会输出
LlamaFactory 版本和 GPU 剩余显存，并设置 PyTorch 可扩展显存段，降低显存碎片导致的
偶发 OOM。

## 下载 Qwen3-8B-Base

官方模型约 16.4GB。考虑 checkpoint、预处理缓存和以后合并模型，建议服务器至少准备
50GB 可用磁盘空间。下载脚本最低只强制检查 20GiB，以保证原始模型可以保存。

先进入环境：

```bash
cd /服务器上的路径/recipe-fine-tuning
conda activate llamafactory
```

查看下载命令但不实际下载：

```bash
DRY_RUN=1 bash finetune_scripts/download_model.sh
```

从 Hugging Face 下载，支持中断后重新执行并续传：

```bash
bash finetune_scripts/download_model.sh
```

默认保存到：

```text
$HOME/models/Qwen3-8B-Base
```

中国大陆服务器访问 Hugging Face 较慢时，可以改用 ModelScope：

```bash
python -m pip install modelscope
DOWNLOAD_SOURCE=modelscope bash finetune_scripts/download_model.sh
```

自定义模型保存位置：

```bash
MODEL_DIR=/data/models/Qwen3-8B-Base \
bash finetune_scripts/download_model.sh
```

自定义目录下载完成后，训练时需要使用同一路径：

```bash
MODEL_PATH=/data/models/Qwen3-8B-Base \
bash finetune_scripts/train.sh
```

Hugging Face 命令不存在时，在当前 Conda 环境安装或更新下载工具：

```bash
python -m pip install -U huggingface_hub
```

模型下载完成后，脚本会读取权重索引，逐个检查所有 safetensors 分片，并检查权重
总大小。也可以随时单独校验：

```bash
python finetune_scripts/verify_model.py "$HOME/models/Qwen3-8B-Base"
```

## 服务器启动命令

```bash
cd /服务器上的路径/recipe-fine-tuning
conda activate llamafactory
bash finetune_scripts/download_model.sh
```

先进行 dry-run，只检查目录、数据和最终命令，不启动训练：

```bash
DRY_RUN=1 bash finetune_scripts/train.sh
```

确认输出正确后正式训练：

```bash
bash finetune_scripts/train.sh
```

正式训练默认通过 `nohup` 自动转入后台，服务器存在 `setsid` 时还会脱离当前会话；命令
返回后会显示 PID 和时间戳日志路径，可以直接关闭终端，不需要手工追加 `&`。查看状态：

```bash
RECIPE_TRAIN_PID=$(cat finetune_scripts/logs/train.pid)
ps -p "$RECIPE_TRAIN_PID" -o pid,etime,stat,cmd

RECIPE_TRAIN_LOG=$(ls -1t finetune_scripts/logs/train_*.log | head -n 1)
tail -f "$RECIPE_TRAIN_LOG"
```

`DRY_RUN=1` 始终在前台显示检查结果。需要在前台调试正式训练时使用：

```bash
TRAIN_RUN_MODE=foreground bash finetune_scripts/train.sh
```

普通 SSH 服务器可使用上述后台模式；Slurm、Kubernetes 或启用了登录退出进程清理策略的
服务器，应使用对应平台的作业调度命令。

默认 checkpoint 输出目录：

```text
finetune_scripts/outputs/qwen3-8b-base/recipe-lora-sft/
```

## 使用其他本地模型目录

训练脚本默认使用 `$HOME/models/Qwen3-8B-Base`。如果模型下载到了其他位置：

```bash
MODEL_PATH=/data/models/Qwen3-8B-Base \
bash finetune_scripts/train.sh
```

`MODEL_PATH` 不应改成其他模型架构；训练模板固定为 `qwen3_nothink`。

## 指定 GPU

使用 GPU 0：

```bash
GPU_IDS=0 bash finetune_scripts/train.sh
```

使用 GPU 1：

```bash
GPU_IDS=1 bash finetune_scripts/train.sh
```

如指定多张 GPU，每张 GPU 的 `per_device_train_batch_size` 仍然是 4，全局有效 batch
size 会变成 `4 × 梯度累积 4 × GPU 数量`。

## 指定训练数据

默认读取：

```text
training_sample/recipe_train_sample_100000.jsonl
```

服务器上的路径不同时：

```bash
DATA_FILE=/data/recipe/recipe_train_sample_100000.jsonl \
bash finetune_scripts/train.sh
```

数据必须为逐行 JSON，并包含字符串类型的 `instruction`、`input` 和 `output`。

## 从断点继续训练

```bash
RESUME_FROM_CHECKPOINT=/path/to/recipe-lora-sft/checkpoint-5000 \
bash finetune_scripts/train.sh
```

## 自定义输出目录

```bash
OUTPUT_DIR=/data/checkpoints/recipe-qwen3-8b-base \
bash finetune_scripts/train.sh
```

## 启动前检查

```bash
test -d "$HOME/LlamaFactory"
test -f training_sample/recipe_train_sample_100000.jsonl
conda activate llamafactory
command -v llamafactory-cli
llamafactory-cli version
nvidia-smi
python finetune_scripts/verify_model.py "$HOME/models/Qwen3-8B-Base"
DRY_RUN=1 bash finetune_scripts/train.sh
```

如果仍然出现 CUDA OOM，先检查是否有其他进程占用 GPU，再把单卡 batch size 从 4
降到 2，并把梯度累积从 4 提高到 8，从而保持有效 batch size 16。不要直接改成
QLoRA，因为当前 H100 90GB 和最终方案都适合 BF16 LoRA。

## 最终评估方案

训练时的 10% 验证集用于观察 `eval_loss`，最终挑选 checkpoint 使用单独固定留出集。
这样可以保留并公平比较 `checkpoint-1400` 到最后一个 checkpoint，也能与未微调基础模型
作同条件对照。

当前已从 `pipeline_output/recipe_train_clean.jsonl` 生成 1000 条留出数据：

- 全量扫描 1,476,060 条清洗数据；
- 按规范化后的 `instruction/input/output` 指纹排除全部 100,000 条训练样本；
- 剩余 1,376,060 条唯一候选中使用固定种子的蓄水池随机采样；
- 输出与训练集零重合，SHA-256 和完整统计保存在
  `data/eval_holdout_report.json`。

如全量数据或训练样本发生变化，应重新生成评估集：

```bash
python3 finetune_scripts/prepare_eval_data.py
```

自定义留出数量和随机种子：

```bash
python3 finetune_scripts/prepare_eval_data.py --count 2000 --seed 20260811
```

### `prepare_eval_data.py` 的作用

这个脚本只负责生成“最终模型选择用的固定留出评估集”，不参与训练，也不做错别字纠正、
规则清洗或训练集抽样。具体流程是：

1. 读取 10 万条训练数据，对去除字段首尾空白后的
   `instruction/input/output` 计算 BLAKE2b-128 指纹；
2. 逐行流式扫描清洗后的全量 JSONL，不把 147 万条数据一次性读入内存；
3. 排除无效行、与训练集重合的记录以及全量数据内部重复记录；
4. 用固定随机种子的蓄水池算法，从剩余候选中等概率采样指定数量；
5. 原子写入评估 JSONL，并生成包含来源、数量、排除统计和 SHA-256 的报告。

因此，它解决的是“评估数据泄漏”和“各 checkpoint 公平复用同一批题目”两个问题。
只有全量清洗文件、训练样本、采样数量或随机种子发生变化时才需要重新运行；每次训练前
不需要重复执行。当前指纹是严格文本匹配，不能识别语义近似的同一道食谱，若后续要求
更严格的近重复隔离，应在采样阶段另加菜名归一化或相似度聚类。

### 安装评估依赖

官方 `vllm_infer.py` 需要 vLLM，BLEU/ROUGE 需要官方 metrics 依赖。建议训练完成后
再安装，或克隆一个独立 Conda 环境，避免安装 vLLM 时改变正在训练的 PyTorch 环境：

```bash
conda create --name llamafactory-eval --clone llamafactory
conda activate llamafactory-eval
cd "$HOME/LlamaFactory"
python -m pip install -r requirements/metrics.txt
python -m pip install vllm
python -m pip check
python -c 'import vllm, jieba, nltk, rouge_chinese; print("评估依赖正常")'
```

应让 vLLM、PyTorch 和服务器 CUDA 版本相互兼容；如果服务器已经有验证过的 vLLM
环境，不要为了追求最新版本强制升级。

### 评估全部 checkpoint

先检查命令，不加载模型：

```bash
DRY_RUN=1 bash finetune_scripts/evaluate_checkpoints.sh
```

正式评估：

```bash
bash finetune_scripts/evaluate_checkpoints.sh
```

默认流程会先评估未微调基础模型，再逐个扫描
`outputs/qwen3-8b-base/recipe-lora-sft/checkpoint-*`，最后把输出根目录中训练完整 3 epoch
的适配器作为 `final_model` 一并评估。所有模型使用相同的
`qwen3_nothink` 模板、greedy 解码、固定随机种子和同一批 1000 条留出数据。

评估脚本会读取每个适配器的 `adapter_config.json`。默认普通 LoRA 直接使用 vLLM 的
LoRA 加载路径；如果是通过 `USE_DORA=1` 训练的 checkpoint，由于 vLLM 会拒绝
`use_dora: true`，脚本才对该 checkpoint 执行以下流程：

1. 用 LlamaFactory 官方 `export` 将 DoRA 临时合并进未量化的基础模型；
2. 用官方 `scripts/vllm_infer.py` 加载该临时完整模型并评估；
3. 只保留预测与指标，立即删除该次约 16.4GB 的临时合并模型；
4. 再处理下一个 checkpoint。

这样不会为约 12 个 checkpoint 长期占用约 197GB 额外空间，但临时合并目录所在磁盘仍需
至少 20GiB 可用空间。中断时脚本也只会清理它自己通过 `mktemp` 创建的目录。默认
`MERGE_DEVICE=auto`，在 H100 上通常会使用 GPU 加快合并；也可以改为 CPU：

```bash
MERGE_DEVICE=cpu bash finetune_scripts/evaluate_checkpoints.sh
```

普通 LoRA 无需上述临时合并，并保留 `max_lora_rank: 32`。因此当前默认训练方案评估
中间节点时速度更快、磁盘开销也更小。

如果 checkpoint 在其他磁盘：

```bash
CHECKPOINT_ROOT=/data/checkpoints/recipe-qwen3-8b-base \
bash finetune_scripts/evaluate_checkpoints.sh
```

只评估一个指定 checkpoint：

```bash
CHECKPOINT_DIR=/data/checkpoints/recipe-qwen3-8b-base/checkpoint-8000 \
INCLUDE_BASE_MODEL=0 \
bash finetune_scripts/evaluate_checkpoints.sh
```

中断后直接重新执行即可：已有完整预测和指标的 checkpoint 会自动跳过。需要覆盖旧结果时：

```bash
FORCE_EVAL=1 bash finetune_scripts/evaluate_checkpoints.sh
```

显存不足时降低 vLLM 批量和显存占用比例：

```bash
VLLM_BATCH_SIZE=64 VLLM_GPU_MEMORY_UTILIZATION=0.80 \
bash finetune_scripts/evaluate_checkpoints.sh
```

### 评估输出

默认写入 `finetune_scripts/evaluation/qwen3-8b-base-recipe/`：

- `base_model/` 和各 `checkpoint-N/`：官方生成结果 `generated_predictions.jsonl`
  与 `metrics.json`；
- `final_model/`：训练结束后输出根目录中最终适配器的结果；
- `checkpoint_ranking.csv`：便于表格查看的全部指标；
- `checkpoint_ranking.json`：程序可读的排序结果；
- `evaluation_report.md`：最终对比表和选择建议；
- `best_checkpoint.txt`：自动排名第一的 checkpoint 完整路径。

自动排序依次使用 ROUGE-L、ROUGE-2、食谱格式合规率和 BLEU-4。脚本同时统计空输出、
编号步骤、`<think>` 泄漏以及预测/参考答案长度比。BLEU/ROUGE 只能衡量和参考食谱的
文本重合度；最终应从排名前 3 的 checkpoint 固定抽取相同样例，人工检查食材遗漏、
用量合理性、步骤可执行性和食品安全后再决定部署版本。

## 合并并导出最终模型

先完成全部 checkpoint 评估和人工复核，再合并最终选中的适配器。脚本默认读取
`evaluation/qwen3-8b-base-recipe/best_checkpoint.txt`：

```bash
DRY_RUN=1 bash finetune_scripts/export_best_model.sh
bash finetune_scripts/export_best_model.sh
```

人工选择了其他 checkpoint 时直接覆盖：

```bash
ADAPTER_PATH=/data/checkpoints/recipe-lora-sft/checkpoint-8000 \
EXPORT_DIR=/data/models/qwen3-8b-base-recipe-final \
bash finetune_scripts/export_best_model.sh
```

脚本使用官方 `llamafactory-cli export` 流程、同一个 `qwen3_nothink` 模板和未量化的
Qwen3-8B-Base 权重，默认按 5GB 分片导出。导出目录非空时会停止，避免覆盖已有模型；
正式导出后会校验权重分片完整性。普通 LoRA 可不合并就用支持适配器的推理框架部署，
但合并模型更便于作为独立目录交付；DoRA 若交给当前 vLLM 部署则必须先合并。

## 与官方源码的对应关系

- Qwen3 LoRA SFT 的阶段、模板、训练入口和 YAML 结构参考官方
  [examples/train_lora](https://github.com/hiyouga/LlamaFactory/tree/main/examples/train_lora)
  及其中的
  [qwen3_lora_sft.yaml](https://github.com/hiyouga/LlamaFactory/blob/main/examples/train_lora/qwen3_lora_sft.yaml)；
- LoRA+ 的 `loraplus_lr_ratio: 16.0` 参考官方
  [examples/extras/loraplus](https://github.com/hiyouga/LlamaFactory/tree/main/examples/extras/loraplus)；
- SFT 的命令行覆盖、数据、模板和训练参数参考官方中文
  [监督微调文档](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/sft.html)；
- 最终模型导出和未量化基础模型要求参考官方中文
  [LoRA 合并文档](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/merge_lora.html)；
- `lora_alpha`、DoRA、LoRA+、packing、梯度检查点等参数语义参考官方中文
  [参数介绍](https://llamafactory.readthedocs.io/zh-cn/latest/advanced/arguments.html)；
- 批量推理、LoRA 加载、`qwen3_nothink` 和指标输出参数直接使用
  [scripts/vllm_infer.py](https://github.com/hiyouga/LlamaFactory/blob/main/scripts/vllm_infer.py)；
- 中文 BLEU/ROUGE 算法沿用
  [scripts/eval_bleu_rouge.py](https://github.com/hiyouga/LlamaFactory/blob/main/scripts/eval_bleu_rouge.py)；
- Alpaca 数据字段映射遵循
  [data/README.md](https://github.com/hiyouga/LlamaFactory/blob/main/data/README.md)。

DoRA 的幅度/方向分解及其实验收益来自原始
[DoRA 论文](https://arxiv.org/abs/2402.09353)。DoRA 必须先合并再交给 vLLM 的判断来自当前
[PEFT 兼容检查源码](https://github.com/vllm-project/vllm/blob/main/vllm/lora/peft_helper.py)，
其中会把 `use_dora` 明确列为不支持的适配器特性。

训练和评估必须使用同一个 `qwen3_nothink` 模板。Qwen3 Base 的 EOS 处理在较新的
LlamaFactory 源码中修复过，因此服务器不应使用早期 Qwen3 初版代码。脚本会检查模板
和官方评估参数是否存在；建议训练前在 `$HOME/LlamaFactory` 更新到经过测试的新版源码，
重新执行 `python -m pip install -e .`，再通过 `llamafactory-cli version` 记录实际版本。
