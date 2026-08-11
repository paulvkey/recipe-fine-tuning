# 食谱数据处理命令手册

本文档汇总当前项目中可以直接使用的命令。所有命令均在项目根目录执行：

```bash
cd /Users/shirsen/Project/recipe-fine-tuning
```

项目只依赖 Python 3 标准库，不需要额外安装依赖。

## 1. 推荐：执行完整清洗流程

对原始食谱执行随机采样建库、纠错规则生成、保护词生成、全量清洗、质量分流、
精确去重和训练格式转换：

```bash
python3 -m recipe_pipeline recipe_corpus.json
```

默认输出到 `pipeline_output/`：

```text
pipeline_output/
├── generated_config/
│   ├── bootstrap_report.json
│   ├── typo_rules.generated.json
│   └── protected_words.generated.txt
├── recipe_train_clean.jsonl
├── recipe_review.jsonl
├── recipe_rejected.jsonl
├── clean_report.json
└── manifest.json
```

指定输出目录：

```bash
python3 -m recipe_pipeline recipe_corpus.json \
  --output-dir pipeline_output_new
```

## 2. 完整清洗并生成额外审计报告

`--with-audit` 会增加一次原始文件全量扫描，并生成独立审计报告：

```bash
python3 -m recipe_pipeline recipe_corpus.json \
  --output-dir pipeline_output_audit \
  --with-audit
```

额外生成：

```text
pipeline_output_audit/audit_report.json
```

## 3. 清洗时直接限定最终训练数据数量

例如从清洗、质量检查和去重后的合格数据中随机输出 100 万条：

```bash
python3 -m recipe_pipeline recipe_corpus.json \
  --output-dir pipeline_output_100w \
  --target-count 1000000 \
  --selection-seed 20260722
```

- `--target-count`：最终训练文件需要的记录数。
- `--selection-seed`：最终数据选择的随机种子。
- 抽取发生在清洗、分流和去重之后。
- 有效数据少于目标数量时，会输出全部有效数据。

## 4. 完整流程参数示例

```bash
python3 -m recipe_pipeline recipe_corpus.json \
  --output-dir pipeline_output \
  --sample-size 100000 \
  --seed 20260722 \
  --min-typo-count 3 \
  --min-protected-count 10 \
  --min-ingredient-protected-count 30 \
  --progress-every 100000
```

参数说明：

- `--sample-size`：生成纠错规则和保护词时使用的随机样本量，默认 100000；它不是最终训练数据数量。
- `--seed`：规则建库的随机种子。
- `--min-typo-count`：自动启用纠错规则的最低样本命中次数。
- `--min-protected-count`：菜名保护词的最低样本频次。
- `--min-ingredient-protected-count`：食材保护词的最低样本频次。
- `--progress-every`：每处理多少条显示一次进度。

## 5. 小规模预览完整流程

只扫描前 10000 条原始数据，用其中最多 2000 条生成规则，最终输出 100 条训练数据：

```bash
python3 -m recipe_pipeline recipe_corpus.json \
  --output-dir pipeline_preview \
  --scan-limit 10000 \
  --sample-size 2000 \
  --target-count 100
```

该命令适合在修改规则后快速查看效果，不适合作为正式全量训练集。

## 6. 从已经清洗好的数据中抽取 10 万条

当前使用的命令：

```bash
python3 -m recipe_pipeline.sample \
  pipeline_output/recipe_train_clean.jsonl \
  --sample-size 100000 \
  --output-dir training_sample \
  --seed 20260811
```

输出内容：

```text
training_sample/
├── recipe_train_sample_100000.jsonl
└── sample_report.json
```

如果使用默认清洗文件、默认数量和默认种子，可以简化为：

```bash
python3 -m recipe_pipeline.sample
```

此时默认输出到 `training_sample_output/`。

自定义抽取数量：

```bash
python3 -m recipe_pipeline.sample \
  pipeline_output/recipe_train_clean.jsonl \
  --sample-size 200000 \
  --output-dir training_sample_200k \
  --seed 20260811
```

自定义文件名并使用 gzip 压缩：

```bash
python3 -m recipe_pipeline.sample \
  pipeline_output/recipe_train_clean.jsonl \
  --sample-size 100000 \
  --output-dir training_sample_100k_gzip \
  --output-name recipe_train_100k.jsonl.gz
```

输出文件已存在时，脚本默认拒绝覆盖。确认需要覆盖时使用：

```bash
python3 -m recipe_pipeline.sample \
  pipeline_output/recipe_train_clean.jsonl \
  --sample-size 100000 \
  --output-dir training_sample_100k \
  --seed 20260810 \
  --overwrite
```

采样脚本使用两遍顺序扫描和恒定内存算法。相同源文件、数量和随机种子会得到相同结果。
输出保留被选记录在源文件中的相对顺序，正式训练时仍建议启用训练框架的数据打乱功能。

## 7. 只审计原始数据

只生成问题统计和示例，不修改原始数据：

```bash
python3 -m recipe_pipeline.audit \
  recipe_corpus.json \
  recipe_audit.json
```

限制审计数量和每类问题的示例数：

```bash
python3 -m recipe_pipeline.audit \
  recipe_corpus.json \
  recipe_audit_preview.json \
  --limit 10000 \
  --examples 10
```

使用自动生成的纠错规则和保护词进行审计：

```bash
python3 -m recipe_pipeline.audit \
  recipe_corpus.json \
  recipe_audit.json \
  --typo-rules pipeline_output/generated_config/typo_rules.generated.json \
  --noise-rules data/base/noise_rules.json \
  --protected-words pipeline_output/generated_config/protected_words.generated.txt
```

## 8. 只生成纠错规则和保护词

不执行正式清洗，只对原始数据采样并生成配置：

```bash
python3 -m recipe_pipeline.bootstrap \
  recipe_corpus.json \
  generated_config_preview \
  --sample-size 100000 \
  --seed 20260722 \
  --min-typo-count 3 \
  --min-protected-count 10 \
  --min-ingredient-protected-count 30
```

只扫描前 10000 条进行测试：

```bash
python3 -m recipe_pipeline.bootstrap \
  recipe_corpus.json \
  generated_config_preview \
  --scan-limit 10000 \
  --sample-size 2000
```

## 9. 只执行底层清洗器

使用基础规则清洗、分流、去重和转换：

```bash
python3 -m recipe_pipeline.clean \
  recipe_corpus.json \
  recipe_train_clean.jsonl \
  --review recipe_review.jsonl \
  --rejected recipe_rejected.jsonl \
  --report recipe_clean_report.json
```

使用一键流程生成的纠错规则和保护词：

```bash
python3 -m recipe_pipeline.clean \
  recipe_corpus.json \
  recipe_train_clean.jsonl \
  --review recipe_review.jsonl \
  --rejected recipe_rejected.jsonl \
  --report recipe_clean_report.json \
  --typo-rules pipeline_output/generated_config/typo_rules.generated.json \
  --noise-rules data/base/noise_rules.json \
  --protected-words pipeline_output/generated_config/protected_words.generated.txt
```

底层清洗时直接随机输出指定数量：

```bash
python3 -m recipe_pipeline.clean \
  recipe_corpus.json \
  recipe_train_100k.jsonl \
  --target-count 100000 \
  --selection-seed 20260810
```

测试时只处理前 10000 条：

```bash
python3 -m recipe_pipeline.clean \
  recipe_corpus.json \
  recipe_train_preview.jsonl \
  --scan-limit 10000
```

默认开启精确去重。只有明确需要保留重复食谱时才增加：

```bash
--no-deduplicate
```

正式生成训练数据时优先使用 `python3 -m recipe_pipeline`，因为它会先自动生成纠错规则和保护词。

## 10. 只做基础格式转换

转换前 10 条用于查看 `instruction/input/output` 格式：

```bash
python3 -m recipe_pipeline.simple_convert \
  recipe_corpus.json \
  preview.jsonl \
  --limit 10
```

转换完整文件，并保存无法解析或转换的原始行：

```bash
python3 -m recipe_pipeline.simple_convert \
  recipe_corpus.json \
  recipe_train.jsonl \
  --errors recipe_rejected.jsonl
```

输入和输出文件名以 `.gz` 结尾时，会自动读取或生成 gzip 压缩文件：

```bash
python3 -m recipe_pipeline.simple_convert \
  recipe_corpus.jsonl.gz \
  recipe_train.jsonl.gz \
  --errors recipe_rejected.jsonl.gz
```

该转换器不执行错字修正、质量分流和去重，因此只适合格式预览或兼容用途，不建议直接用于正式训练。

## 11. 查看命令帮助

```bash
python3 -m recipe_pipeline --help
python3 -m recipe_pipeline.sample --help
python3 -m recipe_pipeline.bootstrap --help
python3 -m recipe_pipeline.audit --help
python3 -m recipe_pipeline.clean --help
python3 -m recipe_pipeline.simple_convert --help
```

## 12. 常见参数区别

| 使用位置     | 参数               | 作用                               |
| ------------ | ------------------ | ---------------------------------- |
| 完整流程     | `--sample-size`    | 生成纠错规则和保护词时的采样量     |
| 完整流程     | `--target-count`   | 清洗完成后最终输出的训练记录数     |
| 完整流程     | `--seed`           | 规则建库的随机种子                 |
| 完整流程     | `--selection-seed` | 最终训练数据选择的随机种子         |
| 清洗结果采样 | `--sample-size`    | 从现有干净训练集中抽取的记录数     |
| 清洗结果采样 | `--seed`           | 从现有干净训练集中抽取时的随机种子 |
| 测试流程     | `--scan-limit`     | 限制处理的原始非空记录数量         |
| 审计工具     | `--limit`          | 限制审计的原始非空记录数量         |

最常用的三个命令是：

```bash
# 重新执行完整清洗
python3 -m recipe_pipeline recipe_corpus.json

# 从现有清洗结果中抽取 10 万条
python3 -m recipe_pipeline.sample \
  pipeline_output/recipe_train_clean.jsonl \
  --sample-size 100000 \
  --output-dir training_sample_100k_new

# 修改规则后快速预览
python3 -m recipe_pipeline recipe_corpus.json \
  --output-dir pipeline_preview \
  --scan-limit 10000 \
  --sample-size 2000 \
  --target-count 100
```
