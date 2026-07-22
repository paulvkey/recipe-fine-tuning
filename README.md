# recipe-fine-tuning

## 一键执行

所有脚本只使用 Python 3 标准库，不需要安装第三方包。原始 JSONL 会逐行处理，
适合当前 GB 级语料。

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

一键流程只扫描原文件两遍：第一遍使用蓄水池算法获得无偏随机样本并生成规则数据，
第二遍执行全量清洗、风险分流、SQLite 精确去重和格式转换。默认采样 100,000 条，
对于约 150 万条数据相当于 6.7%：

```bash
python3 -m recipe_pipeline recipe_corpus.json \
  --output-dir pipeline_output \
  --sample-size 100000 \
  --min-typo-count 3 \
  --min-protected-count 10 \
  --min-ingredient-protected-count 30
```

如需额外生成独立全量审计报告，可增加 `--with-audit`，此时会多扫描一遍原文件。

默认输出全部清洗合格的数据。如果只需要随机转换 100 万条：

```bash
python3 -m recipe_pipeline recipe_corpus.json \
  --output-dir pipeline_output_100w \
  --target-count 1000000 \
  --selection-seed 20260722
```

随机定量选择发生在清洗、风险分流和精确去重之后，因此只会从真正可用于训练的
数据中抽取，并能在有效数据充足时精确输出指定数量。该模式会使用输出目录中的
临时磁盘文件，不会把 100 万条训练数据载入内存；运行结束后临时文件自动删除。
应预留大约“完整清洗候选文件 + 最终抽取文件”的临时磁盘空间。如果有效数据少于
请求数量，脚本会输出全部有效数据并给出明显提示。

## 项目结构

```text
recipe_pipeline/
├── __main__.py                 # `python -m recipe_pipeline` 统一入口
├── orchestrator.py             # 编排全部阶段
├── bootstrap.py                # 随机采样，生成纠错规则与保护词
├── audit.py                    # 可选的独立审计
├── clean.py                    # 清洗、分流、去重、转换
├── quality.py                  # 共享规则引擎与质量检测
└── simple_convert.py           # 不做质检的基础/兼容转换器
data/base/                      # 人工维护的小型基础配置
```

## 自动扩充逻辑

- 纠错规则：使用 `canonical_terms.json` 的标准烹饪词和 `confusion_map.json` 的
  单向易混字映射，只提升采样中真实出现且达到阈值的候选。
- 保护词：从随机样本中的高频 `dish` 和食材词自动提取，再与基础保护词合并；
  菜名默认至少出现 10 次，食材默认至少出现 30 次，并排除单位、厨具和操作说明。
- 自动产物与基础配置分离，不会反向覆盖人工规则。
- 数字、温度、时间和用量不会自动修改，只会进入审核文件。

这种方式可以自动覆盖“绰水→焯水”“耗油→蚝油”等高置信度变体，同时避免
开放式拼写模型把地方菜名、品牌名或正常近义词改错。

## 当前质量规则

- 使用 Unicode NFC，保留中文全角标点、波浪号和颜文字，不进行破坏性兼容转换。
- 自动修复 `克糖8→8克糖`、`个鸡蛋2→2个鸡蛋` 等数量顺序颠倒。
- URL 会从描述、食材和步骤中清除；联系方式和推广行会优先删除，而不是整条丢弃。
- 纯“看图/如图”步骤会删除；`如图切块` 会清理为 `切块`。至少两个且一半步骤
  完全依赖图片时才隔离整份食谱。
- 食材中的 `看图花椒`、`一小把（如图示）花椒` 会清理为可独立理解的食材；
  “材料见图”“配料看视频”等关键内容缺失的食谱进入审核。
- 输出不足 35 字符，或步骤只有数字、“好吃”“图片整理”“下次补上”等无操作内容时
  进入审核；正常的简短食谱仍会保留。
- 食材大比例缺少数量时进入审核文件，不猜测或编造原始数量。
- 烹饪时长与腌制、发酵、酿酒、风干、保存时长分开判断；“180天以上的鸡”等
  禽类日龄也不会作为烹饪时长误报。
- 输出超过 8,000 字符的日记型、合集型食谱进入审核文件，避免超过训练上下文。
- 菜名超过 60 字且存在简短 `dish` 时，使用 `dish` 生成更自然的提问。
- “传图专用”“杂七杂八的记录”“作品合集”等占位标题优先改用有效 `dish`，
  没有可用 `dish` 时进入审核。

## 分阶段执行

通常只需使用一键入口。需要单独检查某个阶段时，可执行以下命令。

### 1. 先审计原始文件

```bash
python3 -m recipe_pipeline.audit recipe_corpus.json recipe_audit.json
```

`recipe_audit.json` 会统计结构错误、异常温度/时长、规则命中次数，并为每类问题
保留少量上下文示例。审计不会修改原文件。

### 2. 基础规则数据

- `data/base/typo_rules.json`：人工确认的高置信度错字替换。
- `data/base/noise_rules.json`：平台编号、末尾图片提示、网页链接清理规则。
- `data/base/protected_words.txt`：基础食材名、地方名和烘焙术语。
- `data/base/canonical_terms.json`：自动纠错使用的标准词。
- `data/base/confusion_map.json`：方向明确的易混字映射。

不要把不确定的纠错加入 `typo_rules.json`，尤其不要自动修改温度、时间和用量。

### 3. 转换少量数据进行预览

```bash
python3 -m recipe_pipeline recipe_corpus.json \
  --output-dir pipeline_preview \
  --scan-limit 10000 \
  --sample-size 2000 \
  --target-count 100
```

### 4. 直接调用底层清洗器

下面的命令只使用 `data/base/` 中的基础规则，不执行自动采样建库。正式生成训练集
时应优先使用文档开头的一键脚本。

```bash
python3 -m recipe_pipeline.clean \
  recipe_corpus.json \
  recipe_train_clean.jsonl \
  --review recipe_review.jsonl \
  --rejected recipe_rejected.jsonl \
  --report recipe_clean_report.json
```

结果文件：

- `recipe_train_clean.jsonl`：通过检查且精确去重后的训练数据。
- `recipe_review.jsonl`：温度、时间、联系方式或图片依赖等高风险数据。
- `recipe_rejected.jsonl`：JSON 错误或缺少菜名、食材、步骤的数据。
- `recipe_clean_report.json`：清洗规则、风险原因及数量统计；分别提供影响记录数和
  实际命中次数，避免一份食谱多次命中造成误读。

精确去重默认开启，使用临时 SQLite 数据库，因此不会随数据量增长而大量占用内存。
只有明确需要保留重复数据时才使用 `--no-deduplicate`。

## 基础转换脚本

`recipe_pipeline.simple_convert` 会逐行读取 JSONL，每次内存中只保留一份食谱，适合转换
GB 级文件。输出字段固定为 `output`。

它不执行错字修正、风险隔离和去重。正式训练数据推荐使用上面的
`python3 -m recipe_pipeline`。

先转换 10 条检查效果：

```bash
python3 -m recipe_pipeline.simple_convert recipe_corpus.json preview.jsonl --limit 10
```

转换完整文件，并单独保存不合格的原始行：

```bash
python3 -m recipe_pipeline.simple_convert \
  recipe_corpus.json \
  recipe_train.jsonl \
  --errors recipe_rejected.jsonl
```

输入和输出文件名以 `.gz` 结尾时会自动使用 gzip 压缩。

### 字段映射

- `instruction`：根据食谱名确定性选择一个日常口吻的问法。
- `input`：固定为空字符串，与只输入菜名提问的实际使用方式保持一致。
- `output`：食谱简介、完整食材列表和编号后的制作步骤。
- `dish`：仅在 `name` 缺失时作为备用菜名，不写入训练样本。
- `keywords`、`author`：忽略。

一份食谱只产生一条训练样本，不根据多个近义关键词复制样本，从而降低重复数据
导致的过拟合风险。
