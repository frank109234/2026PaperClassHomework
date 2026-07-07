# L-Defense_EFND 项目复现正式报告

## 一、复现任务概述

本次复现的对象为 WWW 2024 论文 *Explainable Fake News Detection With Large Language Model via Defense Among Competing Wisdom* 对应的开源项目 `L-Defense_EFND-main`。本次工作的目标是在 `RAWFC` 数据集上完成论文方法的完整三阶段流程复现，包括证据提取、解释生成与最终分类，并尽可能贴近论文设定获得可比结果。

论文方法整体由三个阶段构成：

1. `Step1`：训练 evidence extractor，从多篇报道中提取与 claim 相关的证据句。
2. `Step2`：基于提取出的正反向证据，生成 label-oriented explanations。
3. `Step3`：将 claim 与 explanations 共同输入分类模型，完成最终真假判定。

本次复现不仅要求完整跑通工程链路，还重点考察：在本地修复若干代码问题、替换 LLM 接口实现后，模型最终性能与论文报告结果的接近程度。

## 二、实验环境

本次复现实验运行于远程 GPU 服务器环境，主要配置如下：

- 平台环境：AutoDL 容器
- GPU：`NVIDIA GeForce RTX 3090 24GB`
- Python：`3.8.20`
- PyTorch：`2.4.1+cu121`
- CUDA（Torch 对应版本）：`12.1`
- Transformers：`4.46.3`

项目运行目录为：

```bash
/root/完整复现/L-Defense_EFND-main
```

预训练模型本地路径为：

```bash
/root/models/roberta-base
/root/models/roberta-large
```

由于服务器环境无法直接连接 Hugging Face 下载模型，因此 `roberta-base` 与 `roberta-large` 均采用“本地下载后上传服务器”的方式准备。

## 三、复现过程中发现的问题与修复

在正式运行论文流程之前，项目原始代码存在若干影响运行的问题，需要先行修复。

### 1. 数据路径硬编码问题

原始代码中部分数据路径被写死，导致数据读取时无法适配当前实际目录结构，运行时会报找不到路径的错误。为解决该问题，对 `dataset.py` 进行了修改，将数据根目录由硬编码方式改为根据项目所在位置动态推断，并分别引入原始数据目录与 Step2 阶段数据目录的统一管理逻辑。

### 2. 非法字符导致语法错误

在 `source/dataset.py` 中发现中文全角引号，如：

```python
“explanation”
```

该问题会直接导致 Python 解释器抛出语法错误，阻塞后续运行。修复时将所有异常引号替换为标准英文半角引号。

### 3. Hugging Face 模型在线下载失败

服务器在加载 `roberta-base` 时出现无法连接 Hugging Face 的错误，导致 `from_pretrained` 失败。为解决这一问题，先在本地使用 `huggingface_hub` 下载所需模型文件，再通过 `Xftp` 上传到服务器本地目录，后续训练与推理统一使用本地模型路径。

### 4. Step2 调用方式需兼容非 OpenAI 官方接口

原始 `step2_explanation_generation.py` 中调用方式与 API 管理方式不够灵活，不便于切换到兼容 OpenAI SDK 的第三方接口。为适配本次实验所采用的 `DeepSeek API`，将该脚本重构为通过环境变量读取：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

从而支持 OpenAI-compatible 推理接口。

### 5. Step2 数据目录出现嵌套“套娃”问题

在运行 `Step2` 后，`RAWFC_step2` 目录实际出现了嵌套结构，如：

```bash
dataset/RAWFC_step2/RAWFC_step2/gpt/...
```

这导致 explanation 文件虽然生成成功，但下游 `Step3` 读取路径时可能对不上。为保证代码鲁棒性，对 `dataset.py` 和 `step2_explanation_generation.py` 进行了进一步补强，使其能够自动识别并适配该类嵌套目录结构，避免后续运行中因路径不一致造成读取失败。

## 四、复现过程

### 1. Step1：训练 evidence extractor 并提取 evidences

首先使用 `roberta-base` 训练 evidence extractor，并确认训练链路可正常运行。随后执行证据导出，成功生成以下文件：

- `train_10_evidence_details.json`
- `eval_10_evidence_details.json`
- `test_10_evidence_details.json`

这说明 Step1 已顺利完成，其输出将作为 Step2 的输入数据。

### 2. Step2：生成 label-oriented explanations

在 Step2 中，论文原始方法使用 ChatGPT 生成正反向 explanations。本次复现未使用论文中的原始 ChatGPT 接口，而是采用 `DeepSeek API` 作为替代方案，模型设置为：

```bash
deepseek-v4-flash
```

实验先进行了小规模试跑，以验证：

- API 能否正确连接
- explanation 文件能否正常生成
- 目录写入逻辑是否正确

在试跑成功后，继续执行全量 explanation generation。最终得到 explanation 数量如下：

- `eval = 400`
- `test = 400`
- `train = 3224`

原因在于每条 claim 会分别生成两条 explanations（true-oriented / false-oriented）：

- `eval` 共 200 条 claim，对应 400 条 explanations
- `test` 共 200 条 claim，对应 400 条 explanations
- `train` 共 1612 条 claim，对应 3224 条 explanations

这表明 Step2 已完整完成。

### 3. Step3：最终分类

Step3 使用 `roberta-large` 作为最终分类器。在正式训练前，先通过 smoke test 验证：

- 解释文件是否可正确读取
- claim 与 explanations 是否能正常拼接成模型输入
- 训练、评估、预测与结果保存流程是否完整可用

在 smoke test 成功后，进行了两组正式实验。

#### （1）保守显存配置实验

考虑到 `RTX 3090 24GB` 显存限制，首先采用较保守的配置：

- `train_batch_size = 2`
- `eval_batch_size = 4`
- `learning_rate = 5e-6`
- `num_train_epochs = 5`

最终结果为：

- `precision = 55.92%`
- `recall = 56.05%`
- `macro-f1 = 55.62%`
- `accuracy = 56.0%`

该结果说明：虽然工程流程完全打通，但与论文结果仍有较明显差距。

#### （2）接近论文设定实验

之后，为尽可能贴近论文中的 Step3 设定，采用如下配置重新运行：

- `train_batch_size = 8`
- `eval_batch_size = 32`
- `learning_rate = 5e-6`
- `num_train_epochs = 5`

最终结果为：

- `precision = 60.78%`
- `recall = 60.55%`
- `macro-f1 = 60.25%`
- `accuracy = 60.5%`

相比保守配置实验，`macro-f1` 提升了 `4.63` 个点，说明 Step3 的 batch size 等训练设定对最终性能影响显著。

## 五、与论文结果的对比分析

根据论文 Table 2，作者在 `RAWFC` 数据集上的主结果约为：

- `precision = 61.72%`
- `recall = 61.01%`
- `macro-f1 = 61.20%`

而本次在接近论文设定下得到的结果为：

- `precision = 60.78%`
- `recall = 60.55%`
- `macro-f1 = 60.25%`

二者差距为：

- `macro-f1` 差约 `0.95`
- `precision` 差约 `0.94`
- `recall` 差约 `0.46`

从结果来看，本次复现已经非常接近论文报告值，可以认为是一次较高质量的近似复现。

## 六、差距来源分析

虽然结果已经接近论文，但仍存在不到 1 个点的差距。结合本次实验条件，可能原因主要包括以下几点：

### 1. Step2 使用的 LLM 与论文不一致

论文中 Step2 使用的是 ChatGPT，而本次复现采用的是 `DeepSeek API`。由于 Step2 所生成的 explanations 会直接影响 Step3 的输入质量，不同 LLM 的表达风格、内容准确性、推理偏好都可能带来最终分类性能上的差异。

### 2. 论文报告通常取多次运行中的最佳值

论文附录中说明，最终 veracity prediction 结果通常取多次运行中的最佳结果，而本次实验主要记录的是单次运行结果。因此，单次实验结果略低于论文报告值是合理现象。

### 3. 依赖版本与环境存在差异

本次复现使用的 PyTorch 与 Transformers 版本高于 README 中推荐版本。虽然运行无误，但版本差异仍可能影响训练过程中的一些数值行为和最终性能。

### 4. 数据目录与项目代码曾被修复调整

虽然修复操作并未改变论文方法核心逻辑，但项目在实际运行时经过了路径与兼容性调整，这些工程层面的修补理论上也可能对部分运行细节产生轻微影响。

## 七、结论

本次复现成功完成了论文方法在 `RAWFC` 数据集上的完整三阶段流程：

1. 训练 extractor 并导出 evidences
2. 基于 evidences 生成正反向 explanations
3. 使用 explanations 完成最终真假分类

在保守显存配置下，最终结果为：

- `macro-f1 = 55.62%`

在接近论文设定下，最终结果提升至：

- `macro-f1 = 60.25%`

与论文报告的 `61.20%` 相比，仅差约 `0.95` 个点。综合考虑 Step2 的 LLM 替换、运行次数差异及环境版本差异，可以认为本次复现已经较高程度地还原了论文在 `RAWFC` 数据集上的性能表现。

因此，本次实验结论为：

**该项目在当前环境下已成功复现，且在接近论文设定时能够获得与论文结果非常接近的实验性能，复现结果总体可信。**

## 八、实验启示

本次复现的主要经验如下：

1. **应先用 smoke test 验证流程完整性，再进行大规模训练。**  
   这样能显著降低排查成本，提高复现效率。

2. **Step2 explanation generation 对整体性能具有关键影响。**  
   最终效果不仅依赖分类器本身，也高度依赖 explanation 质量。

3. **Step3 的 batch size 与训练设定对性能影响明显。**  
   本次实验中，仅调整 Step3 为更接近论文设定，就带来了超过 4 个点的 `macro-f1` 提升。

4. **对于无法联网下载模型的环境，本地下载再上传是可行方案。**  
   对远程 GPU 服务器复现工作具有较强实用价值。
