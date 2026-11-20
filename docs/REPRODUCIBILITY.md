# 复现说明

## 两种复核范围

`scripts/verify_results.py`从发布CSV重算34组最终测试指标，不依赖原数据和检查点。`scripts/verify_bootstrap.py`使用发布的12份2×2预测与回答组编号复算五项区间；后者仅需numpy。两者证明已保存结果之间的一致性，不能替代重训复现。

完整重训需要QEvasion／QNLI数据、RoBERTa和MiniLM模型，并重新生成数据清洗、模型检查点、开发预测和补充分析。所有输出都写在忽略目录，发布的results不是训练输入。

## 环境

建议Python 3.11。先安装与设备相容的PyTorch，再安装其余依赖，例如原设备对应CUDA 12.8：

```bash
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-ml.txt
```

CPU环境可按PyTorch官方说明安装CPU版本。原编码器训练和固定表示探测使用CUDA bf16；两组GPU补充训练入口不会静默切换为不同精度的CPU实验。只核对已发布指标不需要torch。

`requirements.txt`和`requirements-ml.txt`给出依赖范围；`requirements-lock.reference.txt`保留原运行记录，包括torch 2.11.0+cu128、transformers 5.19.0、sentence-transformers 6.1.0及scikit-learn 1.7.2。跨版本和跨硬件完全一致未被保证，不应把原版本文件称为适用于任意设备的一键安装锁。

## 输入获取与核对

原任务为[ailsntua/QEvasion](https://huggingface.co/datasets/ailsntua/QEvasion)，目标为[nyu-mll/glue的qnli配置](https://huggingface.co/datasets/nyu-mll/glue)。预训练模型为[roberta-base](https://huggingface.co/roberta-base)和[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)。

在仓库根目录执行：

```bash
python experiment/src/clean_data.py
python experiment/src/fetch_model.py roberta-base
python experiment/src/fetch_model.py sentence-transformers/all-MiniLM-L6-v2
python scripts/record_inputs.py
```

clean_data.py在缺少本地原始数据时取得数据并清洗；fetch_model.py下载模型到experiment/models。QNLI由迁移阶段取得、适配及分组。下载器默认使用官方站点，可用HF_ENDPOINT选择其他入口。

`provenance/reference_input_checksums.json`记录原实验两个原始parquet及两套模型配置／权重的字节哈希。可将已合法取得的同一原始快照放入experiment/data/raw，并将模型放入experiment/models，再执行：

```bash
python scripts/record_inputs.py --strict-reference
```

普通登记会记录本次实际取得的文件哈希，并报告是否与原快照一致；严格模式不一致就失败。由不同parquet写入版本重新序列化，即使记录相同也可能改变字节哈希。原运行没有记录所有上游commit，本仓库不虚构固定revision；取得当前main版本不等于取得原快照。

正式清洗入口是clean_data.py；data_prep.py只做字段规范化，单独执行会产生未完成文本清洗的数据。

## 运行顺序

先在新工作副本中查看全流程：

```bash
python run_study.py --stage all --dry-run
```

执行全部阶段：

```bash
python run_study.py --stage all
```

原任务阶段包括多数类、四个TF-IDF候选、MiniLM、三类三种子、九类三种子、QNLI源初始化与原始初始化的冻结／整体微调、长度基线、结果汇总及核验。模型与配置均按开发指标选择；不将测试分数用于选择主配置。

补充阶段包括层级开发推理、数据重叠和窗口审计、六组不加权训练、开发预测、2×2分析与配对区间、三源种子的固定表示＋LR、原始细类冲突回查，以及六组歧义拟合排除重训和分析。

可单独执行原任务或补充阶段：

```bash
python run_study.py --stage primary
python run_study.py --stage supplement
```

补充阶段依赖已完成的原任务数据、开发划分及模型。除明确校验续跑计划的权重对照脚本外，补充分析多数要求新输出目录；重复执行已完成的全流程通常会因目录存在而停止。需要局部重跑时，查看--dry-run列出的对应命令并指定新的--output，依赖参数同时指向新产物。

analyze_ambiguity_controls.py还读取同一仓库补充实验目录中的原权重消融产物，这是完整2×2比较的输入。不要只复制该脚本到别处运行。核验中的assert是实验约束，运行不要加python -O。

## 发布结果与新运行

发布预测去除完整问题／回答，只保留索引、金标、预测和概率；34份对应指标位于results/metrics。原始导出来源哈希保存在results/index.json。公开回答组编号保留同一回答的组结构而不复制回答正文。

provenance/run_configs保存原模型训练设置，绝对路径已替换为仓库相对位置。它们是既有运行的配置记录，不是新运行的配置：源代码调整、工作目录和依赖变化会改变新清单中的哈希。新实验应保留自己的数据、源代码、运行ID和输出清单，不能用新哈希覆盖既有结果的来源记录。
