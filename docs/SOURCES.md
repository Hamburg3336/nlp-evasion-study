# 数据 模型与第三方来源

本项目自行编写实验组织、分析和复核代码；研究中使用下列公开数据、模型与软件。仓库没有重新分发完整原始数据、预训练／训练权重或课程模板。

| 来源 | 用途 | 入口 |
|---|---|---|
| QEvasion | 政治访谈问答清晰度三分类及九类策略标签 | https://huggingface.co/datasets/ailsntua/QEvasion |
| GLUE QNLI | question／sentence二分类目标任务 | https://huggingface.co/datasets/nyu-mll/glue |
| SQuAD | QNLI的原始问答数据来源 | https://aclanthology.org/D16-1264/ |
| RoBERTa-base | 句对编码器及迁移初始化 | https://huggingface.co/roberta-base |
| all-MiniLM-L6-v2 | 冻结句向量基线 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| PyTorch、Transformers、Sentence Transformers | 训练、分词和推理 | 各项目官方发布渠道 |
| scikit-learn、numpy、pandas、matplotlib | 基线、统计、数据处理和绘图 | 各项目官方发布渠道 |

数据和模型的使用与再分发遵守来源仓库及模型卡所列条款；软件使用各项目许可证。本仓库不为第三方资源授予新的许可。项目自编代码当前没有附加开源许可证，公开托管不自动代表MIT、Apache等许可。

发布CSV只保留预测所需的索引、标签和概率。图表与少量案例摘录用于说明本研究结果，不把完整数据集作为项目自有材料发布。参考输入哈希、导出前预测哈希和训练配置均用于追溯，不代表上游版本永久可用。
