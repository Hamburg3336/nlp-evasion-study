"""全局配置：路径、随机种子、标签体系与超参。

所有脚本统一从这里取配置，保证「同一数据划分、同一标签映射、同一主指标」。
"""

import os
from pathlib import Path

# ---------------------------------------------------------------- 路径
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
# 修订实验独立落盘，避免与早期不同评测协议的指标混用。
OUT = Path(os.environ.get("NLP_OUTPUT_DIR", str(ROOT / "outputs" / "revised")))
FIG = OUT / "figures"
METRICS = OUT / "metrics"
PRED = OUT / "predictions"
CKPT = OUT / "models"  # 训练好的权重落盘处（迁移实验要复用）


def ensure_directories():
    """Create runtime output folders only for an explicitly executed stage."""
    for directory in (DATA_RAW, DATA_PROC, FIG, METRICS, PRED, CKPT):
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 随机种子
# 模板要求：至少固定 Python / NumPy / 框架的种子，多次运行给均值与标准差
SEED = 42
SEEDS = [42, 2024, 3407]

# ---------------------------------------------------------------- 标签体系
# 3 类高层 clarity（与数据集中 clarity_label 的原始字符串完全一致）
CLARITY_LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]

# 9 类细粒度 evasion（仅 train 有标签；官方 test 的 evasion_label 全为空字符串）
EVASION_LABELS = [
    "Explicit",
    "Implicit",
    "Dodging",
    "General",
    "Deflection",
    "Partial/half-answer",
    "Declining to answer",
    "Claims ignorance",
    "Clarification",
]

# 论文 taxonomy 的确定性映射：细粒度 evasion -> 高层 clarity
# 依据：Thomas et al. (2024) Findings of EMNLP，已用 train 集计数交叉验证
EVASION_TO_CLARITY = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent",
    "Dodging": "Ambivalent",
    "General": "Ambivalent",
    "Deflection": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply",
}

# 绝对不能作为输入特征的列：GPT-3.5 的摘要与预测，喂进模型会造成标签泄漏
LEAKY_COLS = ["gpt3.5_summary", "gpt3.5_prediction"]

# ---------------------------------------------------------------- 文本字段
FIELD_TEXT_A = "question"  # 子问题，平均约 85 字符
FIELD_TEXT_B = "interview_answer"  # 完整回答，平均约 1663 字符
FIELD_GROUP = "title"  # 标题组；同名title可能跨日期/总统，不等于访谈场次

# ---------------------------------------------------------------- 文本截断
# answer 极长（p95=5025 字符），截断策略必须写进报告的「表2 训练配置记录」
MAX_LEN = 512  # token 上限（BERT/RoBERTa）
# 截断方式。默认 longest_first 是 HF 原生策略，也是已完成实验所用设置，
# 改动会改变训练行为、使已有指标与报告失效，因此只在做消融时用 --trunc 显式覆盖。
#   longest_first -> 优先缩短最长序列；默认右侧截断，保留其开头
#   head          -> 只保留回答开头
#   tail          -> 只保留回答结尾
#   head_tail     -> 保留开头 75% + 结尾 25%，避免长回答的结论句被截掉
TRUNC_MODE = "longest_first"
HEAD_RATIO = 0.75  # head_tail 模式下头部占比，剩余留给尾部

# ---------------------------------------------------------------- 基线 A：TF-IDF + LR
TFIDF_CONFIGS = [
    {"name": "tfidf_uni", "ngram_range": (1, 1), "class_weight": None},
    {"name": "tfidf_uni_bal", "ngram_range": (1, 1), "class_weight": "balanced"},
    {"name": "tfidf_uni_bi", "ngram_range": (1, 2), "class_weight": None},
    {"name": "tfidf_uni_bi_bal", "ngram_range": (1, 2), "class_weight": "balanced"},
]
TFIDF_MAX_FEATURES = 50000
TFIDF_MIN_DF = 2
SEP = " [SEP] "  # 问题与回答之间的显式分隔符

# ---------------------------------------------------------------- 基线 B：预训练模型
# 本地优先：本机 huggingface_hub 经代理下载会写出 0 字节文件，直接联网必失败。
# 因此若 models/ 下已有该权重就指向本地目录，没有才回退到 HF 仓库名（换机器时可自动下载）。
_MODELS_DIR = ROOT / "models"
_ST_LOCAL = _MODELS_DIR / "all-MiniLM-L6-v2"
_ENC_LOCAL = _MODELS_DIR / "roberta-base"

ST_MODEL = str(_ST_LOCAL) if _ST_LOCAL.exists() else "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_MODEL = str(_ENC_LOCAL) if _ENC_LOCAL.exists() else "roberta-base"
ENCODER_LR = 2e-5
ENCODER_BATCH = 8
ENCODER_EPOCHS = 5
ENCODER_WARMUP_RATIO = 0.1
ENCODER_EARLY_STOP = 2  # 连续多少轮验证集 Macro-F1 不升就停

# ---------------------------------------------------------------- 评测
MAIN_METRIC = "macro_f1"  # 主指标
REPORT_METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]

# ---------------------------------------------------------------- 中文绘图字体（Windows）
MPL_FONT = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
