"""
features_runtime.py — 训练与推理共用的「单一事实来源」

为什么需要它：
    模型是在 evaluate_v2.py / evaluate_v3.py 产出的分数上训练的。
    如果 predict.py 自己另写一套特征提取（采样率、裁剪长度、hop、分桶方式
    只要差一点），分数分布就会漂移，模型等于白训。
    所以训练侧与推理侧必须调用同一个函数。

与训练数据完全一致的参数：
    sr = 22050，mono，每首取前 60 秒
    sim_timbre / sim_melody : core_v2.extract + core_v2.similarity
    landmark                : core_v3.peaks + core_v3.landmark_score
"""

from __future__ import annotations

import warnings
from pathlib import Path

import librosa

warnings.filterwarnings("ignore")

import core_v2
import core_v3

# 这两个值必须与 evaluate_dataset.py / evaluate_v2.py / evaluate_v3.py 一致。
# 不 import 评测脚本，是为了让推理侧保持轻量、不依赖标注表。
SR = core_v2.SR                 # 22050
CLIP_SECONDS = 60              # 每首取前 60 秒

FEATURE_ORDER = ["landmark", "sim_timbre", "sim_melody"]


def load_audio(path, sr: int = SR, clip_seconds: int | None = CLIP_SECONDS):
    """与 evaluate_v2/v3 完全相同的加载方式（含解码失败的降级）。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"audio file not found: {p}")
    try:
        y, sr_out = librosa.load(str(p), sr=sr, mono=True, duration=clip_seconds)
    except Exception:
        y, sr_out = librosa.load(str(p), sr=sr, mono=True)
        if clip_seconds:
            y = y[: int(sr_out * clip_seconds)]
    return y, sr_out


def compute_features(path_qry, path_ref) -> dict:
    """返回模型所需的三路分数，键名与训练 CSV 严格一致。"""
    y1, sr1 = load_audio(path_qry)
    y2, sr2 = load_audio(path_ref)

    st, sm = core_v2.similarity(core_v2.extract(y1, sr1), core_v2.extract(y2, sr2))
    lm = core_v3.landmark_score(core_v3.peaks(y1, sr1), core_v3.peaks(y2, sr2))
    return {"landmark": float(lm), "sim_timbre": float(st), "sim_melody": float(sm)}


def as_vector(feat: dict):
    """按模型训练时的列顺序生成一行特征。"""
    return [float(feat[k]) for k in FEATURE_ORDER]
