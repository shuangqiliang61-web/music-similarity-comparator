"""
core_v6.py — 换「信号表示」：谐波分离 + CQT 色度 + 离散音级轮廓

v4/v5 的教训：只要还在同一个 STFT-CQT 余弦/DTW 代价上做变换，
新特征就只是旧分数的变体（相关系数 0.99），加进去只会过拟合。

v6 改的是**信号从哪儿来、怎么表示**，三件事都是项目里从未出现过的：

1) 谐波/打击分离 (librosa.effects.harmonic)
   翻唱常换鼓点与编曲，打击乐会污染色度。先只保留谐波部分，
   再提色度，等价于简化的 HPCP —— 翻唱识别（MIREX cover song task）
   的经典前端。此前所有版本都是在原始混合音频上算色度。

2) CQT 色度 (chroma_cqt) 代替 STFT 色度
   CQT 的频率轴按音程等比划分，低音区音准分辨率远高于线性 STFT，
   对变调 / 换乐器更稳。此前 v2 用的是 chroma_stft，v5 用的是 CENS
   （CENS 是 CQT 色度的平滑量化版，但仍在混合音频上算）。

3) 离散音级轮廓 (pitch class sequence)
   把每 0.5 秒桶里最强的音级取出来，得到一串 0~11 的整数，
   用「滑窗精确匹配率」比对（允许 ±1 半音容差、12 个移位）。
   这是**离散符号比对**，与所有连续余弦/DTW 分数本质不同，
   理论上对音色、编曲、录音差异最不敏感 —— 正是翻唱需要的。

依赖仍只有 numpy / librosa，不下载任何模型权重。
"""
from __future__ import annotations

import numpy as np
import librosa

from core_v2 import subseq_dtw_cost, _pool_to_bins, HOP

SR = 22050


def _harmonic(y: np.ndarray):
    """谐波分离；若该 librosa 版本不支持则退化为原音频（并如实记录）。"""
    try:
        return librosa.effects.harmonic(y, margin=3.0), True
    except Exception:
        return y, False


def _pitch_seq(bins: np.ndarray) -> np.ndarray:
    """>(n_bins, 12) -> (n_bins,) 每桶最强音级的整数编号。"""
    return np.argmax(bins, axis=1).astype(np.int16)


def _sliding_match(a: np.ndarray, b: np.ndarray) -> float:
    """短序列在长序列里滑动，取最大「精确 / ±1 半音」匹配率。"""
    if len(a) > len(b):
        a, b = b, a
    n, m = len(a), len(b)
    if n == 0:
        return 0.0
    best = 0.0
    for off in range(0, max(1, m - n + 1)):
        seg = b[off:off + n]
        exact = float(np.mean(a == seg))
        d = np.abs(a.astype(int) - seg.astype(int))
        fuzzy = float(np.mean(np.minimum(d, 12 - d) <= 1))
        best = max(best, 0.5 * exact + 0.5 * fuzzy)
    return float(best)


def extract(y, sr: int = SR) -> dict:
    y_h, ok = _harmonic(y)
    # CQT 色度：hop_length 与 v2 对齐，便于 _pool_to_bins
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr, hop_length=HOP)
    bins = _pool_to_bins(chroma)
    hratio = float(np.linalg.norm(y_h) / (np.linalg.norm(y) + 1e-9))
    return {
        "hpcp": bins,
        "pitch": _pitch_seq(bins),
        "hratio": hratio,
        "harmonic_ok": ok,
    }


def _sim(cost: float) -> float:
    return float(np.clip(1.0 - cost / 2.0, 0.0, 1.0))


def features(a: dict, b: dict) -> dict:
    X, Y = a["hpcp"], b["hpcp"]
    if len(X) > len(Y):
        X, Y = Y, X
    costs = np.array([subseq_dtw_cost(X, np.roll(Y, k, axis=1))
                      for k in range(12)])
    best_k = int(costs.argmin())
    best = float(costs.min())
    second = float(np.partition(costs, 1)[1])

    # 不移位时的全局平均色度余弦（绝对调相同会更像）
    ga = X.mean(axis=0)
    gb = Y.mean(axis=0)
    flat = float(np.dot(ga, gb) / (np.linalg.norm(ga) * np.linalg.norm(gb) + 1e-9))

    # 离散音级轮廓：12 个移位 × 滑窗，取最优
    pa, pb = a["pitch"], b["pitch"]
    pscores = np.array([
        _sliding_match((pa - k) % 12, pb) for k in range(12)
    ])
    pk = int(pscores.argmax())

    ra, rb = a["hratio"], b["hratio"]
    return {
        "hpcp_sim": _sim(best),
        "hpcp_gap": float(second - best),
        "hpcp_flat": flat,
        "hpcp_shift_zero": float(best_k == 0),
        "pitch_match": float(pscores.max()),
        "pitch_shift_zero": float(pk == 0),
        "hratio_sim": float(1.0 - abs(ra - rb) / (max(ra, rb) + 1e-9)),
    }


FEATURE_NAMES = ["hpcp_sim", "hpcp_gap", "hpcp_flat", "hpcp_shift_zero",
                 "pitch_match", "pitch_shift_zero", "hratio_sim"]
