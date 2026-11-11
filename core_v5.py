"""
core_v5.py — 引入两个「全新信息源」，而不是旧分数的变体

v4 的教训：tempo_best / seg_median 与 sim_melody 相关系数 0.99，
因为它们全是同一组 DTW 代价的重新组合，没有新信息。

v5 补的是两个此前完全没用过的维度：

1) CENS 色度（librosa.feature.chroma_cens）
   专为「音频匹配 / 翻唱识别」设计：在 CQT 色度基础上做量化、平滑、
   时间池化，对音色、演奏力度、编曲差异比普通 STFT-Chroma 更鲁棒。
   用 12 半音循环移位 + 子序列 DTW 得到 sim_cens，另取移位代价的
   best/second 差作为 cens_gap（转调确凿度）。

2) 节奏 / 起音（onset envelope）
   此前所有特征只看「频谱形状 + 音高分布」，节奏信息从未进入模型。
   翻唱通常会保留节奏骨架，不相关歌曲则不同。这里用
     - 每 0.5 秒一桶的起音形状序列（8 维/桶）做子序列 DTW -> sim_rhythm
     - onset 自相关的主周期比 -> tempo_ratio（对变速有一定容忍）
     - onset 自相关曲线的余弦相似度 -> rhythm_acf_sim

依赖仍是 numpy / librosa，不引入新库。
"""

from __future__ import annotations

import numpy as np
import librosa

from core_v2 import subseq_dtw_cost, _pool_to_bins, HOP, FRAMES_PER_BIN

SR = 22050
RHYTHM_DIM = 8          # 每个 0.5 秒桶内保留的起音形状维度
ACF_MIN_SEC = 0.25      # 自相关主周期搜索范围
ACF_MAX_SEC = 1.50


def _onset_bins(onset: np.ndarray) -> np.ndarray:
    """(T,) 起音强度 -> (n_bins, RHYTHM_DIM)，逐桶 L2 归一化。

    保留桶内的起音形状（而不是只取均值），否则单维归一化后会退化成 ±1。
    """
    n = max(1, len(onset) // FRAMES_PER_BIN)
    out = np.zeros((n, RHYTHM_DIM))
    src = np.arange(FRAMES_PER_BIN, dtype=float)
    dst = np.linspace(0, FRAMES_PER_BIN - 1, RHYTHM_DIM)
    for i in range(n):
        seg = onset[i * FRAMES_PER_BIN:(i + 1) * FRAMES_PER_BIN]
        if len(seg) < FRAMES_PER_BIN:
            seg = np.pad(seg, (0, FRAMES_PER_BIN - len(seg)))
        out[i] = np.interp(dst, src, seg)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)


def _onset_acf(onset: np.ndarray, sr: int = SR) -> tuple[int, np.ndarray]:
    """onset 包络的自相关；返回 (主周期帧数, 归一化自相关曲线)。"""
    o = onset - float(onset.mean())
    if len(o) < 8 or float(np.dot(o, o)) < 1e-12:
        return 0, np.zeros(64)
    full = np.correlate(o, o, mode="full")[len(o) - 1:]
    ac = full / (full[0] + 1e-9)
    lo = max(1, int(ACF_MIN_SEC * sr / HOP))
    hi = min(len(ac) - 1, int(ACF_MAX_SEC * sr / HOP))
    if hi <= lo:
        return 0, ac[:64]
    k = lo + int(np.argmax(ac[lo:hi]))
    return k, ac[:64]


def extract(y, sr: int = SR) -> dict:
    """从波形提取 v5 所需的序列特征（与 v2/v3 的特征分开缓存）。"""
    cens = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=HOP)      # (12, T)
    # librosa 0.11 里叫 onset_strength（旧文档里的 onset_envelope 已不存在）
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)     # (T,)
    k, acf = _onset_acf(onset, sr)
    return {
        "cens": _pool_to_bins(cens),
        "rhythm": _onset_bins(onset),
        "onset_lag": k,
        "onset_acf": acf / (np.linalg.norm(acf) + 1e-9),
    }


def _sim(cost: float) -> float:
    return float(np.clip(1.0 - cost / 2.0, 0.0, 1.0))


def features(a: dict, b: dict) -> dict:
    """返回 5 个 v5 特征，全部越大越像。"""
    # 1) CENS 色度：12 半音循环移位取最优，同时记录 best 与次好的差距
    X, Y = a["cens"], b["cens"]
    if len(X) > len(Y):
        X, Y = Y, X
    costs = np.array([subseq_dtw_cost(X, np.roll(Y, k, axis=1)) for k in range(12)])
    best = float(costs.min())
    second = float(np.partition(costs, 1)[1])

    # 2) 节奏形状序列
    rhythm_cost = subseq_dtw_cost(a["rhythm"], b["rhythm"])

    # 3) 速度 / 周期性
    la, lb = a["onset_lag"], b["onset_lag"]
    tempo_ratio = (min(la, lb) / max(la, lb)) if (la > 0 and lb > 0) else 0.0
    acf_sim = float(np.dot(a["onset_acf"], b["onset_acf"]))

    return {
        "sim_cens": _sim(best),
        "cens_gap": float(second - best),
        "sim_rhythm": _sim(rhythm_cost),
        "tempo_ratio": float(tempo_ratio),
        "rhythm_acf_sim": acf_sim,
    }


FEATURE_NAMES = ["sim_cens", "cens_gap", "sim_rhythm", "tempo_ratio", "rhythm_acf_sim"]
