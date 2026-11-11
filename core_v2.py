"""
core_v2.py — 改进版特征：保留时序 + 子序列 DTW 对齐

v1 的问题（见 评测报告.md）：
    把整首歌的 MFCC / Chroma 求时间平均，压成 24 维向量再算余弦，
    任意两首真实音乐的平均频谱包络都很像，相似度普遍 0.85~1.0，三类分不开。

v2 的改动：
    1. 不再求全曲平均，改成「按固定时长分桶的特征序列」（约每 0.5 秒一帧）
       -> 保留时序/旋律走向。
    2. 用子序列 DTW 对齐：允许较短的查询音频在较长的参考音频里滑动匹配
       -> 应对片段截取、变速、前后裁剪。
    3. Chroma 做 12 个半音的循环移位取最优代价
       -> 对「翻唱换了调」不变。
    4. 输出两个分数而不是一个：
         sim_timbre (MFCC 序列)  -> 判断是否同一录音（搬运）
         sim_melody (Chroma 序列) -> 判断是否同一旋律（翻唱）
       两级判定：timbre 高 -> copy；否则 melody 高 -> cover；否则 unrelated。

依赖与 core.py 完全一致（numpy / librosa），不引入新库。
"""

from __future__ import annotations

import numpy as np
import librosa

SR = 22050
HOP = 2048                 # STFT 跳长
BIN_SECONDS = 0.5          # 每个时间桶的时长
FRAMES_PER_BIN = max(1, int(BIN_SECONDS * SR / HOP))   # ≈ 5 帧 / 桶

# 默认阈值（在标注集上调参前先用这组；evaluate_v2.py 会网格搜索最优值）
TH_TIMBRE = 0.90           # sim_timbre >= 此值 -> copy
TH_MELODY = 0.80           # 否则 sim_melody >= 此值 -> cover


# ---------- 1. 特征序列 ----------

def _pool_to_bins(mat: np.ndarray) -> np.ndarray:
    """把 (d, T) 的特征矩阵按固定帧数分桶 -> (n_bins, d)，每桶做 L2 归一化。

    归一化是逐桶做的：去掉响度影响，但保留「这一刻的音色/音高分布」。
    """
    d, T = mat.shape
    if T == 0:
        return np.zeros((1, d))
    n = max(1, T // FRAMES_PER_BIN)
    out = np.zeros((n, d))
    for i in range(n):
        seg = mat[:, i * FRAMES_PER_BIN: (i + 1) * FRAMES_PER_BIN]
        v = seg.mean(axis=1) if seg.shape[1] else np.zeros(d)
        out[i] = v / (np.linalg.norm(v) + 1e-9)
    return out


def extract(y, sr: int = SR) -> dict:
    """返回两段序列特征：chroma 旋律序列、mfcc 音色序列。"""
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP)       # (12, T)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=HOP)[1:]  # (19, T)
    return {"chroma": _pool_to_bins(chroma), "mfcc": _pool_to_bins(mfcc)}


# ---------- 2. 子序列 DTW ----------

def subseq_dtw_cost(X: np.ndarray, Y: np.ndarray) -> float:
    """子序列 DTW 的「平均每步代价」。

    允许较短的 X 在较长的 Y 中任意位置起始匹配（acc[0, :] = 0），
    终点取最后一行的最小值。代价 = 1 - 逐帧余弦相似度，范围 [0, 2]。
    返回按对齐路径长度归一化后的平均代价，0 表示完全一致。
    """
    if len(X) > len(Y):
        X, Y = Y, X
    n, m = len(X), len(Y)
    if n == 0 or m == 0:
        return 2.0

    D = 1.0 - np.clip(X @ Y.T, -1.0, 1.0)        # (n, m)
    INF = 1e12
    acc = np.full((n + 1, m + 1), INF)
    acc[0, :] = 0.0                                # 任意起点免费
    for i in range(1, n + 1):
        di = D[i - 1]
        prev = acc[i - 1]
        cur = acc[i]
        for j in range(1, m + 1):
            cur[j] = di[j - 1] + min(prev[j - 1], prev[j], cur[j - 1])

    end_j = int(np.argmin(acc[n, 1:])) + 1
    total = float(acc[n, end_j])
    if not np.isfinite(total):
        return 2.0

    # 回溯求路径长度，用于归一化
    i, j, steps = n, end_j, 0
    while i > 0 and j > 0 and steps <= n + m:
        steps += 1
        k = int(np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]]))
        if k == 0:
            i -= 1
            j -= 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    return total / max(steps, 1)


def best_shift_cost(X: np.ndarray, Y: np.ndarray) -> float:
    """Chroma 序列：12 个半音循环移位取最小 DTW 代价（对转调不变）。"""
    best = 2.0
    for k in range(12):
        c = subseq_dtw_cost(X, np.roll(Y, k, axis=1))
        if c < best:
            best = c
    return best


# ---------- 3. 相似度与分类 ----------

def similarity(a: dict, b: dict) -> tuple[float, float]:
    """返回 (sim_timbre, sim_melody)，都在 0~1 之间，越大越像。"""
    cost_t = subseq_dtw_cost(a["mfcc"], b["mfcc"])
    cost_m = best_shift_cost(a["chroma"], b["chroma"])
    st = float(np.clip(1.0 - cost_t / 2.0, 0.0, 1.0))
    sm = float(np.clip(1.0 - cost_m / 2.0, 0.0, 1.0))
    return st, sm


def classify(sim_timbre: float, sim_melody: float,
             th_timbre: float = TH_TIMBRE, th_melody: float = TH_MELODY) -> str:
    """两级判定：先认「同一录音」，再认「同一旋律」。"""
    if sim_timbre >= th_timbre:
        return "copy"
    if sim_melody >= th_melody:
        return "cover"
    return "unrelated"
