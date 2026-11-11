"""
core_v4.py — 针对「翻唱」补充的旋律级特征

v3 的瓶颈：翻唱 recall 只有 0.18。三路分数里
    landmark     认「同一录音」，翻唱天然低
    sim_timbre   认音色，翻唱换人唱就低
    sim_melody   认旋律，但分数被压在 0.92~0.99，分不开

v2 的 sim_melody 是「12 个半音循环移位的 DTW 代价取最小值」。
问题就在这个「取最小」：不相关的两首歌也有 12 次撞运气的机会，
总能挑出一个偏小的代价，于是所有歌的分数都被抬到接近 1。
而真正同一旋律的歌，正确那个移位应该**明显优于其余 11 个**。

v4 把被丢掉的信息捡回来，输出 7 个新特征（全部只用 v2 的特征序列，不引入新库）：

    shift_gap     第二好的代价 - 最好的代价      （越大 = 这个调越「确凿」）
    shift_margin  12 个代价的均值 - 最好代价    （同上，更稳）
    shift_std     12 个代价的标准差              （同上）
    shift_is_zero 最优移位是否为 0（没有转调）
    tempo_best    在 0.85/1.0/1.18 倍时间伸缩下取最优旋律代价（应对变速）
    seg_median    把查询切成 3 段各自去参考里匹配，取代价中位数（应对片段/局部翻唱）
    seg_worst     3 段里最差的代价（衡量整首是否都吻合，而非只有一段撞上）

另保留 sim_melody（= 1 - 最优代价/2）作为与 v2/v3 的对齐基准。
"""

from __future__ import annotations

import numpy as np

from core_v2 import subseq_dtw_cost

TEMPO_SCALES = (0.85, 1.0, 1.18)
N_SEGMENTS = 3


def shift_costs(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """12 个半音循环移位各自的 DTW 代价，长度 12 的向量。"""
    return np.array([subseq_dtw_cost(X, np.roll(Y, k, axis=1)) for k in range(12)])


def resample_seq(X: np.ndarray, factor: float) -> np.ndarray:
    """把 (n_bins, d) 序列沿时间轴线性插值到 factor 倍长度，并逐行重新归一化。"""
    n = len(X)
    new_n = max(3, int(round(n * factor)))
    if new_n == n:
        return X
    idx = np.linspace(0.0, n - 1, new_n)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (idx - lo)[:, None]
    out = X[lo] * (1.0 - w) + X[hi] * w
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)


def features(a: dict, b: dict) -> dict:
    """输入 core_v2.extract() 的结果，返回 v4 特征字典（越大越像同一旋律）。"""
    X, Y = a["chroma"], b["chroma"]
    if len(X) > len(Y):
        X, Y = Y, X          # X = 较短的查询，Y = 较长的参考

    costs = shift_costs(X, Y)
    k_best = int(np.argmin(costs))
    best, second = float(costs[k_best]), float(np.partition(costs, 1)[1])

    # 变速：只在最优移位上试 3 个时间尺度，控制计算量
    Yr = np.roll(Y, k_best, axis=1)
    tempo_costs = [subseq_dtw_cost(resample_seq(X, s), Yr) for s in TEMPO_SCALES]
    tempo_best = float(min(tempo_costs))

    # 局部片段：短查询切 3 段，各自在完整参考里找最佳对齐
    seg_costs = [subseq_dtw_cost(seg, Yr)
                 for seg in np.array_split(X, N_SEGMENTS) if len(seg) >= 3]
    if not seg_costs:
        seg_costs = [best]

    def sim(c: float) -> float:
        return float(np.clip(1.0 - c / 2.0, 0.0, 1.0))

    return {
        "sim_melody": sim(best),                       # 与 v2/v3 同一口径
        "shift_gap": float(second - best),
        "shift_margin": float(costs.mean() - best),
        "shift_std": float(costs.std()),
        "shift_is_zero": 1.0 if k_best == 0 else 0.0,
        "tempo_best": sim(tempo_best),
        "seg_median": sim(float(np.median(seg_costs))),
        "seg_worst": sim(float(np.max(seg_costs))),
    }


FEATURE_NAMES = ["sim_melody", "shift_gap", "shift_margin", "shift_std",
                 "shift_is_zero", "tempo_best", "seg_median", "seg_worst"]
