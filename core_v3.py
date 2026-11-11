"""
core_v3.py — 频谱峰值指纹（landmark）匹配，专攻「搬运」判定

为什么需要它：
    v1 用全曲平均向量、v2 用归一化特征序列 + DTW，
    两者都基于「连续特征的余弦相似度」，而归一化后任意两段音乐的
    余弦值本来就接近 1，分数全挤在 0.92~0.99，区分度被压扁。

思路（Shazam 式离散指纹）：
    1. 对音频做 STFT，在每个时间帧取若干个最强的频谱峰 (t, f)
    2. 对 qry 与 ref 中「频率相同」的峰，统计时间差 Δ = t_ref - t_qry
    3. 若是同一段录音（搬运），大量峰会落在同一个 Δ 上，形成尖锐的直方图峰；
       若是不同录音（翻唱 / 不相关），Δ 分布散乱，峰值很低
    4. 归一化得分 = 直方图最大值 / 较少的峰总数

这个分数是「离散匹配计数」，不受余弦相似度饱和的影响，
对片段截取、不同压缩格式（mp3/mp4/m4a/ogg）都鲁棒。

依赖：numpy / librosa（与 core.py 一致），不引入新库。
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import librosa

SR = 22050
N_FFT = 2048
HOP = 512
TIME_STRIDE = 4          # 每 4 个 STFT 帧取一帧，降低时间分辨率
TOP_K = 4                # 每帧最多保留 4 个频率峰
FREQ_LO_HZ = 120         # 只看 120~5000 Hz，避开直流与高频噪声
FREQ_HI_HZ = 5000

# 默认阈值（evaluate_v3.py 会在标注集上网格搜索最优值）
TH_LANDMARK = 0.08       # landmark 分数 >= 此值 -> copy
TH_MELODY = 0.94         # 否则 sim_melody >= 此值 -> cover


def peaks(y, sr: int = SR):
    """返回 (按频率分组的峰时间表, 峰总数)。

    形如 {freq_bin: [t1, t2, ...]}，t 是降采样后的帧序号。
    """
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    if S.shape[1] == 0:
        return {}, 0
    S = S[:, ::TIME_STRIDE]

    lo = max(1, int(FREQ_LO_HZ * N_FFT / sr))
    hi = min(S.shape[0] - 1, int(FREQ_HI_HZ * N_FFT / sr))
    band = S[lo:hi, :]
    if band.shape[1] == 0 or band.max(initial=0) <= 0:
        return {}, 0

    by_freq = defaultdict(list)
    for t in range(band.shape[1]):
        col = band[:, t]
        if col.max() <= 0:
            continue
        k = min(TOP_K, len(col))
        cand = np.argpartition(col, -k)[-k:]
        for i in cand:
            i = int(i)
            # 只要频率方向的局部极大，避免一个宽峰贡献多个 bin
            if i > 0 and col[i] < col[i - 1]:
                continue
            if i < len(col) - 1 and col[i] < col[i + 1]:
                continue
            by_freq[i].append(t)

    return dict(by_freq), sum(len(v) for v in by_freq.values())


def landmark_score(a, b) -> float:
    """两首歌的频谱峰在最佳时间偏移下的匹配率，越大越像同一录音。"""
    ba, na = a
    bb, nb = b
    if na == 0 or nb == 0:
        return 0.0

    offset_counts = Counter()
    for f, times_a in ba.items():
        times_b = bb.get(f)
        if not times_b:
            continue
        for ta in times_a:
            for tb in times_b:
                offset_counts[tb - ta] += 1

    if not offset_counts:
        return 0.0
    best = max(offset_counts.values())
    return float(best) / min(na, nb)


def classify(landmark: float, sim_melody: float,
             th_landmark: float = TH_LANDMARK,
             th_melody: float = TH_MELODY) -> str:
    """两级判定：先用离散指纹认「同一录音」，再用旋律相似度认「翻唱」。"""
    if landmark >= th_landmark:
        return "copy"
    if sim_melody >= th_melody:
        return "cover"
    return "unrelated"
