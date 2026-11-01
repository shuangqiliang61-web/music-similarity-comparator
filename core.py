"""
core.py — 音乐相似度比对的核心算法（模块化沉淀版）

包含：
  - fingerprint(): 把一段音频变成固定长度"指纹向量"
  - cosine_sim():  计算两个指纹的余弦相似度
  - classify():    按阈值把相似度判成 搬运 / 翻唱 / 不相关

设计要点（详见开发笔记 / 论文 Limitations）：
  朴素特征会把 MFCC 第 0 维(log-energy, 响度)算进去，
  导致相似度几乎只比"谁更响"，淹没音高/音色差异。
  这里去掉第 0 维，并把 MFCC 块与 Chroma 块分别做 L2 归一化，
  使两块的贡献均衡，区分度显著提升。
"""

import numpy as np
import librosa

SR = 22050            # 统一重采样率
TH_COPY = 0.85        # 相似度 >= 0.85 判为"搬运"
TH_COVER = 0.55       # 0.55 <= 相似度 < 0.85 判为"翻唱"；< 0.55 判为"不相关"


def fingerprint(y, sr=SR):
    """把一段音频 y 变成固定长度指纹向量（24 维）。

    返回 = [MFCC 第1~12维的均值(归一化)] + [Chroma 12维均值(归一化)]
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mf = mfcc[1:, :].mean(axis=1)          # 去掉第 0 维（能量）
    ch = chroma.mean(axis=1)
    mf = mf / (np.linalg.norm(mf) + 1e-9)  # 分块归一化，避免某块主导
    ch = ch / (np.linalg.norm(ch) + 1e-9)
    return np.concatenate([mf, ch])


def cosine_sim(a, b):
    """余弦相似度：1=完全一样，0=不相关，-1=完全相反。"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def classify(sim):
    """把相似度分数映射成类别标签（英文，论文可用）。"""
    if sim >= TH_COPY:
        return "copy (搬运)"
    elif sim >= TH_COVER:
        return "cover (翻唱)"
    else:
        return "unrelated (不相关)"
