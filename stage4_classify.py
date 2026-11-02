"""
阶段 4 · 自动分类（给项目装"裁判大脑"）
思路：
  1) 用阶段3的指纹 + 余弦相似度，算出两首歌的相似分数 sim (范围 -1 ~ 1)
  2) 按阈值自动判类：
       sim >= 0.85  -> 搬运 (copy / plagiarism)   几乎一模一样
       0.55 <= sim < 0.85 -> 翻唱 (cover)         同一首歌的不同演绎
       sim < 0.55   -> 不相关 (unrelated)         完全不同的歌
运行：在 (music) 环境，cd 到项目目录，python stage4_classify.py
说明：若同目录有 song1.mp3 / song2.mp3 就比这两首并判类；否则自动演示
      copy / cover / unrelated 三个场景。
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

AUDIO1 = "song1.mp3"
AUDIO2 = "song2.mp3"

# 分类阈值（来自项目指导）
TH_COPY = 0.85      # >= 0.85 判搬运
TH_COVER = 0.55     # 0.55~0.85 判翻唱；< 0.55 判不相关


def make_tone(path, freqs, sr=22050, duration=5.0):
    """生成一段由若干音符组成的旋律，存成 wav"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    seg = len(t) // len(freqs)
    y = np.zeros_like(t)
    for i, f in enumerate(freqs):
        s = slice(i * seg, (i + 1) * seg)
        y[s] = 0.5 * np.sin(2 * np.pi * f * t[s]) * np.hanning(seg)
    sf.write(path, y.astype(np.float32), sr)


def fingerprint(path):
    """把一首歌变成固定长度指纹向量。

    关键设计（经过调参验证）：
      - 去掉 MFCC 第 0 维（log-energy），否则相似度几乎只比"响度"，
        会淹没音高/音色差异（详见开发笔记 / 论文 Limitations）。
      - MFCC 块与 Chroma 块各自归一化到单位范数，避免某一块主导余弦相似度。
    返回 24 维向量 = (MFCC 1..12 均值) + (Chroma 12 维均值)。
    """
    y, sr = librosa.load(path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mf = mfcc[1:, :].mean(axis=1)
    ch = chroma.mean(axis=1)
    mf = mf / (np.linalg.norm(mf) + 1e-9)
    ch = ch / (np.linalg.norm(ch) + 1e-9)
    return np.concatenate([mf, ch])


def cosine_sim(a, b):
    """余弦相似度：1=完全一样，-1=完全相反，0=不相关"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def classify(sim):
    """按阈值返回类别标签（英文，论文可用）"""
    if sim >= TH_COPY:
        return "copy"
    elif sim >= TH_COVER:
        return "cover"
    else:
        return "unrelated"


def main():
    if os.path.exists(AUDIO1) and os.path.exists(AUDIO2):
        # —— 真实模式 ——
        fa = fingerprint(AUDIO1)
        fb = fingerprint(AUDIO2)
        sim = cosine_sim(fa, fb)
        print(f"✅ Compare {AUDIO1} vs {AUDIO2}")
        print(f"   similarity = {sim:.3f}  ->  {classify(sim)}")
        labels = [f"{AUDIO1}\nvs\n{AUDIO2}"]
        vals = [sim]
        colors = ["#cc4444"]
    else:
        # —— 演示模式：造三个场景，验证分类逻辑 ——
        print("⚠️ No song1.mp3 / song2.mp3 found, demo mode…")
        # 原曲 A
        make_tone("demo_A.wav", [440, 494, 523, 587, 659])

        # 场景1：搬运 = 原曲加一点噪声（几乎一模一样）
        y, sr = librosa.load("demo_A.wav", sr=22050)
        y_copy = y + 0.02 * np.random.randn(len(y))
        sf.write("demo_copy.wav", y_copy.astype(np.float32), sr)

        # 场景2：翻唱 = 原曲升 3 个半音（同一首歌的不同调式演绎）
        # 用改进后的指纹，此变换稳定落在 0.55~0.85 的"翻唱"区间
        y_cover = librosa.effects.pitch_shift(y, sr=sr, n_steps=3)
        sf.write("demo_cover.wav", y_cover.astype(np.float32), sr)

        # 场景3：不相关 = 白噪声
        sf.write("demo_noise.wav",
                 (0.3 * np.random.randn(int(sr * 5))).astype(np.float32), sr)

        sim_copy = cosine_sim(fingerprint("demo_A.wav"), fingerprint("demo_copy.wav"))
        sim_cover = cosine_sim(fingerprint("demo_A.wav"), fingerprint("demo_cover.wav"))
        sim_noise = cosine_sim(fingerprint("demo_A.wav"), fingerprint("demo_noise.wav"))

        rows = [
            ("A vs copy", sim_copy),
            ("A vs cover", sim_cover),
            ("A vs noise", sim_noise),
        ]
        for name, s in rows:
            print(f"   {name:28s} sim = {s:+.3f}  ->  {classify(s)}")

        labels = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        # 按类别上色：搬运红、翻唱橙、不相关灰
        colors = ["#cc4444" if v >= TH_COPY else
                  "#dd9944" if v >= TH_COVER else "#888780" for v in vals]

    # 画柱状图（论文级：英文标签 + 完整 y 轴）
    plt.figure(figsize=(9, 4.5))
    bars = plt.bar(labels, vals, color=colors)
    plt.axhline(TH_COPY, color="#cc4444", ls="--", lw=1)
    plt.axhline(TH_COVER, color="#dd9944", ls="--", lw=1)
    plt.text(len(vals) - 0.5, TH_COPY + 0.02, "copy >= 0.85", color="#cc4444", fontsize=9, ha="right")
    plt.text(len(vals) - 0.5, TH_COVER + 0.02, "cover >= 0.55", color="#dd9944", fontsize=9, ha="right")
    plt.ylim(-1.1, 1.1)
    plt.ylabel("Cosine similarity")
    plt.title("Similarity and classification")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.06),
                 f"{v:+.3f}", ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig("stage4_classify.png", dpi=120)
    plt.show()
    print("✅ Saved stage4_classify.png")


if __name__ == "__main__":
    main()
