"""
阶段 3 · 计算两首歌的“相似度分数”（项目核心算法）
思路：
  1) 把每首歌压成一个固定长度的指纹向量 = MFCC 均值 + 色谱图均值
  2) 用余弦相似度(cosine similarity)比较两首歌的指纹
     - 越接近 1.0 越像；越接近 0 越不像
运行：在 (music) 环境，cd 到项目目录，python stage3_similarity.py
说明：若同目录有 song1.mp3 / song2.mp3 就比这两首；否则自动生成
      demo_A(原曲) / demo_B(翻唱) / demo_C(不相关噪声) 做演示。
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

AUDIO1 = "song1.mp3"
AUDIO2 = "song2.mp3"


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
    """余弦相似度：1=完全一样，0=完全不相关"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    if os.path.exists(AUDIO1) and os.path.exists(AUDIO2):
        # —— 真实模式：比较两首用户提供的歌 ——
        fa = fingerprint(AUDIO1)
        fb = fingerprint(AUDIO2)
        sim = cosine_sim(fa, fb)
        print(f"✅ 比较 {AUDIO1} vs {AUDIO2}")
        print(f"   相似度分数 = {sim:.3f}")
        labels, vals = [f"{AUDIO1}\nvs\n{AUDIO2}"], [sim]

    else:
        # —— 演示模式：自动造三首，验证算法能分辨“像”和“不像” ——
        print("⚠️ 没找到 song1.mp3 / song2.mp3，进入演示模式…")
        # A：原曲
        make_tone("demo_A.wav", [440, 494, 523, 587, 659])
        # B：A 的“翻唱”——变调 + 变速（模拟同一首歌的不同演绎）
        y, sr = librosa.load("demo_A.wav", sr=22050)
        y_b = librosa.effects.pitch_shift(
            librosa.effects.time_stretch(y, rate=1.2), sr=sr, n_steps=2)
        sf.write("demo_B.wav", y_b.astype(np.float32), sr)
        # C：完全不相关的白噪声
        sf.write("demo_C.wav", (0.3 * np.random.randn(int(sr * 5))).astype(np.float32), sr)

        fa = fingerprint("demo_A.wav")
        fb = fingerprint("demo_B.wav")
        fc = fingerprint("demo_C.wav")
        sim_ab = cosine_sim(fa, fb)   # 原曲 vs 翻唱 → 应该较高
        sim_ac = cosine_sim(fa, fc)   # 原曲 vs 噪声 → 应该很低
        print(f"   相似度 A(原曲) vs B(翻唱)  = {sim_ab:.3f}")
        print(f"   相似度 A(原曲) vs C(不相关)= {sim_ac:.3f}")
        labels = ["A (original) vs B (cover)", "A (original) vs C (noise)"]
        vals = [sim_ab, sim_ac]

    # 画一张对比柱状图
    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, vals, color=["#639922", "#888780"])
    plt.ylim(-1.1, 1.1)
    plt.ylabel("Cosine similarity")
    plt.title("Song similarity score")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                 ha="center", fontsize=12)
    plt.tight_layout()
    plt.savefig("stage3_similarity.png", dpi=120)
    plt.show()
    print("✅ 已保存 stage3_similarity.png")


if __name__ == "__main__":
    main()
