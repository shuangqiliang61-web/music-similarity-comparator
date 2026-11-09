"""
main.py — 音乐歌曲比对项目 · 主入口

用法
----
  比较两首歌：
      python main.py song1.mp3 song2.mp3
  演示模式（自动生成样本，验证三类分类）：
      python main.py

输出
----
  - 命令行打印相似度分数与分类结果
  - 保存结果图 result.png

依赖：numpy, librosa, matplotlib, soundfile（建议在 conda 环境 music 中运行）
"""

import os
import sys
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")  # 无图形界面也能保存图片
import matplotlib.pyplot as plt
import soundfile as sf

from core import fingerprint, cosine_sim, classify, TH_COPY, TH_COVER


def make_tone(path, freqs, sr=22050, duration=5.0):
    """生成一段由若干音符组成的旋律，存成 wav（仅演示用）。"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    seg = len(t) // len(freqs)
    y = np.zeros_like(t)
    for i, f in enumerate(freqs):
        s = slice(i * seg, (i + 1) * seg)
        y[s] = 0.5 * np.sin(2 * np.pi * f * t[s]) * np.hanning(seg)
    sf.write(path, y.astype(np.float32), sr)


def compare(path1, path2):
    """读取两首歌，返回余弦相似度分数。"""
    y1, sr1 = librosa.load(path1, sr=22050)
    y2, sr2 = librosa.load(path2, sr=22050)
    return cosine_sim(fingerprint(y1, sr1), fingerprint(y2, sr2))


def plot_bars(labels, vals, out_path):
    """画分类柱状图：按类别上色，并画出两条阈值虚线。"""
    colors = ["#cc4444" if v >= TH_COPY else
              "#dd9944" if v >= TH_COVER else "#888780" for v in vals]
    plt.figure(figsize=(9, 4.5))
    bars = plt.bar(labels, vals, color=colors)
    plt.axhline(TH_COPY, color="#cc4444", ls="--", lw=1)
    plt.axhline(TH_COVER, color="#dd9944", ls="--", lw=1)
    plt.text(len(vals) - 0.5, TH_COPY + 0.02, "copy >= 0.85",
             color="#cc4444", fontsize=9, ha="right")
    plt.text(len(vals) - 0.5, TH_COVER + 0.02, f"cover >= {TH_COVER:.2f}",
             color="#dd9944", fontsize=9, ha="right")
    plt.ylim(-1.1, 1.1)
    plt.ylabel("Cosine similarity")
    plt.title("Song similarity and classification")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.06),
                 f"{v:+.3f}", ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"✅ 已保存结果图：{out_path}")


def run_demo():
    """演示模式：造 copy / cover / noise 三个场景，验证分类逻辑。"""
    print("⚠️ 未提供两首歌，进入演示模式…")
    make_tone("demo_A.wav", [440, 494, 523, 587, 659])
    y, sr = librosa.load("demo_A.wav", sr=22050)

    # 搬运：原曲加一点点噪声
    sf.write("demo_copy.wav",
             (y + 0.02 * np.random.randn(len(y))).astype(np.float32), sr)
    # 翻唱：原曲升 3 个半音（同一首歌的不同调式演绎）
    sf.write("demo_cover.wav",
             librosa.effects.pitch_shift(y, sr=sr, n_steps=3).astype(np.float32), sr)
    # 不相关：白噪声
    sf.write("demo_noise.wav",
             (0.3 * np.random.randn(int(sr * 5))).astype(np.float32), sr)

    sim_copy = compare("demo_A.wav", "demo_copy.wav")
    sim_cover = compare("demo_A.wav", "demo_cover.wav")
    sim_noise = compare("demo_A.wav", "demo_noise.wav")

    rows = [
        ("A vs copy", sim_copy),
        ("A vs cover", sim_cover),
        ("A vs noise", sim_noise),
    ]
    for name, s in rows:
        print(f"   {name:12s} similarity = {s:+.3f}  ->  {classify(s)}")

    plot_bars([r[0] for r in rows], [r[1] for r in rows], "result.png")


def run_pair(path1, path2):
    """真实模式：比较用户给出的两首歌。"""
    sim = compare(path1, path2)
    print(f"✅ Compare {path1} vs {path2}")
    print(f"   similarity = {sim:+.3f}  ->  {classify(sim)}")
    plot_bars([f"{os.path.basename(path1)}\nvs\n{os.path.basename(path2)}"],
              [sim], "result.png")


def main():
    if len(sys.argv) == 3 and os.path.exists(sys.argv[1]) and os.path.exists(sys.argv[2]):
        run_pair(sys.argv[1], sys.argv[2])
    else:
        if len(sys.argv) != 1:
            print("提示：用法为 `python main.py song1 song2`，或 `python main.py` 进入演示模式。")
        run_demo()


if __name__ == "__main__":
    main()
