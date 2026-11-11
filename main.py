"""
main.py — 教学演示入口（v1：全曲平均指纹 + 余弦相似度）

重要
----
  这个文件保留的是**最初的版本**，用来演示「声音 -> 24 维指纹 -> 余弦相似度
  -> 阈值分类」这条最基础的链路，方便讲解和复现踩坑过程。
  但 v1 在 100 对真实标注数据上只有 35% 准确率，已被证伪。

  实际比对两首歌请用改进版：
      python predict.py song1.mp3 song2.mp3
  它使用 v3 特征 + 训练好的分类器（见 train_model.py / model_card.md）。
  本文件在提供两首歌时也会自动转发到 predict.py。

用法
----
  python main.py song1.mp3 song2.mp3    # 转发到改进版推理
  python main.py                        # v1 合成演示（仅教学用途）

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

LABEL_ZH = {"copy": "搬运", "cover": "翻唱", "unrelated": "不相关"}


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
    plt.text(len(vals) - 0.5, TH_COPY + 0.02, f"copy >= {TH_COPY:.2f}",
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
    print("⚠️ 进入 v1 教学演示模式（合成正弦波 vs 白噪声）。")
    print("   这里漂亮的分类结果不能外推到真实歌曲；真实比对请用 predict.py。")
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
        lab = classify(s)
        print(f"   {name:12s} similarity = {s:+.3f}  ->  {lab} ({LABEL_ZH[lab]})")

    plot_bars([r[0] for r in rows], [r[1] for r in rows], "result.png")


def run_pair(path1, path2):
    """真实模式：优先使用训练好的改进版模型；模型缺失才退回 v1。"""
    try:
        from predict import predict_pair
        r = predict_pair(path1, path2)
    except SystemExit as e:
        print(f"⚠️ 改进版模型不可用（{e}）")
        print("   先运行 `python train_model.py`；本次退回 v1 演示算法。")
        print("   ⚠️ v1 在真实数据上只有 35% 准确率，结果仅供参考。")
    else:
        print(f"✅ Compare {path1} vs {path2}  (v3 特征 + 逻辑回归)")
        print(f"   判定：{r['prediction_zh']} ({r['prediction']})"
              f"   置信度 {r['confidence']:.1%}")
        print("   三路分数：" + "  ".join(
            f"{k}={v:+.4f}" for k, v in r["features"].items()))
        print("   各类概率：" + "  ".join(
            f"{k} {v:.1%}" for k, v in r["probabilities"].items()))
        return

    # v1 回退路径
    sim = compare(path1, path2)
    lab = classify(sim)
    print(f"   v1 similarity = {sim:+.3f}  ->  {lab} ({LABEL_ZH[lab]})")
    plot_bars([f"{os.path.basename(path1)}\nvs\n{os.path.basename(path2)}"],
              [sim], "result.png")


def main():
    if len(sys.argv) == 1:
        run_demo()
        return
    if len(sys.argv) == 3:
        p1, p2 = sys.argv[1], sys.argv[2]
        missing = [p for p in (p1, p2) if not os.path.isfile(p)]
        if missing:
            raise SystemExit(f"[ERROR] audio file(s) not found: {missing}")
        run_pair(p1, p2)
        return
    print("提示：用法为 `python main.py song1 song2`，或 `python main.py` 进入演示模式。")
    print("     真实比对推荐直接用 `python predict.py song1 song2`。")


if __name__ == "__main__":
    main()
