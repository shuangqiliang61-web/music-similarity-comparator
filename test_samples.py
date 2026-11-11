"""
test_samples.py — 真实歌曲样本批量测试

用法（在 Anaconda Prompt 里，先 conda activate music）：
    python test_samples.py                    # 默认扫描 samples/ 目录
    python test_samples.py "D:\\Downloads\\歌曲"  # 指定别的文件夹

做什么：
    把文件夹里的音频两两配对，算余弦相似度 + 分类，
    打印结果表，并保存 samples_result.csv 和 samples_result.png。

注意：
    - 音频样本只放本地，不要提交到 GitHub（可能有版权）。
    - 结果文件（csv / png）不含音频内容，可以作为作品集、论文的实验素材。
"""

import os
import sys
import csv
import itertools
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import fingerprint, cosine_sim, classify, TH_COPY, TH_COVER

# 支持的音频格式
EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma")

# 只取每首歌前 N 秒可以加速测试；None = 用整首歌（与 core.py 定义一致）
# 歌曲很长、跑得慢时，可把它改成 60
CLIP_SECONDS = None


def load_fingerprint(path, sr=22050):
    """读一个音频文件，返回 24 维指纹。"""
    y, sr = librosa.load(path, sr=sr, mono=True)
    if CLIP_SECONDS is not None:
        y = y[: int(sr * CLIP_SECONDS)]
    return fingerprint(y, sr)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "samples"
    if not os.path.isdir(folder):
        print(f"[X] 找不到文件夹：{folder}")
        print("    请在项目目录下新建 samples 文件夹，把下载好的歌放进去；")
        print("    或者把文件夹路径作为参数传进来，例如：")
        print('    python test_samples.py "D:\\Downloads\\歌曲"')
        return

    files = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(EXTS)
    )

    if len(files) < 2:
        print(f"[X] 在 {folder} 里只找到 {len(files)} 个音频文件，至少需要 2 个才能比对。")
        print(f"    支持的格式：{', '.join(EXTS)}")
        return

    print(f"[OK] 找到 {len(files)} 个音频文件：")
    for i, f in enumerate(files, 1):
        print(f"   [{i}] {os.path.basename(f)}")

    # 提取指纹：单个文件失败不影响其他文件
    fps = {}
    for f in files:
        try:
            fps[f] = load_fingerprint(f)
            print(f"   [OK] 特征提取完成：{os.path.basename(f)}")
        except Exception as e:
            print(f"   [FAIL] 读取失败：{os.path.basename(f)} -> {type(e).__name__}: {e}")
            print("          （mp3 / m4a 若报解码错误，先转成 .wav 再试）")

    if len(fps) < 2:
        print("[X] 能成功读取的音频不足 2 个，测试中止。")
        return

    # 两两比对
    rows = []
    for a, b in itertools.combinations(sorted(fps), 2):
        sim = cosine_sim(fps[a], fps[b])
        rows.append((os.path.basename(a), os.path.basename(b), sim, classify(sim)))

    print("\n=== 比对结果 ===")
    print(f"{'A':<30} {'B':<30} {'相似度':>8}  判定")
    for ra, rb, sim, label in rows:
        print(f"{ra:<30} {rb:<30} {sim:>+8.3f}  {label}")

    # 存 CSV：utf-8-sig 让 Excel 直接打开中文不乱码
    with open("samples_result.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file_A", "file_B", "cosine_similarity", "classification"])
        for ra, rb, sim, label in rows:
            w.writerow([ra, rb, f"{sim:.4f}", label])
    print("\n[OK] 结果已存：samples_result.csv")

    # 画图：横轴用编号，避免中文文件名在 matplotlib 里显示成方块
    labels = [str(i + 1) for i in range(len(rows))]
    vals = [r[2] for r in rows]
    colors = ["#cc4444" if v >= TH_COPY else
              "#dd9944" if v >= TH_COVER else "#888780" for v in vals]
    plt.figure(figsize=(max(7, len(rows) * 1.2), 4.5))
    bars = plt.bar(labels, vals, color=colors)
    plt.axhline(TH_COPY, color="#cc4444", ls="--", lw=1)
    plt.axhline(TH_COVER, color="#dd9944", ls="--", lw=1)
    plt.ylim(-1.1, 1.1)
    plt.xlabel("pair number (see samples_result.csv)")
    plt.ylabel("Cosine similarity")
    plt.title("Real-song sample pairs: similarity and classification")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.08),
                 f"{v:+.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("samples_result.png", dpi=120)
    plt.close()
    print("[OK] 结果图已存：samples_result.png")


if __name__ == "__main__":
    main()
