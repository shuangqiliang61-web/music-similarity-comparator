"""
阶段 2 · 提取"音乐指纹"特征：MFCC + Chromagram（色谱图）
这是整个项目最关键的一步——把"听起来的像不像"变成数字矩阵。
运行：在 (music) 环境，cd 到项目目录，运行 python stage2_feature_extraction.py
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

# ===== 想用真歌就改成你的文件路径，例如 "周杰伦-晴天.mp3" =====
AUDIO_PATH = "song.mp3"


def make_demo_tone(path, sr=22050, duration=5.0):
    """自动生成一段简单旋律，避免'没歌可跑'的尴尬"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    notes = [440.0, 493.9, 523.3, 587.3, 659.3]  # A4 B4 C5 D5 E5
    seg = len(t) // len(notes)
    y = np.zeros_like(t)
    for i, f in enumerate(notes):
        s = slice(i * seg, (i + 1) * seg)
        env = np.hanning(seg)
        y[s] = 0.5 * np.sin(2 * np.pi * f * t[s]) * env
    sf.write(path, y.astype(np.float32), sr)
    return y, sr


def main():
    if os.path.exists(AUDIO_PATH):
        y, sr = librosa.load(AUDIO_PATH, sr=None)
        print(f"✅ 读取成功：{AUDIO_PATH}，采样率={sr}，时长={len(y)/sr:.2f}s")
    else:
        print("⚠️ 没找到 song.mp3，自动生成 demo 音频用于测试…")
        y, sr = make_demo_tone("demo_tone.wav")
        print(f"✅ 已生成 demo_tone.wav，采样率={sr}，时长={len(y)/sr:.2f}s")

    # 1) MFCC（梅尔频率倒谱系数）—— 描述"音色/音高轮廓"，最常用
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print(f"MFCC 形状 (系数数 × 时间帧): {mfcc.shape}")

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis="time")
    plt.colorbar(format="%+2.0f")
    plt.title("MFCC (Mel-Frequency Cepstral Coefficients)")
    plt.tight_layout()
    plt.savefig("stage2_mfcc.png", dpi=120)
    plt.show()

    # 2) Chromagram（色谱图）—— 把音高映射到 12 个半音，擅长捕捉"旋律/和弦"
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    print(f"Chroma 形状 (12 半音 × 时间帧): {chroma.shape}")

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, x_axis="time", y_axis="chroma")
    plt.colorbar()
    plt.title("Chromagram (12 pitch classes)")
    plt.tight_layout()
    plt.savefig("stage2_chroma.png", dpi=120)
    plt.show()

    print("✅ 两张特征图已保存：stage2_mfcc.png / stage2_chroma.png")


if __name__ == "__main__":
    main()
