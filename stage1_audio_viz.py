"""
阶段 1 · 读一首歌 + 画波形 & 频谱（最小可运行示例，零门槛版）
运行前：先装库  python -m pip install numpy librosa matplotlib soundfile
运行：        python stage1_audio_viz.py

说明：
- 如果同目录有 song.mp3（或你改下面的 AUDIO_PATH），就读它；
- 没有的话会自动生成一个 demo_tone.wav 让你先看到图，之后把真歌放进来即可。
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
    notes = [440.0, 493.9, 523.3, 587.3, 659.3]  # A4 B4 C5 D5 E5 五个音
    seg = len(t) // len(notes)
    y = np.zeros_like(t)
    for i, f in enumerate(notes):
        s = slice(i * seg, (i + 1) * seg)
        env = np.hanning(seg)            # 平滑包络，避免爆音
        y[s] = 0.5 * np.sin(2 * np.pi * f * t[s]) * env
    sf.write(path, y.astype(np.float32), sr)
    return y, sr


def main():
    if os.path.exists(AUDIO_PATH):
        y, sr = librosa.load(AUDIO_PATH, sr=None)   # sr=None 保留原始采样率
        print(f"✅ 读取成功：{AUDIO_PATH}，采样率={sr}，时长={len(y)/sr:.2f}s")
    else:
        print(f"⚠️ 没找到 {AUDIO_PATH}，自动生成一个 demo 音频用于测试…")
        y, sr = make_demo_tone("demo_tone.wav")
        print(f"✅ 已生成 demo_tone.wav，采样率={sr}，时长={len(y)/sr:.2f}s")

    # 1) 波形（Waveform）
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("stage1_waveform.png", dpi=120)
    plt.show()

    # 2) 梅尔频谱（Mel-spectrogram）
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel-spectrogram")
    plt.tight_layout()
    plt.savefig("stage1_melspec.png", dpi=120)
    plt.show()

    print("✅ 两张图已保存：stage1_waveform.png / stage1_melspec.png")


if __name__ == "__main__":
    main()
