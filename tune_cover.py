"""调参 v2：改进指纹(去掉MFCC第0维+分块归一化)后，找能落进 cover 区间的变换。"""
import numpy as np
import librosa

sr = 22050
t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False)
freqs = [440, 494, 523, 587, 659]
seg = len(t) // len(freqs)
y = np.zeros_like(t)
for i, f in enumerate(freqs):
    s = slice(i * seg, (i + 1) * seg)
    y[s] = 0.5 * np.sin(2 * np.pi * f * t[s]) * np.hanning(seg)


def fingerprint(y, variant="v2"):
    y = np.asfortranarray(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    if variant == "v1_old":
        return np.concatenate([mfcc.mean(axis=1), chroma.mean(axis=1)])
    # 改进：去掉 MFCC 第0维(能量)，两块各自归一化到单位范数
    mf = mfcc[1:, :].mean(axis=1)
    ch = chroma.mean(axis=1)
    mf = mf / (np.linalg.norm(mf) + 1e-9)
    ch = ch / (np.linalg.norm(ch) + 1e-9)
    if variant == "v2_mfcc":
        return mf
    if variant == "v2_chroma":
        return ch
    return np.concatenate([mf, ch])


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


fa = fingerprint(y, "v2")
print("baseline fa norm (v2):", np.linalg.norm(fa))

candidates = {}
candidates["copy_02nz"] = y + 0.02 * np.random.randn(len(y))
candidates["noise"] = 0.3 * np.random.randn(len(y))
# 翻唱候选：不同变调 + 少量噪声
for ns in [3, 5, 7, 9]:
    for nz in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
        yt = librosa.effects.pitch_shift(y, sr=sr, n_steps=ns)
        if nz > 0:
            yt = yt + nz * np.random.randn(len(yt))
        candidates[f"shift{ns}_nz{nz}"] = yt
# 升7半音对半混音
ys = librosa.effects.pitch_shift(y, sr=sr, n_steps=7)
candidates["blend07_50"] = 0.5 * y + 0.5 * ys
candidates["blend07_30"] = 0.3 * y + 0.7 * ys
# 不同旋律
freqs2 = [277, 311, 349, 415, 466]
seg2 = len(t) // len(freqs2)
y2 = np.zeros_like(t)
for i, f in enumerate(freqs2):
    s = slice(i * seg2, (i + 1) * seg2)
    y2[s] = 0.5 * np.sin(2 * np.pi * f * t[s]) * np.hanning(seg2)
candidates["mix_other_50"] = 0.5 * y + 0.5 * y2

print("\n=== v2 scores (improved fingerprint) ===")
rows = []
for name, sig in candidates.items():
    try:
        fb = fingerprint(sig, "v2")
        s = cosine(fa, fb)
        if s >= 0.85:
            tag = "COPY(red)"
        elif s >= 0.55:
            tag = "COVER(orange) <== want"
        else:
            tag = "UNRELATED(gray)"
        rows.append((s, name, tag))
    except Exception as e:
        print(name, "ERR", e)
for s, name, tag in sorted(rows, reverse=True):
    print(f"{name:18s} {s:+.3f}  {tag}")
