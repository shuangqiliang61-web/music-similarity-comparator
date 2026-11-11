# 我用 Python 写了个能分辨「搬运 / 翻唱」的音频比对工具（附完整踩坑过程）

> 建议标签：`#showdev` `#python` `#机器学习` `#音频处理` `#新手`
> 封面图：用仓库里的 `result.png`（文章里也会嵌同一张图）

一句话介绍：给两首歌，它能算出「有多像」，并自动判成 **搬运 / 翻唱 / 不相关**。底层是 MFCC + Chroma 特征 + 余弦相似度。

![三类分类结果](result.png)

---

## 为什么做这个

刷视频时常看到「这歌是不是搬运 / 翻唱」的争论。人耳能听个大概，但没法量化。我就想：能不能用一段 Python，把「像不像」变成一个 0~1 之间的数字？

这成了我第一个正经的作品集项目——也是第一次完整走完「想法 → 代码 → 踩坑 → 修好 → 写出来」的全过程。

## 它能做什么

- 输入两首歌（mp3 / wav），输出一个**相似度分数** + **分类**：
  - `≥ 0.85` → 搬运（copy，几乎一模一样）
  - `0.55 ~ 0.85` → 翻唱（cover，同一首歌的不同演绎）
  - `< 0.55` → 不相关（unrelated，完全不同的歌）
- 跑 `python main.py` 进入演示模式，自动造「搬运 / 翻唱 / 不相关」三个样本验证分类，并画出上面的柱状图（两条虚线是阈值）。

## 核心思路

把一首歌压成一段固定长度的「指纹向量」，再和另一首的指纹算**余弦相似度**——越接近 1 越像。

指纹 = **24 维**：
- MFCC 第 1~12 维的均值（描述音色 / 音高轮廓）
- Chroma 12 维的均值（描述旋律，对转调敏感）

```python
import numpy as np, librosa

def fingerprint(y, sr=22050):
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mf = mfcc[1:, :].mean(axis=1)        # 去掉第 0 维（能量维）
    ch = chroma.mean(axis=1)
    mf = mf / (np.linalg.norm(mf) + 1e-9)   # 分块 L2 归一化
    ch = ch / (np.linalg.norm(ch) + 1e-9)
    return np.concatenate([mf, ch])      # 24 维指纹
```

分类就是拿分数对阈值「切三刀」：

```python
def classify(sim):
    if   sim >= 0.85: return "copy (搬运)"
    elif sim >= 0.55: return "cover (翻唱)"
    else:             return "unrelated (不相关)"
```

## 真正的坑：MFCC 第 0 维

最早我直接把**全部 MFCC（含第 0 维 log-energy）**和 Chroma 拼一起求平均。结果翻唱和原曲的相似度低得离谱，三类根本拉不开——只有「几乎一样」和「完全无关」两档，**中间的「翻唱」死活出不来**。

原因后来才想明白：MFCC 第 0 维是**响度（能量）**，它数值大、方差也大，直接把余弦相似度「绑架」成了「比谁更响」，把音高 / 音色的差异全淹了。

修复就两步：
1. **丢掉 MFCC 第 0 维**；
2. 把 MFCC 块和 Chroma 块**分别做 L2 归一化**，让两块贡献均衡。

改完，三类分数干净地拉开了（演示结果：搬运 ≈ 0.98、翻唱 ≈ 0.72、不相关 ≈ 0.16）。这个 bug 比「把项目做出来」本身更让我学到东西——**特征怎么选，比模型多复杂更重要**。

## 怎么跑

```bash
conda create -n music python=3.11 -y
conda activate music
conda install -c conda-forge librosa matplotlib -y
pip install soundfile

python main.py song1.mp3 song2.mp3   # 比两首歌
python main.py                       # 演示模式（自动造样本）
```

> 小提示：Windows 上 Python 3.14 的 pip 环境会因「应用控制策略」拦截 numba 的 DLL，所以统一用 conda 环境最稳。

## 局限 & 下一步

- 现在用的是「全曲平均特征」，对**变调翻唱**区分力有限 → 下一步可上 **DTW 时间对齐**的 Chroma 序列比对。
- 阈值（0.85 / 0.55）是经验值，应该在带标注的数据集上用混淆矩阵 / ROC 调优。
- 真实歌曲样本还不够多，分类边界有待更多样本验证。

## 仓库

👉 https://github.com/shuangqiliang61-web/music-similarity-comparator

欢迎 Star、提 Issue，或者告诉我你踩过的音频处理坑。我是新手，正在攒作品集准备申请海外实习，任何反馈都超有价值 🙏

---

*如果你也想从零做一个能写进简历的小项目，这个「选特征 → 写相似度 → 调阈值 → 踩坑」的套路其实很好复用，欢迎一起交流。*
