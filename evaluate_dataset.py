"""
evaluate_dataset.py — 在「带真值标注」的音频比对测试库上评测 core.py 的分类器

用法（Anaconda Prompt 里先 conda activate music）：
    python evaluate_dataset.py                # 全量 100 对
    python evaluate_dataset.py --limit 10     # 先试跑 10 对

它做什么：
    1. 读 samples/音乐比对测试库/audio_list.xlsx（100 条标注：搬运/翻唱/不相关）
    2. 按清单把 qry 与 ref 配对，用 core.py 提取指纹、算余弦相似度、分类
    3. 与标准答案对比，输出准确率 / 混淆矩阵 / 每类指标 / 分数分布
    4. 网格搜索最优阈值，检验经验值 0.85 / 0.55 是否合理

产出：
    samples/eval_results.csv   每对明细（含原始文件名，已被 .gitignore 忽略，不要外传）
    eval_summary.json          聚合统计（不含文件名，可放进作品集/论文）
    eval_distribution.png      三类相似度分布 + 阈值线
    eval_confusion.png         混淆矩阵

隐私提醒：
    测试库文件名含业务 ID，原始清单路径来自内部系统。
    只公开聚合数字与图，绝不把音频、明细 CSV 或原始文件名上传到公开仓库或文章。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import warnings
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import fingerprint, cosine_sim, classify, TH_COPY, TH_COVER

warnings.filterwarnings("ignore")

SR = 22050
DATA_ROOT = Path("samples/音乐比对测试库")
LABEL_XLSX = DATA_ROOT / "audio_list.xlsx"

# 每首只取前 60 秒：加速、可复现，也是音频评测的常见做法。
# 想用整首歌做严格对比，改成 None 再跑一次即可。
CLIP_SECONDS = 60

DETAIL_CSV = Path("samples/eval_results.csv")   # 含文件名 -> 放在被忽略的 samples/ 内
SUMMARY_JSON = Path("eval_summary.json")
DIST_PNG = Path("eval_distribution.png")
CONF_PNG = Path("eval_confusion.png")

CLASSES = ["copy", "cover", "unrelated"]
IDX = {c: i for i, c in enumerate(CLASSES)}
ZH2EN = {"搬运": "copy", "翻唱": "cover", "不相关": "unrelated"}
COLORS = {"copy": "#cc4444", "cover": "#dd9944", "unrelated": "#888780"}
M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


# ---------- 1. 读标注表（纯标准库解 xlsx，不依赖 openpyxl） ----------

def _col_index(ref: str) -> int:
    """把单元格引用 'C12' 转成 0 基列号，避免空单元格导致列错位。"""
    m = re.match(r"([A-Z]+)", ref or "A")
    n = 0
    for ch in m.group(1):
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_labels(path: Path):
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(M + "si"):
                shared.append("".join(t.text or "" for t in si.iter(M + "t")))

        root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in root.iter(M + "row"):
            cells, max_i = {}, -1
            for c in row.findall(M + "c"):
                i = _col_index(c.get("r"))
                t = c.get("t")
                v, inline = c.find(M + "v"), c.find(M + "is")
                if t == "s" and v is not None:
                    val = shared[int(v.text)]
                elif t == "inlineStr" and inline is not None:
                    val = "".join(x.text or "" for x in inline.iter(M + "t"))
                elif v is not None:
                    val = v.text
                else:
                    val = ""
                cells[i] = val
                max_i = max(max_i, i)
            rows.append([cells.get(i, "") for i in range(max_i + 1)])

    items = []
    for r in rows[1:]:                       # 跳过表头
        r = r + [""] * (7 - len(r))
        zh = r[1].strip()
        if zh not in ZH2EN:
            continue
        # 清单里的路径是 Linux 内部路径，只取它的目录名映射到本地
        q_dir = Path(r[3]).parent.name
        r_dir = Path(r[5]).parent.name
        items.append({
            "id": r[0].strip(),
            "truth": ZH2EN[zh],
            "truth_zh": zh,
            "qry": DATA_ROOT / q_dir / r[2].strip(),
            "ref": DATA_ROOT / r_dir / r[4].strip(),
            "note": r[6].strip(),
        })
    return items


# ---------- 2. 指纹提取（带缓存：很多 ref 被多对复用） ----------

def load_fp(path: Path):
    try:
        y, sr = librosa.load(str(path), sr=SR, mono=True, duration=CLIP_SECONDS)
    except Exception:
        y, sr = librosa.load(str(path), sr=SR, mono=True)
        if CLIP_SECONDS:
            y = y[: int(sr * CLIP_SECONDS)]
    return fingerprint(y, sr)


# ---------- 3. 指标计算 ----------

def metrics(valid):
    cm = [[0, 0, 0] for _ in range(3)]
    for r in valid:
        cm[IDX[r["truth"]]][IDX[r["pred"]]] += 1
    per_class = {}
    for c in CLASSES:
        i = IDX[c]
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(3)) - tp
        fn = sum(cm[i]) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "support": sum(cm[i])}
    acc = sum(cm[i][i] for i in range(3)) / max(1, len(valid))
    macro = sum(v["f1"] for v in per_class.values()) / 3
    return cm, per_class, round(acc, 4), round(macro, 4)


def search_thresholds(valid):
    """网格搜索：在当前特征下，最优的两个阈值是多少、准确率能到多少。"""
    sims = [r["sim"] for r in valid]
    truth = [r["truth"] for r in valid]
    best = []
    for tc in np.arange(0.30, 0.991, 0.01):
        for tv in np.arange(0.05, tc, 0.01):
            n_ok = 0
            for s, t in zip(sims, truth):
                p = "copy" if s >= tc else ("cover" if s >= tv else "unrelated")
                n_ok += (p == t)
            best.append((n_ok / len(truth), round(float(tc), 2), round(float(tv), 2)))
    best.sort(reverse=True)
    return best


# ---------- 4. 画图（英文标签，避开中文字体缺失） ----------

def plot_distribution(valid):
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(0)
    for i, c in enumerate(CLASSES):
        vals = [r["sim"] for r in valid if r["truth"] == c]
        if not vals:
            continue
        x = i + rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(x, vals, s=26, alpha=0.7, color=COLORS[c], label=f"{c} (n={len(vals)})",
                   edgecolors="white", linewidths=0.5)
        ax.hlines(np.mean(vals), i - 0.28, i + 0.28, color=COLORS[c], lw=2.5)
    ax.axhline(TH_COPY, color="#cc4444", ls="--", lw=1.2, label=f"TH_COPY = {TH_COPY}")
    ax.axhline(TH_COVER, color="#dd9944", ls="--", lw=1.2, label=f"TH_COVER = {TH_COVER}")
    ax.set_xticks(range(3), CLASSES)
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Similarity distribution by ground-truth class")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(DIST_PNG, dpi=130)
    plt.close(fig)


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    arr = np.array(cm, dtype=float)
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Confusion matrix (rows = truth)")
    vmax = arr.max() or 1
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=12,
                    color="white" if arr[i, j] > vmax * 0.55 else "#222222")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(CONF_PNG, dpi=130)
    plt.close(fig)


# ---------- 5. 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only evaluate first N pairs (smoke test)")
    args = ap.parse_args()

    if not LABEL_XLSX.exists():
        print(f"[ERROR] label file not found: {LABEL_XLSX}")
        return
    items = read_labels(LABEL_XLSX)
    if args.limit:
        items = items[:args.limit]

    print(f"[INFO] pairs={len(items)}  clip={CLIP_SECONDS}s  "
          f"thresholds copy>={TH_COPY} cover>={TH_COVER}", flush=True)

    cache, failed, rows = {}, [], []
    for n, it in enumerate(items, 1):
        print(f"  [{n:>3}/{len(items)}] #{it['id']:<4} {it['truth']}  loading...", flush=True)
        sim, err = None, ""
        try:
            for key in ("qry", "ref"):
                k = str(it[key])
                if k not in cache:
                    cache[k] = load_fp(it[key])
            sim = cosine_sim(cache[str(it["qry"])], cache[str(it["ref"])])
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:90]}"
            failed.append((it["id"], it["truth_zh"], it["qry"].name, it["ref"].name, err))

        pred = classify(sim).split()[0] if sim is not None else "ERROR"
        rows.append({
            "id": it["id"], "truth_zh": it["truth_zh"], "truth": it["truth"],
            "pred": pred, "sim": sim, "correct": int(pred == it["truth"]),
            "qry": it["qry"].name, "ref": it["ref"].name,
            "note": it["note"], "error": err,
        })
        if sim is None:
            print(f"        -> FAILED: {err[:80]}", flush=True)

    # 明细 CSV（含业务文件名，只放本地 samples/）
    DETAIL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with DETAIL_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["序号", "真值", "预测", "相似度", "是否正确", "查询音频", "参考音频", "备忘", "错误"])
        for r in rows:
            w.writerow([r["id"], r["truth_zh"], r["pred"],
                        "" if r["sim"] is None else f"{r['sim']:.4f}",
                        r["correct"], r["qry"], r["ref"], r["note"], r["error"]])

    valid = [r for r in rows if r["pred"] != "ERROR"]
    if not valid:
        print("[ERROR] no pair decoded successfully")
        return

    cm, per_class, acc, macro = metrics(valid)
    stats = {}
    for c in CLASSES:
        v = [r["sim"] for r in valid if r["truth"] == c]
        if not v:                       # --limit 试跑时可能某类一个样本都没有
            stats[c] = {"n": 0, "mean": None, "median": None, "min": None, "max": None}
            continue
        stats[c] = {"n": len(v), "mean": round(float(np.mean(v)), 4),
                    "median": round(float(np.median(v)), 4),
                    "min": round(float(np.min(v)), 4), "max": round(float(np.max(v)), 4)}

    best = search_thresholds(valid)

    summary = {
        "dataset": "labeled qry/ref audio pairs (3 classes, balanced)",
        "n_pairs": len(items), "n_valid": len(valid), "n_failed": len(failed),
        "clip_seconds_per_track": CLIP_SECONDS,
        "thresholds_used": {"copy": TH_COPY, "cover": TH_COVER},
        "accuracy": acc, "macro_f1": macro,
        "per_class": per_class,
        "confusion_matrix": {"labels": CLASSES, "matrix": cm},
        "similarity_stats": stats,
        "top_threshold_candidates": [
            {"accuracy": round(a, 4), "th_copy": tc, "th_cover": tv} for a, tc, tv in best[:5]
        ],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_distribution(valid)
    plot_confusion(cm)

    print("\n========== EVALUATION REPORT ==========", flush=True)
    print(f"pairs: {len(items)}   valid: {len(valid)}   failed: {len(failed)}", flush=True)
    print(f"accuracy (0.85 / 0.55): {acc:.4f}    macro-F1: {macro:.4f}", flush=True)
    print("confusion matrix (rows=truth, cols=pred) " + str(CLASSES), flush=True)
    for c, row in zip(CLASSES, cm):
        print(f"   {c:<10} {row}", flush=True)
    print("per-class:", flush=True)
    for c, v in per_class.items():
        print(f"   {c:<10} P={v['precision']:.3f} R={v['recall']:.3f} "
              f"F1={v['f1']:.3f} n={v['support']}", flush=True)
    print("similarity by truth class:", flush=True)
    for c, v in stats.items():
        if not v["n"]:
            print(f"   {c:<10} (no samples)", flush=True)
            continue
        print(f"   {c:<10} mean={v['mean']:+.3f} median={v['median']:+.3f} "
              f"range=[{v['min']:+.3f},{v['max']:+.3f}]", flush=True)
    print("best thresholds from grid search:", flush=True)
    for a, tc, tv in best[:5]:
        print(f"   copy>={tc:.2f}  cover>={tv:.2f}  -> accuracy {a:.4f}", flush=True)
    if failed:
        print(f"[WARN] {len(failed)} pairs failed to decode; see {DETAIL_CSV}", flush=True)
    print(f"\n[OK] saved: {DETAIL_CSV}, {SUMMARY_JSON}, {DIST_PNG}, {CONF_PNG}", flush=True)


if __name__ == "__main__":
    main()
