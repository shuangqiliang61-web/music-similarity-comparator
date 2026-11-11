"""
evaluate_v3.py — 频谱峰值指纹（core_v3）+ v2 旋律分数，在同一套 100 对上评测

用法（Anaconda Prompt 里先 conda activate music）：
    python evaluate_v3.py

复用 v2 已算好的 sim_melody（读 samples/eval_v2_results.csv），
只新算 landmark 分数，因此跑得很快。

产出：
    samples/eval_v3_results.csv    每对明细（含原始文件名，仅本地，勿外传）
    eval_summary_v3.json           聚合统计 + v1/v2/v3 对比
    eval_v3_scatter.png            landmark vs melody 二维散点
    eval_confusion_v3.png          混淆矩阵

评估口径与 evaluate_v2.py 一致：默认阈值 / 同数据调优 / 5 折交叉验证。
另附一个 scikit-learn 逻辑回归作对照，看「从数据学分类器」还能再提多少。
"""

from __future__ import annotations

import csv
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import core_v3
from evaluate_dataset import read_labels, metrics, CLASSES, LABEL_XLSX, SR, CLIP_SECONDS

warnings.filterwarnings("ignore")

V2_CSV = Path("samples/eval_v2_results.csv")
V1_JSON = Path("eval_summary.json")
V2_JSON = Path("eval_summary_v2.json")
DETAIL_CSV = Path("samples/eval_v3_results.csv")
SUMMARY_JSON = Path("eval_summary_v3.json")
SCATTER_PNG = Path("eval_v3_scatter.png")
CONF_PNG = Path("eval_confusion_v3.png")
COLORS = {"copy": "#cc4444", "cover": "#dd9944", "unrelated": "#888780"}


def load_peaks(path: Path, cache: dict):
    k = str(path)
    if k not in cache:
        try:
            y, sr = librosa.load(str(path), sr=SR, mono=True, duration=CLIP_SECONDS)
        except Exception:
            y, sr = librosa.load(str(path), sr=SR, mono=True)
            if CLIP_SECONDS:
                y = y[: int(sr * CLIP_SECONDS)]
        cache[k] = core_v3.peaks(y, sr)
    return cache[k]


def predict(r, tl, tm):
    return "copy" if r["lm"] >= tl else ("cover" if r["sm"] >= tm else "unrelated")


def accuracy(rows, tl, tm):
    return sum(predict(r, tl, tm) == r["truth"] for r in rows) / len(rows)


def grid_best(rows):
    best = (0.0, core_v3.TH_LANDMARK, core_v3.TH_MELODY)
    for tl in np.arange(0.005, 0.501, 0.005):
        for tm in np.arange(0.85, 1.0001, 0.005):
            a = accuracy(rows, float(tl), float(tm))
            if a > best[0]:
                best = (a, round(float(tl), 3), round(float(tm), 3))
    return best


def cross_val(rows, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    parts = np.array_split(order, folds)
    scores = []
    for held in parts:
        held_set = set(held.tolist())
        train = [rows[i] for i in range(len(rows)) if i not in held_set]
        _, tl, tm = grid_best(train)
        held_rows = [rows[i] for i in held]
        scores.append(accuracy(held_rows, tl, tm))
    return round(float(np.mean(scores)), 4), [round(float(s), 4) for s in scores]


def sklearn_baseline(rows):
    """对照：用 (landmark, sim_timbre, sim_melody) 学一个逻辑回归，5 折 CV。"""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold
    except Exception as e:
        return None, f"sklearn unavailable: {type(e).__name__}"
    X = np.array([[r["lm"], r["st"], r["sm"]] for r in rows])
    y = np.array([r["truth"] for r in rows])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000)
        clf.fit(sc.transform(X[tr]), y[tr])
        scores.append(float(clf.score(sc.transform(X[te]), y[te])))
    return round(float(np.mean(scores)), 4), [round(s, 4) for s in scores]


def plot_scatter(rows, tl, tm):
    fig, ax = plt.subplots(figsize=(7.2, 6))
    for c in CLASSES:
        pts = [r for r in rows if r["truth"] == c]
        ax.scatter([r["lm"] for r in pts], [r["sm"] for r in pts],
                   s=34, alpha=0.78, color=COLORS[c], label=f"{c} (n={len(pts)})",
                   edgecolors="white", linewidths=0.5)
    ax.axvline(tl, color="#333333", ls="--", lw=1.2, label=f"th_landmark = {tl}")
    ax.axhline(tm, color="#555555", ls=":", lw=1.2, label=f"th_melody = {tm}")
    ax.set_xlabel("landmark match rate (spectral peak fingerprint)")
    ax.set_ylabel("sim_melody (Chroma DTW, key-invariant)")
    ax.set_title("v3 features: discrete fingerprint vs melody")
    ax.set_ylim(0.85, 1.01)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(SCATTER_PNG, dpi=130)
    plt.close(fig)


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    arr = np.asarray(cm, dtype=float)
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("v3 confusion matrix (rows = truth)")
    vmax = arr.max() or 1
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=12,
                    color="white" if arr[i, j] > vmax * 0.55 else "#222222")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(CONF_PNG, dpi=130)
    plt.close(fig)


def main():
    if not V2_CSV.exists():
        print(f"[ERROR] run evaluate_v2.py first ({V2_CSV} not found)")
        return
    v2map = {}
    with V2_CSV.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            v2map[r["序号"].strip()] = (float(r["sim_timbre"]), float(r["sim_melody"]))

    items = read_labels(LABEL_XLSX)
    print(f"[INFO] pairs={len(items)}  clip={CLIP_SECONDS}s  method=v3 (landmark + v2 melody)", flush=True)

    cache, rows, failed = {}, [], 0
    for n, it in enumerate(items, 1):
        print(f"  [{n:>3}/{len(items)}] #{it['id']:<4} {it['truth']}", flush=True)
        try:
            lm = core_v3.landmark_score(load_peaks(it["qry"], cache),
                                        load_peaks(it["ref"], cache))
            st, sm = v2map[it["id"]]
        except Exception as e:
            failed += 1
            print(f"        -> FAILED: {type(e).__name__}: {str(e)[:70]}", flush=True)
            continue
        rows.append({
            "id": it["id"], "truth": it["truth"], "truth_zh": it["truth_zh"],
            "lm": lm, "st": st, "sm": sm,
            "pred": core_v3.classify(lm, sm),
            "qry": it["qry"].name, "ref": it["ref"].name, "note": it["note"],
        })

    if not rows:
        print("[ERROR] no pair evaluated")
        return

    DETAIL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with DETAIL_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["序号", "真值", "预测", "landmark", "sim_timbre", "sim_melody",
                    "查询音频", "参考音频", "备忘"])
        for r in rows:
            w.writerow([r["id"], r["truth_zh"], r["pred"], f"{r['lm']:.4f}",
                        f"{r['st']:.4f}", f"{r['sm']:.4f}", r["qry"], r["ref"], r["note"]])

    acc_default = accuracy(rows, core_v3.TH_LANDMARK, core_v3.TH_MELODY)
    acc_tuned, tl, tm = grid_best(rows)
    cv_mean, cv_folds = cross_val(rows)
    sk_mean, sk_folds = sklearn_baseline(rows)

    tuned_rows = [dict(r, pred=predict(r, tl, tm)) for r in rows]
    cm, per_class, _, macro = metrics(tuned_rows)

    stats = {}
    for c in CLASSES:
        v = [r["lm"] for r in rows if r["truth"] == c]
        if not v:
            continue
        stats[c] = {"n": len(v), "landmark_mean": round(float(np.mean(v)), 4),
                    "landmark_median": round(float(np.median(v)), 4),
                    "landmark_range": [round(float(np.min(v)), 4), round(float(np.max(v)), 4)]}

    def read_acc(p, key):
        try:
            return json.loads(Path(p).read_text(encoding="utf-8")).get(key)
        except Exception:
            return None

    summary = {
        "method": "v3: spectral-peak landmark matching + v2 key-invariant Chroma DTW",
        "n_pairs": len(items), "n_valid": len(rows), "n_failed": failed,
        "clip_seconds_per_track": CLIP_SECONDS,
        "default_thresholds": {"th_landmark": core_v3.TH_LANDMARK, "th_melody": core_v3.TH_MELODY},
        "accuracy_default_thresholds": round(acc_default, 4),
        "tuned_thresholds": {"th_landmark": tl, "th_melody": tm},
        "accuracy_tuned_in_sample": round(acc_tuned, 4),
        "accuracy_5fold_cv": cv_mean, "cv_fold_scores": cv_folds,
        "sklearn_logreg_5fold_cv": sk_mean, "sklearn_fold_scores": sk_folds,
        "macro_f1_tuned": macro, "per_class_tuned": per_class,
        "confusion_matrix_tuned": {"labels": CLASSES, "matrix": cm},
        "landmark_stats_by_class": stats,
        "comparison": {
            "v1_accuracy": read_acc(V1_JSON, "accuracy"),
            "v2_accuracy_5fold_cv": read_acc(V2_JSON, "accuracy_5fold_cv"),
            "v3_accuracy_5fold_cv": cv_mean,
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_scatter(rows, tl, tm)
    plot_confusion(cm)

    print("\n========== V3 EVALUATION REPORT ==========", flush=True)
    print(f"pairs: {len(items)}   valid: {len(rows)}   failed: {failed}", flush=True)
    print(f"v1 accuracy:                            {summary['comparison']['v1_accuracy']}", flush=True)
    print(f"v2 accuracy, 5-fold CV:                 {summary['comparison']['v2_accuracy_5fold_cv']}", flush=True)
    print(f"v3 accuracy, default thresholds:        {acc_default:.4f}", flush=True)
    print(f"v3 accuracy, tuned on same data:        {acc_tuned:.4f}  "
          f"(th_landmark={tl}, th_melody={tm})", flush=True)
    print(f"v3 accuracy, 5-fold CV:                 {cv_mean:.4f}  <- honest estimate", flush=True)
    print(f"  fold scores: {cv_folds}", flush=True)
    print(f"sklearn LogReg 5-fold CV (reference):   {sk_mean}  {sk_folds}", flush=True)
    print("landmark score by truth class:", flush=True)
    for c, v in stats.items():
        print(f"   {c:<10} mean={v['landmark_mean']:.4f} median={v['landmark_median']:.4f} "
              f"range={v['landmark_range']}", flush=True)
    print("confusion matrix (tuned, rows=truth, cols=pred) " + str(CLASSES), flush=True)
    for c, row in zip(CLASSES, cm):
        print(f"   {c:<10} {row}", flush=True)
    print("per-class (tuned):", flush=True)
    for c, v in per_class.items():
        print(f"   {c:<10} P={v['precision']:.3f} R={v['recall']:.3f} "
              f"F1={v['f1']:.3f} n={v['support']}", flush=True)
    print(f"\n[OK] saved: {DETAIL_CSV}, {SUMMARY_JSON}, {SCATTER_PNG}, {CONF_PNG}", flush=True)


if __name__ == "__main__":
    main()
