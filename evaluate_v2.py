"""
evaluate_v2.py — 用 core_v2 在同一套 100 对标注数据上评测，并与 v1 对比

用法（Anaconda Prompt 里先 conda activate music）：
    python evaluate_v2.py                # 全量 100 对
    python evaluate_v2.py --limit 10     # 试跑

产出：
    samples/eval_v2_results.csv   每对明细（含原始文件名，仅本地，勿外传）
    eval_summary_v2.json          聚合统计 + 与 v1 的对比
    eval_v2_scatter.png           两维相似度散点（看三类是否分开）
    eval_confusion_v2.png         混淆矩阵

评估口径：
    - default thresholds : 直接用 core_v2 里的默认阈值，最诚实
    - tuned thresholds   : 在全部 100 对上网格搜索，属于乐观上界
    - 5-fold CV          : 用 4 折调阈值、在剩下 1 折上测，取平均，作为可信估计
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import core_v2
from evaluate_dataset import (read_labels, metrics, CLASSES, IDX,
                              LABEL_XLSX, SR, CLIP_SECONDS)

warnings.filterwarnings("ignore")

DETAIL_CSV = Path("samples/eval_v2_results.csv")
SUMMARY_JSON = Path("eval_summary_v2.json")
SCATTER_PNG = Path("eval_v2_scatter.png")
CONF_PNG = Path("eval_confusion_v2.png")
V1_JSON = Path("eval_summary.json")
COLORS = {"copy": "#cc4444", "cover": "#dd9944", "unrelated": "#888780"}


def load_features(path: Path, cache: dict):
    k = str(path)
    if k not in cache:
        try:
            y, sr = librosa.load(str(path), sr=SR, mono=True, duration=CLIP_SECONDS)
        except Exception:
            y, sr = librosa.load(str(path), sr=SR, mono=True)
            if CLIP_SECONDS:
                y = y[: int(sr * CLIP_SECONDS)]
        cache[k] = core_v2.extract(y, sr)
    return cache[k]


def predict(rows, th_timbre, th_melody):
    return [core_v2.classify(r["st"], r["sm"], th_timbre, th_melody) for r in rows]


def accuracy(rows, th_timbre, th_melody):
    preds = predict(rows, th_timbre, th_melody)
    return sum(p == r["truth"] for p, r in zip(preds, rows)) / len(rows)


def grid_best(rows):
    best = (0.0, core_v2.TH_TIMBRE, core_v2.TH_MELODY)
    for tt in np.arange(0.50, 1.001, 0.01):
        for tm in np.arange(0.50, 1.001, 0.01):
            a = accuracy(rows, float(tt), float(tm))
            if a > best[0]:
                best = (a, round(float(tt), 2), round(float(tm), 2))
    return best


def cross_val(rows, folds=5, seed=0):
    """用 4 折调阈值、在留出折上测，避免调参与评估用同一批数据。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    parts = np.array_split(order, folds)
    scores = []
    for held in parts:
        train = [rows[i] for i in range(len(rows)) if i not in set(held.tolist())]
        _, tt, tm = grid_best(train)
        held_rows = [rows[i] for i in held]
        scores.append(accuracy(held_rows, tt, tm))
    return round(float(np.mean(scores)), 4), [round(s, 4) for s in scores]


def plot_scatter(valid, tt, tm):
    fig, ax = plt.subplots(figsize=(7.2, 6))
    for c in CLASSES:
        pts = [r for r in valid if r["truth"] == c]
        ax.scatter([r["st"] for r in pts], [r["sm"] for r in pts],
                   s=32, alpha=0.75, color=COLORS[c], label=f"{c} (n={len(pts)})",
                   edgecolors="white", linewidths=0.5)
    ax.axvline(tt, color="#333333", ls="--", lw=1.2, label=f"th_timbre = {tt}")
    ax.axhline(tm, color="#555555", ls=":", lw=1.2, label=f"th_melody = {tm}")
    ax.set_xlabel("sim_timbre (MFCC sequence DTW)")
    ax.set_ylabel("sim_melody (Chroma DTW, key-invariant)")
    ax.set_title("v2 features: timbre vs melody similarity")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower left")
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
    ax.set_title("v2 confusion matrix (rows = truth)")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = read_labels(LABEL_XLSX)
    if args.limit:
        items = items[:args.limit]
    print(f"[INFO] pairs={len(items)}  clip={CLIP_SECONDS}s  method=v2 (sequence + DTW)", flush=True)

    cache, rows, failed = {}, [], 0
    for n, it in enumerate(items, 1):
        print(f"  [{n:>3}/{len(items)}] #{it['id']:<4} {it['truth']}", flush=True)
        try:
            a = load_features(it["qry"], cache)
            b = load_features(it["ref"], cache)
            st, sm = core_v2.similarity(a, b)
        except Exception as e:
            failed += 1
            print(f"        -> FAILED: {type(e).__name__}: {str(e)[:70]}", flush=True)
            continue
        pred = core_v2.classify(st, sm)
        rows.append({
            "id": it["id"], "truth": it["truth"], "truth_zh": it["truth_zh"],
            "pred": pred, "st": st, "sm": sm, "correct": int(pred == it["truth"]),
            "qry": it["qry"].name, "ref": it["ref"].name, "note": it["note"],
        })

    if not rows:
        print("[ERROR] no pair evaluated")
        return

    DETAIL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with DETAIL_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["序号", "真值", "预测", "sim_timbre", "sim_melody",
                    "是否正确", "查询音频", "参考音频", "备忘"])
        for r in rows:
            w.writerow([r["id"], r["truth_zh"], r["pred"], f"{r['st']:.4f}",
                        f"{r['sm']:.4f}", r["correct"], r["qry"], r["ref"], r["note"]])

    acc_default = accuracy(rows, core_v2.TH_TIMBRE, core_v2.TH_MELODY)
    acc_tuned, tt, tm = grid_best(rows)
    cv_mean, cv_folds = cross_val(rows)

    tuned_rows = [dict(r, pred=p) for r, p in zip(rows, predict(rows, tt, tm))]
    cm, per_class, acc_check, macro = metrics(tuned_rows)

    stats = {}
    for c in CLASSES:
        st_v = [r["st"] for r in rows if r["truth"] == c]
        sm_v = [r["sm"] for r in rows if r["truth"] == c]
        if not st_v:
            continue
        stats[c] = {
            "n": len(st_v),
            "sim_timbre_mean": round(float(np.mean(st_v)), 4),
            "sim_timbre_range": [round(float(np.min(st_v)), 4), round(float(np.max(st_v)), 4)],
            "sim_melody_mean": round(float(np.mean(sm_v)), 4),
            "sim_melody_range": [round(float(np.min(sm_v)), 4), round(float(np.max(sm_v)), 4)],
        }

    v1_acc = None
    if V1_JSON.exists():
        try:
            v1_acc = json.loads(V1_JSON.read_text(encoding="utf-8")).get("accuracy")
        except Exception:
            pass

    summary = {
        "method": "v2: per-bin sequences + subsequence DTW (key-invariant chroma)",
        "n_pairs": len(items), "n_valid": len(rows), "n_failed": failed,
        "clip_seconds_per_track": CLIP_SECONDS,
        "default_thresholds": {"th_timbre": core_v2.TH_TIMBRE, "th_melody": core_v2.TH_MELODY},
        "accuracy_default_thresholds": round(acc_default, 4),
        "tuned_thresholds": {"th_timbre": tt, "th_melody": tm},
        "accuracy_tuned_in_sample": round(acc_tuned, 4),
        "accuracy_5fold_cv": cv_mean,
        "cv_fold_scores": cv_folds,
        "macro_f1_tuned": macro,
        "per_class_tuned": per_class,
        "confusion_matrix_tuned": {"labels": CLASSES, "matrix": cm},
        "score_stats": stats,
        "comparison": {"v1_accuracy": v1_acc,
                       "v2_accuracy_default": round(acc_default, 4),
                       "v2_accuracy_5fold_cv": cv_mean},
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_scatter(rows, tt, tm)
    plot_confusion(cm)

    print("\n========== V2 EVALUATION REPORT ==========", flush=True)
    print(f"pairs: {len(items)}   valid: {len(rows)}   failed: {failed}", flush=True)
    print(f"v1 accuracy (for reference):            {v1_acc}", flush=True)
    print(f"v2 accuracy, default thresholds:        {acc_default:.4f}", flush=True)
    print(f"v2 accuracy, tuned on same data:        {acc_tuned:.4f}  "
          f"(th_timbre={tt}, th_melody={tm})  <- optimistic upper bound", flush=True)
    print(f"v2 accuracy, 5-fold cross-validation:   {cv_mean:.4f}  <- honest estimate", flush=True)
    print(f"  fold scores: {cv_folds}", flush=True)
    print("confusion matrix (tuned, rows=truth, cols=pred) " + str(CLASSES), flush=True)
    for c, row in zip(CLASSES, cm):
        print(f"   {c:<10} {row}", flush=True)
    print("per-class (tuned):", flush=True)
    for c, v in per_class.items():
        print(f"   {c:<10} P={v['precision']:.3f} R={v['recall']:.3f} "
              f"F1={v['f1']:.3f} n={v['support']}", flush=True)
    print("score means by truth class:", flush=True)
    for c, v in stats.items():
        print(f"   {c:<10} timbre={v['sim_timbre_mean']:.3f}  melody={v['sim_melody_mean']:.3f}", flush=True)
    print(f"\n[OK] saved: {DETAIL_CSV}, {SUMMARY_JSON}, {SCATTER_PNG}, {CONF_PNG}", flush=True)


if __name__ == "__main__":
    main()
