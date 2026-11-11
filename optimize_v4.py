"""
optimize_v4.py — 验证 v4 新特征是否真的带来提升（严格同口径对比）

对比三组特征，其余条件完全一致：
    A  v3 原始 3 特征：landmark, sim_timbre, sim_melody
    B  A + v4 全部 8 个旋律特征（去掉旧的 sim_melody，共 10 维）
    C  精简 6 维：landmark, sim_timbre, shift_gap, shift_is_zero, tempo_best, seg_worst
       （样本只有 100 对，特征少一些更不容易过拟合）

评估协议与 optimize_model.py 一致，且用同一个 seed=42 的测试集划分：
    25% 分层独立测试集全程不参与模型选择；
    剩 75 对用 RepeatedStratifiedKFold(5x4) 按 macro-F1 选模型；
    最后在测试集上只评一次；
    再做 20 种不同划分给出准确率的波动范围（判断"是不是运气"）。

输出：v4_model_report.json（纯聚合数字，可公开）
      v4_confusion.png（最佳方案在独立测试集上的混淆矩阵）
      v4_feature_comparison.csv
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split, RepeatedStratifiedKFold, cross_validate,
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from optimize_model import candidates, CLASSES, ZH2EN, SEED, plot_confusion, CONF_PNG

V4_CSV = Path("samples/eval_v4_results.csv")
REPORT_JSON = Path("v4_model_report.json")
COMPARE_CSV = Path("v4_feature_comparison.csv")
V4_CONF_PNG = Path("v4_confusion.png")

SETS = {
    "A_v3_baseline": ["landmark", "sim_timbre", "sim_melody"],
    "B_v3_plus_v4": ["landmark", "sim_timbre", "shift_gap", "shift_margin",
                     "shift_std", "shift_is_zero", "tempo_best", "seg_median",
                     "seg_worst", "sim_melody"],
    "C_compact": ["landmark", "sim_timbre", "shift_gap", "shift_is_zero",
                  "tempo_best", "seg_worst"],
}


def load():
    rows = []
    with V4_CSV.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({k: (v if k == "真值" else float(v))
                         for k, v in r.items()
                         if k not in ("序号", "查询音频", "参考音频")})
    return rows


def evaluate(rows, cols, name):
    X = np.array([[r[c] for c in cols] for r in rows], float)
    y = np.array([ZH2EN[r["真值"].strip()] for r in rows])
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0)
    model_scores, models = [], {}
    for mname, model in candidates().items():
        res = cross_validate(model, Xtr, ytr, cv=cv,
                             scoring=["accuracy", "f1_macro"], n_jobs=1)
        model_scores.append((float(res["test_f1_macro"].mean()),
                             float(res["test_accuracy"].mean()), mname))
        models[mname] = model
    _, _, best = max(model_scores)

    clf = models[best]
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    cm = confusion_matrix(yte, pred, labels=CLASSES)
    rep = classification_report(yte, pred, labels=CLASSES,
                                output_dict=True, zero_division=0)

    spread = []
    for s in range(20):
        a, b, c, d = train_test_split(X, y, test_size=0.25, stratify=y, random_state=s)
        m = candidates()[best]
        m.fit(a, c)
        spread.append(round(float(accuracy_score(d, m.predict(b))), 4))

    return {
        "features": cols, "n_features": len(cols),
        "cv_model_scores": [
            {"model": m, "cv_macro_f1": round(f, 4), "cv_accuracy": round(a, 4)}
            for f, a, m in sorted(model_scores, reverse=True)
        ],
        "selected_model": best,
        "cv_macro_f1_selected": round(max(f for f, _, _ in model_scores), 4),
        "held_out_accuracy": round(float(accuracy_score(yte, pred)), 4),
        "held_out_macro_f1": round(float(f1_score(yte, pred, average="macro")), 4),
        "held_out_confusion": {"labels": CLASSES, "matrix": cm.tolist()},
        "held_out_per_class": {
            c: {k: round(float(v), 4) for k, v in rep[c].items()} for c in CLASSES
        },
        "stability_20_splits": {
            "mean": round(float(np.mean(spread)), 4),
            "min": min(spread), "max": max(spread), "scores": spread,
        },
    }, (best, cm, cols)


def main():
    rows = load()
    print(f"[INFO] pairs={len(rows)}  same hold-out split as optimize_model.py (seed={SEED})\n",
          flush=True)

    results = {}
    for name, cols in SETS.items():
        r, extra = evaluate(rows, cols, name)
        results[name] = r
        print(f"{name:<16} dim={len(cols):>2}  selected={r['selected_model']:<22} "
              f"CV_F1={r['cv_macro_f1_selected']:.3f}  "
              f"TEST_acc={r['held_out_accuracy']:.3f}  "
              f"TEST_F1={r['held_out_macro_f1']:.3f}  "
              f"20-split mean={r['stability_20_splits']['mean']:.3f} "
              f"[{r['stability_20_splits']['min']:.2f}~{r['stability_20_splits']['max']:.2f}]",
              flush=True)

    winner = max(results, key=lambda k: results[k]["stability_20_splits"]["mean"])
    print(f"\n[WINNER by 20-split mean] {winner}", flush=True)
    w = results[winner]
    print(f"  held-out confusion (rows=truth) {CLASSES}", flush=True)
    for c, row in zip(CLASSES, w["held_out_confusion"]["matrix"]):
        print(f"   {c:<10} {row}", flush=True)
    print("  per-class on held-out:", flush=True)
    for c in CLASSES:
        v = w["held_out_per_class"][c]
        print(f"   {c:<10} P={v['precision']:.3f} R={v['recall']:.3f} "
              f"F1={v['f1-score']:.3f} n={int(v['support'])}", flush=True)

    # 画获胜方案的混淆矩阵
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(w["held_out_confusion"]["matrix"])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"Held-out test — {winner} ({w['selected_model']})", fontsize=9)
    vmax = cm.max() or 1
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > vmax * 0.55 else "#222222")
    fig.tight_layout()
    fig.savefig(V4_CONF_PNG, dpi=130)
    plt.close(fig)

    with COMPARE_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["feature_set", "n_features", "selected_model", "cv_macro_f1",
                     "held_out_accuracy", "held_out_macro_f1",
                     "mean_acc_20_splits", "min", "max"])
        for k, v in results.items():
            s = v["stability_20_splits"]
            wr.writerow([k, v["n_features"], v["selected_model"], v["cv_macro_f1_selected"],
                         v["held_out_accuracy"], v["held_out_macro_f1"],
                         s["mean"], s["min"], s["max"]])

    REPORT_JSON.write_text(json.dumps({
        "protocol": "same as optimize_model.py: 25% untouched stratified hold-out; "
                    "RepeatedStratifiedKFold(5x4) on 75% for model selection by macro-F1; "
                    "single held-out evaluation; 20 alternative splits for stability",
        "n_pairs": len(rows), "seed": SEED,
        "winner_by_20_split_mean": winner,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] saved: {REPORT_JSON}, {COMPARE_CSV}, {V4_CONF_PNG}", flush=True)


if __name__ == "__main__":
    main()
