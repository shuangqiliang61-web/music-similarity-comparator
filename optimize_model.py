"""
optimize_model.py — v4 第一步：规范化模型选择 + 独立测试集评估

背景：
    v3 把 (landmark, sim_timbre, sim_melody) 三路分数交给逻辑回归，
    分层 5 折 CV 得到 72%。但「在同一批 100 对上既挑模型又报准确率」
    仍然偏乐观，而且翻唱 recall 只有 0.18。

本脚本做两件更严谨的事：
    1. 先切出 25% 的**独立测试集**（分层，固定 seed），全程不参与模型选择；
    2. 只在剩下 75 对上，用重复分层交叉验证比较多个候选模型，
       以 macro-F1（对三类一视同仁）选模型，最后在测试集上只评一次。
    3. 另做 20 次不同划分的 holdout，给出准确率的波动范围，
       说明 100 对样本下数字本身有多不稳定。

输入：samples/eval_v3_results.csv（含原始文件名，仅本地，勿外传）
输出：model_selection_report.json（纯聚合数字，可公开）
      model_confusion.png（独立测试集混淆矩阵）
      model_comparison.csv（各候选模型 CV 成绩）

用法（Anaconda Prompt）：
    conda activate music
    python optimize_model.py
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    train_test_split, RepeatedStratifiedKFold, cross_validate,
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

DETAIL_CSV = Path("samples/eval_v3_results.csv")
REPORT_JSON = Path("model_selection_report.json")
COMPARE_CSV = Path("model_comparison.csv")
CONF_PNG = Path("model_confusion.png")
CLASSES = ["copy", "cover", "unrelated"]
ZH2EN = {"搬运": "copy", "翻唱": "cover", "不相关": "unrelated"}
SEED = 42


def load():
    X, y, ids = [], [], []
    with DETAIL_CSV.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            X.append([float(r["landmark"]), float(r["sim_timbre"]), float(r["sim_melody"])])
            y.append(ZH2EN[r["真值"].strip()])
            ids.append(r["序号"].strip())
    return np.asarray(X, float), np.asarray(y), ids


def candidates():
    """每个候选都写成 Pipeline：标准化只在训练折里 fit，避免数据泄漏。"""
    return {
        "LogReg (v3 baseline)": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000)),
        "LogReg balanced": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
        "LogReg + poly2": make_pipeline(
            StandardScaler(), PolynomialFeatures(2, include_bias=False),
            StandardScaler(), LogisticRegression(max_iter=5000)),
        "LogReg poly2 balanced": make_pipeline(
            StandardScaler(), PolynomialFeatures(2, include_bias=False),
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced")),
        "SVC rbf C=1": make_pipeline(
            StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale")),
        "SVC rbf C=10 balanced": make_pipeline(
            StandardScaler(), SVC(C=10, kernel="rbf", gamma="scale",
                                  class_weight="balanced")),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=7, l2_regularization=1.0, random_state=SEED),
        "kNN k=5": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }


def plot_confusion(cm, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    arr = np.asarray(cm, float)
    ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title, fontsize=10)
    vmax = arr.max() or 1
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=13,
                    color="white" if arr[i, j] > vmax * 0.55 else "#222222")
    fig.tight_layout()
    fig.savefig(CONF_PNG, dpi=130)
    plt.close(fig)


def main():
    X, y, ids = load()
    print(f"[INFO] pairs={len(X)}  class counts={dict(zip(*np.unique(y, return_counts=True)))}",
          flush=True)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)
    print(f"[INFO] model-selection set={len(Xtr)}   UNTOUCHED test set={len(Xte)}", flush=True)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0)
    rows, fitted = [], {}
    for name, model in candidates().items():
        res = cross_validate(model, Xtr, ytr, cv=cv,
                             scoring=["accuracy", "f1_macro"], n_jobs=1)
        acc, f1 = float(res["test_accuracy"].mean()), float(res["test_f1_macro"].mean())
        rows.append({
            "model": name,
            "cv_accuracy_mean": round(acc, 4),
            "cv_accuracy_std": round(float(res["test_accuracy"].std()), 4),
            "cv_macro_f1_mean": round(f1, 4),
            "cv_macro_f1_std": round(float(res["test_f1_macro"].std()), 4),
        })
        fitted[name] = model
        print(f"  {name:<26} acc={acc:.3f}±{res['test_accuracy'].std():.3f}  "
              f"macroF1={f1:.3f}", flush=True)

    with COMPARE_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    best = max(rows, key=lambda r: r["cv_macro_f1_mean"])["model"]
    print(f"\n[SELECT] {best}  (criterion: macro-F1 on model-selection set only)", flush=True)

    clf = fitted[best]
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc_te = float(accuracy_score(yte, pred))
    f1_te = float(f1_score(yte, pred, average="macro"))
    cm = confusion_matrix(yte, pred, labels=CLASSES)
    report = classification_report(yte, pred, labels=CLASSES,
                                   output_dict=True, zero_division=0)
    plot_confusion(cm, f"Held-out test (n={len(Xte)}) — {best}")

    # 稳健性：换 20 种划分重跑同一流程，看数字波动多大
    spread = []
    for s in range(20):
        a, b, c, d = train_test_split(X, y, test_size=0.25, stratify=y, random_state=s)
        m = candidates()[best]
        m.fit(a, c)
        spread.append(round(float(accuracy_score(d, m.predict(b))), 4))

    print("\n========== HELD-OUT TEST (seen only once) ==========", flush=True)
    print(f"accuracy : {acc_te:.4f}   macro-F1: {f1_te:.4f}   n={len(Xte)}", flush=True)
    print("confusion (rows=truth, cols=pred) " + str(CLASSES), flush=True)
    for c, row in zip(CLASSES, cm):
        print(f"   {c:<10} {list(row)}", flush=True)
    print("per-class:", flush=True)
    for c in CLASSES:
        v = report[c]
        print(f"   {c:<10} P={v['precision']:.3f} R={v['recall']:.3f} "
              f"F1={v['f1-score']:.3f} n={int(v['support'])}", flush=True)
    print(f"20 random splits: mean={np.mean(spread):.3f}  "
          f"min={min(spread):.2f}  max={max(spread):.2f}  <- 样本量小，数字会晃", flush=True)

    out = {
        "n_pairs": int(len(X)),
        "features": ["landmark", "sim_timbre", "sim_melody"],
        "protocol": "25% stratified hold-out untouched during model selection; "
                    "RepeatedStratifiedKFold(5x4) on the remaining 75% for selection; "
                    "single evaluation on hold-out; 20 alternative splits for stability",
        "seed": SEED,
        "n_selection": int(len(Xtr)), "n_test": int(len(Xte)),
        "model_comparison_cv": rows,
        "selected_model": best,
        "selection_criterion": "mean macro-F1 over repeated 5-fold CV",
        "held_out_accuracy": round(acc_te, 4),
        "held_out_macro_f1": round(f1_te, 4),
        "held_out_confusion": {"labels": CLASSES, "matrix": cm.tolist()},
        "held_out_per_class": {
            c: {k: round(float(v), 4) for k, v in report[c].items()} for c in CLASSES
        },
        "stability_20_splits": {
            "scores": spread, "mean": round(float(np.mean(spread)), 4),
            "min": min(spread), "max": max(spread),
        },
        "v3_logreg_5fold_cv_for_reference": 0.72,
    }
    REPORT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] saved: {REPORT_JSON}, {COMPARE_CSV}, {CONF_PNG}", flush=True)


if __name__ == "__main__":
    main()
