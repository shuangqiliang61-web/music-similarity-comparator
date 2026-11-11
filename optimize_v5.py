"""
optimize_v5.py — v5 新特征是否真提升？用嵌套评估避免「挑到好看的那次」

两层协议：

【展示层】固定 seed=42 的 25% 独立测试集，对比 5 组特征 × 多个模型，
          给出直观表格与混淆矩阵。

【结论层】嵌套评估（nested selection），跑 20 种划分：
          每种划分先用训练部分做 5 折 CV，从「特征集 × 模型」的候选里
          选 macro-F1 最好的；选完才在测试部分评一次。
          对照组只能用 v3 老特征选模型，实验组可以选新特征。
          两组的 20 次准确率均值之差，才是「新特征有没有用」的无偏估计。

输出：v5_model_report.json、v5_feature_comparison.csv、v5_confusion.png
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from optimize_model import ZH2EN, CLASSES, SEED

V5_CSV = Path("samples/eval_v5_results.csv")
REPORT_JSON = Path("v5_model_report.json")
COMPARE_CSV = Path("v5_feature_comparison.csv")
CONF_PNG = Path("v5_confusion.png")

SETS = {
    "A_v3_baseline": ["landmark", "sim_timbre", "sim_melody"],
    "B_add_rhythm_acf": ["landmark", "sim_timbre", "sim_melody", "rhythm_acf_sim"],
    "C_add_two_independent": ["landmark", "sim_timbre", "sim_melody",
                              "rhythm_acf_sim", "tempo_ratio"],
    "D_rhythm_all": ["landmark", "sim_timbre", "sim_melody", "rhythm_acf_sim",
                     "tempo_ratio", "sim_rhythm"],
    "E_all_8": ["landmark", "sim_timbre", "sim_melody", "sim_cens", "cens_gap",
                "sim_rhythm", "tempo_ratio", "rhythm_acf_sim"],
}


def models(light: bool = False):
    """标准化放进 Pipeline，保证只在训练折里 fit。"""
    d = {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "LogReg balanced": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
        "SVC rbf": make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale")),
    }
    if not light:
        d["SVC rbf balanced"] = make_pipeline(
            StandardScaler(), SVC(C=3, kernel="rbf", gamma="scale",
                                  class_weight="balanced"))
        d["RandomForest"] = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    return d


def load():
    rows = list(csv.DictReader(V5_CSV.open(encoding="utf-8-sig")))
    y = np.array([ZH2EN[r["真值"].strip()] for r in rows])
    all_cols = list(SETS["E_all_8"])
    X = {c: np.array([float(r[c]) for r in rows]) for c in all_cols}
    return X, y, all_cols


def matrix(X, cols):
    return np.column_stack([X[c] for c in cols])


def fixed_holdout(X, y):
    """展示层：同一测试集上逐组对比。"""
    out = []
    for name, cols in SETS.items():
        M = matrix(X, cols)
        a, b, c, d = train_test_split(M, y, test_size=0.25, stratify=y,
                                      random_state=SEED)
        best = None
        for mname, model in models().items():
            s = float(cross_val_score(model, a, c, cv=StratifiedKFold(
                5, shuffle=True, random_state=0), scoring="f1_macro").mean())
            if best is None or s > best[0]:
                best = (s, mname, model)
        clf = best[2]
        clf.fit(a, c)
        pred = clf.predict(b)
        out.append({
            "features": name, "n_features": len(cols), "selected_model": best[1],
            "cv_macro_f1": round(best[0], 4),
            "held_out_accuracy": round(float(accuracy_score(d, pred)), 4),
            "held_out_macro_f1": round(float(f1_score(d, pred, average="macro")), 4),
            "confusion": confusion_matrix(d, pred, labels=CLASSES).tolist(),
            "per_class": {k: {kk: round(float(vv), 4) for kk, vv in v.items()}
                          for k, v in classification_report(
                              d, pred, labels=CLASSES, output_dict=True,
                              zero_division=0).items() if k in CLASSES},
        })
        print(f"  {name:<24} dim={len(cols)}  model={best[1]:<18} "
              f"CV_F1={best[0]:.3f}  TEST={out[-1]['held_out_accuracy']:.3f}", flush=True)
    return out


def nested(X, y, allowed_sets, tag):
    """结论层：每次划分只在训练部分选特征集+模型，测试部分只评一次。"""
    scores, choices = [], {}
    pool = models(light=True)
    for s in range(20):
        best = None
        for sname in allowed_sets:
            M = matrix(X, SETS[sname])
            a, b, c, d = train_test_split(M, y, test_size=0.25, stratify=y,
                                          random_state=s)
            cv = StratifiedKFold(5, shuffle=True, random_state=s)
            for mname, model in pool.items():
                sc = float(cross_val_score(model, a, c, cv=cv,
                                           scoring="f1_macro").mean())
                if best is None or sc > best[0]:
                    best = (sc, sname, mname, M, a, b, c, d)
        _, sname, mname, _, a, b, c, d = best
        clf = pool[mname]
        clf.fit(a, c)
        pred = clf.predict(b)
        acc = float(accuracy_score(d, pred))
        scores.append(round(acc, 4))
        key = f"{sname}|{mname}"
        choices[key] = choices.get(key, 0) + 1
    print(f"  {tag:<24} mean={np.mean(scores):.4f}  "
          f"min={min(scores):.2f}  max={max(scores):.2f}", flush=True)
    return {
        "scores": scores, "mean": round(float(np.mean(scores)), 4),
        "std": round(float(np.std(scores)), 4),
        "min": min(scores), "max": max(scores),
        "selected_combinations": choices,
    }


def plot_conf(cm, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arr = np.asarray(cm, float)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title, fontsize=9)
    vmax = arr.max() or 1
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=13,
                    color="white" if arr[i, j] > vmax * 0.55 else "#222222")
    fig.tight_layout()
    fig.savefig(CONF_PNG, dpi=130)
    plt.close(fig)


def main():
    X, y, _ = load()
    print(f"[INFO] pairs={len(y)}  class counts={dict(zip(*np.unique(y, return_counts=True)))}",
          flush=True)

    print("\n=== 展示层：固定 seed=42 独立测试集 ===", flush=True)
    table = fixed_holdout(X, y)

    print("\n=== 结论层：嵌套评估 20 次划分（模型/特征选择都在训练部分内） ===", flush=True)
    base = nested(X, y, ["A_v3_baseline"], "A: v3 baseline")
    full = nested(X, y, list(SETS.keys()), "A~E: v5 可选新特征")
    delta = round(full["mean"] - base["mean"], 4)
    print(f"\n[RESULT] nested delta = {delta:+.4f}", flush=True)

    winner = max(table, key=lambda r: r["held_out_accuracy"])
    plot_conf(winner["confusion"],
              f"Held-out test — {winner['features']} ({winner['selected_model']})")

    with COMPARE_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["feature_set", "n_features", "selected_model", "cv_macro_f1",
                    "held_out_accuracy", "held_out_macro_f1"])
        for r in table:
            w.writerow([r["features"], r["n_features"], r["selected_model"],
                        r["cv_macro_f1"], r["held_out_accuracy"],
                        r["held_out_macro_f1"]])

    REPORT_JSON.write_text(json.dumps({
        "n_pairs": int(len(y)), "seed_fixed_holdout": SEED,
        "protocol": "Fixed 25% hold-out for display; nested 20-split evaluation "
                    "where BOTH feature set and model are selected by inner 5-fold CV "
                    "on the training portion only; test portion evaluated once.",
        "feature_sets": SETS,
        "fixed_holdout_comparison": table,
        "nested_baseline_v3": base,
        "nested_v5": full,
        "nested_delta": delta,
        "interpretation": "Only the nested delta can be cited as evidence that "
                          "the new features help; single-split numbers are unstable.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] saved: {REPORT_JSON}, {COMPARE_CSV}, {CONF_PNG}", flush=True)


if __name__ == "__main__":
    main()
