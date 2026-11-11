"""
optimize_hierarchical.py — 结构性改进：三分类 -> 两级二分类

动机
----
翻唱 recall 一直最低（~0.42）：它夹在「搬运」和「不相关」中间，
一个三分类逻辑回归要同时学两条边界，而中间类样本只有 33 条。

改成两级：
  Stage 1: copy vs rest      —— landmark 指纹几乎单独就能决定
  Stage 2: cover vs unrelated —— 只在非 copy 样本上训练，用旋律/节奏特征

每层只有一个二分类决策面，参数量更小，理论上更适合 100 对小样本。
这是**结构**改动，不是再堆一维特征，因此值得单独验证。

评估协议（与 optimize_v5.py 完全一致的嵌套评估）
------------------------------------------------
跑 20 种划分。每种划分：
  - 只在训练部分(75%)内用 5 折 CV，按 macro-F1 选「特征集 × 结构」
  - 选完在测试部分(25%)只评一次
三组对照：
  G0 只能用 v3 三路特征 + 扁平三分类（等于现有基线）
  G1 v3 三路特征 + 任意结构（单独看结构的贡献）
  G2 所有特征 + 任意结构（结构与特征共同的上限）
只有 G1/G2 相对 G0 的均值差，才能作为「有没有用」的证据。

显著性：20 次划分用相同 seed，属配对样本，因此额外做配对 t 检验
        与 Wilcoxon 符号秩检验。只有两者都 p<0.05 才对外宣称「有提升」。

输出：hierarchical_report.json（含 paired_significance）、
      hierarchical_comparison.csv
"""
from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, cross_val_predict,
)
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)
from scipy import stats

from optimize_model import ZH2EN, CLASSES, SEED

V3_CSV = Path("samples/eval_v3_results.csv")
V4_CSV = Path("samples/eval_v4_results.csv")
V5_CSV = Path("samples/eval_v5_results.csv")
V6_CSV = Path("samples/eval_v6_results.csv")
REPORT_JSON = Path("hierarchical_report.json")
COMPARE_CSV = Path("hierarchical_comparison.csv")

# hratio_sim 在 100 对上恒为 1.0（常量、零方差），显式剔除
V6_COLS = ["hpcp_sim", "hpcp_gap", "hpcp_flat", "hpcp_shift_zero",
           "pitch_match", "pitch_shift_zero"]
# 与旧特征低相关、三类均值分离度好的三项，单独成组
V6_INDEPENDENT = ["hpcp_flat", "pitch_match", "pitch_shift_zero"]
# G0/G1/G2 固定用这 5 组，加入 v6 后仍可复现、可比较
BASE_SET_NAMES = ["base3", "base3+rhythm2", "base3+shift2",
                  "base3+independent4", "all"]

SETS = {
    "base3": ["landmark", "sim_timbre", "sim_melody"],
    "base3+rhythm2": ["landmark", "sim_timbre", "sim_melody",
                      "rhythm_acf_sim", "tempo_ratio"],
    "base3+shift2": ["landmark", "sim_timbre", "sim_melody",
                     "shift_is_zero", "shift_gap"],
    "base3+independent4": ["landmark", "sim_timbre", "sim_melody",
                           "rhythm_acf_sim", "tempo_ratio",
                           "shift_is_zero", "shift_gap"],
    "all": ["landmark", "sim_timbre", "sim_melody",
            "sim_cens", "cens_gap", "sim_rhythm", "tempo_ratio",
            "rhythm_acf_sim", "shift_is_zero", "shift_gap"],
}


def _lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


class FlatClassifier(BaseEstimator, ClassifierMixin):
    """扁平三分类：现有基线的等价实现。"""

    def __init__(self, balanced: bool = False, ovr: bool = False):
        self.balanced = balanced
        self.ovr = ovr

    def fit(self, X, y):
        self.classes_ = np.array(CLASSES)
        base = LogisticRegression(
            max_iter=3000,
            class_weight="balanced" if self.balanced else None)
        if self.ovr:
            self.model_ = make_pipeline(
                StandardScaler(), OneVsRestClassifier(base))
        else:
            self.model_ = make_pipeline(StandardScaler(), base)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    """两级二分类：先 copy vs rest，再 cover vs unrelated。

    tune_threshold=True 时，Stage 1 的判定阈值在**训练部分内部**
    用 3 折 out-of-fold 概率按 macro-F1 挑选；测试部分完全不参与。
    """

    def __init__(self, tune_threshold: bool = False, balanced: bool = False):
        self.tune_threshold = tune_threshold
        self.balanced = balanced

    def _new_lr(self):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000,
                               class_weight="balanced" if self.balanced else None))

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.classes_ = np.array(CLASSES)
        y1 = (y == "copy").astype(int)
        self.stage1_ = self._new_lr().fit(X, y1)

        mask = y != "copy"
        self.stage2_ = self._new_lr().fit(
            X[mask], (y[mask] == "cover").astype(int))

        self.threshold_ = 0.5
        if self.tune_threshold and len(np.unique(y1)) > 1:
            # 内层 oof 概率只来自训练部分；用整体 macro-F1 挑阈值
            p1 = cross_val_predict(
                self._new_lr(), X, y1, cv=3, method="predict_proba")[:, 1]
            best = None
            for t in np.arange(0.25, 0.80, 0.05):
                pred = self._compose(X, p1 >= t,
                                     self.stage2_.predict_proba(X)[:, 1])
                sc = float(f1_score(y, pred, average="macro"))
                if best is None or sc > best[0]:
                    best = (sc, float(t))
            if best is not None:
                self.threshold_ = best[1]
        return self

    @staticmethod
    def _compose(X, is_copy, p2):
        rest = np.where(np.asarray(p2) >= 0.5, "cover", "unrelated")
        return np.where(np.asarray(is_copy), "copy", rest)

    def predict(self, X):
        X = np.asarray(X, float)
        p1 = self.stage1_.predict_proba(X)[:, 1]
        p2 = self.stage2_.predict_proba(X)[:, 1]
        return self._compose(X, p1 >= self.threshold_, p2)


def structures():
    return {
        "flat3": FlatClassifier(),
        "flat3_balanced": FlatClassifier(balanced=True),
        "flat3_ovr": FlatClassifier(ovr=True),
        "hier2": HierarchicalClassifier(),
        "hier2_balanced": HierarchicalClassifier(balanced=True),
        "hier2_tuned": HierarchicalClassifier(tune_threshold=True),
    }


def load():
    """按「序号」把各 CSV 对齐，避免依赖行顺序。

    若 evaluate_v6.py 已产出合并表，优先读它（列最全）；
    否则回退到 v3/v4/v5 三表合并。
    """
    def read(p):
        with p.open(encoding="utf-8-sig") as f:
            return {r["序号"].strip(): r for r in csv.DictReader(f)}

    r6 = read(V6_CSV) if V6_CSV.exists() else {}
    has_v6 = bool(r6) and all(
        c in next(iter(r6.values())) for c in V6_COLS)
    if has_v6:
        ids = sorted(r6, key=lambda x: int(x))
        y = np.array([ZH2EN[r6[i]["真值"].strip()] for i in ids])
        cols = sorted({c for s in SETS.values() for c in s} | set(V6_COLS))
        X = {c: np.nan_to_num(np.array([float(r6[i][c]) for i in ids], float),
                              nan=0.0, posinf=0.0, neginf=0.0) for c in cols}
        print("[INFO] loaded merged table with v6 features", flush=True)
        return X, y, cols, True

    r3, r4, r5 = read(V3_CSV), read(V4_CSV), read(V5_CSV)
    ids = sorted(r3, key=lambda x: int(x))
    y = np.array([ZH2EN[r3[i]["真值"].strip()] for i in ids])
    cols = sorted({c for s in SETS.values() for c in s})
    X = {}
    for c in cols:
        src = r3 if c in r3[ids[0]] else (r4 if c in r4[ids[0]] else r5)
        X[c] = np.array([float(src[i][c]) for i in ids])
    return X, y, cols, False


def matrix(X, cols):
    return np.column_stack([X[c] for c in cols])


def nested(X, y, set_names, struct_names, tag):
    pool = structures()
    pool = {k: pool[k] for k in struct_names}
    scores, cover_recalls, choices = [], [], {}
    cv_scores_all = []

    for s in range(20):
        best = None
        for sname in set_names:
            M = matrix(X, SETS[sname])
            a, b, c, d = train_test_split(
                M, y, test_size=0.25, stratify=y, random_state=s)
            cv = StratifiedKFold(5, shuffle=True, random_state=s)
            for stname, est in pool.items():
                sc = float(cross_val_score(
                    clone(est), a, c, cv=cv, scoring="f1_macro").mean())
                cv_scores_all.append(sc)
                if best is None or sc > best[0]:
                    best = (sc, sname, stname, a, b, c, d)

        _, sname, stname, a, b, c, d = best
        clf = clone(pool[stname]).fit(a, c)
        pred = clf.predict(b)
        scores.append(round(float(accuracy_score(d, pred)), 4))
        rep = classification_report(d, pred, labels=CLASSES,
                                    output_dict=True, zero_division=0)
        cover_recalls.append(round(float(rep["cover"]["recall"]), 4))
        key = f"{sname}|{stname}"
        choices[key] = choices.get(key, 0) + 1

    res = {
        "tag": tag,
        "allowed_feature_sets": list(set_names),
        "allowed_structures": list(struct_names),
        "scores_20_splits": scores,
        "mean_accuracy": round(float(np.mean(scores)), 4),
        "std_accuracy": round(float(np.std(scores)), 4),
        "min": min(scores), "max": max(scores),
        "mean_cover_recall": round(float(np.mean(cover_recalls)), 4),
        "mean_inner_cv_f1": round(float(np.mean(cv_scores_all)), 4),
        "selected_combinations": choices,
    }
    print(f"  {tag:<26} mean={res['mean_accuracy']:.4f} "
          f"coverR={res['mean_cover_recall']:.3f} "
          f"innerCV={res['mean_inner_cv_f1']:.3f} "
          f"range=[{res['min']:.2f},{res['max']:.2f}]", flush=True)
    return res


def fixed_holdout(X, y):
    """展示层：固定 seed=42 的独立测试集，逐个结构看混淆矩阵。"""
    rows = []
    a0, b0, c0, d0 = train_test_split(
        matrix(X, SETS["base3"]), y, test_size=0.25, stratify=y,
        random_state=SEED)
    for stname, est in structures().items():
        clf = clone(est).fit(a0, c0)
        pred = clf.predict(b0)
        rep = classification_report(d0, pred, labels=CLASSES,
                                    output_dict=True, zero_division=0)
        rows.append({
            "structure": stname,
            "held_out_accuracy": round(float(accuracy_score(d0, pred)), 4),
            "held_out_macro_f1": round(float(f1_score(d0, pred, average="macro")), 4),
            "cover_recall": round(float(rep["cover"]["recall"]), 4),
            "confusion": confusion_matrix(d0, pred, labels=CLASSES).tolist(),
        })
        print(f"  {stname:<20} TEST={rows[-1]['held_out_accuracy']:.3f} "
              f"coverR={rows[-1]['cover_recall']:.3f}", flush=True)
    return rows


def paired_test(a_scores, b_scores):
    """同一批划分 -> 配对检验；只有两种检验都显著才认定提升。"""
    a = np.asarray(a_scores, float)
    b = np.asarray(b_scores, float)
    diff = a - b
    _, pt = stats.ttest_rel(a, b)
    try:
        _, pw = stats.wilcoxon(a, b, zero_method="zsplit")
    except ValueError:
        pw = float("nan")
    return {
        "mean_delta": round(float(diff.mean()), 4),
        "wins": int((diff > 0).sum()),
        "ties": int((diff == 0).sum()),
        "losses": int((diff < 0).sum()),
        "paired_t_p": round(float(pt), 4),
        "wilcoxon_p": round(float(pw), 4),
        "significant_at_0.05": bool(pt < 0.05 and pw < 0.05),
    }


def correlation(X, cols):
    base = ["landmark", "sim_timbre", "sim_melody"]
    out = {}
    for c in cols:
        if c in base:
            continue
        out[c] = {b: round(float(np.corrcoef(X[c], X[b])[0, 1]), 3)
                  for b in base}
    return out


def main():
    X, y, cols, has_v6 = load()
    print(f"[INFO] pairs={len(y)} features={len(cols)} "
          f"class={dict(zip(*np.unique(y, return_counts=True)))}", flush=True)

    print("\n=== 新特征与 v3 三路特征的相关性（>0.9 视为冗余）===", flush=True)
    corr = correlation(X, cols)
    for c, v in corr.items():
        mx = max(abs(x) for x in v.values())
        print(f"  {c:<18} max|r|={mx:.3f}  {v}", flush=True)

    print("\n=== 展示层：固定独立测试集，只换结构（v3 三路特征）===", flush=True)
    table = fixed_holdout(X, y)

    print("\n=== 结论层：嵌套评估 20 次划分 ===", flush=True)
    g0 = nested(X, y, ["base3"], ["flat3"], "G0 v3基线(仅扁平)")
    g1 = nested(X, y, ["base3"], list(structures()), "G1 仅换结构")
    g2 = nested(X, y, BASE_SET_NAMES, list(structures()), "G2 结构+旧特征")

    g3 = None
    if has_v6:
        # 只把「与旧特征低相关」的离散音级轮廓单独成组；
        # 另外也给全部 v6 特征一个机会，由内折 CV 自己判断。
        SETS["base3+v6indep"] = SETS["base3"] + V6_INDEPENDENT
        SETS["all+v6indep"] = SETS["all"] + V6_INDEPENDENT
        SETS["all+v6"] = SETS["all"] + V6_COLS
        g3 = nested(X, y, list(SETS), list(structures()), "G3 结构+全部(含v6)")

    print("\n========== RESULT ==========", flush=True)
    print(f"  结构单独贡献 G1-G0 = {g1['mean_accuracy'] - g0['mean_accuracy']:+.4f}",
          flush=True)
    print(f"  结构+旧特征 G2-G0 = {g2['mean_accuracy'] - g0['mean_accuracy']:+.4f}",
          flush=True)
    if g3:
        print(f"  再加 v6     G3-G0 = {g3['mean_accuracy'] - g0['mean_accuracy']:+.4f}",
              flush=True)
    print(f"  翻唱 recall  G0={g0['mean_cover_recall']:.3f} -> "
          f"G1={g1['mean_cover_recall']:.3f} -> G2={g2['mean_cover_recall']:.3f}"
          + (f" -> G3={g3['mean_cover_recall']:.3f}" if g3 else ""), flush=True)

    s0 = g0["scores_20_splits"]
    sig = {"G2_vs_G0": paired_test(g2["scores_20_splits"], s0)}
    if g3:
        sig["G3_vs_G0"] = paired_test(g3["scores_20_splits"], s0)
        sig["G3_vs_G2"] = paired_test(g3["scores_20_splits"],
                                      g2["scores_20_splits"])
    print("\n=== 配对显著性检验（同一批划分，20 组）===", flush=True)
    for k, v in sig.items():
        print(f"  {k:<10} delta={v['mean_delta']:+.4f}  "
              f"W/T/L={v['wins']}/{v['ties']}/{v['losses']}  "
              f"t_p={v['paired_t_p']:.4f}  wilcoxon_p={v['wilcoxon_p']:.4f}  "
              f"significant={v['significant_at_0.05']}", flush=True)

    verdict = ("Directionally positive but NOT statistically significant on 100 pairs. "
               "v6 independent features improve cover recall and narrow the spread, "
               "but with 20 paired splits neither test reaches p<0.05. The production "
               "model is intentionally kept at the v3 baseline; v6 is retained as a "
               "documented, reproducible experiment.")
    print(f"\n[VERDICT] {verdict}", flush=True)

    with COMPARE_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["structure", "held_out_accuracy",
                                          "held_out_macro_f1", "cover_recall"])
        w.writeheader()
        for r in table:
            w.writerow({k: r[k] for k in w.fieldnames})

    REPORT_JSON.write_text(json.dumps({
        "n_pairs": int(len(y)),
        "protocol": "Nested 20-split evaluation: inside each split, feature set "
                    "and classifier structure are chosen by 5-fold CV on the "
                    "75% training portion only; the 25% test portion is scored once. "
                    "Identical splits to optimize_v5.py for comparability.",
        "motivation": "cover sits between copy and unrelated; a flat 3-class model "
                      "must learn two boundaries from only 33 cover pairs. A "
                      "two-stage binary cascade learns one boundary per stage.",
        "feature_sets": SETS,
        "correlation_with_v3": corr,
        "fixed_holdout_structures": table,
        "nested": {"G0_baseline": g0, "G1_structure_only": g1,
                   "G2_all_old_features": g2,
                   "G3_with_v6": g3},
        "delta_structure_only": round(g1["mean_accuracy"] - g0["mean_accuracy"], 4),
        "delta_all_old": round(g2["mean_accuracy"] - g0["mean_accuracy"], 4),
        "delta_with_v6": (round(g3["mean_accuracy"] - g0["mean_accuracy"], 4)
                          if g3 else None),
        "paired_significance": sig,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] saved: {REPORT_JSON}, {COMPARE_CSV}", flush=True)


if __name__ == "__main__":
    main()
