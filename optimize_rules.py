"""
optimize_rules.py — 反方向尝试：用「参数极少的可解释规则」代替黑箱模型

动机：
    v5 加了真有信息的新特征，训练集内 CV 从 0.623 涨到 0.770，
    但嵌套 20 次划分均值反而 -0.008。原因不是特征没用，而是
    100 对样本撑不住 8 维特征 + 随机森林的高方差。

    那就反过来：只用 2~3 个参数、物理含义清楚的规则。
    参数少 => 方差小 => 小样本上更可能稳。

规则族（统一形式，只有两个阈值）：
    recording_score = landmark          越高越像「同一段录音」
    song_score      = 若干旋律/节奏特征的标准化平均   越高越像「同一首歌」
    if recording_score >= t1 -> copy
    elif song_score >= t2    -> cover
    else                     -> unrelated

    6 种 song_score 组合，全部在训练折上用网格搜索定阈值（按 macro-F1），
    测试折只评一次；再跑 20 次划分做嵌套评估。

输出：rules_report.json、rules_comparison.csv
"""

from __future__ import annotations

import csv
import json
import itertools
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from optimize_model import ZH2EN, CLASSES, SEED

V5_CSV = Path("samples/eval_v5_results.csv")
REPORT_JSON = Path("rules_report.json")
COMPARE_CSV = Path("rules_comparison.csv")

COMBOS = {
    "R1_melody_only_v3": ["sim_melody"],
    "R2_rhythm_acf": ["rhythm_acf_sim"],
    "R3_melody_plus_rhythm": ["sim_melody", "rhythm_acf_sim"],
    "R4_three": ["sim_melody", "rhythm_acf_sim", "tempo_ratio"],
    "R5_four": ["sim_melody", "rhythm_acf_sim", "tempo_ratio", "sim_rhythm"],
    "R6_cens_plus_rhythm": ["sim_cens", "rhythm_acf_sim", "tempo_ratio"],
}


def load():
    rows = list(csv.DictReader(V5_CSV.open(encoding="utf-8-sig")))
    y = np.array([ZH2EN[r["真值"].strip()] for r in rows])
    cols = ["landmark", "sim_melody", "rhythm_acf_sim", "tempo_ratio",
            "sim_rhythm", "sim_cens"]
    X = np.column_stack([[float(r[c]) for r in rows] for c in cols])
    return X, y, {c: i for i, c in enumerate(cols)}


def prep(Xtr, Xte, idx, combo):
    """标准化只在训练折 fit；song_score = 选中列的 z 分数均值。"""
    sub_tr, sub_te = Xtr[:, idx], Xte[:, idx]
    mu, sd = sub_tr.mean(axis=0), sub_tr.std(axis=0) + 1e-9
    ztr, zte = (sub_tr - mu) / sd, (sub_te - mu) / sd
    return Xtr[:, 0], Xte[:, 0], ztr.mean(axis=1), zte.mean(axis=1)


def fit_thresholds(lm, song, ytr):
    """按 macro-F1 在训练折上网格搜索两个阈值。"""
    t1s = np.unique(np.quantile(lm, np.linspace(0.05, 0.95, 19)))
    t2s = np.unique(np.quantile(song, np.linspace(0.05, 0.95, 19)))
    best = (-1.0, float(t1s[0]), float(t2s[0]))
    for a, b in itertools.product(t1s, t2s):
        pred = np.where(lm >= a, "copy", np.where(song >= b, "cover", "unrelated"))
        f = float(f1_score(ytr, pred, average="macro", zero_division=0))
        if f > best[0]:
            best = (f, float(a), float(b))
    return best


def predict(lm, song, t1, t2):
    return np.where(lm >= t1, "copy", np.where(song >= t2, "cover", "unrelated"))


def fixed_holdout(X, y, idx):
    a, b, c, d = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    out = []
    for name, combo in COMBOS.items():
        ci = [idx[k] for k in combo]
        lm_tr, lm_te, s_tr, s_te = prep(a, b, ci, combo)
        f, t1, t2 = fit_thresholds(lm_tr, s_tr, c)
        pred = predict(lm_te, s_te, t1, t2)
        out.append({
            "rule": name, "song_score_features": combo,
            "train_cv_macro_f1": round(f, 4),
            "thresholds": {"th_landmark": round(t1, 4), "th_song": round(t2, 4)},
            "held_out_accuracy": round(float(accuracy_score(d, pred)), 4),
            "held_out_macro_f1": round(float(f1_score(d, pred, average="macro")), 4),
            "confusion": confusion_matrix(d, pred, labels=CLASSES).tolist(),
        })
        print(f"  {name:<26} trainF1={f:.3f}  TEST={out[-1]['held_out_accuracy']:.3f} "
              f"(t1={t1:.3f}, t2={t2:.3f})", flush=True)
    return out


def nested(X, y, idx, only=None, n_splits=20):
    """每次划分：训练部分内选规则+阈值，测试部分只评一次。"""
    names = only or list(COMBOS.keys())
    scores, picks = [], {}
    for s in range(n_splits):
        a, b, c, d = train_test_split(X, y, test_size=0.25, stratify=y, random_state=s)
        best = None
        for name in names:
            ci = [idx[k] for k in COMBOS[name]]
            lm_tr, _, s_tr, _ = prep(a, b, ci, COMBOS[name])
            f, t1, t2 = fit_thresholds(lm_tr, s_tr, c)
            if best is None or f > best[0]:
                best = (f, name, t1, t2, ci)
        _, name, t1, t2, ci = best
        lm_tr, lm_te, s_tr, s_te = prep(a, b, ci, COMBOS[name])
        pred = predict(lm_te, s_te, t1, t2)
        scores.append(round(float(accuracy_score(d, pred)), 4))
        picks[name] = picks.get(name, 0) + 1
    return {
        "scores": scores, "mean": round(float(np.mean(scores)), 4),
        "std": round(float(np.std(scores)), 4),
        "min": min(scores), "max": max(scores),
        "selected_rules": picks,
    }


def main():
    X, y, idx = load()
    print(f"[INFO] pairs={len(y)}\n", flush=True)

    print("=== 展示层：固定 seed=42 独立测试集 ===", flush=True)
    table = fixed_holdout(X, y, idx)

    print("\n=== 结论层：嵌套 20 次划分 ===", flush=True)
    base = nested(X, y, idx, only=["R1_melody_only_v3"])
    print(f"  v3 原规则 (R1)        mean={base['mean']:.4f}  "
          f"[{base['min']:.2f}~{base['max']:.2f}]", flush=True)
    full = nested(X, y, idx)
    print(f"  6 规则族内自动选择    mean={full['mean']:.4f}  "
          f"[{full['min']:.2f}~{full['max']:.2f}]", flush=True)
    print(f"\n[RESULT] nested delta = {full['mean'] - base['mean']:+.4f}", flush=True)
    print("  rule picks:", full["selected_rules"], flush=True)

    with COMPARE_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rule", "song_score_features", "train_cv_macro_f1",
                    "th_landmark", "th_song", "held_out_accuracy", "held_out_macro_f1"])
        for r in table:
            w.writerow([r["rule"], "+".join(r["song_score_features"]),
                        r["train_cv_macro_f1"], r["thresholds"]["th_landmark"],
                        r["thresholds"]["th_song"], r["held_out_accuracy"],
                        r["held_out_macro_f1"]])

    REPORT_JSON.write_text(json.dumps({
        "n_pairs": int(len(y)), "seed_fixed_holdout": SEED,
        "rule_form": "landmark >= t1 -> copy; elif standardized song_score >= t2 -> cover; else unrelated",
        "combos": COMBOS,
        "fixed_holdout_comparison": table,
        "nested_baseline_v3_rule": base,
        "nested_all_rules": full,
        "nested_delta": round(full["mean"] - base["mean"], 4),
        "reference": {"v3_logreg_20split_mean": 0.672,
                      "nested_model_baseline": 0.660,
                      "nested_v5_features_delta": -0.008},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] saved: {REPORT_JSON}, {COMPARE_CSV}", flush=True)


if __name__ == "__main__":
    main()
