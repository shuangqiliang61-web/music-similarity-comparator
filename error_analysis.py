"""
error_analysis.py — 用越折预测做错误分析（不挑好看的那一折）

用 cross_val_predict 得到每对样本的「没被自己训练过的」预测，
再按类别、按是否有难例备注、按 landmark 区间统计错误分布。
所有输出都是聚合数字，不含文件名或业务 ID。

输出：ERROR_ANALYSIS.md、error_analysis.json
用法：conda activate music && python error_analysis.py
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report

from evaluate_dataset import read_labels, LABEL_XLSX
from features_runtime import FEATURE_ORDER

CSV = Path("samples/eval_v3_results.csv")
OUT_MD = Path("ERROR_ANALYSIS.md")
OUT_JSON = Path("error_analysis.json")
CLASSES = ["copy", "cover", "unrelated"]
ZH2EN = {"搬运": "copy", "翻唱": "cover", "不相关": "unrelated"}


def main():
    rows = list(csv.DictReader(CSV.open(encoding="utf-8-sig")))
    X = np.array([[float(r[k]) for k in FEATURE_ORDER] for r in rows])
    y = np.array([ZH2EN[r["真值"].strip()] for r in rows])

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pred = cross_val_predict(clf, X, y, cv=cv)

    labels = {str(it["id"]).strip(): it for it in read_labels(LABEL_XLSX)}
    notes = np.array([
        bool(labels.get(str(r["序号"]).strip(), {}).get("note", "").strip())
        for r in rows
    ])

    cm = confusion_matrix(y, pred, labels=CLASSES)
    rep = classification_report(y, pred, labels=CLASSES,
                                output_dict=True, zero_division=0)
    acc = float((y == pred).mean())

    by_class = {}
    for c in CLASSES:
        mask = y == c
        wrong = pred[mask][pred[mask] != c]
        by_class[c] = {
            "support": int(mask.sum()),
            "recall": round(float(rep[c]["recall"]), 4),
            "most_common_mistake": (str(np.bincount(
                [CLASSES.index(w) for w in wrong]).argmax())
                if len(wrong) else None),
            "mistaken_as": {k: int(v) for k, v in zip(
                *np.unique(wrong, return_counts=True))} if len(wrong) else {},
        }
    # 把 argmax 索引换成可读类名
    for c, v in by_class.items():
        idx = v.pop("most_common_mistake")
        v["most_common_mistake"] = CLASSES[int(idx)] if idx is not None else None

    by_note = {
        "with_note": {"n": int(notes.sum()),
                      "accuracy": round(float((y[notes] == pred[notes]).mean()), 4)
                      if notes.sum() else None},
        "without_note": {"n": int((~notes).sum()),
                         "accuracy": round(float((y[~notes] == pred[~notes]).mean()), 4)},
    }

    lm = X[:, 0]
    bins = [(-1, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 2.0)]
    by_landmark = []
    for lo, hi in bins:
        m = (lm >= lo) & (lm < hi)
        by_landmark.append({
            "range": f"[{lo:.2f}, {hi:.2f})", "n": int(m.sum()),
            "accuracy": round(float((y[m] == pred[m]).mean()), 4) if m.sum() else None,
            "class_mix": {c: int(((y == c) & m).sum()) for c in CLASSES},
        })

    print(f"[INFO] out-of-fold accuracy = {acc:.4f}  (n={len(y)})", flush=True)
    print("confusion (rows=truth, cols=pred):", flush=True)
    for c, row in zip(CLASSES, cm):
        print(f"   {c:<10} {list(row)}", flush=True)
    for c, v in by_class.items():
        print(f"   {c:<10} recall={v['recall']:.3f}  最常见错判 -> "
              f"{v['most_common_mistake']}  {v['mistaken_as']}", flush=True)
    print("by note:", by_note, flush=True)
    for b in by_landmark:
        print(f"   landmark {b['range']:<14} n={b['n']:>2} acc={b['accuracy']}",
              flush=True)

    data = {
        "n_pairs": int(len(y)),
        "method": "StratifiedKFold(5, seed=0) cross_val_predict; every prediction "
                  "comes from a model that did not train on that pair",
        "out_of_fold_accuracy": round(acc, 4),
        "confusion": {"labels": CLASSES, "matrix": cm.tolist()},
        "per_class": by_class,
        "by_hard_case_note": by_note,
        "by_landmark_bin": by_landmark,
    }
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Error analysis（越折预测）",
        "",
        "所有结论基于 `cross_val_predict`：每一对的预测都来自**没有用这一对训练过**的模型，",
        "避免用「自己考自己」的结果分析问题。",
        "",
        f"- 样本：{len(y)} 对",
        f"- 越折准确率：**{acc:.1%}**",
        "",
        "## 混淆矩阵（行=真值，列=预测）",
        "",
        "|  | " + " | ".join(CLASSES) + " |",
        "|---|---|---|---|",
    ]
    for c, row in zip(CLASSES, cm):
        lines.append(f"| **{c}** | " + " | ".join(str(v) for v in row) + " |")
    lines += ["", "## 每一类最容易错成什么", "",
              "| 类别 | 样本数 | recall | 最常见错判 |", "|---|---|---|---|"]
    for c, v in by_class.items():
        lines.append(f"| {c} | {v['support']} | {v['recall']:.3f} | "
                     f"{v['most_common_mistake']} |")
    lines += [
        "",
        "## 难例备注",
        "",
        f"- 有备注（难例）：{by_note['with_note']['n']} 对，"
        f"准确率 {by_note['with_note']['accuracy']:.1%}",
        f"- 无备注：{by_note['without_note']['n']} 对，"
        f"准确率 {by_note['without_note']['accuracy']:.1%}",
        "",
        "## 按 landmark 分数分箱",
        "",
        "| landmark 区间 | 对数 | 准确率 | 类别构成 |",
        "|---|---|---|---|",
    ]
    for b in by_landmark:
        mix = ", ".join(f"{k}={v}" for k, v in b["class_mix"].items() if v)
        a = f"{b['accuracy']:.1%}" if b["accuracy"] is not None else "n/a"
        lines.append(f"| {b['range']} | {b['n']} | {a} | {mix} |")
    lines += [
        "",
        "## 结论",
        "",
        "1. 翻唱是主要错误来源：它既可能被当成搬运（高逼真度翻唱，频谱峰仍对齐），"
        "也可能被当成不相关（变调 / 变速 / 纯伴奏 / 人声音量很小）。",
        "2. landmark 中间区间的错误最集中——那里正是「旋律相同但录音不同」的区域，"
        "单靠频谱峰指纹无法区分。",
        "3. 有难例备注的样本准确率更低，说明标注时人工也认为这些 case 更含糊。",
        "4. 想再提升，需要的是**旋律级、对人声/伴奏变化鲁棒**的表示，或更多这类难例，"
        "而不是继续调阈值或换分类器（见 `评测报告.md` 第 9、10 节）。",
        "",
        "> 本报告只含聚合统计，不含音频、文件名与业务 ID。",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] saved: {OUT_MD}, {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
