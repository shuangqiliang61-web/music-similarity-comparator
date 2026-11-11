"""
train_model.py — 训练并导出最终可部署的分类器

协议：
    1. 用 100 对标注数据的 (landmark, sim_timbre, sim_melody) 三路分数
    2. 分层重复 5 折交叉验证估计泛化性能（仅用于报告，不是最终模型分数）
    3. 用全部 100 对重新训练最终模型（标准做法：CV 估性能，全量训模型）
    4. 导出 model_bundle.joblib 与 model_card.json / model_card.md

隐私：模型只含 3 组系数与标准化统计量，不含任何音频、文件名、业务 ID，
      可以安全提交到公开仓库。

用法：conda activate music && python train_model.py
"""

from __future__ import annotations

import csv
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix

from features_runtime import FEATURE_ORDER

ZH2EN = {"搬运": "copy", "翻唱": "cover", "不相关": "unrelated"}
CLASSES = ["copy", "cover", "unrelated"]

TRAIN_CSV = Path("samples/eval_v3_results.csv")
BUNDLE = Path("model_bundle.joblib")
CARD_JSON = Path("model_card.json")
CARD_MD = Path("model_card.md")


def main():
    if not TRAIN_CSV.exists():
        raise SystemExit(f"[ERROR] {TRAIN_CSV} not found; run evaluate_v3.py first")

    X, y = [], []
    with TRAIN_CSV.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            X.append([float(r[k]) for k in FEATURE_ORDER])
            y.append(ZH2EN[r["真值"].strip()])
    X, y = np.asarray(X, float), np.asarray(y)
    print(f"[INFO] training pairs={len(y)}  features={FEATURE_ORDER}", flush=True)
    print(f"[INFO] class counts={dict(zip(*np.unique(y, return_counts=True)))}", flush=True)

    def build():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0)
    res = cross_validate(build(), X, y, cv=cv,
                         scoring=["accuracy", "f1_macro"], n_jobs=1)
    cv_acc = round(float(res["test_accuracy"].mean()), 4)
    cv_f1 = round(float(res["test_f1_macro"].mean()), 4)
    print(f"[INFO] CV estimate: accuracy={cv_acc:.4f}  macro-F1={cv_f1:.4f}", flush=True)

    clf = build()
    clf.fit(X, y)
    in_sample = float(clf.score(X, y))
    report = classification_report(y, clf.predict(X), labels=CLASSES,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y, clf.predict(X), labels=CLASSES)

    bundle = {
        "pipeline": clf,
        "feature_order": FEATURE_ORDER,
        "classes": list(clf.named_steps["logisticregression"].classes_),
        "trained_on_pairs": int(len(y)),
        "cv_accuracy_mean": cv_acc,
        "cv_macro_f1_mean": cv_f1,
        "sr": 22050,
        "clip_seconds": 60,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "evaluate_v2.py + evaluate_v3.py scores on 100 labelled pairs",
    }
    joblib.dump(bundle, BUNDLE)

    card = {
        "model": "StandardScaler + LogisticRegression (multinomial, 3 classes)",
        "features": FEATURE_ORDER,
        "feature_meanings": {
            "landmark": "spectral-peak fingerprint match rate (same recording)",
            "sim_timbre": "MFCC sequence sub-sequence DTW similarity (timbre)",
            "sim_melody": "key-invariant Chroma DTW similarity (melody)",
        },
        "training_pairs": int(len(y)),
        "class_distribution": {c: int((y == c).sum()) for c in CLASSES},
        "cross_validation": {
            "scheme": "RepeatedStratifiedKFold(5 splits x 4 repeats, seed=0)",
            "accuracy_mean": cv_acc,
            "accuracy_std": round(float(res["test_accuracy"].std()), 4),
            "macro_f1_mean": cv_f1,
        },
        "in_sample_accuracy_not_for_reporting": round(in_sample, 4),
        "in_sample_confusion": {"labels": CLASSES, "matrix": cm.tolist()},
        "in_sample_per_class": {
            c: {k: round(float(v), 4) for k, v in report[c].items()} for c in CLASSES
        },
        "audio_preprocessing": "mono, sr=22050, first 60 seconds",
        "privacy": "Contains model coefficients only. No audio, filenames or IDs.",
        "limitations": [
            "Only 100 labelled pairs from a single source; accuracy varies "
            "0.52-0.88 across 20 different train/test splits (mean ~0.67).",
            "Cover detection is the weakest class (see README / 评测报告.md).",
            "Not suitable for production copyright decisions.",
        ],
        "reproduce": "conda activate music && python train_model.py",
    }
    CARD_JSON.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Model card",
        "",
        f"- Model: `{card['model']}`",
        f"- Features: `{', '.join(FEATURE_ORDER)}`",
        f"- Training data: {len(y)} labelled pairs "
        f"({card['class_distribution']})",
        f"- Audio: mono, 22050 Hz, first 60 s",
        f"- **Cross-validated accuracy: {cv_acc:.1%}** "
        f"(macro-F1 {cv_f1:.3f}, 5-fold x 4 repeats)",
        f"- In-sample accuracy {in_sample:.1%} is shown only as a sanity check, "
        "**not** a performance claim",
        "",
        "## Feature meanings",
        "",
    ]
    for k, v in card["feature_meanings"].items():
        lines.append(f"- `{k}`: {v}")
    lines += [
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {s}" for s in card["limitations"]]
    lines += [
        "",
        "## Privacy",
        "",
        card["privacy"],
        "",
        "## Reproduce",
        "",
        "```bash",
        "conda activate music",
        "python train_model.py",
        "python predict.py song1.mp3 song2.mp3",
        "```",
        "",
    ]
    CARD_MD.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== in-sample confusion (rows=truth) ===", flush=True)
    for c, row in zip(CLASSES, cm):
        print(f"   {c:<10} {list(row)}", flush=True)
    print(f"\n[OK] saved: {BUNDLE}, {CARD_JSON}, {CARD_MD}", flush=True)


if __name__ == "__main__":
    main()
