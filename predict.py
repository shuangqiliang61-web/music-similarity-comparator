"""
predict.py — 项目真正的推理入口（v3 特征 + 训练好的分类器）

用法（Anaconda Prompt：conda activate music）：
    python predict.py song1.mp3 song2.mp3
    python predict.py song1.mp3 song2.mp3 --json

输出：三分类结果 + 每一类的概率 + 三路中间分数。
若尚未训练模型，会提示先运行 `python train_model.py`。

注意：特征提取统一走 features_runtime.py，与训练侧严格同源。
      main.py 保留为 v1 教学演示，不代表当前算法性能。
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np

from features_runtime import compute_features, as_vector, FEATURE_ORDER

BUNDLE = Path("model_bundle.joblib")
ZH = {"copy": "搬运", "cover": "翻唱", "unrelated": "不相关"}


def load_bundle(path: Path):
    if not path.is_file():
        raise SystemExit(
            f"[ERROR] model not found: {path}\n"
            "        Run first:  conda activate music && python train_model.py")
    return joblib.load(path)


def predict_pair(path_qry: str, path_ref: str, bundle_path: Path = BUNDLE) -> dict:
    bundle = load_bundle(bundle_path)
    feat = compute_features(path_qry, path_ref)
    x = np.asarray([as_vector(feat)], float)

    clf = bundle["pipeline"]
    order = list(bundle["feature_order"])
    if order != FEATURE_ORDER:
        raise SystemExit(f"[ERROR] feature order mismatch: model={order}, runtime={FEATURE_ORDER}")

    proba = clf.predict_proba(x)[0]
    classes = list(clf.named_steps["logisticregression"].classes_) \
        if hasattr(clf, "named_steps") else list(clf.classes_)
    label = str(classes[int(np.argmax(proba))])
    return {
        "query": str(path_qry),
        "reference": str(path_ref),
        "prediction": label,
        "prediction_zh": ZH.get(label, label),
        "confidence": round(float(np.max(proba)), 4),
        "probabilities": {str(c): round(float(p), 4) for c, p in zip(classes, proba)},
        "features": {k: round(float(v), 6) for k, v in feat.items()},
        "model_cv_accuracy_estimate": bundle.get("cv_accuracy_mean"),
    }


def main():
    ap = argparse.ArgumentParser(description="Classify a pair of songs (copy / cover / unrelated)")
    ap.add_argument("songs", nargs=2, help="query audio and reference audio")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--model", default=str(BUNDLE), help="path to model_bundle.joblib")
    args = ap.parse_args()

    for p in args.songs:
        if not Path(p).is_file():
            raise SystemExit(f"[ERROR] audio file not found: {p}")

    try:
        result = predict_pair(args.songs[0], args.songs[1], Path(args.model))
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(f"[ERROR] failed to analyze pair: {type(e).__name__}: {e}")

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("\n========== 音乐比对结果 ==========")
    print(f"  判定：{result['prediction_zh']} ({result['prediction']})"
          f"   置信度 {result['confidence']:.1%}")
    print("  各类概率：" + "  ".join(
        f"{ZH.get(k, k)} {v:.1%}" for k, v in result["probabilities"].items()))
    print("  中间分数：")
    for k, v in result["features"].items():
        print(f"    {k:<14} {v:+.4f}")
    print(f"\n  模型交叉验证准确率估计：{result['model_cv_accuracy_estimate']:.1%}")
    print("  ⚠️ 仅 100 对标注数据训练，翻唱识别仍是弱项；请勿用于正式版权判定。")
    print("==================================\n")


if __name__ == "__main__":
    main()
