"""
evaluate_v4.py — 在 100 对标注数据上计算 core_v4 的旋律级新特征

用法（Anaconda Prompt：conda activate music）：
    python evaluate_v4.py --limit 8     # 先试跑，测速度
    python evaluate_v4.py               # 全量 100 对

复用：landmark / sim_timbre 直接读 samples/eval_v3_results.csv（不重算），
      只新算 v4 的 8 个旋律特征，因此比从零跑便宜。

产出：
    samples/eval_v4_results.csv   每对明细（含原始文件名，仅本地，勿外传）
    eval_v4_feature_stats.json    各特征按真值类别的聚合统计（纯数字，可公开）

之后运行 optimize_v4.py 做「独立测试集」口径的模型对比。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import librosa

import core_v2
import core_v4
from evaluate_dataset import read_labels, LABEL_XLSX, SR, CLIP_SECONDS

V3_CSV = Path("samples/eval_v3_results.csv")
DETAIL_CSV = Path("samples/eval_v4_results.csv")
STATS_JSON = Path("eval_v4_feature_stats.json")


def load_features(path: Path, cache: dict):
    key = str(path)
    if key not in cache:
        try:
            y, sr = librosa.load(str(path), sr=SR, mono=True, duration=CLIP_SECONDS)
        except Exception:
            y, sr = librosa.load(str(path), sr=SR, mono=True)
            if CLIP_SECONDS:
                y = y[: int(sr * CLIP_SECONDS)]
        cache[key] = core_v2.extract(y, sr)
    return cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 对，用于试跑")
    args = ap.parse_args()

    v3map = {}
    with V3_CSV.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            v3map[r["序号"].strip()] = (float(r["landmark"]), float(r["sim_timbre"]))

    items = read_labels(LABEL_XLSX)
    if args.limit:
        items = items[: args.limit]
    print(f"[INFO] pairs={len(items)}  clip={CLIP_SECONDS}s  method=v4 melodic features",
          flush=True)

    cache, rows, failed = {}, [], 0
    t0 = time.time()
    for n, it in enumerate(items, 1):
        try:
            fa = load_features(it["qry"], cache)
            fb = load_features(it["ref"], cache)
            feat = core_v4.features(fa, fb)
            lm, st = v3map[it["id"]]
        except Exception as e:
            failed += 1
            print(f"  [{n:>3}] #{it['id']} FAILED {type(e).__name__}: {str(e)[:70]}",
                  flush=True)
            continue
        rows.append({"id": it["id"], "truth": it["truth"], "truth_zh": it["truth_zh"],
                     "lm": lm, "st": st, **feat,
                     "qry": it["qry"].name, "ref": it["ref"].name})
        if n % 5 == 0 or n == len(items):
            print(f"  [{n:>3}/{len(items)}] elapsed={time.time()-t0:.1f}s  "
                  f"last: {it['truth']} shift_gap={feat['shift_gap']:.4f} "
                  f"seg_median={feat['seg_median']:.4f}", flush=True)

    if not rows:
        print("[ERROR] no pair evaluated")
        return

    cols = ["序号", "真值", "landmark", "sim_timbre"] + core_v4.FEATURE_NAMES + \
           ["查询音频", "参考音频"]
    DETAIL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with DETAIL_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["id"], r["truth_zh"], f"{r['lm']:.4f}", f"{r['st']:.4f}"] +
                       [f"{r[k]:.6f}" for k in core_v4.FEATURE_NAMES] +
                       [r["qry"], r["ref"]])

    stats = {}
    for c in ["copy", "cover", "unrelated"]:
        sub = [r for r in rows if r["truth"] == c]
        stats[c] = {"n": len(sub)}
        for k in core_v4.FEATURE_NAMES:
            v = [r[k] for r in sub]
            stats[c][k] = round(float(np.mean(v)), 5) if v else None
    STATS_JSON.write_text(json.dumps({
        "n_pairs": len(items), "n_valid": len(rows), "n_failed": failed,
        "elapsed_seconds": round(time.time() - t0, 1),
        "feature_names": core_v4.FEATURE_NAMES,
        "mean_by_class": stats,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 各特征按真值的均值（看有没有区分力） ===", flush=True)
    print(f"{'feature':<16}{'copy':>10}{'cover':>10}{'unrelated':>12}", flush=True)
    for k in core_v4.FEATURE_NAMES:
        print(f"{k:<16}{stats['copy'][k] or 0:>10.4f}{stats['cover'][k] or 0:>10.4f}"
              f"{stats['unrelated'][k] or 0:>12.4f}", flush=True)
    print(f"\n[OK] saved: {DETAIL_CSV}, {STATS_JSON}  ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
