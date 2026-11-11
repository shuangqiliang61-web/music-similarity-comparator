"""
evaluate_v5.py — 在 100 对标注数据上计算 core_v5 的 CENS / 节奏特征

用法（Anaconda Prompt：conda activate music）：
    python evaluate_v5.py --limit 8    # 试跑测速
    python evaluate_v5.py              # 全量 100 对

复用 samples/eval_v3_results.csv 里的 landmark / sim_timbre / sim_melody（不重算），
只新算 v5 的 5 个特征，并**立刻检查与旧特征的相关性**——
v4 就是因为没做这一步，白忙一轮。

产出：
    samples/eval_v5_results.csv   每对明细（含原始文件名，仅本地，勿外传）
    eval_v5_feature_stats.json    按类别均值 + 新旧特征相关矩阵（纯数字，可公开）
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

import core_v5
from evaluate_dataset import read_labels, LABEL_XLSX, SR, CLIP_SECONDS

V3_CSV = Path("samples/eval_v3_results.csv")
DETAIL_CSV = Path("samples/eval_v5_results.csv")
STATS_JSON = Path("eval_v5_feature_stats.json")
OLD = ["landmark", "sim_timbre", "sim_melody"]
ALL = OLD + core_v5.FEATURE_NAMES


def load_features(path: Path, cache: dict):
    key = str(path)
    if key not in cache:
        try:
            y, sr = librosa.load(str(path), sr=SR, mono=True, duration=CLIP_SECONDS)
        except Exception:
            y, sr = librosa.load(str(path), sr=SR, mono=True)
            if CLIP_SECONDS:
                y = y[: int(sr * CLIP_SECONDS)]
        cache[key] = core_v5.extract(y, sr)
    return cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    v3map = {}
    with V3_CSV.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            v3map[r["序号"].strip()] = (float(r["landmark"]),
                                       float(r["sim_timbre"]),
                                       float(r["sim_melody"]))

    items = read_labels(LABEL_XLSX)
    if args.limit:
        items = items[: args.limit]
    print(f"[INFO] pairs={len(items)}  clip={CLIP_SECONDS}s  method=v5 CENS + rhythm", flush=True)

    cache, rows, failed = {}, [], 0
    t0 = time.time()
    for n, it in enumerate(items, 1):
        try:
            fa = load_features(it["qry"], cache)
            fb = load_features(it["ref"], cache)
            feat = core_v5.features(fa, fb)
            lm, st, sm = v3map[it["id"]]
        except Exception as e:
            failed += 1
            print(f"  [{n:>3}] #{it['id']} FAILED {type(e).__name__}: {str(e)[:80]}",
                  flush=True)
            continue
        rows.append({"id": it["id"], "truth": it["truth"], "truth_zh": it["truth_zh"],
                     "landmark": lm, "sim_timbre": st, "sim_melody": sm, **feat,
                     "qry": it["qry"].name, "ref": it["ref"].name})
        if n % 5 == 0 or n == len(items):
            print(f"  [{n:>3}/{len(items)}] {time.time()-t0:.1f}s  last: {it['truth']} "
                  f"sim_cens={feat['sim_cens']:.4f} sim_rhythm={feat['sim_rhythm']:.4f}",
                  flush=True)

    if not rows:
        print("[ERROR] no pair evaluated")
        return

    DETAIL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with DETAIL_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["序号", "真值"] + ALL + ["查询音频", "参考音频"])
        for r in rows:
            w.writerow([r["id"], r["truth_zh"]] + [f"{r[k]:.6f}" for k in ALL] +
                       [r["qry"], r["ref"]])

    stats = {}
    for c in ["copy", "cover", "unrelated"]:
        sub = [r for r in rows if r["truth"] == c]
        stats[c] = {"n": len(sub)}
        for k in core_v5.FEATURE_NAMES:
            v = [r[k] for r in sub]
            stats[c][k] = round(float(np.mean(v)), 5) if v else None

    X = np.array([[r[k] for k in ALL] for r in rows])
    corr = np.corrcoef(X, rowvar=False)

    print("\n=== 新特征按真值的均值 ===", flush=True)
    print(f"{'feature':<18}{'copy':>10}{'cover':>10}{'unrelated':>12}", flush=True)
    for k in core_v5.FEATURE_NAMES:
        print(f"{k:<18}{stats['copy'][k] or 0:>10.4f}{stats['cover'][k] or 0:>10.4f}"
              f"{stats['unrelated'][k] or 0:>12.4f}", flush=True)
    print("\n=== 与旧特征的最大绝对相关（越接近 1 越冗余） ===", flush=True)
    redundancy = {}
    for i, k in enumerate(core_v5.FEATURE_NAMES, start=len(OLD)):
        vals = [abs(corr[i, j]) for j in range(len(OLD))]
        redundancy[k] = round(float(max(vals)), 4)
        print(f"   {k:<18} max|r| = {max(vals):.3f}  (vs {OLD[int(np.argmax(vals))]})",
              flush=True)

    STATS_JSON.write_text(json.dumps({
        "n_pairs": len(items), "n_valid": len(rows), "n_failed": failed,
        "elapsed_seconds": round(time.time() - t0, 1),
        "columns": ALL, "new_features": core_v5.FEATURE_NAMES,
        "mean_by_class": stats,
        "correlation_matrix": np.round(corr, 3).tolist(),
        "max_abs_corr_with_old_features": redundancy,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] saved: {DETAIL_CSV}, {STATS_JSON}  ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
