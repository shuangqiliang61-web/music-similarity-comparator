"""
evaluate_v6.py — 在 100 对标注数据上计算 core_v6 的 HPCP / 离散音级特征

复用 v3/v4/v5 已算好的分数（不重算音频特征），只新算 v6 的 7 维，
并把所有特征合并成一张表 samples/eval_v6_results.csv，
供 optimize_hierarchical.py 做统一的嵌套评估。

用法（Anaconda Prompt：conda activate music）：
    python evaluate_v6.py --limit 6    # 试跑测速
    python evaluate_v6.py              # 全量 100 对

产出：
    samples/eval_v6_results.csv  全特征明细（含原始文件名，仅本地，勿外传）
    eval_v6_feature_stats.json   按类别均值 + 新旧特征相关性（纯数字，可公开）
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

import core_v6
from evaluate_dataset import read_labels, LABEL_XLSX, SR, CLIP_SECONDS

V3_CSV = Path("samples/eval_v3_results.csv")
V4_CSV = Path("samples/eval_v4_results.csv")
V5_CSV = Path("samples/eval_v5_results.csv")
DETAIL_CSV = Path("samples/eval_v6_results.csv")
STATS_JSON = Path("eval_v6_feature_stats.json")

OLD = ["landmark", "sim_timbre", "sim_melody",
       "sim_cens", "cens_gap", "sim_rhythm", "tempo_ratio", "rhythm_acf_sim",
       "shift_is_zero", "shift_gap"]
ALL = OLD + core_v6.FEATURE_NAMES


def _read(p: Path):
    with p.open(encoding="utf-8-sig") as f:
        return {r["序号"].strip(): r for r in csv.DictReader(f)}


def load_features(path: Path, cache: dict, flags: dict):
    key = str(path)
    if key not in cache:
        try:
            y, sr = librosa.load(str(path), sr=SR, mono=True, duration=CLIP_SECONDS)
        except Exception:
            y, sr = librosa.load(str(path), sr=SR, mono=True)
            if CLIP_SECONDS:
                y = y[: int(sr * CLIP_SECONDS)]
        f = core_v6.extract(y, sr)
        flags["n_harmonic_ok"] += int(f["harmonic_ok"])
        cache[key] = f
    return cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    r3, r4, r5 = _read(V3_CSV), _read(V4_CSV), _read(V5_CSV)
    items = read_labels(LABEL_XLSX)
    if args.limit:
        items = items[: args.limit]
    print(f"[INFO] pairs={len(items)} clip={CLIP_SECONDS}s method=v6 harmonic+CQT+pitch",
          flush=True)

    cache, flags, rows, failed = {}, {"n_harmonic_ok": 0}, [], 0
    t0 = time.time()
    for n, it in enumerate(items, 1):
        try:
            fa = load_features(it["qry"], cache, flags)
            fb = load_features(it["ref"], cache, flags)
            feat = core_v6.features(fa, fb)
            old = {}
            for c in OLD:
                src = r3 if c in r3[it["id"]] else (r4 if c in r4[it["id"]] else r5)
                old[c] = float(src[it["id"]][c])
        except Exception as e:
            failed += 1
            print(f"  [{n:>3}] #{it['id']} FAILED {type(e).__name__}: {str(e)[:90]}",
                  flush=True)
            continue
        rows.append({"id": it["id"], "truth": it["truth"], "truth_zh": it["truth_zh"],
                     **old, **feat,
                     "qry": it["qry"].name, "ref": it["ref"].name})
        if n % 5 == 0 or n == len(items):
            print(f"  [{n:>3}/{len(items)}] {time.time()-t0:.1f}s  last={it['truth']} "
                  f"hpcp={feat['hpcp_sim']:.4f} pitch={feat['pitch_match']:.4f}",
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
        for k in core_v6.FEATURE_NAMES:
            v = [r[k] for r in sub]
            stats[c][k] = round(float(np.mean(v)), 5) if v else None

    X = np.array([[r[k] for k in ALL] for r in rows])
    corr = np.corrcoef(X, rowvar=False)
    redundancy = {}
    for i, k in enumerate(core_v6.FEATURE_NAMES, start=len(OLD)):
        vals = [abs(corr[i, j]) for j in range(len(OLD))]
        redundancy[k] = {
            "max_abs_r": round(float(max(vals)), 4),
            "vs": OLD[int(np.argmax(vals))],
        }

    print("\n=== v6 新特征按真值的均值 ===", flush=True)
    print(f"{'feature':<20}{'copy':>10}{'cover':>10}{'unrelated':>12}", flush=True)
    for k in core_v6.FEATURE_NAMES:
        print(f"{k:<20}{stats['copy'][k] or 0:>10.4f}{stats['cover'][k] or 0:>10.4f}"
              f"{stats['unrelated'][k] or 0:>12.4f}", flush=True)
    print("\n=== 与所有旧特征的最大绝对相关（>0.9 视为冗余） ===", flush=True)
    for k, v in redundancy.items():
        print(f"   {k:<20} max|r|={v['max_abs_r']:.3f}  (vs {v['vs']})", flush=True)

    STATS_JSON.write_text(json.dumps({
        "n_pairs": len(items), "n_valid": len(rows), "n_failed": failed,
        "n_unique_files": len(cache), "n_harmonic_separation_ok": flags["n_harmonic_ok"],
        "elapsed_seconds": round(time.time() - t0, 1),
        "columns": ALL, "new_features": core_v6.FEATURE_NAMES,
        "mean_by_class": stats,
        "correlation_matrix": np.round(corr, 3).tolist(),
        "max_abs_corr_with_old_features": redundancy,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] saved: {DETAIL_CSV}, {STATS_JSON}  ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
