"""
diagnose_audio_quality.py — 检查评测集的音频本身有没有系统性问题

动机：
    所有改进都集中在「特征 / 模型」，但如果部分文件前 60 秒是大段静音、
    片头、或极低音量，任何特征都提取不到有效信息。这类问题属于输入质量，
    修它带来的是真实提升，而不是在 100 条数据上背答案。

检查项（全部只读，不改动音频）：
    总时长、前 60 秒的每秒 RMS、静音秒数、首个有效秒、峰值/削波、
    并按「真值类别」和「是否被模型判错」分组对比。

输出：
    samples/audio_quality_detail.csv   明细（含原始文件名，仅本地，勿外传）
    audio_quality_report.json          聚合统计（纯数字，可公开）

用法：conda activate music && python diagnose_audio_quality.py
"""

from __future__ import annotations

import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import librosa

warnings.filterwarnings("ignore")

from evaluate_dataset import read_labels, LABEL_XLSX, SR, CLIP_SECONDS

DETAIL_CSV = Path("samples/audio_quality_detail.csv")
REPORT_JSON = Path("audio_quality_report.json")
V4_CSV = Path("samples/eval_v4_results.csv")
SILENCE_REL = 0.02        # 低于本曲最大秒 RMS 的 2% 视为静音
SILENCE_ABS = 1e-4        # 且绝对值低于此


def main():
    items = read_labels(LABEL_XLSX)
    paths = {}
    for it in items:
        for tag in ("qry", "ref"):
            p = it[tag]
            paths[str(p)] = p
    print(f"[INFO] pairs={len(items)}  unique files={len(paths)}", flush=True)

    pred_by_id = {}
    if V4_CSV.exists():
        with V4_CSV.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                pred_by_id[r["序号"].strip()] = r["真值"].strip()

    info, failed = {}, []
    for n, (key, p) in enumerate(sorted(paths.items()), 1):
        try:
            # 不用 sf.info：本机 libsndfile 对 mp4/m4a 的元数据读取会抛
            # LibsndfileError，但 librosa.load 走另一条解码路径是正常的。
            y, sr = librosa.load(str(p), sr=SR, mono=True, duration=CLIP_SECONDS)
            n_sec = int(len(y) // sr)
            total_sec = float(n_sec)   # 即实际取到的秒数（上限 60）
            rms = np.array([float(np.sqrt(np.mean(y[i*sr:(i+1)*sr] ** 2)))
                            for i in range(n_sec)]) if n_sec else np.array([0.0])
            peak = float(np.max(np.abs(y))) if len(y) else 0.0
            mx = float(rms.max()) if len(rms) else 0.0
            silent = rms < max(SILENCE_ABS, mx * SILENCE_REL)
            first = int(np.argmax(~silent)) if (~silent).any() else n_sec
            info[key] = {
                "total_sec": round(total_sec, 2),
                "analyzed_sec": n_sec,
                "silence_sec": int(silent.sum()),
                "silence_ratio": round(float(silent.mean()), 4),
                "first_active_sec": first,
                "peak": round(peak, 5),
                "clipped": peak >= 0.999,
                "rms_mean": round(float(rms.mean()), 6),
            }
        except Exception as e:
            failed.append((str(p.name), type(e).__name__, str(e)[:70]))
        if n % 40 == 0:
            print(f"  [{n}/{len(paths)}] ok={len(info)} failed={len(failed)}", flush=True)
    if failed:
        from collections import Counter
        print("  failure types:", dict(Counter(f[1] for f in failed)), flush=True)
        print("  first failures:", failed[:3], flush=True)

    DETAIL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with DETAIL_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["文件", "总时长秒", "分析秒数", "静音秒数", "静音比例",
                    "首个有效秒", "峰值", "削波", "平均RMS"])
        for k, v in info.items():
            w.writerow([Path(k).name, v["total_sec"], v["analyzed_sec"],
                        v["silence_sec"], v["silence_ratio"], v["first_active_sec"],
                        v["peak"], v["clipped"], v["rms_mean"]])

    def summarize(vals, label):
        if not vals:
            return {}
        arr = {k: np.array([v[k] for v in vals], float) for k in
               ["total_sec", "silence_sec", "silence_ratio", "first_active_sec", "rms_mean"]}
        return {
            "n": len(vals),
            "short_under_60s": int((arr["total_sec"] < 60).sum()),
            "silence_over_10s": int((arr["silence_sec"] > 10).sum()),
            "silence_over_30s": int((arr["silence_sec"] > 30).sum()),
            "leading_silence_over_5s": int((arr["first_active_sec"] > 5).sum()),
            "very_quiet": int((arr["rms_mean"] < 0.005).sum()),
            "mean_silence_ratio": round(float(arr["silence_ratio"].mean()), 4),
            "mean_first_active_sec": round(float(arr["first_active_sec"].mean()), 3),
            "clipped_count": sum(1 for v in vals if v["clipped"]),
            "label": label,
        }

    by_class = defaultdict(list)
    for it in items:
        for tag in ("qry", "ref"):
            v = info.get(str(it[tag]))
            if v:
                by_class[it["truth"]].append(v)

    overall = summarize(list(info.values()), "all")
    print("\n=== 全部音频质量 ===", flush=True)
    for k, v in overall.items():
        print(f"   {k:<28} {v}", flush=True)
    print("\n=== 按真值类别 ===", flush=True)
    for c in ["copy", "cover", "unrelated"]:
        s = summarize(by_class[c], c)
        print(f"   {c:<10} n={s.get('n')} 静音均值={s.get('mean_silence_ratio')} "
              f"起始静音>5s={s.get('leading_silence_over_5s')} "
              f"静音>30s={s.get('silence_over_30s')} 极安静={s.get('very_quiet')}",
              flush=True)

    # 判错的样本是否更可能有质量问题：按 pair 取 qry/ref 静音比例最大值
    wrong, right = [], []
    for it in items:
        a, b = info.get(str(it["qry"])), info.get(str(it["ref"]))
        if not (a and b):
            continue
        merged = {"total_sec": min(a["total_sec"], b["total_sec"]),
                  "silence_sec": max(a["silence_sec"], b["silence_sec"]),
                  "silence_ratio": max(a["silence_ratio"], b["silence_ratio"]),
                  "first_active_sec": max(a["first_active_sec"], b["first_active_sec"]),
                  "rms_mean": min(a["rms_mean"], b["rms_mean"]),
                  "clipped": a["clipped"] or b["clipped"]}
        # v4 明细里「真值」列就是标签；预测对错用 v3 的 tuned 判定另算不了，
        # 这里按「该 pair 是否属于难例备注」分组，备注来自标注表
        (wrong if it.get("note") else right).append(merged)
    note_summary = {
        "hard_cases_with_note": summarize(wrong, "with note"),
        "others": summarize(right, "without note"),
    }
    print("\n=== 难例备注 vs 其他 ===", flush=True)
    for k, s in note_summary.items():
        print(f"   {k:<24} n={s.get('n')} 静音均值={s.get('mean_silence_ratio')} "
              f"起始静音>5s={s.get('leading_silence_over_5s')}", flush=True)

    REPORT_JSON.write_text(json.dumps({
        "n_pairs": len(items), "n_unique_files": len(paths),
        "n_failed_decode": len(failed), "failed_examples": failed[:5],
        "clip_seconds": CLIP_SECONDS,
        "thresholds": {"silence_rel": SILENCE_REL, "silence_abs": SILENCE_ABS},
        "overall": overall,
        "by_class": {c: summarize(by_class[c], c)
                     for c in ["copy", "cover", "unrelated"]},
        "note_comparison": note_summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] saved: {DETAIL_CSV}, {REPORT_JSON}", flush=True)


if __name__ == "__main__":
    main()
