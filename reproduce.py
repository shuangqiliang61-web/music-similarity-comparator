"""
reproduce.py — 一键复现项目里的全部实验（作品集可复现性）

用法（Anaconda Prompt：conda activate music）：
    python reproduce.py --list          # 只看会跑什么
    python reproduce.py --run           # 跑快速组（不需要重新解码音频，约 1-2 分钟）
    python reproduce.py --run --full    # 连音频重算一起跑（约 8-12 分钟）

说明：
    快速组基于 samples/ 下已有的每对分数明细（本地、不入库），
    完整组会重新解码音频并重算所有特征。
    所有步骤都通过 sys.executable 调用，保证用的是当前 conda 环境。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

QUICK = [
    ("单元测试", ["-m", "unittest", "discover", "-s", "tests", "-v"]),
    ("规范化模型选择 / 独立测试集", ["optimize_model.py"]),
    ("v4 特征集严格对比", ["optimize_v4.py"]),
    ("v4 冗余诊断", ["diagnose_v4.py"]),
    ("v5 新特征嵌套评估", ["optimize_v5.py"]),
    ("可解释规则族嵌套评估", ["optimize_rules.py"]),
    ("越折错误分析", ["error_analysis.py"]),
    ("训练并导出最终模型", ["train_model.py"]),
]

FULL = [
    ("v1 全量评测", ["evaluate_dataset.py"]),
    ("v2 特征序列评测", ["evaluate_v2.py"]),
    ("v3 landmark 评测", ["evaluate_v3.py"]),
    ("音频质量诊断", ["diagnose_audio_quality.py"]),
    ("v4 旋律特征重算", ["evaluate_v4.py"]),
    ("v5 CENS/节奏特征重算", ["evaluate_v5.py"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--full", action="store_true", help="包含需要重新解码音频的步骤")
    args = ap.parse_args()

    steps = (FULL + QUICK) if args.full else QUICK
    if args.list or not args.run:
        print(f"复现步骤（{'完整' if args.full else '快速'}组，共 {len(steps)} 步）：")
        for i, (name, cmd) in enumerate(steps, 1):
            print(f"  {i:>2}. {name:<28} python {' '.join(cmd)}")
        print("\n加 --run 执行；加 --run --full 连音频重算一起跑。")
        return

    root = Path(__file__).resolve().parent
    t0 = time.time()
    failed = []
    for i, (name, cmd) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {name} ...", flush=True)
        r = subprocess.run([sys.executable, "-u"] + cmd, cwd=str(root),
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        tail = (r.stdout or "").strip().splitlines()[-8:]
        for line in tail:
            print("   " + line, flush=True)
        if r.returncode != 0:
            failed.append((name, r.returncode, (r.stderr or "")[-300:]))
            print(f"   [FAILED] exit={r.returncode}", flush=True)
        else:
            print("   [OK]", flush=True)

    print(f"\n==== 完成，用时 {time.time()-t0:.1f}s ====", flush=True)
    if failed:
        for name, code, err in failed:
            print(f"  FAILED: {name} (exit {code})\n{err}", flush=True)
        sys.exit(1)
    print("  全部步骤通过。", flush=True)


if __name__ == "__main__":
    main()
