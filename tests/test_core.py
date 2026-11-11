"""
test_core.py — 核心算法不变量的自动化测试（不需要真实音频文件）

运行（Anaconda Prompt）：
    conda activate music
    python -m unittest discover -s tests -v

这些测试锁住的是「数学上必须成立」的性质，防止以后改特征或重构时
悄悄破坏基础行为。它们不测准确率（那是评测脚本的职责）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core
import core_v2
import core_v3
import core_v4
import features_runtime


def tone(freq=440.0, sec=1.0, sr=22050):
    t = np.linspace(0, sec, int(sr * sec), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestV1Fingerprint(unittest.TestCase):
    def test_cosine_self_is_one(self):
        v = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(core.cosine_sim(v, v), 1.0, places=6)

    def test_cosine_opposite_is_minus_one(self):
        v = np.array([1.0, 0.0])
        self.assertAlmostEqual(core.cosine_sim(v, -v), -1.0, places=6)

    def test_fingerprint_is_24_dimensional(self):
        fp = core.fingerprint(tone(), 22050)
        self.assertEqual(fp.shape, (24,))

    def test_fingerprint_blocks_are_each_unit_norm(self):
        """两块分别 L2 归一化 —— 这正是修 MFCC[0] bug 时加的约束。"""
        fp = core.fingerprint(tone(), 22050)
        self.assertAlmostEqual(float(np.linalg.norm(fp[:12])), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(fp[12:])), 1.0, places=5)

    def test_classify_thresholds(self):
        self.assertEqual(core.classify(core.TH_COPY), "copy")
        self.assertEqual(core.classify(core.TH_COVER), "cover")
        self.assertEqual(core.classify(0.0), "unrelated")


class TestV2SequenceFeatures(unittest.TestCase):
    def test_pool_shape_and_row_norm(self):
        mat = np.random.default_rng(0).random((12, core_v2.FRAMES_PER_BIN * 7))
        out = core_v2._pool_to_bins(mat)
        self.assertEqual(out.shape, (7, 12))
        np.testing.assert_allclose(np.linalg.norm(out, axis=1),
                                   np.ones(7), atol=1e-5)

    def test_dtw_identical_sequence_has_zero_cost(self):
        rng = np.random.default_rng(1)
        X = rng.random((10, 12))
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        self.assertLess(core_v2.subseq_dtw_cost(X, X), 1e-6)

    def test_similarity_self_is_high(self):
        f = core_v2.extract(tone(), 22050)
        st, sm = core_v2.similarity(f, f)
        self.assertGreater(st, 0.99)
        self.assertGreater(sm, 0.99)


class TestV3Landmark(unittest.TestCase):
    def test_self_match_is_one(self):
        p = core_v3.peaks(tone(), 22050)
        self.assertAlmostEqual(core_v3.landmark_score(p, p), 1.0, places=6)

    def test_empty_peaks_returns_zero(self):
        self.assertEqual(core_v3.landmark_score(({}, 0), ({}, 0)), 0.0)

    def test_unrelated_signals_score_low(self):
        a = core_v3.peaks(tone(220.0), 22050)
        b = core_v3.peaks(tone(880.0, sec=1.3), 22050)
        self.assertLess(core_v3.landmark_score(a, b), 0.2)


class TestRuntimeConsistency(unittest.TestCase):
    def test_feature_order_matches_core_modules(self):
        self.assertEqual(features_runtime.FEATURE_ORDER,
                         ["landmark", "sim_timbre", "sim_melody"])

    def test_sr_agrees_with_feature_modules(self):
        self.assertEqual(features_runtime.SR, core_v2.SR)
        self.assertEqual(features_runtime.SR, core_v3.SR)

    def test_v4_feature_names_are_stable(self):
        self.assertEqual(len(core_v4.FEATURE_NAMES), 8)
        self.assertIn("shift_gap", core_v4.FEATURE_NAMES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
