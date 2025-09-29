"""
BLS 特徵萃取模組單元測試
"""
import unittest
import numpy as np
import sys
from pathlib import Path

# 添加 app 目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from bls_features import (
    run_bls,
    extract_features,
    compute_odd_even_difference,
    compute_transit_symmetry,
    compute_periodicity_strength,
    compute_skewness,
    compute_kurtosis
)


class TestBLSFeatures(unittest.TestCase):
    """BLS 特徵萃取測試"""

    def setUp(self):
        """設定測試資料"""
        np.random.seed(42)
        self.n_points = 1000

        # 產生乾淨的凌日信號
        self.time = np.linspace(0, 30, self.n_points)
        self.period = 3.5
        self.depth = 0.01
        self.duration = 0.2

        # 建立凌日模型
        self.flux = np.ones(self.n_points)
        phase = (self.time % self.period) / self.period
        transit_mask = np.abs(phase - 0.5) < (self.duration / self.period / 2)
        self.flux[transit_mask] *= (1 - self.depth)

        # 加入雜訊
        self.flux += 0.001 * np.random.randn(self.n_points)

    def test_run_bls(self):
        """測試 BLS 搜尋"""
        result = run_bls(self.time, self.flux, min_period=1.0, max_period=10.0)

        self.assertIsInstance(result, dict)
        self.assertIn('period', result)
        self.assertIn('power', result)
        self.assertIn('depth', result)
        self.assertIn('duration', result)
        self.assertIn('t0', result)
        self.assertIn('snr', result)

        # 檢查找到的週期是否接近真實值
        self.assertAlmostEqual(result['period'], self.period, delta=0.5)
        self.assertAlmostEqual(result['depth'], self.depth, delta=0.005)

    def test_extract_features_basic(self):
        """測試基本特徵萃取"""
        bls_result = {
            'period': self.period,
            'power': 100,
            'depth': self.depth,
            'duration': self.duration,
            't0': 0,
            'snr': 10
        }

        features = extract_features(
            self.time, self.flux, bls_result,
            compute_advanced=False
        )

        # 檢查基本特徵
        expected_features = [
            'bls_period', 'bls_power', 'bls_depth',
            'bls_duration', 'bls_t0', 'bls_snr',
            'duration_over_period', 'flux_std', 'flux_mad'
        ]

        for feat in expected_features:
            self.assertIn(feat, features)
            self.assertIsNotNone(features[feat])

    def test_extract_features_advanced(self):
        """測試進階特徵萃取"""
        bls_result = {
            'period': self.period,
            'power': 100,
            'depth': self.depth,
            'duration': self.duration,
            't0': 0,
            'snr': 10
        }

        features = extract_features(
            self.time, self.flux, bls_result,
            compute_advanced=True
        )

        # 檢查進階特徵
        advanced_features = [
            'odd_even_depth_diff',
            'transit_symmetry',
            'periodicity_strength',
            'flux_skewness',
            'flux_kurtosis'
        ]

        for feat in advanced_features:
            self.assertIn(feat, features)
            self.assertIsNotNone(features[feat])

    def test_compute_odd_even_difference(self):
        """測試奇偶深度差異計算"""
        # 建立奇偶不同的凌日
        flux_odd_even = np.ones(self.n_points)
        phase = (self.time % self.period) / self.period

        # 奇數凌日較深
        for i in range(0, int(30 / self.period), 2):
            mask = (self.time >= i * self.period) & \
                   (self.time < i * self.period + self.duration)
            flux_odd_even[mask] *= 0.99

        # 偶數凌日較淺
        for i in range(1, int(30 / self.period), 2):
            mask = (self.time >= i * self.period) & \
                   (self.time < i * self.period + self.duration)
            flux_odd_even[mask] *= 0.995

        diff = compute_odd_even_difference(
            self.time, flux_odd_even, self.period, self.duration
        )

        # 應該檢測到差異
        self.assertGreater(np.abs(diff), 0.003)

    def test_compute_transit_symmetry(self):
        """測試凌日對稱性計算"""
        # 建立不對稱凌日
        flux_asym = np.ones(self.n_points)
        phase = (self.time % self.period) / self.period
        transit_mask = np.abs(phase - 0.5) < (self.duration / self.period / 2)

        # 入凌陡峭，出凌平緩
        for i, in_transit in enumerate(transit_mask):
            if in_transit:
                phase_in_transit = (phase[i] - 0.5) / (self.duration / self.period / 2)
                if phase_in_transit < 0:  # 入凌
                    flux_asym[i] *= (1 - self.depth * (1 + phase_in_transit))
                else:  # 出凌
                    flux_asym[i] *= (1 - self.depth * (1 - phase_in_transit * 0.5))

        symmetry = compute_transit_symmetry(
            self.time, flux_asym, self.period, 0, self.duration
        )

        # 不對稱性應該被檢測到
        self.assertLess(symmetry, 0.95)

    def test_compute_periodicity_strength(self):
        """測試週期性強度計算"""
        # 強週期信號
        strength_strong = compute_periodicity_strength(
            self.time, self.flux, self.period, harmonics=2
        )

        # 隨機雜訊
        noise = np.random.randn(self.n_points) * 0.01 + 1.0
        strength_noise = compute_periodicity_strength(
            self.time, noise, self.period, harmonics=2
        )

        # 週期信號應該更強
        self.assertGreater(strength_strong, strength_noise * 2)

    def test_skewness_kurtosis(self):
        """測試偏度與峰度計算"""
        # 正常分布
        normal = np.random.randn(1000)
        skew_normal = compute_skewness(normal)
        kurt_normal = compute_kurtosis(normal)

        self.assertAlmostEqual(skew_normal, 0, delta=0.5)
        self.assertAlmostEqual(kurt_normal, 0, delta=0.5)

        # 偏斜分布
        skewed = np.concatenate([
            np.random.randn(800),
            np.random.randn(200) + 3
        ])
        skew_skewed = compute_skewness(skewed)
        self.assertGreater(np.abs(skew_skewed), 0.5)

    def test_feature_consistency(self):
        """測試特徵計算一致性"""
        bls_result = run_bls(self.time, self.flux)

        # 多次計算應該得到相同結果
        features1 = extract_features(self.time, self.flux, bls_result)
        features2 = extract_features(self.time, self.flux, bls_result)

        for key in features1:
            if isinstance(features1[key], (int, float)):
                self.assertAlmostEqual(
                    features1[key], features2[key],
                    msg=f"Feature {key} not consistent"
                )

    def test_edge_cases(self):
        """測試邊界情況"""
        # 空資料
        with self.assertRaises(Exception):
            run_bls(np.array([]), np.array([]))

        # 單點資料
        with self.assertRaises(Exception):
            run_bls(np.array([1]), np.array([1]))

        # 常數通量
        constant_flux = np.ones(100)
        result = run_bls(np.arange(100), constant_flux)
        self.assertLess(result['snr'], 3)  # 應該沒有顯著信號

    def test_nan_handling(self):
        """測試 NaN 值處理"""
        # 加入 NaN 值
        flux_with_nan = self.flux.copy()
        flux_with_nan[::10] = np.nan

        # 應該能處理 NaN
        clean_time = self.time[~np.isnan(flux_with_nan)]
        clean_flux = flux_with_nan[~np.isnan(flux_with_nan)]

        result = run_bls(clean_time, clean_flux)
        self.assertIsNotNone(result)

        features = extract_features(clean_time, clean_flux, result)
        self.assertIsNotNone(features)


class TestBLSPerformance(unittest.TestCase):
    """BLS 效能測試"""

    def test_large_dataset(self):
        """測試大資料集處理"""
        # 產生大資料集
        np.random.seed(42)
        n_points = 10000
        time = np.linspace(0, 100, n_points)
        flux = np.random.randn(n_points) * 0.001 + 1.0

        # 應該能在合理時間內完成
        import time as timer
        start = timer.time()
        result = run_bls(time, flux, min_period=0.5, max_period=10.0)
        elapsed = timer.time() - start

        self.assertLess(elapsed, 30)  # 應該在 30 秒內完成
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()