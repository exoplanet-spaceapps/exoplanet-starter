"""
Unit tests for injection module
"""
import unittest
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.injection import (
    inject_box_transit,
    generate_synthetic_dataset,
    save_synthetic_dataset,
    load_synthetic_dataset,
    generate_transit_parameters
)


class TestInjection(unittest.TestCase):

    def setUp(self):
        """設置測試資料"""
        # 創建模擬時間序列
        self.time = np.linspace(0, 30, 1000)  # 30天，1000個資料點
        self.flux = np.ones(1000) + np.random.normal(0, 0.0001, 1000)  # 基礎流量

    def test_inject_box_transit_basic(self):
        """測試基本的箱形凌日注入"""
        period = 3.0
        depth = 0.01  # 1% 深度
        duration = 0.1  # 0.1 天
        t0 = 1.5

        injected_flux = inject_box_transit(
            self.time, self.flux, period, depth, duration, t0
        )

        # 檢查注入後的流量有下降
        self.assertTrue(np.min(injected_flux) < np.min(self.flux))

        # 檢查深度大約正確
        expected_min = np.mean(self.flux) * (1 - depth)
        actual_min = np.min(injected_flux)
        self.assertAlmostEqual(actual_min, expected_min, places=2)

    def test_inject_box_transit_multiple_transits(self):
        """測試多次凌日"""
        period = 2.0
        depth = 0.005
        duration = 0.05
        t0 = 0.5

        injected_flux = inject_box_transit(
            self.time, self.flux, period, depth, duration, t0
        )

        # 計算預期的凌日次數
        n_expected_transits = int((self.time[-1] - t0) / period) + 1

        # 計算實際的凌日次數（通過計數低於閾值的連續段）
        in_transit = injected_flux < (np.mean(self.flux) * (1 - depth/2))
        transit_starts = np.where(np.diff(np.concatenate(([False], in_transit, [False]))))[0]
        n_actual_transits = len(transit_starts) // 2

        # 允許有1個凌日的誤差（邊界效應）
        self.assertAlmostEqual(n_actual_transits, n_expected_transits, delta=1)

    def test_generate_synthetic_dataset(self):
        """測試合成資料集生成"""
        n_positive = 10
        n_negative = 10

        samples_df, labels_df = generate_synthetic_dataset(
            self.time,
            self.flux,
            n_positive=n_positive,
            n_negative=n_negative,
            seed=42
        )

        # 檢查樣本數量
        self.assertEqual(len(samples_df), n_positive + n_negative)
        self.assertEqual(len(labels_df), n_positive + n_negative)

        # 檢查標籤分布
        self.assertEqual(sum(labels_df['label'] == 1), n_positive)
        self.assertEqual(sum(labels_df['label'] == 0), n_negative)

        # 檢查正樣本有參數
        positive_labels = labels_df[labels_df['label'] == 1]
        self.assertTrue(all(positive_labels['period'].notna()))
        self.assertTrue(all(positive_labels['depth'].notna()))

        # 檢查負樣本沒有參數
        negative_labels = labels_df[labels_df['label'] == 0]
        self.assertTrue(all(negative_labels['period'].isna()))
        self.assertTrue(all(negative_labels['depth'].isna()))

    def test_save_and_load_dataset(self):
        """測試資料集儲存和載入"""
        samples_df, labels_df = generate_synthetic_dataset(
            self.time,
            self.flux,
            n_positive=5,
            n_negative=5,
            seed=42
        )

        # 儲存資料集
        paths = save_synthetic_dataset(
            samples_df,
            labels_df,
            output_dir="tests/temp_data",
            format="parquet"
        )

        # 檢查檔案是否存在
        self.assertTrue(os.path.exists(paths['samples']))
        self.assertTrue(os.path.exists(paths['labels']))
        self.assertTrue(os.path.exists(paths['metadata']))

        # 載入資料集
        loaded_samples, loaded_labels = load_synthetic_dataset("tests/temp_data")

        # 檢查載入的資料與原始資料一致
        self.assertEqual(len(loaded_samples), len(samples_df))
        self.assertEqual(len(loaded_labels), len(labels_df))
        self.assertTrue(all(loaded_labels['label'] == labels_df['label']))

        # 清理測試檔案
        import shutil
        if os.path.exists("tests/temp_data"):
            shutil.rmtree("tests/temp_data")

    def test_generate_transit_parameters(self):
        """測試凌日參數生成"""
        n_samples = 100
        params_df = generate_transit_parameters(n_samples, seed=42)

        # 檢查樣本數量
        self.assertEqual(len(params_df), n_samples)

        # 檢查參數範圍
        self.assertTrue(all(params_df['period_days'] >= 0.6))
        self.assertTrue(all(params_df['period_days'] <= 10.0))
        self.assertTrue(all(params_df['depth_relative'] >= 0.0005))
        self.assertTrue(all(params_df['depth_relative'] <= 0.02))

        # 檢查衍生參數
        self.assertTrue(all(params_df['depth_ppm'] > 0))
        self.assertTrue(all(params_df['duration_hours'] > 0))

    def test_inject_no_transit_when_duration_zero(self):
        """測試當持續時間為零時不注入凌日"""
        period = 3.0
        depth = 0.01
        duration = 0.0  # 零持續時間
        t0 = 1.5

        injected_flux = inject_box_transit(
            self.time, self.flux, period, depth, duration, t0
        )

        # 流量應該保持不變
        np.testing.assert_array_almost_equal(injected_flux, self.flux)

    def test_inject_deep_transit(self):
        """測試深度凌日注入"""
        period = 5.0
        depth = 0.1  # 10% 深度（很深）
        duration = 0.2
        t0 = 2.0

        injected_flux = inject_box_transit(
            self.time, self.flux, period, depth, duration, t0
        )

        # 檢查最小值
        expected_min = np.mean(self.flux) * (1 - depth)
        actual_min = np.min(injected_flux)
        self.assertAlmostEqual(actual_min, expected_min, places=2)


if __name__ == '__main__':
    unittest.main()