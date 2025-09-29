"""
Tests for Standardized Output Schema (TDD RED phase)
"""
# UTF-8 fix
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


class TestOutputSchema:
    """Test suite for standardized output schema"""

    @pytest.fixture
    def sample_candidates(self):
        """Create sample candidate data"""
        target_ids = [f"TIC{i}" for i in range(100, 105)]
        missions = ["TESS"] * 5
        bls_results = {
            "TIC100": {"period": 3.5, "duration": 2.1, "depth": 1200, "snr": 8.5, "power": 15.2, "sector": "S01"},
            "TIC101": {"period": 7.2, "duration": 3.5, "depth": 850, "snr": 12.3, "power": 22.1, "sector": "S01"},
            "TIC102": {"period": 1.8, "duration": 1.2, "depth": 2100, "snr": 6.8, "power": 11.5, "sector": "S02"},
            "TIC103": {"period": 12.5, "duration": 5.1, "depth": 450, "snr": 15.2, "power": 28.3, "sector": "S02"},
            "TIC104": {"period": 5.6, "duration": 2.8, "depth": 1500, "snr": 9.1, "power": 18.7, "sector": "S03"}
        }
        model_predictions = {
            "TIC100": 0.85,
            "TIC101": 0.92,
            "TIC102": 0.67,
            "TIC103": 0.78,
            "TIC104": 0.88
        }

        return target_ids, missions, bls_results, model_predictions

    def test_candidate_dataframe_creation(self, sample_candidates):
        """Test: Should create DataFrame with standardized schema"""
        from utils.output_schema import create_candidate_dataframe

        target_ids, missions, bls_results, model_predictions = sample_candidates

        df = create_candidate_dataframe(
            target_ids=target_ids,
            missions=missions,
            bls_results=bls_results,
            model_predictions=model_predictions
        )

        # Check required columns exist
        required_cols = [
            "target_id", "mission", "sector_or_quarter",
            "bls_period_d", "bls_duration_hr", "bls_depth_ppm",
            "snr", "power", "model_score", "score_uncalibrated",
            "is_eb_flag", "toi_crossmatch", "quality_flags",
            "run_id", "model_version", "data_source_url"
        ]

        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_dataframe_sorted_by_score(self, sample_candidates):
        """Test: DataFrame should be sorted by model_score descending"""
        from utils.output_schema import create_candidate_dataframe

        target_ids, missions, bls_results, model_predictions = sample_candidates

        df = create_candidate_dataframe(
            target_ids=target_ids,
            missions=missions,
            bls_results=bls_results,
            model_predictions=model_predictions
        )

        # Check sorting
        assert df["model_score"].is_monotonic_decreasing

    def test_csv_export(self, sample_candidates, tmp_path):
        """Test: Should export to CSV with timestamp"""
        from utils.output_schema import create_candidate_dataframe, export_candidates_csv

        target_ids, missions, bls_results, model_predictions = sample_candidates

        df = create_candidate_dataframe(
            target_ids=target_ids,
            missions=missions,
            bls_results=bls_results,
            model_predictions=model_predictions
        )

        output_path = export_candidates_csv(df, output_dir=tmp_path, timestamp="20250930")

        assert output_path.exists()
        assert output_path.name == "candidates_20250930.csv"

        # Verify CSV is readable
        df_loaded = pd.read_csv(output_path)
        assert len(df_loaded) == len(df)

    def test_jsonl_export(self, sample_candidates, tmp_path):
        """Test: Should export to JSONL format"""
        from utils.output_schema import create_candidate_dataframe, export_candidates_jsonl

        target_ids, missions, bls_results, model_predictions = sample_candidates

        df = create_candidate_dataframe(
            target_ids=target_ids,
            missions=missions,
            bls_results=bls_results,
            model_predictions=model_predictions
        )

        output_path = export_candidates_jsonl(df, output_dir=tmp_path, timestamp="20250930")

        assert output_path.exists()
        assert output_path.name == "candidates_20250930.jsonl"

        # Verify JSONL is readable
        df_loaded = pd.read_json(output_path, lines=True)
        assert len(df_loaded) == len(df)


if __name__ == "__main__":
    print("🧪 Running Output Schema Tests (TDD RED Phase)")
    print("="*60)
    pytest.main([__file__, '-v'])