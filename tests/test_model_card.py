"""
Tests for Model Card Generator (TDD RED phase)
"""
# UTF-8 fix
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pytest
from pathlib import Path
import json

src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


class TestModelCard:
    """Test suite for Model Card generation"""

    @pytest.fixture
    def sample_model_info(self):
        """Create sample model information"""
        return {
            "model_name": "XGBoost_Exoplanet_Test",
            "model_version": "v1.0.0",
            "training_date": "2025-09-30",
            "metrics": {
                "pr_auc": 0.85,
                "roc_auc": 0.92,
                "brier_score": 0.12
            },
            "features": ["bls_period", "bls_depth", "snr"],
            "calibration_method": "isotonic",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1
            },
            "dataset_info": {
                "name": "TOI + KOI Combined",
                "n_samples": 1000,
                "n_positives": 100,
                "n_negatives": 900
            }
        }

    def test_model_card_creation(self, sample_model_info):
        """Test: Model card should be created with all required fields"""
        from utils.model_card import create_model_card

        card = create_model_card(**sample_model_info)

        assert "model_details" in card
        assert "intended_use" in card
        assert "training_data" in card
        assert "model_architecture" in card
        assert "performance" in card
        assert "ethical_considerations" in card
        assert "metadata" in card

    def test_model_card_contains_metrics(self, sample_model_info):
        """Test: Model card should contain performance metrics"""
        from utils.model_card import create_model_card

        card = create_model_card(**sample_model_info)

        assert card["performance"]["metrics"]["pr_auc"] == 0.85
        assert card["performance"]["metrics"]["roc_auc"] == 0.92
        assert card["performance"]["metrics"]["brier_score"] == 0.12

    def test_model_card_save_and_load(self, sample_model_info, tmp_path):
        """Test: Model card should be saveable and loadable"""
        from utils.model_card import create_model_card, save_model_card, load_model_card

        card = create_model_card(**sample_model_info)
        output_path = tmp_path / "model_card.json"

        save_model_card(card, output_path)

        assert output_path.exists()

        loaded_card = load_model_card(output_path)
        assert loaded_card["model_details"]["name"] == "XGBoost_Exoplanet_Test"
        assert loaded_card["model_details"]["version"] == "v1.0.0"


class TestProvenance:
    """Test suite for Provenance tracking"""

    @pytest.fixture
    def sample_provenance_info(self):
        """Create sample provenance information"""
        return {
            "run_id": "20250930_143022",
            "data_source": "MAST TAP",
            "mission": "TESS",
            "query_params": {"target": "TIC123456", "sector": 1},
            "model_info": {
                "version": "v1.0.0",
                "path": "/models/xgb_model.pkl",
                "calibration": "isotonic",
                "random_state": 42
            },
            "processing_steps": ["Download", "BLS", "Feature extraction", "Inference"],
            "output_files": ["outputs/candidates_20250930.csv"]
        }

    def test_provenance_creation(self, sample_provenance_info):
        """Test: Provenance record should be created"""
        from utils.provenance import create_provenance_record

        prov = create_provenance_record(**sample_provenance_info)

        assert "run_info" in prov
        assert "data_source" in prov
        assert "model" in prov
        assert "dependencies" in prov
        assert "processing" in prov
        assert "outputs" in prov

    def test_provenance_contains_versions(self, sample_provenance_info):
        """Test: Provenance should track package versions"""
        from utils.provenance import create_provenance_record

        prov = create_provenance_record(**sample_provenance_info)

        assert "lightkurve" in prov["dependencies"]
        assert "xgboost" in prov["dependencies"]
        assert "scikit-learn" in prov["dependencies"]

    def test_provenance_save_and_load(self, sample_provenance_info, tmp_path):
        """Test: Provenance should be saveable to YAML"""
        from utils.provenance import create_provenance_record, save_provenance, load_provenance

        prov = create_provenance_record(**sample_provenance_info)
        output_path = tmp_path / "provenance.yaml"

        save_provenance(prov, output_path)

        assert output_path.exists()

        loaded_prov = load_provenance(output_path)
        assert loaded_prov["run_info"]["run_id"] == "20250930_143022"
        assert loaded_prov["data_source"]["mission"] == "TESS"


if __name__ == "__main__":
    print("🧪 Running Model Card & Provenance Tests (TDD RED Phase)")
    print("="*60)
    pytest.main([__file__, '-v'])