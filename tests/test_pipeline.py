"""
Tests for Sklearn Pipeline (Phase 3) - TDD RED phase
Write tests FIRST before implementation
"""
# UTF-8 fix
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


class TestExoplanetPipeline:
    """Test suite for exoplanet detection pipeline"""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_samples = 100

        data = {
            'bls_period': np.random.uniform(0.5, 20, n_samples),
            'bls_depth_ppm': np.random.uniform(100, 5000, n_samples),
            'bls_snr': np.random.uniform(5, 50, n_samples),
            'tls_period': np.random.uniform(0.5, 20, n_samples),
            'tls_depth_ppm': np.random.uniform(100, 5000, n_samples),
            'tls_sde': np.random.uniform(5, 50, n_samples),
            'label': np.random.randint(0, 2, n_samples),
            'target_id': [f'TIC{i}' for i in range(n_samples)]
        }

        # Add some NaN values
        data['bls_depth_ppm'][::10] = np.nan

        return pd.DataFrame(data)

    def test_pipeline_exists(self):
        """Test: Pipeline module should exist"""
        try:
            from models import pipeline
            assert True
        except ImportError:
            pytest.fail("models.pipeline module not found")

    def test_create_pipeline_function_exists(self):
        """Test: create_exoplanet_pipeline function should exist"""
        from models.pipeline import create_exoplanet_pipeline
        assert callable(create_exoplanet_pipeline)

    def test_pipeline_has_preprocessor(self, sample_data):
        """Test: Pipeline should have preprocessing step"""
        from models.pipeline import create_exoplanet_pipeline

        features = ['bls_period', 'bls_depth_ppm', 'bls_snr']
        pipeline = create_exoplanet_pipeline(numerical_features=features)

        assert hasattr(pipeline, 'named_steps')
        assert 'preprocessor' in pipeline.named_steps

    def test_pipeline_has_classifier(self, sample_data):
        """Test: Pipeline should have XGBoost classifier"""
        from models.pipeline import create_exoplanet_pipeline

        features = ['bls_period', 'bls_depth_ppm', 'bls_snr']
        pipeline = create_exoplanet_pipeline(numerical_features=features)

        assert 'classifier' in pipeline.named_steps
        assert pipeline.named_steps['classifier'].__class__.__name__ == 'XGBClassifier'

    def test_pipeline_handles_missing_values(self, sample_data):
        """Test: Pipeline should handle missing values"""
        from models.pipeline import create_exoplanet_pipeline

        features = ['bls_period', 'bls_depth_ppm', 'bls_snr']
        pipeline = create_exoplanet_pipeline(numerical_features=features)

        # Fit with data containing NaNs
        pipeline.fit(sample_data[features], sample_data['label'])

        # Predict should not fail
        predictions = pipeline.predict(sample_data[features])
        assert len(predictions) == len(sample_data)

    def test_pipeline_uses_random_state(self, sample_data):
        """Test: Pipeline should use random_state for reproducibility"""
        from models.pipeline import create_exoplanet_pipeline

        features = ['bls_period', 'bls_depth_ppm', 'bls_snr']

        # Create two identical pipelines
        pipeline1 = create_exoplanet_pipeline(numerical_features=features, random_state=42)
        pipeline2 = create_exoplanet_pipeline(numerical_features=features, random_state=42)

        # Train both
        pipeline1.fit(sample_data[features], sample_data['label'])
        pipeline2.fit(sample_data[features], sample_data['label'])

        # Predictions should be identical
        pred1 = pipeline1.predict_proba(sample_data[features])
        pred2 = pipeline2.predict_proba(sample_data[features])

        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_pipeline_accepts_gpu_params(self, sample_data):
        """Test: Pipeline should accept GPU parameters"""
        from models.pipeline import create_exoplanet_pipeline

        features = ['bls_period', 'bls_depth_ppm', 'bls_snr']
        gpu_params = {'device': 'cuda', 'tree_method': 'hist'}

        pipeline = create_exoplanet_pipeline(
            numerical_features=features,
            xgb_params=gpu_params
        )

        classifier = pipeline.named_steps['classifier']
        assert classifier.get_params()['device'] == 'cuda'
        assert classifier.get_params()['tree_method'] == 'hist'

    def test_pipeline_can_be_saved_and_loaded(self, sample_data, tmp_path):
        """Test: Pipeline should be serializable"""
        import joblib
        from models.pipeline import create_exoplanet_pipeline

        features = ['bls_period', 'bls_depth_ppm', 'bls_snr']
        pipeline = create_exoplanet_pipeline(numerical_features=features)

        # Train
        pipeline.fit(sample_data[features], sample_data['label'])

        # Save
        model_path = tmp_path / "test_pipeline.joblib"
        joblib.dump(pipeline, model_path)

        # Load
        loaded_pipeline = joblib.load(model_path)

        # Predictions should match
        pred_original = pipeline.predict_proba(sample_data[features])
        pred_loaded = loaded_pipeline.predict_proba(sample_data[features])

        np.testing.assert_array_almost_equal(pred_original, pred_loaded)


if __name__ == "__main__":
    print("🧪 Running Pipeline Tests (TDD RED Phase)")
    print("="*60)
    pytest.main([__file__, '-v'])