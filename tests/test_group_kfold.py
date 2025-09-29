"""
Tests for StratifiedGroupKFold (Phase 4) - TDD RED phase
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

src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


class TestStratifiedGroupKFold:
    """Test suite for grouped cross-validation"""

    @pytest.fixture
    def sample_data_with_groups(self):
        """Create sample data with groups"""
        np.random.seed(42)
        n_groups = 20
        samples_per_group = 5

        data = []
        for group_id in range(n_groups):
            for _ in range(samples_per_group):
                data.append({
                    'feature1': np.random.rand(),
                    'feature2': np.random.rand(),
                    'feature3': np.random.rand(),
                    'label': group_id % 2,  # Alternating labels
                    'target_id': f'TIC{group_id}'
                })

        return pd.DataFrame(data)

    def test_no_leakage_between_folds(self, sample_data_with_groups):
        """Test: Same target_id should not appear in train and test"""
        from sklearn.model_selection import StratifiedGroupKFold

        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

        X = sample_data_with_groups[['feature1', 'feature2', 'feature3']]
        y = sample_data_with_groups['label']
        groups = sample_data_with_groups['target_id']

        for train_idx, test_idx in cv.split(X, y, groups):
            train_groups = set(groups.iloc[train_idx])
            test_groups = set(groups.iloc[test_idx])

            # No overlap between train and test groups
            assert len(train_groups & test_groups) == 0, "Data leakage detected!"

    def test_stratification_preserved(self, sample_data_with_groups):
        """Test: Label distribution should be similar across folds"""
        from sklearn.model_selection import StratifiedGroupKFold

        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

        X = sample_data_with_groups[['feature1', 'feature2', 'feature3']]
        y = sample_data_with_groups['label']
        groups = sample_data_with_groups['target_id']

        label_ratios = []
        for train_idx, test_idx in cv.split(X, y, groups):
            test_labels = y.iloc[test_idx]
            ratio = test_labels.mean()  # Proportion of label=1
            label_ratios.append(ratio)

        # All folds should have similar label ratios
        assert np.std(label_ratios) < 0.2, "Stratification not preserved"

    def test_cv_with_pipeline_integration(self, sample_data_with_groups):
        """Test: Cross-validation should work with pipeline"""
        from sklearn.model_selection import StratifiedGroupKFold, cross_validate
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier

        # Create simple pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])

        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

        X = sample_data_with_groups[['feature1', 'feature2', 'feature3']]
        y = sample_data_with_groups['label']
        groups = sample_data_with_groups['target_id']

        # Should not raise error
        cv_results = cross_validate(
            pipeline, X, y, groups=groups,
            cv=cv,
            return_train_score=True,
            n_jobs=1
        )

        assert 'test_score' in cv_results
        assert len(cv_results['test_score']) == 3


if __name__ == "__main__":
    print("🧪 Running GroupKFold Tests (TDD RED Phase)")
    print("="*60)
    pytest.main([__file__, '-v'])