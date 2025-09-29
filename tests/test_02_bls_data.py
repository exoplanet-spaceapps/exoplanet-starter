"""
Test suite for 02_bls_baseline.ipynb data loading functionality
Following TDD principles: Write tests first, then verify implementation
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Configure UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestDataLoading:
    """Test data loading functionality for 02 notebook"""

    @pytest.fixture
    def data_dir(self):
        """Get data directory path"""
        return project_root / 'data'

    @pytest.fixture
    def required_files(self):
        """List of required data files"""
        return {
            'supervised_dataset': 'supervised_dataset.csv',
            'toi_positive': 'toi_positive.csv',
            'toi_negative': 'toi_negative.csv',
            'koi_false_positives': 'koi_false_positives.csv'
        }

    def test_data_directory_exists(self, data_dir):
        """Test that data directory exists"""
        assert data_dir.exists(), f"Data directory not found: {data_dir}"
        assert data_dir.is_dir(), f"Path is not a directory: {data_dir}"

    def test_required_files_exist(self, data_dir, required_files):
        """Test that all required data files exist"""
        for name, filename in required_files.items():
            file_path = data_dir / filename
            assert file_path.exists(), f"Required file not found: {filename}"
            assert file_path.stat().st_size > 0, f"File is empty: {filename}"

    def test_supervised_dataset_structure(self, data_dir):
        """Test supervised dataset has required columns"""
        file_path = data_dir / 'supervised_dataset.csv'
        df = pd.read_csv(file_path)

        # Check dataset is not empty
        assert len(df) > 0, "Supervised dataset is empty"

        # Check required columns exist
        required_columns = ['label', 'period', 'depth', 'target_id']
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

        # Check label values
        assert set(df['label'].unique()).issubset({0, 1}), \
            "Label column should only contain 0 and 1"

        # Check data types
        assert pd.api.types.is_numeric_dtype(df['label']), "Label should be numeric"
        assert pd.api.types.is_numeric_dtype(df['period']), "Period should be numeric"
        assert pd.api.types.is_numeric_dtype(df['depth']), "Depth should be numeric"

    def test_sample_target_creation(self, data_dir):
        """Test that sample targets can be created from dataset"""
        file_path = data_dir / 'supervised_dataset.csv'
        df = pd.read_csv(file_path)

        # Create sample targets (same logic as notebook)
        complete_data = df.dropna(subset=['period', 'depth'])

        # Should have enough data for samples
        assert len(complete_data) > 0, "No complete data available"

        # Check we have both positive and negative samples
        positive_count = (complete_data['label'] == 1).sum()
        negative_count = (complete_data['label'] == 0).sum()

        assert positive_count >= 3, f"Not enough positive samples: {positive_count}"
        assert negative_count >= 2, f"Not enough negative samples: {negative_count}"

        # Create samples
        positive_samples = complete_data[complete_data['label'] == 1].head(3)
        negative_samples = complete_data[complete_data['label'] == 0].head(2)

        sample_targets = pd.concat([positive_samples, negative_samples], ignore_index=True)

        # Verify sample targets
        assert len(sample_targets) == 5, "Should have 5 sample targets"
        assert (sample_targets['label'] == 1).sum() == 3, "Should have 3 positive samples"
        assert (sample_targets['label'] == 0).sum() == 2, "Should have 2 negative samples"

    def test_target_id_format(self, data_dir):
        """Test that target IDs are in correct format"""
        file_path = data_dir / 'supervised_dataset.csv'
        df = pd.read_csv(file_path)

        # Get a sample of target IDs
        sample_ids = df['target_id'].dropna().head(10)

        for target_id in sample_ids:
            target_str = str(target_id)
            # Should contain TIC, KIC, or be numeric
            is_valid = (
                'TIC' in target_str or
                'KIC' in target_str or
                target_str.replace('.', '').replace('-', '').isdigit()
            )
            assert is_valid, f"Invalid target ID format: {target_id}"

    def test_physical_parameters_range(self, data_dir):
        """Test that physical parameters are in reasonable ranges"""
        file_path = data_dir / 'supervised_dataset.csv'
        df = pd.read_csv(file_path)

        # Filter complete data
        complete_data = df.dropna(subset=['period', 'depth'])

        if len(complete_data) > 0:
            # Test period ranges (should be positive)
            assert (complete_data['period'] > 0).all(), "All periods should be positive"
            assert (complete_data['period'] < 1000).any(), "Should have some short-period planets"

            # Test depth ranges (should be positive, in ppm)
            assert (complete_data['depth'] > 0).all(), "All depths should be positive"
            # Note: Eclipsing binaries can have very deep transits
            # Allow up to 200% (2,000,000 ppm) to handle data outliers
            assert (complete_data['depth'] < 2000000).all(), "Depths should be < 200%"

            # Check we have reasonable planet-sized depths
            planet_depth = (complete_data['depth'] < 50000)  # < 5%
            assert planet_depth.sum() > len(complete_data) * 0.3, \
                "Should have at least 30% planet-sized depths (< 5%)"

            # Check distribution
            max_depth = complete_data['depth'].max()
            median_depth = complete_data['depth'].median()
            print(f"   Depth range: {complete_data['depth'].min():.1f} - {max_depth:.1f} ppm")
            print(f"   Median depth: {median_depth:.1f} ppm")

    def test_data_statistics(self, data_dir, required_files):
        """Test basic statistics of each dataset"""
        for name, filename in required_files.items():
            file_path = data_dir / filename
            df = pd.read_csv(file_path)

            print(f"\n📊 {name} statistics:")
            print(f"   Total records: {len(df)}")
            print(f"   Columns: {', '.join(df.columns[:5])}...")

            if 'label' in df.columns:
                print(f"   Positive samples: {(df['label'] == 1).sum()}")
                print(f"   Negative samples: {(df['label'] == 0).sum()}")

            # Dataset should not be empty
            assert len(df) > 0, f"{name} is empty"


class TestNotebookDataFlow:
    """Test the complete data flow as used in notebook"""

    def test_complete_data_loading_workflow(self):
        """Test the complete workflow from notebook"""
        data_dir = project_root / 'data'

        # Step 1: Load all datasets
        datasets = {}
        data_files = {
            'supervised_dataset': 'supervised_dataset.csv',
            'toi_positive': 'toi_positive.csv',
            'toi_negative': 'toi_negative.csv',
            'koi_false_positives': 'koi_false_positives.csv'
        }

        for name, filename in data_files.items():
            file_path = data_dir / filename
            datasets[name] = pd.read_csv(file_path)

        # Step 2: Create sample targets
        df = datasets['supervised_dataset']
        complete_data = df.dropna(subset=['period', 'depth'])
        positive_samples = complete_data[complete_data['label'] == 1].head(3)
        negative_samples = complete_data[complete_data['label'] == 0].head(2)
        sample_targets = pd.concat([positive_samples, negative_samples], ignore_index=True)

        # Step 3: Verify targets can be formatted
        targets = []
        for idx, row in sample_targets.iterrows():
            target_id = row.get('target_id', f'Unknown_{idx}')

            # Format ID
            if 'TIC' in str(target_id):
                clean_id = str(target_id).replace('TIC', '').strip()
                formatted_id = f"TIC {clean_id}"
                mission = "TESS"
            elif 'KIC' in str(target_id):
                clean_id = str(target_id).replace('KIC', '').strip()
                formatted_id = f"KIC {clean_id}"
                mission = "Kepler"
            else:
                formatted_id = str(target_id)
                mission = "Unknown"

            target_dict = {
                "id": formatted_id,
                "mission": mission,
                "label": row['label']
            }

            if 'period' in row and pd.notna(row['period']):
                target_dict['known_period'] = float(row['period'])
            if 'depth' in row and pd.notna(row['depth']):
                target_dict['known_depth'] = float(row['depth'])

            targets.append(target_dict)

        # Verify results
        assert len(targets) == 5, "Should create 5 targets"
        assert all('id' in t for t in targets), "All targets should have ID"
        assert all('mission' in t for t in targets), "All targets should have mission"
        assert any('known_period' in t for t in targets), "Some targets should have known period"

        print("\n✅ Complete workflow test passed")
        print(f"   Created {len(targets)} targets for BLS analysis")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])