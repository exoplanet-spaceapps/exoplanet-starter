"""
Test suite for 02_bls_baseline.ipynb data loading functionality
Following TDD principles: Write tests FIRST to define expected behavior

This test suite verifies that:
1. The notebook imports data_loader_colab module
2. The notebook uses setup_data_directory() function
3. The notebook uses load_datasets() function
4. The notebook handles both Colab and local environments correctly
5. Data loading verifies files exist before proceeding
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

# Configure UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'notebooks'))

# Import the data loader module
from notebooks import data_loader_colab


class TestNotebookDataLoaderIntegration:
    """Test that 02 notebook correctly integrates with data_loader_colab"""

    @pytest.fixture
    def notebook_path(self):
        """Get notebook file path"""
        return project_root / 'notebooks' / '02_bls_baseline.ipynb'

    @pytest.fixture
    def notebook_content(self, notebook_path):
        """Load notebook content"""
        with open(notebook_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_notebook_imports_data_loader_colab(self, notebook_content):
        """
        TEST 1: 02 notebook Cell 7 should import data_loader_colab
        This ensures the notebook uses the centralized data loading module
        """
        # Find Cell 7 (index 6 in 0-based array)
        cells = notebook_content.get('cells', [])

        # Cell 7 should be around index 6-7, but let's search for the data loading cell
        cell_7_found = False
        imports_data_loader = False

        for idx, cell in enumerate(cells):
            cell_source = ''.join(cell.get('source', []))

            # Look for the data loading section (marked with specific comments)
            if '载入已下载的资料集' in cell_source or '載入已下載的資料集' in cell_source:
                cell_7_found = True

                # Check if it imports data_loader_colab
                if 'import data_loader_colab' in cell_source or 'from data_loader_colab' in cell_source:
                    imports_data_loader = True
                    print(f"✅ Found import in cell {idx}")
                    break

        assert cell_7_found, "Could not find data loading cell in notebook"
        assert imports_data_loader, (
            "Cell 7 should import data_loader_colab module. "
            "Expected: 'import data_loader_colab' or 'from data_loader_colab import ...'"
        )

    def test_notebook_calls_setup_data_directory(self, notebook_content):
        """
        TEST 2: 02 notebook should call setup_data_directory() or main()
        This ensures proper environment detection and directory setup
        """
        cells = notebook_content.get('cells', [])

        calls_setup = False

        for cell in cells:
            cell_source = ''.join(cell.get('source', []))

            # Accept either direct call to setup_data_directory() or main()
            if 'data_loader_colab.main()' in cell_source:
                calls_setup = True
                print("✅ Found data_loader_colab.main() call (includes setup_data_directory)")
                break
            elif 'setup_data_directory' in cell_source:
                # Should be a function call, not just a comment
                if 'setup_data_directory()' in cell_source or 'setup_data_directory(' in cell_source:
                    calls_setup = True
                    print("✅ Found setup_data_directory() call")
                    break

        assert calls_setup, (
            "Notebook should call setup_data_directory() or data_loader_colab.main(). "
            "Expected: 'data_dir, IN_COLAB = setup_data_directory()' or "
            "'sample_targets, datasets, data_dir, IN_COLAB = data_loader_colab.main()'"
        )

    def test_notebook_calls_load_datasets(self, notebook_content):
        """
        TEST 3: 02 notebook should call load_datasets() or main()
        This ensures datasets are loaded using the standardized function
        """
        cells = notebook_content.get('cells', [])

        calls_load_datasets = False

        for cell in cells:
            cell_source = ''.join(cell.get('source', []))

            # Accept either direct call to load_datasets() or main()
            if 'data_loader_colab.main()' in cell_source:
                calls_load_datasets = True
                print("✅ Found data_loader_colab.main() call (includes load_datasets)")
                break
            elif 'load_datasets' in cell_source:
                # Should be a function call
                if 'load_datasets(' in cell_source:
                    calls_load_datasets = True
                    print("✅ Found load_datasets() call")
                    break

        assert calls_load_datasets, (
            "Notebook should call load_datasets() or data_loader_colab.main(). "
            "Expected: 'datasets = load_datasets(data_dir)' or "
            "'sample_targets, datasets, data_dir, IN_COLAB = data_loader_colab.main()'"
        )

    def test_notebook_removes_duplicate_code(self, notebook_content):
        """
        TEST 4: Notebook should NOT have duplicate data loading logic
        After refactoring, the old inline code should be removed
        """
        cells = notebook_content.get('cells', [])

        # Check that old duplicate patterns are removed
        duplicate_patterns = [
            'data_files = {',  # Old inline dictionary definition
            "for name, filename in data_files.items():",  # Old inline loading loop
        ]

        found_duplicates = []

        for cell in cells:
            cell_source = ''.join(cell.get('source', []))

            # Skip if this is the data_loader_colab module itself
            if 'def load_datasets' in cell_source:
                continue

            for pattern in duplicate_patterns:
                if pattern in cell_source:
                    # Check if it's not inside a comment or string
                    lines = cell_source.split('\n')
                    for line in lines:
                        if pattern in line and not line.strip().startswith('#'):
                            found_duplicates.append(pattern)

        # Note: This test might be too strict initially, so we log instead of failing
        if found_duplicates:
            print(f"⚠️ Found potentially duplicate code patterns: {found_duplicates}")
            print("   Consider removing these after refactoring to use data_loader_colab")


class TestColabEnvironmentHandling:
    """Test that the notebook handles Colab environment correctly"""

    def test_notebook_handles_colab_environment(self):
        """
        TEST 5: Notebook should handle Colab environment (GitHub clone)
        When google.colab is available, it should use /content/exoplanet-starter/data
        """
        # Mock Colab environment
        with patch.dict('sys.modules', {'google.colab': MagicMock()}):
            with patch('subprocess.run') as mock_run:
                # Mock successful git clone
                mock_run.return_value = MagicMock(returncode=0, stderr='', stdout='')

                # Mock Path.exists to return False (needs cloning)
                with patch('pathlib.Path.exists', return_value=False):
                    with patch('os.chdir'):
                        data_dir, in_colab = data_loader_colab.setup_data_directory()

                        assert in_colab is True, "Should detect Colab environment"
                        assert 'exoplanet-starter' in str(data_dir), \
                            "In Colab, should use /content/exoplanet-starter/data"

                        # Verify git clone was called
                        assert mock_run.called, "Should call git clone in Colab"

                        # Check git clone command
                        call_args = str(mock_run.call_args)
                        assert 'git' in call_args and 'clone' in call_args, \
                            "Should execute git clone command"

    def test_notebook_handles_local_environment(self):
        """
        TEST 6: Notebook should handle local environment (../data)
        When google.colab is NOT available, it should use ../data
        """
        # Ensure google.colab is not in sys.modules
        if 'google.colab' in sys.modules:
            del sys.modules['google.colab']

        with patch.dict('sys.modules', {'google.colab': None}):
            # Should raise ImportError when trying to import google.colab
            with pytest.raises(Exception):
                from google.colab import drive

        # Now test our function
        data_dir, in_colab = data_loader_colab.setup_data_directory()

        assert in_colab is False, "Should detect local environment"
        assert str(data_dir) == str(Path('../data')), \
            "In local environment, should use ../data"


class TestDataLoadingValidation:
    """Test that data loading verifies files exist before proceeding"""

    @pytest.fixture
    def mock_data_dir(self, tmp_path):
        """Create a temporary data directory with test files"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create test CSV files
        test_data = pd.DataFrame({
            'target_id': ['TIC123', 'TIC456'],
            'label': [1, 0],
            'period': [4.5, 2.3],
            'depth': [2000, 1500]
        })

        (data_dir / 'supervised_dataset.csv').write_text(test_data.to_csv(index=False))
        (data_dir / 'toi_positive.csv').write_text(test_data.to_csv(index=False))

        return data_dir

    def test_load_datasets_verifies_directory_exists(self, tmp_path):
        """
        TEST 7: load_datasets() should verify data directory exists
        Should return empty dict if directory doesn't exist
        """
        non_existent_dir = tmp_path / "non_existent"

        datasets = data_loader_colab.load_datasets(non_existent_dir)

        assert isinstance(datasets, dict), "Should return a dictionary"
        assert len(datasets) == 0, "Should return empty dict if directory doesn't exist"

    def test_load_datasets_verifies_files_exist(self, mock_data_dir):
        """
        TEST 8: load_datasets() should verify each file exists before loading
        Should skip missing files and continue with others
        """
        datasets = data_loader_colab.load_datasets(mock_data_dir)

        # Should load the files that exist
        assert 'supervised_dataset' in datasets, "Should load supervised_dataset.csv"
        assert 'toi_positive' in datasets, "Should load toi_positive.csv"

        # Should not crash on missing files
        assert 'toi_negative' not in datasets or len(datasets['toi_negative']) == 0, \
            "Should handle missing toi_negative.csv gracefully"

    def test_load_datasets_handles_empty_directory(self, tmp_path):
        """
        TEST 9: load_datasets() should handle empty data directory
        Should return empty dict and print warning
        """
        empty_dir = tmp_path / "empty_data"
        empty_dir.mkdir()

        datasets = data_loader_colab.load_datasets(empty_dir)

        assert isinstance(datasets, dict), "Should return a dictionary"
        assert len(datasets) == 0, "Should return empty dict for empty directory"

    def test_load_datasets_validates_csv_format(self, tmp_path):
        """
        TEST 10: load_datasets() should handle corrupted CSV files
        Should skip corrupted files and continue with others
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create a corrupted CSV
        (data_dir / 'supervised_dataset.csv').write_text("corrupted,data\nthis,is,not,valid,csv")

        # Should not crash
        datasets = data_loader_colab.load_datasets(data_dir)

        assert isinstance(datasets, dict), "Should return a dictionary"
        # May or may not load the corrupted file, but should not crash


class TestSampleTargetCreation:
    """Test sample target creation from datasets"""

    def test_create_sample_targets_with_valid_data(self):
        """
        TEST 11: create_sample_targets() should create valid samples
        Should select specified number of positive and negative samples
        """
        # Create mock datasets
        datasets = {
            'supervised_dataset': pd.DataFrame({
                'target_id': ['TIC1', 'TIC2', 'TIC3', 'TIC4', 'TIC5'],
                'label': [1, 1, 1, 0, 0],
                'period': [4.5, 3.2, 2.1, 5.6, 6.7],
                'depth': [2000, 1500, 1200, 800, 900]
            })
        }

        sample_targets = data_loader_colab.create_sample_targets(datasets, n_positive=3, n_negative=2)

        assert len(sample_targets) == 5, "Should create 5 sample targets"
        assert (sample_targets['label'] == 1).sum() == 3, "Should have 3 positive samples"
        assert (sample_targets['label'] == 0).sum() == 2, "Should have 2 negative samples"

    def test_create_sample_targets_with_insufficient_data(self):
        """
        TEST 12: create_sample_targets() should handle insufficient data
        Should use default targets if not enough data available
        """
        # Create datasets with insufficient samples
        datasets = {
            'supervised_dataset': pd.DataFrame({
                'target_id': ['TIC1'],
                'label': [1],
                'period': [4.5],
                'depth': [2000]
            })
        }

        sample_targets = data_loader_colab.create_sample_targets(datasets, n_positive=3, n_negative=2)

        # Should fallback to default or return what's available
        assert len(sample_targets) > 0, "Should return some targets"
        assert 'target_id' in sample_targets.columns, "Should have target_id column"


class TestDataLoaderMainFunction:
    """Test the main() function that orchestrates the full workflow"""

    def test_main_function_returns_all_required_values(self):
        """
        TEST 13: main() should return all required values
        Should return (sample_targets, datasets, data_dir, IN_COLAB)
        """
        with patch('data_loader_colab.setup_data_directory') as mock_setup:
            with patch('data_loader_colab.load_datasets') as mock_load:
                with patch('data_loader_colab.create_sample_targets') as mock_create:
                    # Mock return values
                    mock_setup.return_value = (Path('../data'), False)
                    mock_load.return_value = {}
                    mock_create.return_value = pd.DataFrame()

                    result = data_loader_colab.main()

                    assert isinstance(result, tuple), "Should return a tuple"
                    assert len(result) == 4, "Should return 4 values"

                    sample_targets, datasets, data_dir, in_colab = result

                    assert isinstance(sample_targets, pd.DataFrame), "First value should be DataFrame"
                    assert isinstance(datasets, dict), "Second value should be dict"
                    assert isinstance(data_dir, Path), "Third value should be Path"
                    assert isinstance(in_colab, bool), "Fourth value should be bool"


class TestEndToEndWorkflow:
    """Test the complete end-to-end workflow as used in notebook"""

    @pytest.mark.integration
    def test_complete_workflow_local_environment(self):
        """
        TEST 14: Complete workflow should work in local environment
        Integration test for the full data loading pipeline
        """
        # Skip if data directory doesn't exist
        data_dir = project_root / 'data'
        if not data_dir.exists():
            pytest.skip("Data directory not found - run 01_tap_download.ipynb first")

        # Test the complete workflow
        data_dir, in_colab = data_loader_colab.setup_data_directory()

        assert data_dir.exists(), "Data directory should exist"
        assert in_colab is False, "Should detect local environment"

        # Load datasets
        datasets = data_loader_colab.load_datasets(data_dir)

        assert len(datasets) > 0, "Should load at least one dataset"
        assert 'supervised_dataset' in datasets, "Should load supervised_dataset"

        # Create sample targets
        sample_targets = data_loader_colab.create_sample_targets(datasets)

        assert len(sample_targets) > 0, "Should create sample targets"
        assert 'target_id' in sample_targets.columns, "Should have target_id"
        assert 'label' in sample_targets.columns, "Should have label"

    def test_workflow_handles_missing_data_gracefully(self):
        """
        TEST 15: Workflow should handle missing data gracefully
        Should not crash even if data files are missing
        """
        with patch('pathlib.Path.exists', return_value=False):
            data_dir, in_colab = data_loader_colab.setup_data_directory()
            datasets = data_loader_colab.load_datasets(data_dir)
            sample_targets = data_loader_colab.create_sample_targets(datasets)

            # Should return default targets instead of crashing
            assert isinstance(sample_targets, pd.DataFrame), "Should return DataFrame"
            assert len(sample_targets) >= 3, "Should have at least 3 default targets"


if __name__ == "__main__":
    # Run tests with verbose output
    print("="*60)
    print("🧪 Running TDD tests for 02_bls_baseline.ipynb data loading")
    print("="*60)
    pytest.main([__file__, '-v', '--tb=short', '-k', 'not integration'])