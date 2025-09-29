"""
Test Suite for Jupyter Notebook Syntax Validation
Tests all notebook cells for valid Python syntax following TDD principles.
"""

import ast
import json
import pytest
from pathlib import Path
from typing import List, Tuple, Dict


class NotebookSyntaxValidator:
    """Validates Python syntax in Jupyter notebook cells."""

    def __init__(self, notebook_path: Path):
        """
        Initialize validator with notebook path.

        Args:
            notebook_path: Path to .ipynb file
        """
        self.notebook_path = notebook_path
        self.notebook_name = notebook_path.name

    def load_notebook(self) -> dict:
        """Load notebook JSON."""
        with open(self.notebook_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def extract_code_cells(self) -> List[Tuple[int, str]]:
        """
        Extract Python code cells from notebook.

        Returns:
            List of (cell_index, source_code) tuples
        """
        notebook = self.load_notebook()
        code_cells = []

        for idx, cell in enumerate(notebook.get('cells', [])):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if isinstance(source, list):
                    source = ''.join(source)
                code_cells.append((idx, source))

        return code_cells

    def clean_cell_code(self, code: str) -> str:
        """
        Clean Jupyter-specific syntax from code.

        Args:
            code: Raw cell code

        Returns:
            Cleaned Python code
        """
        lines = code.split('\n')
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            # Skip Jupyter magic commands
            if stripped.startswith('!') or stripped.startswith('%'):
                continue
            # Skip empty lines at start/end
            if not stripped and not cleaned_lines:
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def validate_cell_syntax(self, cell_idx: int, code: str) -> Tuple[bool, str]:
        """
        Validate Python syntax of a cell.

        Args:
            cell_idx: Cell index in notebook
            code: Python code to validate

        Returns:
            (is_valid, error_message) tuple
        """
        if not code.strip():
            return True, ""

        cleaned_code = self.clean_cell_code(code)
        if not cleaned_code.strip():
            return True, ""

        try:
            ast.parse(cleaned_code)
            return True, ""
        except SyntaxError as e:
            error_msg = (
                f"Cell {cell_idx}: {e.msg} at line {e.lineno}\n"
                f"Problem: {e.text.strip() if e.text else 'N/A'}"
            )
            return False, error_msg
        except Exception as e:
            error_msg = f"Cell {cell_idx}: Unexpected error: {str(e)}"
            return False, error_msg

    def validate_all_cells(self) -> Dict[str, any]:
        """
        Validate all code cells in notebook.

        Returns:
            Dictionary with validation results
        """
        code_cells = self.extract_code_cells()
        errors = []
        total_cells = len(code_cells)

        for cell_idx, code in code_cells:
            is_valid, error_msg = self.validate_cell_syntax(cell_idx, code)
            if not is_valid:
                errors.append({
                    'cell_index': cell_idx,
                    'error': error_msg
                })

        return {
            'notebook': self.notebook_name,
            'total_cells': total_cells,
            'valid_cells': total_cells - len(errors),
            'invalid_cells': len(errors),
            'errors': errors,
            'is_valid': len(errors) == 0
        }


# Test fixtures
@pytest.fixture
def notebooks_dir():
    """Get notebooks directory path."""
    base_dir = Path(__file__).parent.parent
    return base_dir / 'notebooks'


@pytest.fixture
def notebook_files(notebooks_dir):
    """Get all notebook files."""
    return {
        '01_tap_download': notebooks_dir / '01_tap_download.ipynb',
        '02_bls_baseline': notebooks_dir / '02_bls_baseline.ipynb',
        '03_injection_train': notebooks_dir / '03_injection_train.ipynb',
        '04_newdata_inference': notebooks_dir / '04_newdata_inference.ipynb',
        '05_metrics_dashboard': notebooks_dir / '05_metrics_dashboard.ipynb'
    }


# Test cases for each notebook
class TestNotebookSyntax:
    """Test suite for notebook syntax validation."""

    def test_01_tap_download_syntax(self, notebook_files):
        """Test 01_tap_download.ipynb for syntax errors."""
        notebook_path = notebook_files['01_tap_download']
        validator = NotebookSyntaxValidator(notebook_path)
        result = validator.validate_all_cells()

        assert result['is_valid'], (
            f"\n{result['notebook']} has {result['invalid_cells']} syntax errors:\n" +
            "\n".join([f"  - {err['error']}" for err in result['errors']])
        )

    def test_02_bls_baseline_syntax(self, notebook_files):
        """Test 02_bls_baseline.ipynb for syntax errors (Cell 35 issue)."""
        notebook_path = notebook_files['02_bls_baseline']
        validator = NotebookSyntaxValidator(notebook_path)
        result = validator.validate_all_cells()

        # This test will FAIL initially due to unterminated string in Cell 35
        assert result['is_valid'], (
            f"\n{result['notebook']} has {result['invalid_cells']} syntax errors:\n" +
            "\n".join([f"  - {err['error']}" for err in result['errors']])
        )

    def test_03_injection_train_syntax(self, notebook_files):
        """Test 03_injection_train.ipynb for syntax errors (Cell 65 issue)."""
        notebook_path = notebook_files['03_injection_train']
        validator = NotebookSyntaxValidator(notebook_path)
        result = validator.validate_all_cells()

        # This test will FAIL initially due to unterminated string in Cell 65
        assert result['is_valid'], (
            f"\n{result['notebook']} has {result['invalid_cells']} syntax errors:\n" +
            "\n".join([f"  - {err['error']}" for err in result['errors']])
        )

    def test_04_newdata_inference_syntax(self, notebook_files):
        """Test 04_newdata_inference.ipynb for syntax errors (Cell 25 issue)."""
        notebook_path = notebook_files['04_newdata_inference']
        validator = NotebookSyntaxValidator(notebook_path)
        result = validator.validate_all_cells()

        # This test will FAIL initially due to unterminated string in Cell 25
        assert result['is_valid'], (
            f"\n{result['notebook']} has {result['invalid_cells']} syntax errors:\n" +
            "\n".join([f"  - {err['error']}" for err in result['errors']])
        )

    def test_05_metrics_dashboard_syntax(self, notebook_files):
        """Test 05_metrics_dashboard.ipynb for syntax errors (Cell 24 issue)."""
        notebook_path = notebook_files['05_metrics_dashboard']
        validator = NotebookSyntaxValidator(notebook_path)
        result = validator.validate_all_cells()

        # This test will FAIL initially due to unterminated string in Cell 24
        assert result['is_valid'], (
            f"\n{result['notebook']} has {result['invalid_cells']} syntax errors:\n" +
            "\n".join([f"  - {err['error']}" for err in result['errors']])
        )


# Test for specific error patterns
class TestSpecificSyntaxErrors:
    """Tests for specific syntax error patterns identified."""

    def test_no_unterminated_strings(self, notebook_files):
        """Verify no notebooks have unterminated triple-quoted strings."""
        for name, path in notebook_files.items():
            validator = NotebookSyntaxValidator(path)
            result = validator.validate_all_cells()

            unterminated_errors = [
                err for err in result['errors']
                if 'unterminated' in err['error'].lower() or
                   'EOF while scanning' in err['error']
            ]

            assert len(unterminated_errors) == 0, (
                f"{name} has unterminated string errors:\n" +
                "\n".join([err['error'] for err in unterminated_errors])
            )

    def test_no_indentation_errors(self, notebook_files):
        """Verify no notebooks have unexpected indentation."""
        for name, path in notebook_files.items():
            validator = NotebookSyntaxValidator(path)
            result = validator.validate_all_cells()

            indent_errors = [
                err for err in result['errors']
                if 'indent' in err['error'].lower()
            ]

            assert len(indent_errors) == 0, (
                f"{name} has indentation errors:\n" +
                "\n".join([err['error'] for err in indent_errors])
            )


# Summary test
def test_all_notebooks_valid(notebook_files):
    """
    Master test: All notebooks must have valid Python syntax.
    This test will FAIL until all notebooks are fixed.
    """
    results = []

    for name, path in notebook_files.items():
        validator = NotebookSyntaxValidator(path)
        result = validator.validate_all_cells()
        results.append(result)

    # Collect all errors
    all_errors = []
    for result in results:
        if not result['is_valid']:
            all_errors.append(f"\n{result['notebook']}:")
            for err in result['errors']:
                all_errors.append(f"  - {err['error']}")

    # Generate summary
    total_notebooks = len(results)
    valid_notebooks = sum(1 for r in results if r['is_valid'])

    summary = (
        f"\n{'='*70}\n"
        f"NOTEBOOK SYNTAX VALIDATION SUMMARY\n"
        f"{'='*70}\n"
        f"Total Notebooks: {total_notebooks}\n"
        f"Valid: {valid_notebooks}\n"
        f"Invalid: {total_notebooks - valid_notebooks}\n"
    )

    if all_errors:
        summary += f"\nErrors found:{''.join(all_errors)}\n"

    assert valid_notebooks == total_notebooks, summary


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])