"""
TDD tests for notebook syntax validation
Following TDD: Write tests first, then fix implementation
"""
import pytest
import json
import ast
from pathlib import Path

project_root = Path(__file__).parent.parent


class TestNotebookSyntax:
    """Test that all notebook cells have valid Python syntax"""

    @pytest.fixture
    def notebooks_dir(self):
        return project_root / 'notebooks'

    def extract_python_code(self, cell_source):
        """Extract Python code from cell, removing magic commands"""
        if isinstance(cell_source, list):
            lines = cell_source
        else:
            lines = cell_source.split('\n')

        python_lines = []
        for line in lines:
            # Skip Jupyter magic commands
            stripped = line.lstrip()
            if stripped.startswith('!') or stripped.startswith('%'):
                continue
            python_lines.append(line)

        return ''.join(python_lines) if isinstance(cell_source, list) else '\n'.join(python_lines)

    def test_02_notebook_no_conditional_import(self, notebooks_dir):
        """Test: 02 notebook should not have 'import X if condition else Y' syntax"""
        nb_path = notebooks_dir / '02_bls_baseline.ipynb'

        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        for idx, cell in enumerate(nb['cells']):
            if cell['cell_type'] != 'code':
                continue

            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

            # Test: Should not contain conditional import
            assert 'import torch if' not in source, \
                f"Cell {idx} contains invalid conditional import syntax"
            assert 'import' in source and 'if' in source and 'else' in source and 'None' in source, \
                f"Cell {idx} might still have conditional import pattern"

    def test_02_notebook_torch_import_is_try_except(self, notebooks_dir):
        """Test: 02 notebook should use try-except for torch import"""
        nb_path = notebooks_dir / '02_bls_baseline.ipynb'

        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        # Find the environment setup cell (Cell 3)
        cell = nb['cells'][3]
        assert cell['cell_type'] == 'code', "Cell 3 should be a code cell"

        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

        # Test: Should contain try-except for torch
        assert 'try:' in source, "Cell 3 should have try block"
        assert 'import torch' in source, "Cell 3 should import torch"
        assert 'except ImportError:' in source or 'except:' in source, "Cell 3 should have except block"
        assert 'torch = None' in source, "Cell 3 should set torch = None on import failure"

    def test_all_notebooks_valid_python_syntax(self, notebooks_dir):
        """Test: All notebook code cells should have valid Python syntax"""
        notebooks = list(notebooks_dir.glob('*.ipynb'))
        assert len(notebooks) > 0, "Should find at least one notebook"

        errors = []

        for nb_path in notebooks:
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)

            for idx, cell in enumerate(nb['cells']):
                if cell['cell_type'] != 'code':
                    continue

                # Extract Python code (skip magic commands)
                code = self.extract_python_code(cell['source'])

                if not code.strip():
                    continue

                # Test: Code should compile without syntax errors
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    errors.append({
                        'notebook': nb_path.name,
                        'cell': idx,
                        'error': str(e),
                        'line': e.lineno
                    })

        # Assert no syntax errors found
        if errors:
            error_msg = "\n".join([
                f"  {err['notebook']} Cell {err['cell']}: {err['error']} (line {err['line']})"
                for err in errors
            ])
            pytest.fail(f"Found {len(errors)} syntax errors:\n{error_msg}")

    def test_no_unterminated_strings(self, notebooks_dir):
        """Test: No notebook should have unterminated string literals"""
        notebooks = list(notebooks_dir.glob('*.ipynb'))

        errors = []

        for nb_path in notebooks:
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)

            for idx, cell in enumerate(nb['cells']):
                if cell['cell_type'] != 'code':
                    continue

                code = self.extract_python_code(cell['source'])

                if not code.strip():
                    continue

                # Test: Should not have unterminated triple-quoted strings
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    if 'unterminated' in str(e).lower() and 'string' in str(e).lower():
                        errors.append({
                            'notebook': nb_path.name,
                            'cell': idx,
                            'error': str(e)
                        })

        if errors:
            error_msg = "\n".join([
                f"  {err['notebook']} Cell {err['cell']}: {err['error']}"
                for err in errors
            ])
            pytest.fail(f"Found {len(errors)} unterminated strings:\n{error_msg}")


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])