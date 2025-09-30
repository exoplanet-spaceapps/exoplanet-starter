"""
Unit tests for CheckpointManager

Tests checkpoint persistence, recovery, and merge functionality.
"""

import pytest
from pathlib import Path
import tempfile
import json
import pandas as pd

# Import from src
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.checkpoint_manager import CheckpointManager


class TestCheckpointManager:
    """Test suite for CheckpointManager"""

    @pytest.fixture
    def temp_drive_path(self):
        """Create temporary directory simulating Google Drive"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def checkpoint_manager(self, temp_drive_path):
        """Create CheckpointManager instance"""
        return CheckpointManager(temp_drive_path, batch_size=10)

    def test_initialization(self, temp_drive_path):
        """Test CheckpointManager initialization"""
        manager = CheckpointManager(temp_drive_path, batch_size=50)

        assert manager.drive_path == Path(temp_drive_path)
        assert manager.batch_size == 50
        assert manager.checkpoint_dir.exists()

    def test_save_checkpoint(self, checkpoint_manager):
        """Test saving a checkpoint"""
        features = {
            0: {'period': 3.5, 'depth': 0.01},
            1: {'period': 5.2, 'depth': 0.02}
        }
        failed = [2, 3]

        checkpoint_file = checkpoint_manager.save_checkpoint(
            batch_id=0,
            features=features,
            failed_indices=failed
        )

        # Verify file created
        assert checkpoint_file.exists()

        # Verify content
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)

        assert checkpoint['batch_range'] == [0, 10]
        assert len(checkpoint['completed_indices']) == 2
        assert checkpoint['failed_indices'] == failed
        assert 0 in checkpoint['completed_indices']

    def test_load_latest_checkpoint(self, checkpoint_manager):
        """Test loading most recent checkpoint"""
        # Save multiple checkpoints
        checkpoint_manager.save_checkpoint(0, {0: {'period': 3.5}})
        checkpoint_manager.save_checkpoint(10, {10: {'period': 5.2}})

        # Load latest
        loaded = checkpoint_manager.load_latest_checkpoint()

        assert loaded is not None
        assert loaded['batch_range'] == [10, 20]
        assert 10 in loaded['completed_indices']

    def test_load_no_checkpoint(self, checkpoint_manager):
        """Test loading when no checkpoints exist"""
        loaded = checkpoint_manager.load_latest_checkpoint()
        assert loaded is None

    def test_get_completed_indices(self, checkpoint_manager):
        """Test retrieving all completed indices"""
        # Save multiple checkpoints
        checkpoint_manager.save_checkpoint(0, {0: {'period': 3.5}, 1: {'period': 4.0}})
        checkpoint_manager.save_checkpoint(10, {10: {'period': 5.2}})

        # Get completed
        completed = checkpoint_manager.get_completed_indices()

        assert len(completed) == 3
        assert 0 in completed
        assert 1 in completed
        assert 10 in completed
        assert 5 not in completed  # Not processed

    def test_get_failed_indices(self, checkpoint_manager):
        """Test retrieving all failed indices"""
        checkpoint_manager.save_checkpoint(0, {0: {'period': 3.5}}, failed_indices=[1, 2])
        checkpoint_manager.save_checkpoint(10, {10: {'period': 5.2}}, failed_indices=[11])

        failed = checkpoint_manager.get_failed_indices()

        assert len(failed) == 3
        assert 1 in failed
        assert 2 in failed
        assert 11 in failed

    def test_merge_all_checkpoints(self, checkpoint_manager):
        """Test merging multiple checkpoints"""
        # Save multiple checkpoints with different features
        checkpoint_manager.save_checkpoint(
            0,
            {0: {'period': 3.5, 'depth': 0.01}, 1: {'period': 4.0, 'depth': 0.02}}
        )
        checkpoint_manager.save_checkpoint(
            10,
            {10: {'period': 5.2, 'depth': 0.03}}
        )

        # Merge
        df = checkpoint_manager.merge_all_checkpoints()

        # Verify merged DataFrame
        assert len(df) == 3
        assert '0' in df.index or 0 in df.index
        assert '1' in df.index or 1 in df.index
        assert '10' in df.index or 10 in df.index
        assert 'period' in df.columns
        assert 'depth' in df.columns
        assert df.loc['0' if '0' in df.index else 0, 'period'] == 3.5
        assert df.loc['10' if '10' in df.index else 10, 'period'] == 5.2

    def test_get_progress_summary(self, checkpoint_manager):
        """Test progress summary calculation"""
        checkpoint_manager.save_checkpoint(
            0,
            {0: {'period': 3.5}, 1: {'period': 4.0}},
            failed_indices=[2]
        )

        summary = checkpoint_manager.get_progress_summary(total_samples=100)

        assert summary['total_samples'] == 100
        assert summary['completed'] == 2
        assert summary['failed'] == 1
        assert summary['remaining'] == 98
        assert summary['success_rate'] == 2.0
        assert summary['failure_rate'] == 1.0

    def test_cleanup_checkpoints(self, checkpoint_manager):
        """Test checkpoint cleanup"""
        # Save multiple checkpoints
        checkpoint_manager.save_checkpoint(0, {0: {'period': 3.5}})
        checkpoint_manager.save_checkpoint(10, {10: {'period': 5.2}})

        # Verify they exist
        checkpoints = list(checkpoint_manager.checkpoint_dir.glob("batch_*.json"))
        assert len(checkpoints) == 2

        # Cleanup
        checkpoint_manager.cleanup_checkpoints()

        # Verify they're gone
        checkpoints = list(checkpoint_manager.checkpoint_dir.glob("batch_*.json"))
        assert len(checkpoints) == 0

    def test_empty_merge(self, checkpoint_manager):
        """Test merging when no checkpoints exist"""
        df = checkpoint_manager.merge_all_checkpoints()
        assert len(df) == 0

    def test_checkpoint_metadata(self, checkpoint_manager):
        """Test saving checkpoint with metadata"""
        features = {0: {'period': 3.5}}
        metadata = {'gpu_used': True, 'execution_time': 120.5}

        checkpoint_file = checkpoint_manager.save_checkpoint(
            batch_id=0,
            features=features,
            metadata=metadata
        )

        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)

        assert checkpoint['metadata']['gpu_used'] is True
        assert checkpoint['metadata']['execution_time'] == 120.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])