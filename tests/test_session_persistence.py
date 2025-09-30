"""
Unit tests for SessionPersistence

Tests session state management and recovery functionality.
"""

import pytest
from pathlib import Path
import tempfile
import json
from datetime import datetime, timedelta
import time

# Import from src
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.session_persistence import SessionPersistence


class TestSessionPersistence:
    """Test suite for SessionPersistence"""

    @pytest.fixture
    def temp_drive_path(self):
        """Create temporary directory simulating Google Drive"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def session_persistence(self, temp_drive_path):
        """Create SessionPersistence instance"""
        return SessionPersistence(temp_drive_path)

    def test_initialization(self, temp_drive_path):
        """Test SessionPersistence initialization"""
        persistence = SessionPersistence(temp_drive_path)

        assert persistence.drive_path == Path(temp_drive_path)
        assert persistence.logs_dir.exists()
        assert persistence.state_file == persistence.logs_dir / "session_state.json"

    def test_save_state(self, session_persistence):
        """Test saving session state"""
        state = {
            'current_batch': 0,
            'completed': 50,
            'total': 1000,
            'failed': 2,
            'start_time': datetime.utcnow().isoformat()
        }

        session_persistence.save_state(state)

        # Verify file created
        assert session_persistence.state_file.exists()

        # Verify content
        with open(session_persistence.state_file, 'r') as f:
            saved_state = json.load(f)

        assert saved_state['completed'] == 50
        assert saved_state['total'] == 1000
        assert 'last_updated' in saved_state
        assert 'session_duration' in saved_state

    def test_load_state(self, session_persistence):
        """Test loading session state"""
        # Save state first
        state = {
            'current_batch': 100,
            'completed': 150,
            'total': 1000,
            'start_time': datetime.utcnow().isoformat()
        }
        session_persistence.save_state(state)

        # Load state
        loaded_state = session_persistence.load_state()

        assert loaded_state is not None
        assert loaded_state['completed'] == 150
        assert loaded_state['total'] == 1000
        assert loaded_state['current_batch'] == 100

    def test_load_no_state(self, session_persistence):
        """Test loading when no state exists"""
        loaded_state = session_persistence.load_state()
        assert loaded_state is None

    def test_estimate_remaining_time(self, session_persistence):
        """Test time estimation"""
        # Simulate 1 hour ago
        start_time = (datetime.utcnow() - timedelta(hours=1)).isoformat()

        # 100 samples in 1 hour = 100/hour rate
        # 900 remaining = 9 hours
        estimate = session_persistence.estimate_remaining_time(
            completed=100,
            total=1000,
            start_time=start_time
        )

        # Should be approximately 9h
        assert 'h' in estimate
        assert '8h' in estimate or '9h' in estimate or '10h' in estimate

    def test_estimate_remaining_time_no_progress(self, session_persistence):
        """Test time estimation with no progress"""
        estimate = session_persistence.estimate_remaining_time(
            completed=0,
            total=1000
        )

        assert estimate == "Unknown"

    def test_estimate_session_remaining(self, session_persistence):
        """Test Colab session time estimation"""
        # Simulate 2 hours ago
        start_time = (datetime.utcnow() - timedelta(hours=2)).isoformat()

        # Should have ~10 hours remaining (12 hour limit - 2 elapsed)
        remaining = session_persistence.estimate_session_remaining(
            start_time=start_time,
            session_limit_hours=12.0
        )

        assert 'h' in remaining
        assert '9h' in remaining or '10h' in remaining

    def test_estimate_session_remaining_no_start(self, session_persistence):
        """Test session estimation with no start time"""
        remaining = session_persistence.estimate_session_remaining()
        assert remaining == "12h 0m"

    def test_get_session_metrics(self, session_persistence):
        """Test session metrics calculation"""
        start_time = (datetime.utcnow() - timedelta(hours=1)).isoformat()

        state = {
            'completed': 100,
            'total': 1000,
            'failed': 5,
            'start_time': start_time
        }

        metrics = session_persistence.get_session_metrics(state)

        assert 'elapsed_hours' in metrics
        assert 'samples_per_hour' in metrics
        assert 'success_rate' in metrics
        assert 'estimated_completion' in metrics
        assert 'session_remaining' in metrics

        # Check approximate values
        assert 0.9 <= metrics['elapsed_hours'] <= 1.1  # ~1 hour
        assert 90 <= metrics['samples_per_hour'] <= 110  # ~100/hour
        assert 94 <= metrics['success_rate'] <= 96  # 100/(100+5) ≈ 95%

    def test_get_session_metrics_no_start(self, session_persistence):
        """Test metrics with no start time"""
        state = {
            'completed': 100,
            'total': 1000,
            'failed': 5
        }

        metrics = session_persistence.get_session_metrics(state)
        assert metrics == {}

    def test_auto_save_start(self, session_persistence):
        """Test starting auto-save thread"""
        state_counter = {'value': 0}

        def get_state():
            state_counter['value'] += 1
            return {'completed': state_counter['value'], 'total': 100}

        # Start auto-save with 1 second interval (for testing)
        session_persistence.start_auto_save(get_state, interval_minutes=1/60)

        # Wait for at least one auto-save
        time.sleep(2)

        # Verify state file created
        assert session_persistence.state_file.exists()

    def test_auto_save_already_running(self, session_persistence, capsys):
        """Test starting auto-save when already running"""
        def get_state():
            return {'completed': 0, 'total': 100}

        # Start first time
        session_persistence.start_auto_save(get_state, interval_minutes=10)

        # Try starting again
        session_persistence.start_auto_save(get_state, interval_minutes=10)

        # Should print warning
        captured = capsys.readouterr()
        assert "already running" in captured.out.lower()

    def test_state_persistence_across_sessions(self, session_persistence):
        """Test state persistence across sessions"""
        # Session 1: Save state
        state1 = {
            'completed': 200,
            'total': 1000,
            'start_time': datetime.utcnow().isoformat()
        }
        session_persistence.save_state(state1)

        # Session 2: Create new instance and load
        persistence2 = SessionPersistence(session_persistence.drive_path)
        loaded_state = persistence2.load_state()

        assert loaded_state is not None
        assert loaded_state['completed'] == 200
        assert loaded_state['total'] == 1000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])