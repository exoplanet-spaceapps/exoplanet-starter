"""
Session Persistence Manager for Google Colab

Handles session state across Colab disconnects with automatic recovery.
"""

from pathlib import Path
from typing import Dict, Optional, Callable
from datetime import datetime
import json
import threading
import time


class SessionPersistence:
    """
    Manage session state across Colab disconnects

    Features:
    - Save session state periodically
    - Auto-detect disconnection
    - Resume from last state
    - Progress tracking and estimation
    """

    def __init__(self, drive_path: str):
        """
        Initialize session persistence

        Args:
            drive_path: Path to Google Drive directory
        """
        self.drive_path = Path(drive_path)
        self.logs_dir = self.drive_path / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.logs_dir / "session_state.json"
        self.last_save = None
        self._auto_save_thread = None

    def save_state(self, state: Dict) -> None:
        """
        Save current execution state

        Args:
            state: Dictionary containing session state
                Required keys: current_batch, completed, total
                Optional keys: failed, start_time, etc.
        """
        state['last_updated'] = datetime.utcnow().isoformat()
        state['session_duration'] = self._get_session_duration(
            state.get('start_time')
        )

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

        self.last_save = datetime.utcnow()
        print(f"💾 Session state saved at {self.last_save.strftime('%H:%M:%S')}")

    def load_state(self) -> Optional[Dict]:
        """
        Load previous session state

        Returns:
            State dictionary or None if no previous state exists
        """
        if not self.state_file.exists():
            print("📂 No previous session state found - starting fresh")
            return None

        with open(self.state_file, 'r') as f:
            state = json.load(f)

        print(f"📂 Loaded session from {state['last_updated']}")
        print(f"   Progress: {state.get('completed', 0)}/{state.get('total', 0)}")

        return state

    def estimate_remaining_time(
        self,
        completed: int,
        total: int,
        start_time: Optional[str] = None
    ) -> str:
        """
        Estimate time to completion

        Args:
            completed: Number of completed samples
            total: Total number of samples
            start_time: ISO format start time (optional)

        Returns:
            Human-readable time estimate (e.g., "3h 45m")
        """
        if completed == 0:
            return "Unknown"

        # Calculate elapsed time
        if start_time:
            start = datetime.fromisoformat(start_time)
            elapsed = (datetime.utcnow() - start).total_seconds()
        else:
            # Fallback: assume 1 hour elapsed
            elapsed = 3600

        # Calculate rate and estimate
        rate = completed / elapsed  # samples per second
        remaining = total - completed
        remaining_seconds = remaining / rate if rate > 0 else 0

        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)

        return f"{hours}h {minutes}m"

    def start_auto_save(
        self,
        get_state_func: Callable[[], Dict],
        interval_minutes: int = 10
    ) -> None:
        """
        Start background auto-save thread

        Args:
            get_state_func: Function that returns current state dict
            interval_minutes: Minutes between auto-saves
        """
        if self._auto_save_thread is not None:
            print("⚠️ Auto-save already running")
            return

        def _save_loop():
            while True:
                time.sleep(interval_minutes * 60)
                try:
                    state = get_state_func()
                    self.save_state(state)
                except Exception as e:
                    print(f"⚠️ Auto-save failed: {e}")

        self._auto_save_thread = threading.Thread(target=_save_loop, daemon=True)
        self._auto_save_thread.start()
        print(f"✅ Auto-save started (every {interval_minutes} minutes)")

    def stop_auto_save(self) -> None:
        """Stop auto-save thread"""
        # Thread will stop when main program exits (daemon=True)
        self._auto_save_thread = None
        print("🛑 Auto-save stopped")

    def _get_session_duration(self, start_time: Optional[str] = None) -> float:
        """
        Get current session duration in hours

        Args:
            start_time: ISO format start time

        Returns:
            Duration in hours
        """
        if not start_time:
            return 0.0

        try:
            start = datetime.fromisoformat(start_time)
            elapsed = (datetime.utcnow() - start).total_seconds()
            return elapsed / 3600
        except:
            return 0.0

    def estimate_session_remaining(
        self,
        start_time: Optional[str] = None,
        session_limit_hours: float = 12.0
    ) -> str:
        """
        Estimate time remaining in Colab session (12-hour limit)

        Args:
            start_time: ISO format start time
            session_limit_hours: Colab session limit (default 12)

        Returns:
            Human-readable time remaining (e.g., "8h 30m")
        """
        if not start_time:
            return f"{session_limit_hours:.0f}h 0m"

        elapsed_hours = self._get_session_duration(start_time)
        remaining_hours = max(0, session_limit_hours - elapsed_hours)

        hours = int(remaining_hours)
        minutes = int((remaining_hours % 1) * 60)

        return f"{hours}h {minutes}m"

    def get_session_metrics(self, state: Dict) -> Dict:
        """
        Calculate session performance metrics

        Args:
            state: Current session state

        Returns:
            Dictionary with metrics (rate, efficiency, etc.)
        """
        completed = state.get('completed', 0)
        total = state.get('total', 0)
        failed = state.get('failed', 0)
        start_time = state.get('start_time')

        if not start_time:
            return {}

        elapsed_hours = self._get_session_duration(start_time)
        rate = completed / elapsed_hours if elapsed_hours > 0 else 0

        return {
            "elapsed_hours": round(elapsed_hours, 2),
            "samples_per_hour": round(rate, 1),
            "success_rate": round(completed / (completed + failed) * 100, 1) if (completed + failed) > 0 else 0,
            "estimated_completion": self.estimate_remaining_time(completed, total, start_time),
            "session_remaining": self.estimate_session_remaining(start_time)
        }