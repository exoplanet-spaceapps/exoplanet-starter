"""
Checkpoint Manager for Google Colab Batch Processing

Handles incremental progress with automatic recovery across Colab session disconnects.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set
import json
from datetime import datetime
import pandas as pd


class CheckpointManager:
    """
    Manages incremental progress with automatic recovery

    Features:
    - Save batch progress to Google Drive
    - Resume from last checkpoint after disconnect
    - Merge all checkpoints into final dataset
    - Track failed samples for retry
    """

    def __init__(self, drive_path: str, batch_size: int = 100):
        """
        Initialize checkpoint manager

        Args:
            drive_path: Path to Google Drive directory
            batch_size: Number of samples per batch
        """
        self.drive_path = Path(drive_path)
        self.checkpoint_dir = self.drive_path / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size

    def save_checkpoint(
        self,
        batch_id: int,
        features: Dict[int, Dict],
        failed_indices: Optional[List[int]] = None,
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        Save batch progress to Drive

        Args:
            batch_id: Starting index of batch
            features: Dictionary mapping sample index -> feature dict
            failed_indices: List of indices that failed processing
            metadata: Additional metadata to save

        Returns:
            Path to saved checkpoint file
        """
        checkpoint = {
            "checkpoint_id": f"batch_{batch_id:04d}_{batch_id + self.batch_size:04d}",
            "timestamp": datetime.utcnow().isoformat(),
            "batch_range": [batch_id, batch_id + self.batch_size],
            "completed_indices": list(features.keys()),
            "failed_indices": failed_indices or [],
            "features": features,
            "metadata": metadata or {}
        }

        checkpoint_file = self.checkpoint_dir / f"{checkpoint['checkpoint_id']}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        print(f"💾 Checkpoint saved: {checkpoint_file.name}")
        print(f"   ✅ Completed: {len(features)}")
        print(f"   ❌ Failed: {len(failed_indices) if failed_indices else 0}")

        return checkpoint_file

    def load_latest_checkpoint(self) -> Optional[Dict]:
        """
        Resume from most recent checkpoint

        Returns:
            Checkpoint dictionary or None if no checkpoints exist
        """
        checkpoints = sorted(self.checkpoint_dir.glob("batch_*.json"))
        if not checkpoints:
            print("📂 No checkpoints found - starting fresh")
            return None

        latest = checkpoints[-1]
        with open(latest, 'r') as f:
            checkpoint = json.load(f)

        print(f"📂 Loaded checkpoint: {latest.name}")
        print(f"   Timestamp: {checkpoint['timestamp']}")
        print(f"   Completed: {len(checkpoint['completed_indices'])}")

        return checkpoint

    def get_completed_indices(self) -> Set[int]:
        """
        Get all successfully processed indices across all checkpoints

        Returns:
            Set of completed sample indices
        """
        completed = set()
        for checkpoint_file in self.checkpoint_dir.glob("batch_*.json"):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                completed.update(checkpoint["completed_indices"])
        return completed

    def get_failed_indices(self) -> List[int]:
        """
        Get all failed indices across all checkpoints

        Returns:
            List of failed sample indices
        """
        failed = set()
        for checkpoint_file in self.checkpoint_dir.glob("batch_*.json"):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                failed.update(checkpoint.get("failed_indices", []))
        return sorted(failed)

    def merge_all_checkpoints(self) -> pd.DataFrame:
        """
        Merge all checkpoint features into single DataFrame

        Returns:
            DataFrame with all features from all checkpoints
        """
        all_features = {}

        checkpoint_files = sorted(self.checkpoint_dir.glob("batch_*.json"))
        print(f"\n🔄 Merging {len(checkpoint_files)} checkpoints...")

        for checkpoint_file in checkpoint_files:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                all_features.update(checkpoint["features"])

        df = pd.DataFrame.from_dict(all_features, orient='index')
        print(f"✅ Merged {len(df)} samples")

        return df

    def get_progress_summary(self, total_samples: int) -> Dict:
        """
        Get summary of processing progress

        Args:
            total_samples: Total number of samples to process

        Returns:
            Dictionary with progress statistics
        """
        completed = self.get_completed_indices()
        failed = self.get_failed_indices()

        return {
            "total_samples": total_samples,
            "completed": len(completed),
            "failed": len(failed),
            "remaining": total_samples - len(completed),
            "success_rate": len(completed) / total_samples * 100 if total_samples > 0 else 0,
            "failure_rate": len(failed) / total_samples * 100 if total_samples > 0 else 0
        }

    def cleanup_checkpoints(self) -> None:
        """
        Remove all checkpoint files (use after successful merge)
        """
        count = 0
        for checkpoint_file in self.checkpoint_dir.glob("batch_*.json"):
            checkpoint_file.unlink()
            count += 1

        print(f"🗑️ Cleaned up {count} checkpoint files")