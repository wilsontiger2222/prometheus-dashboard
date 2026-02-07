"""Auto-remediation actions for Watchdog agent."""

import asyncio
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RemediationResult:
    action: str
    target: str
    success: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "target": self.target, "success": self.success, "message": self.message}


async def restart_process(name: str, command: str) -> RemediationResult:
    """Restart a process using the given shell command."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *command.split(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            return RemediationResult(action="restart", target=name, success=True, message=f"Restarted {name}")
        else:
            return RemediationResult(action="restart", target=name, success=False, message=f"Failed to restart {name}: {stderr.decode().strip()}")
    except Exception as e:
        return RemediationResult(action="restart", target=name, success=False, message=f"Error restarting {name}: {e}")


def rotate_log(log_path: str, max_size_mb: int) -> RemediationResult:
    """Rotate a log file if it exceeds max_size_mb."""
    path = Path(log_path)
    if not path.exists():
        return RemediationResult(action="rotate", target=log_path, success=False, message=f"Log file not found: {log_path}")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb < max_size_mb:
        return RemediationResult(action="rotate", target=log_path, success=True, message=f"{log_path} at {size_mb:.1f}MB, below threshold ({max_size_mb}MB)")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rotated_path = f"{log_path}.{timestamp}"
    shutil.move(str(path), rotated_path)
    path.touch()

    return RemediationResult(action="rotate", target=log_path, success=True, message=f"Rotated {log_path} ({size_mb:.1f}MB) to {rotated_path}")
