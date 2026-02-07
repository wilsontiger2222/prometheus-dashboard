"""Tests for Watchdog remediation actions."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from agents.watchdog.remediation import (
    restart_process,
    rotate_log,
    RemediationResult,
)


def test_remediation_result():
    r = RemediationResult(action="restart", target="openclaw", success=True, message="Restarted")
    assert r.to_dict()["success"] is True


@pytest.mark.asyncio
@patch("agents.watchdog.remediation.asyncio")
async def test_restart_process_success(mock_asyncio):
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b"", b"")
    mock_proc.returncode = 0
    mock_asyncio.create_subprocess_exec = AsyncMock(return_value=mock_proc)
    mock_asyncio.subprocess = MagicMock()
    mock_asyncio.subprocess.PIPE = -1

    result = await restart_process("redis-server", "systemctl restart redis-server")
    assert result.success is True
    assert result.action == "restart"


def test_rotate_log_moves_file(tmp_path):
    log_file = tmp_path / "gateway.log"
    log_file.write_text("x" * 1000)

    result = rotate_log(str(log_file), max_size_mb=0)
    assert result.success is True
    assert not log_file.exists() or log_file.stat().st_size == 0
    rotated = list(tmp_path.glob("gateway.log.*"))
    assert len(rotated) == 1


def test_rotate_log_skips_small_file(tmp_path):
    log_file = tmp_path / "gateway.log"
    log_file.write_text("small")

    result = rotate_log(str(log_file), max_size_mb=100)
    assert result.success is True
    assert "below threshold" in result.message


def test_rotate_log_missing_file():
    result = rotate_log("/nonexistent/file.log", max_size_mb=100)
    assert result.success is False
