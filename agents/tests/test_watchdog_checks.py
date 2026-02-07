"""Tests for Watchdog health check functions."""

import pytest
from unittest.mock import patch, MagicMock
from agents.watchdog.checks import (
    CheckResult,
    check_cpu,
    check_ram,
    check_disk,
    check_processes,
)


def test_check_result_ok():
    r = CheckResult(name="cpu", value=45.0, status="ok", message="CPU at 45%")
    assert r.name == "cpu"
    assert r.status == "ok"
    assert r.to_dict()["value"] == 45.0


def test_check_result_to_dict():
    r = CheckResult(name="ram", value=85.0, status="warning", message="RAM high")
    d = r.to_dict()
    assert d == {"name": "ram", "value": 85.0, "status": "warning", "message": "RAM high"}


@patch("agents.watchdog.checks.psutil")
def test_check_cpu_ok(mock_psutil):
    mock_psutil.cpu_percent.return_value = 45.0
    thresholds = {"warning": 80, "critical": 95}
    result = check_cpu(thresholds)
    assert result.status == "ok"
    assert result.value == 45.0


@patch("agents.watchdog.checks.psutil")
def test_check_cpu_warning(mock_psutil):
    mock_psutil.cpu_percent.return_value = 85.0
    thresholds = {"warning": 80, "critical": 95}
    result = check_cpu(thresholds)
    assert result.status == "warning"


@patch("agents.watchdog.checks.psutil")
def test_check_cpu_critical(mock_psutil):
    mock_psutil.cpu_percent.return_value = 97.0
    thresholds = {"warning": 80, "critical": 95}
    result = check_cpu(thresholds)
    assert result.status == "critical"


@patch("agents.watchdog.checks.psutil")
def test_check_ram_ok(mock_psutil):
    mock_psutil.virtual_memory.return_value = MagicMock(percent=50.0)
    thresholds = {"warning": 80, "critical": 90}
    result = check_ram(thresholds)
    assert result.status == "ok"
    assert result.value == 50.0


@patch("agents.watchdog.checks.psutil")
def test_check_ram_critical(mock_psutil):
    mock_psutil.virtual_memory.return_value = MagicMock(percent=92.0)
    thresholds = {"warning": 80, "critical": 90}
    result = check_ram(thresholds)
    assert result.status == "critical"


@patch("agents.watchdog.checks.psutil")
def test_check_disk_ok(mock_psutil):
    mock_psutil.disk_usage.return_value = MagicMock(percent=40.0)
    thresholds = {"warning": 80, "critical": 95}
    result = check_disk(thresholds)
    assert result.status == "ok"
    assert result.value == 40.0


@patch("agents.watchdog.checks.psutil")
def test_check_disk_warning(mock_psutil):
    mock_psutil.disk_usage.return_value = MagicMock(percent=82.0)
    thresholds = {"warning": 80, "critical": 95}
    result = check_disk(thresholds)
    assert result.status == "warning"


@patch("agents.watchdog.checks.psutil")
def test_check_processes_all_running(mock_psutil):
    proc1 = MagicMock()
    proc1.info = {"name": "openclaw", "pid": 100, "status": "running"}
    proc2 = MagicMock()
    proc2.info = {"name": "redis-server", "pid": 200, "status": "running"}
    mock_psutil.process_iter.return_value = [proc1, proc2]
    result = check_processes(["openclaw", "redis-server"])
    assert result.status == "ok"


@patch("agents.watchdog.checks.psutil")
def test_check_processes_missing(mock_psutil):
    proc1 = MagicMock()
    proc1.info = {"name": "openclaw", "pid": 100, "status": "running"}
    mock_psutil.process_iter.return_value = [proc1]
    result = check_processes(["openclaw", "redis-server"])
    assert result.status == "critical"
    assert "redis-server" in result.message
