"""Tests for the Watchdog agent."""

import asyncio
import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from agents.watchdog.agent import WatchdogAgent
from agents.watchdog.checks import CheckResult


@pytest.fixture
def agent(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "name": "watchdog",
        "check_interval_seconds": 1,
        "thresholds": {
            "cpu_percent": {"warning": 80, "critical": 95},
            "ram_percent": {"warning": 80, "critical": 90},
            "disk_percent": {"warning": 80, "critical": 95},
        },
        "monitored_processes": ["openclaw", "redis-server"],
        "log_max_size_mb": 100,
        "log_paths": [],
        "auto_remediate": False,
    }))
    return WatchdogAgent(config_path=str(config))


def test_watchdog_name(agent):
    assert agent.name == "watchdog"


def test_watchdog_loads_thresholds(agent):
    assert agent.config["thresholds"]["cpu_percent"]["warning"] == 80


@pytest.mark.asyncio
async def test_run_health_checks(agent):
    agent.bus = AsyncMock()
    agent.bus.publish = AsyncMock()

    with patch("agents.watchdog.agent.check_cpu") as mock_cpu, \
         patch("agents.watchdog.agent.check_ram") as mock_ram, \
         patch("agents.watchdog.agent.check_disk") as mock_disk, \
         patch("agents.watchdog.agent.check_processes") as mock_procs:

        mock_cpu.return_value = CheckResult("cpu", 45.0, "ok", "CPU at 45%")
        mock_ram.return_value = CheckResult("ram", 50.0, "ok", "RAM at 50%")
        mock_disk.return_value = CheckResult("disk", 30.0, "ok", "Disk at 30%")
        mock_procs.return_value = CheckResult("processes", 0, "ok", "All running")

        results = await agent.run_health_checks()
        assert len(results) == 4
        assert all(r.status == "ok" for r in results)


@pytest.mark.asyncio
async def test_publishes_health_to_bus(agent):
    agent.bus = AsyncMock()
    agent.bus.publish = AsyncMock()

    with patch("agents.watchdog.agent.check_cpu") as mock_cpu, \
         patch("agents.watchdog.agent.check_ram") as mock_ram, \
         patch("agents.watchdog.agent.check_disk") as mock_disk, \
         patch("agents.watchdog.agent.check_processes") as mock_procs:

        mock_cpu.return_value = CheckResult("cpu", 45.0, "ok", "CPU at 45%")
        mock_ram.return_value = CheckResult("ram", 50.0, "ok", "RAM at 50%")
        mock_disk.return_value = CheckResult("disk", 30.0, "ok", "Disk at 30%")
        mock_procs.return_value = CheckResult("processes", 0, "ok", "All running")

        await agent.run_health_checks()
        agent.bus.publish.assert_called()
        channels = [call[0][0] for call in agent.bus.publish.call_args_list]
        assert "watchdog/health" in channels


@pytest.mark.asyncio
async def test_alerts_on_critical(agent):
    agent.bus = AsyncMock()
    agent.bus.publish = AsyncMock()

    with patch("agents.watchdog.agent.check_cpu") as mock_cpu, \
         patch("agents.watchdog.agent.check_ram") as mock_ram, \
         patch("agents.watchdog.agent.check_disk") as mock_disk, \
         patch("agents.watchdog.agent.check_processes") as mock_procs:

        mock_cpu.return_value = CheckResult("cpu", 97.0, "critical", "CPU at 97%")
        mock_ram.return_value = CheckResult("ram", 50.0, "ok", "RAM at 50%")
        mock_disk.return_value = CheckResult("disk", 30.0, "ok", "Disk at 30%")
        mock_procs.return_value = CheckResult("processes", 0, "ok", "All running")

        await agent.run_health_checks()
        channels = [call[0][0] for call in agent.bus.publish.call_args_list]
        assert "watchdog/critical" in channels


@pytest.mark.asyncio
async def test_on_dispatch_status(agent):
    agent.bus = AsyncMock()
    agent.bus.publish = AsyncMock()

    with patch("agents.watchdog.agent.check_cpu") as mock_cpu, \
         patch("agents.watchdog.agent.check_ram") as mock_ram, \
         patch("agents.watchdog.agent.check_disk") as mock_disk, \
         patch("agents.watchdog.agent.check_processes") as mock_procs:

        mock_cpu.return_value = CheckResult("cpu", 45.0, "ok", "CPU at 45%")
        mock_ram.return_value = CheckResult("ram", 50.0, "ok", "RAM at 50%")
        mock_disk.return_value = CheckResult("disk", 30.0, "ok", "Disk at 30%")
        mock_procs.return_value = CheckResult("processes", 0, "ok", "All running")

        message = {"from": "prometheus", "payload": {"task": "status"}}
        await agent.on_dispatch(message)
        assert agent.bus.publish.called
