# Phase 2: Watchdog Agent — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the Watchdog agent — a proactive server health monitor that checks CPU, RAM, disk, and processes every 60 seconds, auto-restarts crashed services, rotates logs, and alerts via Telegram on critical events.

**Architecture:** Watchdog inherits from `BaseAgent` and runs a continuous health check loop. Each check type (CPU, RAM, disk, processes) is a standalone function in `checks/` that returns a `CheckResult`. The main loop collects results, publishes to the bus, and triggers auto-remediation when thresholds are breached. Almost zero AI usage — this is pure Python scripts.

**Tech Stack:** Python 3.10+, `psutil` (system metrics), asyncio, BaseAgent (from Phase 1)

---

### Task 1: Add psutil dependency and Watchdog config

**Files:**
- Modify: `agents/requirements.txt`
- Create: `agents/watchdog/config.json`

**Step 1: Update requirements**

Add to `agents/requirements.txt`:
```
redis>=5.0.0
aiohttp>=3.9.0
psutil>=5.9.0
```

**Step 2: Create default config**

`agents/watchdog/config.json`:
```json
{
  "name": "watchdog",
  "check_interval_seconds": 60,
  "thresholds": {
    "cpu_percent": {"warning": 80, "critical": 95},
    "ram_percent": {"warning": 80, "critical": 90},
    "disk_percent": {"warning": 80, "critical": 95}
  },
  "monitored_processes": ["openclaw", "redis-server"],
  "log_max_size_mb": 100,
  "log_paths": ["~/gateway.log"],
  "auto_remediate": true
}
```

**Step 3: Install psutil**

Run: `python -m pip install psutil`

**Step 4: Commit**

```bash
git add agents/requirements.txt agents/watchdog/config.json
git commit -m "feat(watchdog): add config and psutil dependency"
```

---

### Task 2: Build health check functions

**Files:**
- Create: `agents/watchdog/checks.py`
- Create: `agents/tests/test_watchdog_checks.py`

**Step 1: Write the failing tests**

`agents/tests/test_watchdog_checks.py`:
```python
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


# --- CheckResult ---

def test_check_result_ok():
    r = CheckResult(name="cpu", value=45.0, status="ok", message="CPU at 45%")
    assert r.name == "cpu"
    assert r.status == "ok"
    assert r.to_dict()["value"] == 45.0


def test_check_result_to_dict():
    r = CheckResult(name="ram", value=85.0, status="warning", message="RAM high")
    d = r.to_dict()
    assert d == {
        "name": "ram",
        "value": 85.0,
        "status": "warning",
        "message": "RAM high",
    }


# --- CPU ---

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


# --- RAM ---

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


# --- Disk ---

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


# --- Processes ---

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
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest agents/tests/test_watchdog_checks.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write the implementation**

`agents/watchdog/checks.py`:
```python
"""Health check functions for Watchdog agent.

Each function takes thresholds and returns a CheckResult.
All system calls go through psutil and are mockable for testing.
"""

from dataclasses import dataclass
from typing import Any

import psutil


@dataclass
class CheckResult:
    name: str
    value: float
    status: str  # "ok", "warning", "critical"
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "status": self.status,
            "message": self.message,
        }


def _classify(value: float, thresholds: dict[str, int]) -> str:
    if value >= thresholds["critical"]:
        return "critical"
    elif value >= thresholds["warning"]:
        return "warning"
    return "ok"


def check_cpu(thresholds: dict[str, int]) -> CheckResult:
    value = psutil.cpu_percent(interval=1)
    status = _classify(value, thresholds)
    return CheckResult(
        name="cpu",
        value=value,
        status=status,
        message=f"CPU at {value}%",
    )


def check_ram(thresholds: dict[str, int]) -> CheckResult:
    mem = psutil.virtual_memory()
    value = mem.percent
    status = _classify(value, thresholds)
    return CheckResult(
        name="ram",
        value=value,
        status=status,
        message=f"RAM at {value}%",
    )


def check_disk(thresholds: dict[str, int], path: str = "/") -> CheckResult:
    usage = psutil.disk_usage(path)
    value = usage.percent
    status = _classify(value, thresholds)
    return CheckResult(
        name="disk",
        value=value,
        status=status,
        message=f"Disk at {value}%",
    )


def check_processes(expected: list[str]) -> CheckResult:
    running = set()
    for proc in psutil.process_iter(["name", "pid", "status"]):
        running.add(proc.info["name"])

    missing = [p for p in expected if p not in running]
    if missing:
        return CheckResult(
            name="processes",
            value=len(missing),
            status="critical",
            message=f"Missing processes: {', '.join(missing)}",
        )
    return CheckResult(
        name="processes",
        value=0,
        status="ok",
        message=f"All {len(expected)} monitored processes running",
    )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest agents/tests/test_watchdog_checks.py -v`
Expected: All 11 tests PASS

**Step 5: Commit**

```bash
git add agents/watchdog/checks.py agents/tests/test_watchdog_checks.py
git commit -m "feat(watchdog): add health check functions with tests"
```

---

### Task 3: Build remediation functions

**Files:**
- Create: `agents/watchdog/remediation.py`
- Create: `agents/tests/test_watchdog_remediation.py`

**Step 1: Write the failing tests**

`agents/tests/test_watchdog_remediation.py`:
```python
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


@patch("agents.watchdog.remediation.asyncio")
async def test_restart_process_success(mock_asyncio):
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b"", b"")
    mock_proc.returncode = 0
    mock_asyncio.create_subprocess_exec = AsyncMock(return_value=mock_proc)

    result = await restart_process("redis-server", "systemctl restart redis-server")
    assert result.success is True
    assert result.action == "restart"


def test_rotate_log_moves_file(tmp_path):
    log_file = tmp_path / "gateway.log"
    log_file.write_text("x" * 1000)

    result = rotate_log(str(log_file), max_size_mb=0)  # 0 MB forces rotation
    assert result.success is True
    assert not log_file.exists() or log_file.stat().st_size == 0
    # Rotated file should exist with timestamp suffix
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
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest agents/tests/test_watchdog_remediation.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write the implementation**

`agents/watchdog/remediation.py`:
```python
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
        return {
            "action": self.action,
            "target": self.target,
            "success": self.success,
            "message": self.message,
        }


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
            return RemediationResult(
                action="restart",
                target=name,
                success=True,
                message=f"Restarted {name}",
            )
        else:
            return RemediationResult(
                action="restart",
                target=name,
                success=False,
                message=f"Failed to restart {name}: {stderr.decode().strip()}",
            )
    except Exception as e:
        return RemediationResult(
            action="restart",
            target=name,
            success=False,
            message=f"Error restarting {name}: {e}",
        )


def rotate_log(log_path: str, max_size_mb: int) -> RemediationResult:
    """Rotate a log file if it exceeds max_size_mb."""
    path = Path(log_path)
    if not path.exists():
        return RemediationResult(
            action="rotate",
            target=log_path,
            success=False,
            message=f"Log file not found: {log_path}",
        )

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb < max_size_mb:
        return RemediationResult(
            action="rotate",
            target=log_path,
            success=True,
            message=f"{log_path} at {size_mb:.1f}MB, below threshold ({max_size_mb}MB)",
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rotated_path = f"{log_path}.{timestamp}"
    shutil.move(str(path), rotated_path)
    path.touch()  # Create fresh empty log

    return RemediationResult(
        action="rotate",
        target=log_path,
        success=True,
        message=f"Rotated {log_path} ({size_mb:.1f}MB) to {rotated_path}",
    )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest agents/tests/test_watchdog_remediation.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add agents/watchdog/remediation.py agents/tests/test_watchdog_remediation.py
git commit -m "feat(watchdog): add remediation functions (restart, log rotation)"
```

---

### Task 4: Build the Watchdog agent class

**Files:**
- Create: `agents/watchdog/agent.py`
- Create: `agents/tests/test_watchdog_agent.py`

**Step 1: Write the failing tests**

`agents/tests/test_watchdog_agent.py`:
```python
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
        # Should publish to watchdog/health
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
        # Should publish to watchdog/critical AND alert telegram
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

        message = {
            "from": "prometheus",
            "payload": {"task": "status"},
        }
        await agent.on_dispatch(message)
        # Should have run checks and published results
        assert agent.bus.publish.called
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest agents/tests/test_watchdog_agent.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write the implementation**

`agents/watchdog/agent.py`:
```python
"""Watchdog agent — proactive server health monitoring."""

import asyncio
from pathlib import Path

from agents.shared.base_agent import BaseAgent
from agents.watchdog.checks import (
    CheckResult,
    check_cpu,
    check_ram,
    check_disk,
    check_processes,
)
from agents.watchdog.remediation import restart_process, rotate_log


class WatchdogAgent(BaseAgent):
    """Monitors server health and auto-remediates issues.

    Health check loop runs every check_interval_seconds.
    Publishes results to watchdog/health (normal) and watchdog/critical (alerts).
    Auto-restarts crashed processes when auto_remediate is True.
    """

    def __init__(self, **kwargs):
        super().__init__(name="watchdog", **kwargs)
        self._check_interval = self.config.get("check_interval_seconds", 60)
        self._thresholds = self.config.get("thresholds", {})
        self._monitored = self.config.get("monitored_processes", [])
        self._log_paths = self.config.get("log_paths", [])
        self._log_max_mb = self.config.get("log_max_size_mb", 100)
        self._auto_remediate = self.config.get("auto_remediate", True)

    async def run(self):
        """Main health check loop."""
        while self._running:
            try:
                await self.run_health_checks()
            except Exception as e:
                self.logger.error(f"Health check cycle failed: {e}")
            await asyncio.sleep(self._check_interval)

    async def on_dispatch(self, message: dict):
        """Handle dispatch commands from Prometheus."""
        payload = message.get("payload", {})
        task = payload.get("task", "")

        if task == "status":
            await self.run_health_checks()
        elif task == "restart" and "process" in payload:
            await self._restart(payload["process"])
        elif task == "rotate_logs":
            await self._rotate_all_logs()

    async def run_health_checks(self) -> list[CheckResult]:
        """Run all health checks and publish results."""
        results = [
            check_cpu(self._thresholds.get("cpu_percent", {"warning": 80, "critical": 95})),
            check_ram(self._thresholds.get("ram_percent", {"warning": 80, "critical": 90})),
            check_disk(self._thresholds.get("disk_percent", {"warning": 80, "critical": 95})),
            check_processes(self._monitored),
        ]

        # Publish full health report
        report = [r.to_dict() for r in results]
        await self.bus.publish("watchdog/health", {"checks": report}, sender="watchdog")

        # Handle critical results
        criticals = [r for r in results if r.status == "critical"]
        if criticals:
            alert_msg = "; ".join(r.message for r in criticals)
            await self.bus.publish(
                "watchdog/critical",
                {"alerts": [r.to_dict() for r in criticals]},
                sender="watchdog",
            )
            self.logger.warning(f"CRITICAL: {alert_msg}")

            if self._auto_remediate:
                await self._auto_remediate_criticals(criticals)

        # Log warnings
        warnings = [r for r in results if r.status == "warning"]
        if warnings:
            for w in warnings:
                self.logger.info(f"WARNING: {w.message}")

        return results

    async def _auto_remediate_criticals(self, criticals: list[CheckResult]):
        """Attempt auto-remediation for critical issues."""
        for check in criticals:
            if check.name == "processes":
                # Restart missing processes
                for proc_name in self._monitored:
                    if proc_name in check.message:
                        await self._restart(proc_name)

    async def _restart(self, process_name: str):
        """Restart a process by name."""
        command = f"systemctl restart {process_name}"
        result = await restart_process(process_name, command)
        if result.success:
            self.logger.info(f"Restarted {process_name}")
        else:
            self.logger.error(f"Failed to restart {process_name}: {result.message}")
            await self.bus.publish(
                "watchdog/critical",
                {"remediation_failed": result.to_dict()},
                sender="watchdog",
            )

    async def _rotate_all_logs(self):
        """Rotate all monitored log files."""
        for log_path in self._log_paths:
            expanded = str(Path(log_path).expanduser())
            result = rotate_log(expanded, self._log_max_mb)
            self.logger.info(result.message)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest agents/tests/test_watchdog_agent.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add agents/watchdog/agent.py agents/tests/test_watchdog_agent.py
git commit -m "feat(watchdog): add WatchdogAgent with health checks and auto-remediation"
```

---

### Task 5: Add `__main__.py` entry point

**Files:**
- Create: `agents/watchdog/__main__.py`

**Step 1: Write the entry point**

`agents/watchdog/__main__.py`:
```python
"""Run the Watchdog agent: python -m agents.watchdog"""

import asyncio
import sys
from pathlib import Path

from agents.watchdog.agent import WatchdogAgent

DEFAULT_CONFIG = str(Path(__file__).parent / "config.json")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    agent = WatchdogAgent(config_path=config_path)

    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print("\nWatchdog shutting down...")


if __name__ == "__main__":
    main()
```

**Step 2: Verify it imports correctly**

Run: `python -c "from agents.watchdog.agent import WatchdogAgent; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add agents/watchdog/__main__.py
git commit -m "feat(watchdog): add __main__.py entry point"
```

---

### Task 6: Run all tests and push

**Step 1: Run full test suite**

Run: `python -m pytest agents/tests/ -v`
Expected: All Phase 1 tests still pass + all new Watchdog tests pass

**Step 2: Push to GitHub**

```bash
git push origin main
```

---

## Summary

After Phase 2 is complete, you will have:

| Component | File | Purpose |
|-----------|------|---------|
| **Health Checks** | `agents/watchdog/checks.py` | CPU, RAM, disk, process monitoring |
| **Remediation** | `agents/watchdog/remediation.py` | Auto-restart, log rotation |
| **Agent** | `agents/watchdog/agent.py` | Main agent loop, alert logic |
| **Config** | `agents/watchdog/config.json` | Thresholds, monitored processes |
| **Entry Point** | `agents/watchdog/__main__.py` | `python -m agents.watchdog` |
| **Tests** | `agents/tests/test_watchdog_*.py` | 22 tests total |

**To run on server:**
```bash
cd ~/prometheus-dashboard
pip install -r agents/requirements.txt
python -m agents.watchdog
```

**Next:** Phase 3 — Sentinel (Trading Agent)
