"""Health check functions for Watchdog agent."""

from dataclasses import dataclass
from typing import Any

import psutil


@dataclass
class CheckResult:
    name: str
    value: float
    status: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "status": self.status,
            "message": self.message,
        }


def _classify(value: float, thresholds: dict[str, int]) -> str:
    """Classify a metric value as ok, warning, or critical."""
    if value >= thresholds["critical"]:
        return "critical"
    elif value >= thresholds["warning"]:
        return "warning"
    return "ok"


def check_cpu(thresholds: dict[str, int]) -> CheckResult:
    """Check CPU usage percentage against thresholds."""
    value = psutil.cpu_percent(interval=1)
    status = _classify(value, thresholds)
    return CheckResult(
        name="cpu", value=value, status=status, message=f"CPU at {value}%"
    )


def check_ram(thresholds: dict[str, int]) -> CheckResult:
    """Check RAM usage percentage against thresholds."""
    mem = psutil.virtual_memory()
    value = mem.percent
    status = _classify(value, thresholds)
    return CheckResult(
        name="ram", value=value, status=status, message=f"RAM at {value}%"
    )


def check_disk(thresholds: dict[str, int], path: str = "/") -> CheckResult:
    """Check disk usage percentage against thresholds."""
    usage = psutil.disk_usage(path)
    value = usage.percent
    status = _classify(value, thresholds)
    return CheckResult(
        name="disk", value=value, status=status, message=f"Disk at {value}%"
    )


def check_processes(expected: list[str]) -> CheckResult:
    """Check that all expected processes are running."""
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
