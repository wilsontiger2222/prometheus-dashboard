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
    """Monitors server health and auto-remediates issues."""

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

        report = [r.to_dict() for r in results]
        await self.bus.publish("watchdog/health", {"checks": report}, sender="watchdog")

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

        warnings = [r for r in results if r.status == "warning"]
        if warnings:
            for w in warnings:
                self.logger.info(f"WARNING: {w.message}")

        return results

    async def _auto_remediate_criticals(self, criticals: list[CheckResult]):
        for check in criticals:
            if check.name == "processes":
                for proc_name in self._monitored:
                    if proc_name in check.message:
                        await self._restart(proc_name)

    async def _restart(self, process_name: str):
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
        for log_path in self._log_paths:
            expanded = str(Path(log_path).expanduser())
            result = rotate_log(expanded, self._log_max_mb)
            self.logger.info(result.message)
