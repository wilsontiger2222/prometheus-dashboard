"""Forge agent — code, testing & deployment."""

import asyncio
from agents.shared.base_agent import BaseAgent
from agents.forge.git_ops import GitOps
from agents.forge.deployer import Deployer


class ForgeAgent(BaseAgent):
    """On-demand code and deployment agent."""

    def __init__(self, **kwargs):
        super().__init__(name="forge", **kwargs)

        self._workspace = self.config.get("workspace", "agents/forge/workspace")
        self._auto_test = self.config.get("auto_test", True)
        self._test_command = self.config.get("test_command", "python -m pytest")
        self._test_timeout = self.config.get("test_timeout_seconds", 120)

        self._git = GitOps(workspace=self._workspace)
        self._deployer = Deployer(
            git_ops=self._git,
            test_runner=self._run_tests,
        )

    async def run(self):
        """Forge is on-demand — idle until dispatch."""
        while self._running:
            await asyncio.sleep(5)

    async def on_dispatch(self, message: dict):
        payload = message.get("payload", {})
        task = payload.get("task", "")
        args = payload.get("args", "")

        if task == "status":
            result = await self._git.status()
            await self.bus.publish("forge/status", result.to_dict(), sender="forge")

        elif task == "deploy":
            repo = args or "current"
            deploy_result = await self._deployer.deploy(
                repo=repo,
                run_tests=self._auto_test,
            )
            await self.bus.publish("forge/deploys", deploy_result.to_dict(), sender="forge")

        elif task == "log":
            result = await self._git.log()
            await self.bus.publish("forge/status", result.to_dict(), sender="forge")

        elif task == "pull":
            result = await self._git.pull()
            await self.bus.publish("forge/status", result.to_dict(), sender="forge")

    async def _run_tests(self) -> tuple[bool, str]:
        """Run the test suite and return (success, output)."""
        parts = self._test_command.split()
        try:
            proc = await asyncio.create_subprocess_exec(
                *parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._workspace,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._test_timeout
            )
            output = stdout.decode().strip() or stderr.decode().strip()
            return (proc.returncode == 0, output)
        except asyncio.TimeoutError:
            return (False, "Test suite timed out")
        except Exception as e:
            return (False, str(e))
