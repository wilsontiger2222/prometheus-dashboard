"""Deployment pipeline for the Forge agent.

Handles the pull → test → restart pipeline.
Each step can fail and stop the pipeline.
"""

from dataclasses import dataclass, field
from typing import Callable, Awaitable

from agents.forge.git_ops import GitOps


@dataclass
class DeployResult:
    success: bool
    repo: str
    steps: list[str] = field(default_factory=list)
    output: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "repo": self.repo,
            "steps": self.steps,
            "output": self.output,
        }


class Deployer:
    """Execute the deploy pipeline: pull → test → restart."""

    def __init__(
        self,
        git_ops: GitOps,
        test_runner: Callable[..., Awaitable[tuple[bool, str]]] | None = None,
        restarter: Callable[..., Awaitable[bool]] | None = None,
    ):
        self._git = git_ops
        self._test_runner = test_runner
        self._restarter = restarter

    async def deploy(
        self,
        repo: str,
        run_tests: bool = True,
        restart_services: list[str] | None = None,
    ) -> DeployResult:
        """Run the deploy pipeline."""
        steps = []
        outputs = []

        # Step 1: Pull latest code
        pull_result = await self._git.pull()
        steps.append("pull")
        outputs.append(f"pull: {pull_result.output}")
        if not pull_result.success:
            return DeployResult(
                success=False, repo=repo, steps=steps,
                output="\n".join(outputs),
            )

        # Step 2: Run tests (if enabled and runner available)
        if run_tests and self._test_runner:
            test_ok, test_output = await self._test_runner()
            steps.append("test")
            outputs.append(f"test: {test_output}")
            if not test_ok:
                return DeployResult(
                    success=False, repo=repo, steps=steps,
                    output="\n".join(outputs),
                )

        # Step 3: Restart services
        if restart_services and self._restarter:
            for service in restart_services:
                ok = await self._restarter(service)
                steps.append("restart")
                outputs.append(f"restart {service}: {'ok' if ok else 'failed'}")
                if not ok:
                    return DeployResult(
                        success=False, repo=repo, steps=steps,
                        output="\n".join(outputs),
                    )

        return DeployResult(
            success=True, repo=repo, steps=steps,
            output="\n".join(outputs),
        )
