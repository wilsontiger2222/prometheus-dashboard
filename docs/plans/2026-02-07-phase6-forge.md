# Phase 6: Forge (Code & Deployment) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an on-demand code and deployment agent that handles git operations, runs tests, manages deployments, and publishes results to the message bus.

**Architecture:** Forge is on-demand like Scout. It receives dispatch requests for git ops (clone, commit, push, status), test execution, and deployments. Git operations are wrapped in an async helper. The deployer handles the pull-test-restart pipeline. All operations publish results to `forge/deploys` so Watchdog can monitor.

**Tech Stack:** asyncio.subprocess for shell commands, BaseAgent (existing), pathlib for workspace management

---

### Task 1: Scaffold Forge package and config

**Files:**
- Create: `agents/forge/__init__.py`
- Create: `agents/forge/config.json`

**Step 1: Create `agents/forge/__init__.py`**

```python
"""Forge — code, testing & deployment agent."""
```

**Step 2: Create `agents/forge/config.json`**

```json
{
  "name": "forge",
  "workspace": "agents/forge/workspace",
  "github_user": "wilsontiger2222",
  "default_branch": "main",
  "auto_test": true,
  "deploy_on_push": false,
  "test_command": "python -m pytest",
  "test_timeout_seconds": 120
}
```

**Step 3: Commit**

```bash
git add agents/forge/__init__.py agents/forge/config.json
git commit -m "feat(forge): scaffold package and config"
```

---

### Task 2: Build git_ops module

**Files:**
- Create: `agents/forge/git_ops.py`
- Create: `agents/tests/test_forge_git_ops.py`

**Step 1: Write the failing tests**

`agents/tests/test_forge_git_ops.py`:

```python
"""Tests for the Forge git operations module."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agents.forge.git_ops import GitOps, GitResult


class TestGitResult:
    def test_git_result_success(self):
        result = GitResult(success=True, output="Already up to date.", command="git pull")
        assert result.success is True
        assert result.output == "Already up to date."

    def test_git_result_to_dict(self):
        result = GitResult(success=False, output="error: failed", command="git push")
        d = result.to_dict()
        assert d["success"] is False
        assert d["output"] == "error: failed"
        assert d["command"] == "git push"


class TestGitOps:
    @pytest.mark.asyncio
    async def test_run_command_success(self):
        ops = GitOps(workspace="/tmp/test")
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"output text", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await ops._run_command("git", "status")

        assert result.success is True
        assert result.output == "output text"

    @pytest.mark.asyncio
    async def test_run_command_failure(self):
        ops = GitOps(workspace="/tmp/test")
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"fatal: not a repo"))
        mock_proc.returncode = 128

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await ops._run_command("git", "status")

        assert result.success is False
        assert "fatal" in result.output

    @pytest.mark.asyncio
    async def test_status(self):
        ops = GitOps(workspace="/tmp/test")
        ops._run_command = AsyncMock(return_value=GitResult(
            success=True, output="On branch main\nnothing to commit", command="git status"
        ))
        result = await ops.status()
        assert result.success is True
        assert "main" in result.output

    @pytest.mark.asyncio
    async def test_pull(self):
        ops = GitOps(workspace="/tmp/test")
        ops._run_command = AsyncMock(return_value=GitResult(
            success=True, output="Already up to date.", command="git pull"
        ))
        result = await ops.pull()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_commit(self):
        ops = GitOps(workspace="/tmp/test")
        ops._run_command = AsyncMock(return_value=GitResult(
            success=True, output="[main abc1234] test commit", command="git commit"
        ))
        result = await ops.commit("test commit")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_push(self):
        ops = GitOps(workspace="/tmp/test")
        ops._run_command = AsyncMock(return_value=GitResult(
            success=True, output="To github.com:user/repo.git\nmain -> main", command="git push"
        ))
        result = await ops.push()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_clone(self):
        ops = GitOps(workspace="/tmp/test")
        ops._run_command = AsyncMock(return_value=GitResult(
            success=True, output="Cloning into 'repo'...", command="git clone"
        ))
        result = await ops.clone("https://github.com/user/repo.git")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_log(self):
        ops = GitOps(workspace="/tmp/test")
        ops._run_command = AsyncMock(return_value=GitResult(
            success=True, output="abc1234 Initial commit\ndef5678 Add feature", command="git log"
        ))
        result = await ops.log(count=2)
        assert result.success is True
        assert "abc1234" in result.output
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest agents/tests/test_forge_git_ops.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

`agents/forge/git_ops.py`:

```python
"""Git operations wrapper for the Forge agent.

Provides async wrappers around git commands using asyncio.subprocess.
All operations run in a configurable workspace directory.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GitResult:
    success: bool
    output: str
    command: str

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output": self.output,
            "command": self.command,
        }


class GitOps:
    """Async git operations in a workspace directory."""

    def __init__(self, workspace: str):
        self._workspace = Path(workspace)

    async def _run_command(self, *args: str) -> GitResult:
        """Run a shell command and return the result."""
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._workspace),
        )
        stdout, stderr = await proc.communicate()
        output = stdout.decode().strip() or stderr.decode().strip()
        return GitResult(
            success=proc.returncode == 0,
            output=output,
            command=" ".join(args),
        )

    async def status(self) -> GitResult:
        """Run git status."""
        return await self._run_command("git", "status")

    async def pull(self, branch: str = "") -> GitResult:
        """Run git pull."""
        args = ["git", "pull"]
        if branch:
            args.extend(["origin", branch])
        return await self._run_command(*args)

    async def commit(self, message: str) -> GitResult:
        """Run git commit with the given message."""
        return await self._run_command("git", "commit", "-m", message)

    async def push(self, branch: str = "") -> GitResult:
        """Run git push."""
        args = ["git", "push"]
        if branch:
            args.extend(["origin", branch])
        return await self._run_command(*args)

    async def clone(self, repo_url: str, dest: str = "") -> GitResult:
        """Clone a repository."""
        args = ["git", "clone", repo_url]
        if dest:
            args.append(dest)
        return await self._run_command(*args)

    async def add(self, *paths: str) -> GitResult:
        """Stage files for commit."""
        return await self._run_command("git", "add", *paths)

    async def log(self, count: int = 10) -> GitResult:
        """Show recent commits."""
        return await self._run_command("git", "log", f"--oneline", f"-{count}")
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest agents/tests/test_forge_git_ops.py -v`
Expected: 9 PASSED

**Step 5: Commit**

```bash
git add agents/forge/git_ops.py agents/tests/test_forge_git_ops.py
git commit -m "feat(forge): add async git operations wrapper"
```

---

### Task 3: Build deployer module

**Files:**
- Create: `agents/forge/deployer.py`
- Create: `agents/tests/test_forge_deployer.py`

**Step 1: Write the failing tests**

`agents/tests/test_forge_deployer.py`:

```python
"""Tests for the Forge deployer module."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from agents.forge.deployer import Deployer, DeployResult
from agents.forge.git_ops import GitResult


class TestDeployResult:
    def test_deploy_result_creation(self):
        result = DeployResult(
            success=True,
            repo="prometheus-dashboard",
            steps=["pull", "test", "restart"],
            output="All steps passed",
        )
        assert result.success is True
        assert result.repo == "prometheus-dashboard"

    def test_deploy_result_to_dict(self):
        result = DeployResult(
            success=False, repo="repo", steps=["pull"], output="failed"
        )
        d = result.to_dict()
        assert d["success"] is False
        assert d["repo"] == "repo"


class TestDeployer:
    @pytest.mark.asyncio
    async def test_deploy_pull_success(self):
        mock_git = AsyncMock()
        mock_git.pull = AsyncMock(return_value=GitResult(
            success=True, output="Already up to date.", command="git pull"
        ))
        mock_runner = AsyncMock(return_value=(True, "OK"))

        deployer = Deployer(git_ops=mock_git, test_runner=mock_runner)
        result = await deployer.deploy(repo="test-repo", run_tests=False, restart_services=[])

        assert result.success is True
        mock_git.pull.assert_called_once()

    @pytest.mark.asyncio
    async def test_deploy_pull_failure_stops_pipeline(self):
        mock_git = AsyncMock()
        mock_git.pull = AsyncMock(return_value=GitResult(
            success=False, output="merge conflict", command="git pull"
        ))

        deployer = Deployer(git_ops=mock_git)
        result = await deployer.deploy(repo="test-repo")

        assert result.success is False
        assert "pull" in result.steps

    @pytest.mark.asyncio
    async def test_deploy_with_tests(self):
        mock_git = AsyncMock()
        mock_git.pull = AsyncMock(return_value=GitResult(
            success=True, output="Updated", command="git pull"
        ))
        mock_runner = AsyncMock(return_value=(True, "5 passed"))

        deployer = Deployer(git_ops=mock_git, test_runner=mock_runner)
        result = await deployer.deploy(repo="test-repo", run_tests=True)

        assert result.success is True
        mock_runner.assert_called_once()
        assert "test" in result.steps

    @pytest.mark.asyncio
    async def test_deploy_test_failure_stops_pipeline(self):
        mock_git = AsyncMock()
        mock_git.pull = AsyncMock(return_value=GitResult(
            success=True, output="Updated", command="git pull"
        ))
        mock_runner = AsyncMock(return_value=(False, "2 failed"))

        deployer = Deployer(git_ops=mock_git, test_runner=mock_runner)
        result = await deployer.deploy(repo="test-repo", run_tests=True)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_deploy_with_restart(self):
        mock_git = AsyncMock()
        mock_git.pull = AsyncMock(return_value=GitResult(
            success=True, output="Updated", command="git pull"
        ))
        mock_runner = AsyncMock(return_value=(True, "OK"))
        mock_restarter = AsyncMock(return_value=True)

        deployer = Deployer(git_ops=mock_git, test_runner=mock_runner, restarter=mock_restarter)
        result = await deployer.deploy(
            repo="test-repo", run_tests=False, restart_services=["sentinel"]
        )

        assert result.success is True
        mock_restarter.assert_called_once_with("sentinel")
        assert "restart" in result.steps
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest agents/tests/test_forge_deployer.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

`agents/forge/deployer.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest agents/tests/test_forge_deployer.py -v`
Expected: 7 PASSED

**Step 5: Commit**

```bash
git add agents/forge/deployer.py agents/tests/test_forge_deployer.py
git commit -m "feat(forge): add deployment pipeline with pull-test-restart"
```

---

### Task 4: Build ForgeAgent class

**Files:**
- Create: `agents/forge/agent.py`
- Create: `agents/tests/test_forge_agent.py`

**Step 1: Write the failing tests**

`agents/tests/test_forge_agent.py`:

```python
"""Tests for the ForgeAgent."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agents.forge.agent import ForgeAgent
from agents.forge.git_ops import GitResult


@pytest.fixture
def forge():
    with patch("agents.shared.base_agent.load_agent_config") as mock_config:
        mock_config.return_value = {
            "name": "forge",
            "workspace": "/tmp/forge_test",
            "github_user": "wilsontiger2222",
            "default_branch": "main",
            "auto_test": True,
            "deploy_on_push": False,
            "test_command": "python -m pytest",
            "test_timeout_seconds": 120,
        }
        agent = ForgeAgent(config_path="dummy.json")
        agent.bus = AsyncMock()
        agent._running = True
        return agent


class TestForgeAgent:
    def test_forge_name(self, forge):
        assert forge.name == "forge"

    def test_loads_config(self, forge):
        assert forge._workspace is not None
        assert forge._auto_test is True

    @pytest.mark.asyncio
    async def test_on_dispatch_status(self, forge):
        """Dispatch with task=status returns git status."""
        forge._git.status = AsyncMock(return_value=GitResult(
            success=True, output="On branch main", command="git status"
        ))
        message = {"payload": {"task": "status"}}
        await forge.on_dispatch(message)
        forge.bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_on_dispatch_deploy(self, forge):
        """Dispatch with task=deploy runs the deploy pipeline."""
        forge._deployer.deploy = AsyncMock(return_value=MagicMock(
            success=True,
            to_dict=lambda: {"success": True, "repo": "test", "steps": ["pull"]},
        ))
        message = {"payload": {"task": "deploy", "args": "prometheus-dashboard"}}
        await forge.on_dispatch(message)
        forge._deployer.deploy.assert_called_once()
        forge.bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_on_dispatch_git_log(self, forge):
        """Dispatch with task=log returns git log."""
        forge._git.log = AsyncMock(return_value=GitResult(
            success=True, output="abc123 commit msg", command="git log"
        ))
        message = {"payload": {"task": "log"}}
        await forge.on_dispatch(message)
        forge.bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_run_is_idle(self, forge):
        """Forge run() should just idle (on-demand agent)."""
        forge._running = False
        await forge.run()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest agents/tests/test_forge_agent.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

`agents/forge/agent.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest agents/tests/test_forge_agent.py -v`
Expected: 6 PASSED

**Step 5: Commit**

```bash
git add agents/forge/agent.py agents/tests/test_forge_agent.py
git commit -m "feat(forge): add ForgeAgent with git ops and deploy pipeline"
```

---

### Task 5: Add Forge `__main__.py` entry point

**Files:**
- Create: `agents/forge/__main__.py`

**Step 1: Write `__main__.py`**

```python
"""Run the Forge agent: python -m agents.forge"""

import asyncio
import sys
from pathlib import Path

from agents.forge.agent import ForgeAgent

DEFAULT_CONFIG = str(Path(__file__).parent / "config.json")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    agent = ForgeAgent(config_path=config_path)

    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print("\nForge shutting down...")


if __name__ == "__main__":
    main()
```

**Step 2: Verify import works**

Run: `python -c "from agents.forge.agent import ForgeAgent; print('OK')"`

**Step 3: Commit**

```bash
git add agents/forge/__main__.py
git commit -m "feat(forge): add __main__.py entry point"
```

---

### Task 6: Final test run and push

**Step 1: Run full test suite**

Run: `python -m pytest agents/tests/ -v`
Expected: All tests pass (no regressions)

**Step 2: Commit plan doc**

```bash
git add docs/plans/2026-02-07-phase6-forge.md
git commit -m "docs: add Phase 6 Forge implementation plan"
```

**Step 3: Push to GitHub**

```bash
git push
```
