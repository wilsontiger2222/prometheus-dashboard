# OpenClaw Hivemind Agents

Subagent system for OpenClaw. See `docs/plans/2026-02-07-openclaw-hivemind-design.md` for full architecture.

## Setup

1. Install Redis: `sudo apt install redis-server`
2. Install Python deps: `pip install -r agents/requirements.txt`
3. Start Redis: `sudo systemctl start redis-server`

## Running an agent

```bash
python -m agents.watchdog
```
