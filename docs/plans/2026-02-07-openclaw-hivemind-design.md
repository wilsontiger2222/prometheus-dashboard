# OpenClaw Hivemind: Subagent Architecture Design

**Date:** 2026-02-07
**Status:** Approved
**Author:** Wilson + Claude

---

## Overview

Transform OpenClaw from a single sequential AI assistant into a multi-agent system ("Hivemind") with specialized subagents that run in parallel, act proactively, and communicate through a lightweight message bus.

### Goals

1. **Parallelism** — Multiple agents working simultaneously instead of one sequential pipeline
2. **Proactive autonomy** — Agents that monitor, detect, and act without being asked
3. **Specialized depth** — Domain-specific agents that excel at one thing

### Constraints

- Server: Digital Ocean Ubuntu droplet (upgrade to 4-8GB RAM)
- AI model: OpenAI Codex GPT-5.2 via ChatGPT OAuth
- Interface: Telegram bot (@MakeMeFkingRichbot)
- Gateway: ws://127.0.0.1:18789
- Token budget: Minimize AI calls — use pure scripts wherever possible

---

## Architecture

```
              YOU (Telegram)
                   │
              ┌────┴────┐
              │PROMETHEUS│  Orchestrator / Dispatcher
              │ (brain)  │
              └────┬─────┘
                   │
        ┌──────────┼──────────────────┐
        │    MESSAGE BUS (Redis)      │
        │    lightweight pub/sub      │
        └──┬────┬────┬────┬────┬─────┘
           │    │    │    │    │
           ▼    ▼    ▼    ▼    ▼
        SENT  WATCH SCOUT FORGE HERALD
```

### Components

| Component | Role | Type |
|-----------|------|------|
| **Prometheus** | Orchestrator — routes messages, dispatches agents, aggregates responses | Core |
| **Sentinel** | Trading & market intelligence | Proactive |
| **Watchdog** | Server health & infrastructure ops | Proactive |
| **Scout** | Research & intelligence gathering | On-demand |
| **Forge** | Code, testing & deployment | On-demand |
| **Herald** | Social media & content creation | Hybrid |
| **Redis Bus** | Inter-agent communication | Infrastructure |

---

## Message Bus

### Technology: Redis Pub/Sub

- ~10MB memory footprint
- Built-in pub/sub channels
- Message persistence via Redis Streams
- Install: `apt install redis-server`

### Channels

```
prometheus/dispatch    → all agents listen (task assignments)
sentinel/alerts       → Prometheus, Herald listen
sentinel/trades       → Prometheus listens
watchdog/health       → Prometheus listens
watchdog/critical     → Prometheus, Telegram listen
scout/reports         → requester listens
forge/deploys         → Prometheus, Watchdog listen
herald/posts          → Prometheus listens
agents/heartbeat      → Prometheus listens (liveness)
```

### Message Format

All messages follow a standard envelope:

```json
{
  "from": "scout",
  "to": "sentinel",
  "channel": "scout/reports",
  "timestamp": "2026-02-07T14:30:00Z",
  "type": "report",
  "payload": {
    "topic": "NVDA earnings",
    "summary": "Beat estimates by 12%...",
    "sources": ["https://..."]
  },
  "request_id": "uuid-here"
}
```

### Example Flow

**User:** "Research NVIDIA and adjust my trading strategy"

1. **Prometheus** receives Telegram message, detects dual intent
2. Publishes to `prometheus/dispatch`: `{task: "research", target: "NVIDIA", then: "sentinel"}`
3. **Scout** picks it up, scrapes earnings data, summarizes (1 AI call)
4. Scout publishes to `scout/reports`: `{report: "NVDA beat earnings...", forward_to: "sentinel"}`
5. **Sentinel** picks up report, adjusts breakout thresholds (1 AI call)
6. Sentinel publishes to `sentinel/alerts`: `{action: "adjusted NVDA strategy"}`
7. **Prometheus** aggregates and sends final summary to Telegram

**Total AI calls:** 3 (routing + summarizing + analyzing). Everything else is JSON over Redis.

---

## Subagent Specifications

### Base Agent Class

All agents inherit from `base_agent.py`:

```python
class BaseAgent:
    def __init__(self, name, config_path):
        self.name = name
        self.bus = RedisBus()
        self.config = load_config(config_path)

    def run(self):
        """Main loop — override per agent."""
        pass

    def on_dispatch(self, message):
        """Handle task from Prometheus."""
        pass

    def on_message(self, channel, message):
        """Handle message from bus."""
        pass

    def heartbeat(self):
        """Publish liveness to agents/heartbeat every 30s."""
        pass

    def call_ai(self, prompt, context=None):
        """Make AI call through OpenClaw gateway. Use sparingly."""
        pass

    def publish(self, channel, payload):
        """Publish message to bus."""
        pass

    def alert_telegram(self, message):
        """Send urgent message directly to Telegram."""
        pass
```

---

### Prometheus (Orchestrator)

**Purpose:** Receive user messages, detect intent, dispatch to agents, aggregate responses.

**Process type:** Always running
**AI usage:** Every inbound Telegram message (for intent routing)
**Token optimization:** Keyword shortcuts bypass AI entirely

**Keyword shortcuts (zero tokens):**
- `/trade [msg]` → Sentinel
- `/status` → Watchdog
- `/research [topic]` → Scout
- `/deploy [repo]` → Forge
- `/post [content]` → Herald
- `/all` → broadcast to all agents

**Responsibilities:**
- Parse Telegram messages and detect which agent(s) to invoke
- Fire multiple agents in parallel when needed
- Aggregate multi-agent responses into a single Telegram reply
- Track agent liveness via heartbeat channel
- Feed all activity to Prometheus Dashboard via WebSocket
- Manage agent lifecycle (start, stop, restart)

---

### Sentinel (Trading & Market Intelligence)

**Purpose:** Autonomous market monitoring and trade execution.

**Process type:** Always running (monitoring loop)
**AI usage:** Low — only for pattern analysis and strategy adjustment
**Loop interval:** Configurable per asset (default: 30s for crypto, 60s for stocks)

**Script layer (no tokens):**
- Price feed polling (exchange APIs)
- Threshold-based alerts (`if price > breakout_level`)
- Trade execution via exchange API
- Position tracking and P&L calculation
- Scheduled backtesting jobs

**AI layer (tokens):**
- Analyzing unusual volume or price patterns
- Adjusting strategy parameters based on Scout research
- Responding to natural language trading questions from user
- Generating trade summaries

**Strategies directory:**
```
sentinel/strategies/
├── breakout.py          ← current strategy (migrate from existing bot)
├── mean_reversion.py
├── momentum.py
└── strategy_base.py     ← base class for strategies
```

**Config:**
```json
{
  "watchlist": ["BTC", "ETH", "NVDA", "AAPL"],
  "mode": "paper",
  "risk_limit_per_trade": 0.02,
  "max_positions": 5,
  "alert_threshold": 0.03
}
```

---

### Watchdog (Server & Infrastructure Ops)

**Purpose:** Keep the server healthy and secure. Auto-remediate issues.

**Process type:** Always running (60s health check loop)
**AI usage:** Almost none — 95% pure scripts

**Health checks:**
- CPU usage (alert > 80%, action > 95%)
- RAM usage (alert > 80%, action > 90%)
- Disk usage (alert > 80%)
- Process monitoring (OpenClaw gateway, Redis, Sentinel, other agents)
- Network connectivity
- Failed SSH attempts (parse auth.log)

**Auto-remediation:**
- Restart crashed services (OpenClaw gateway, agents, Redis)
- Log rotation when gateway.log exceeds 100MB
- Clear temp files and old caches
- Kill runaway processes consuming > 50% CPU for > 5min

**Alerting tiers:**
1. **Info** → Prometheus dashboard only
2. **Warning** → Dashboard + periodic Telegram digest
3. **Critical** → Immediate Telegram alert + auto-remediate

**Config:**
```json
{
  "check_interval_seconds": 60,
  "monitored_processes": ["openclaw", "redis-server", "sentinel", "scout"],
  "cpu_alert_threshold": 80,
  "ram_alert_threshold": 80,
  "disk_alert_threshold": 80,
  "log_max_size_mb": 100
}
```

---

### Scout (Research & Intelligence Gathering)

**Purpose:** Web scraping, data gathering, summarization. Callable by user or other agents.

**Process type:** On-demand (spawns per request, idle otherwise)
**AI usage:** Medium — summarization requires AI

**Capabilities:**
- Web scraping via BeautifulSoup (static) and Playwright (dynamic/JS)
- Article and document summarization
- Multi-source aggregation (combine info from multiple URLs)
- Data extraction (tables, prices, stats)

**Cache system:**
```
scout/cache/
├── index.json           ← topic → cache file mapping
├── nvda_earnings.json   ← cached research, TTL: 24h
└── btc_analysis.json
```
- Cache TTL configurable per topic type (news: 1h, fundamentals: 24h, static: 7d)
- Other agents check cache before requesting new research

**Inter-agent integration:**
- Sentinel requests: "research [ticker] [topic]"
- Herald requests: "find trending topics in [niche]"
- Forge requests: "find documentation for [library]"

**Config:**
```json
{
  "default_cache_ttl_hours": 24,
  "max_cache_size_mb": 500,
  "scraper": "beautifulsoup",
  "fallback_scraper": "playwright",
  "max_concurrent_scrapes": 3
}
```

---

### Forge (Code & Deployment)

**Purpose:** Write, test, and deploy code. Manage git repos and CI pipelines.

**Process type:** On-demand
**AI usage:** Medium — code generation and review require AI

**Capabilities:**
- Code generation and modification
- Git operations (clone, commit, push, PR)
- Deployment pipeline: GitHub push → server pull → service restart
- Test execution and reporting
- Dependency management

**Deploy pipeline:**
```
User request → Forge writes code → runs tests → commits to GitHub
  → pulls on server → restarts affected services → reports status
```

**Integration with Watchdog:**
- Forge publishes to `forge/deploys` after every deployment
- Watchdog monitors newly deployed services for crashes

**Config:**
```json
{
  "workspace": "~/.openclaw/workspace/",
  "github_user": "wilsontiger2222",
  "default_branch": "main",
  "auto_test": true,
  "deploy_on_push": false
}
```

---

### Herald (Social Media & Content)

**Purpose:** Content creation, scheduling, and publishing across platforms.

**Process type:** Hybrid — scheduled posting loop + on-demand creation
**AI usage:** Medium — content creation requires AI

**Capabilities:**
- Content generation from templates and triggers
- Scheduled posting (cron-based)
- Multi-platform support (Twitter/X, Discord, etc.)
- Trend monitoring
- Review queue: drafts sent to Telegram for approval before posting

**Trigger-based content:**
- Sentinel trading win → Herald drafts a post
- Forge successful deploy → Herald announces project update
- Scout finds trending topic → Herald creates relevant content

**Review flow:**
```
Trigger/Request → Herald drafts content → sends draft to Telegram
  → User approves/edits → Herald publishes
```

**Config:**
```json
{
  "platforms": ["twitter"],
  "require_approval": true,
  "scheduled_posts_per_day": 3,
  "content_templates_dir": "herald/templates/"
}
```

---

## Directory Structure

```
~/.openclaw/agents/
├── prometheus/
│   ├── dispatcher.py        ← main orchestrator loop
│   ├── router.py            ← intent detection + agent routing
│   ├── aggregator.py        ← combine multi-agent responses
│   └── config.json
├── sentinel/
│   ├── agent.py             ← main trading loop
│   ├── monitor.py           ← price feed monitoring
│   ├── executor.py          ← trade execution
│   ├── strategies/
│   │   ├── breakout.py
│   │   ├── strategy_base.py
│   │   └── ...
│   └── config.json
├── watchdog/
│   ├── agent.py             ← main health check loop
│   ├── checks/
│   │   ├── cpu.py
│   │   ├── memory.py
│   │   ├── disk.py
│   │   ├── processes.py
│   │   └── security.py
│   ├── remediations.py      ← auto-fix actions
│   └── config.json
├── scout/
│   ├── agent.py             ← request handler
│   ├── scraper.py           ← web scraping engine
│   ├── summarizer.py        ← AI summarization
│   ├── cache/
│   └── config.json
├── forge/
│   ├── agent.py             ← code/deploy handler
│   ├── deployer.py          ← deployment pipeline
│   ├── git_ops.py           ← git operations
│   └── config.json
├── herald/
│   ├── agent.py             ← content/posting handler
│   ├── scheduler.py         ← cron-based posting
│   ├── templates/
│   └── config.json
└── shared/
    ├── bus.py               ← Redis pub/sub wrapper
    ├── base_agent.py        ← base class all agents inherit
    ├── logger.py            ← unified logging → Prometheus dashboard
    ├── ai_client.py         ← shared AI call wrapper (token tracking)
    └── config.py            ← shared configuration loader
```

---

## Token Usage Estimates

| Agent | Proactive Calls/Day | On-Demand Calls/Day | Estimated Tokens/Day |
|-------|---------------------|---------------------|----------------------|
| Prometheus | 0 (keyword bypass) | 20-50 (routing) | ~5,000-15,000 |
| Sentinel | 5-10 (analysis) | 5-10 (questions) | ~3,000-10,000 |
| Watchdog | 0 | ~1 (rare) | ~100 |
| Scout | 0 | 5-15 (summaries) | ~5,000-20,000 |
| Forge | 0 | 3-8 (code tasks) | ~5,000-15,000 |
| Herald | 3-5 (drafts) | 2-5 (on request) | ~3,000-10,000 |
| **Total** | | | **~20,000-70,000/day** |

Note: These are rough estimates. Actual usage depends on how actively you interact. Watchdog is essentially free. Keyword shortcuts on Prometheus reduce routing tokens significantly.

---

## Prometheus Dashboard Integration

Every agent feeds into the Prometheus Dashboard in real-time:

**WebSocket events pushed to dashboard:**
```json
{
  "agent": "sentinel",
  "event": "trade_executed",
  "data": {"ticker": "BTC", "action": "buy", "price": 105000},
  "timestamp": "2026-02-07T14:30:00Z"
}
```

**Dashboard behavior:**
- Core orb drifts toward the active module (Chat, Trading, Web, Files)
- Agent nodes glow when active, dim when idle
- Heartbeat pulse on each agent node (alive/dead indicator)
- Activity log stream at the bottom
- Alert flash on critical Watchdog events

---

## Implementation Phases

| Phase | What | Days | Server Req |
|-------|------|------|------------|
| **1. Foundation** | base_agent, bus.py, Redis, directory structure | 1-2 | 4GB |
| **2. Watchdog** | Health checks, auto-restart, alerts | 3 | 4GB |
| **3. Sentinel** | Migrate trading bot, monitoring loop, AI layer | 4-6 | 4GB |
| **4. Scout** | Scraping, summarization, cache, inter-agent hook | 7-8 | 4GB |
| **5. Prometheus** | Orchestrator, routing, parallel dispatch, dashboard feed | 9-10 | 4GB |
| **6. Forge** | Code gen, git ops, deploy pipeline | 11-12 | 8GB |
| **7. Herald** | Content creation, scheduling, review queue | 13-14 | 8GB |

### Phase order rationale:
- **Foundation first** — everything depends on the bus and base class
- **Watchdog second** — protects everything built after it
- **Sentinel third** — upgrades existing trading bot (not a rebuild)
- **Scout fourth** — makes Sentinel smarter via inter-agent research
- **Prometheus fifth** — by now there are 3 agents to orchestrate
- **Forge sixth** — can deploy and update all previous agents
- **Herald last** — lowest priority, builds on all other agents for triggers

---

## Risk & Mitigation

| Risk | Mitigation |
|------|------------|
| Token burn from chatty agents | Token tracking in `ai_client.py`, daily budgets per agent |
| Runaway agent crashes server | Watchdog monitors all agents, auto-restart + resource limits |
| Redis single point of failure | Watchdog monitors Redis, auto-restart. Agents queue locally if Redis is down |
| OAuth token expiry mid-task | Retry logic with token refresh in `ai_client.py` |
| Agent infinite loop | Timeout on all AI calls, max retries per task, Watchdog kills stuck processes |

---

## Future Considerations (Not in scope)

- **Cortex (Memory Agent):** Persistent memory/knowledge base shared across all agents
- **Multi-model support:** Route different agents to different AI models (e.g., cheaper model for Watchdog)
- **Agent marketplace:** Pluggable agent system where new agents can be dropped in
- **Mobile dashboard:** Prometheus dashboard as a PWA
- **Voice interface:** Whisper integration for voice commands via Telegram voice messages
